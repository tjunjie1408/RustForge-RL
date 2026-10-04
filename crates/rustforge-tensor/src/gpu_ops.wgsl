struct Parameters {
    length: u32,
    operation: u32,
    groups: u32,
    divisor: u32,
    first_columns: u32,
    second_columns: u32,
    offset: u32,
    padding: u32,
}

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> parameters: Parameters;

@compute @workgroup_size(256)
fn elementwise(@builtin(workgroup_id) group: vec3<u32>,
               @builtin(num_workgroups) grid: vec3<u32>,
               @builtin(local_invocation_index) lane: u32) {
    let group_index = group.y * grid.x + group.x;
    if (group_index >= parameters.groups) {
        return;
    }
    let index = group_index * 256u + lane;
    if (index >= parameters.length) {
        return;
    }
    switch parameters.operation {
        case 20u: {
            let width = parameters.first_columns + parameters.second_columns;
            let row = index / width;
            let column = index % width;
            if (column < parameters.first_columns) {
                output[index] = a[row * parameters.first_columns + column];
            } else {
                output[index] = b[row * parameters.second_columns + column - parameters.first_columns];
            }
        }
        case 21u: {
            let row = index / parameters.second_columns;
            let column = index % parameters.second_columns;
            output[index] = a[row * parameters.first_columns + parameters.offset + column];
        }
        case 0u: { output[index] = a[index] + b[index]; }
        case 1u: { output[index] = a[index] * b[index]; }
        // Select zero for NaN, matching Tensor::relu's f32::max.
        case 2u: { output[index] = select(0.0, a[index], a[index] > 0.0); }
        case 3u: { output[index] = a[index] * bitcast<f32>(parameters.divisor); }
        case 4u: { output[index] = b[index] * select(0.0, 1.0, a[index] > 0.0); }
        case 5u: { output[index] = a[0] * bitcast<f32>(parameters.divisor); }
        case 6u: { output[index] = bitcast<f32>(parameters.divisor); }
        case 7u: { output[index] = a[index] + b[index % parameters.divisor]; }
        case 8u: { output[index] = sqrt(a[index]) + bitcast<f32>(parameters.divisor); }
        case 9u: {
            var value = 0.0;
            for (var row = 0u; row < parameters.divisor; row += 1u) {
                value += a[row * parameters.length + index];
            }
            output[index] = value;
        }
        case 10u: { output[index] = a[index] / b[index]; }
        case 11u: { output[index] = exp(a[index]); }
        case 12u: {
            let offset = (index / parameters.divisor) * parameters.divisor;
            var maximum = a[offset];
            for (var col = 0u; col < parameters.divisor; col += 1u) {
                let value = a[offset + col];
                if ((bitcast<u32>(value) & 0x7f800000u) == 0x7f800000u) {
                    output[index] = bitcast<f32>(0x7fc00000u);
                    return;
                }
                maximum = max(maximum, value);
            }
            var total = 0.0;
            for (var col = 0u; col < parameters.divisor; col += 1u) {
                total += exp(a[offset + col] - maximum);
            }
            output[index] = (a[index] - maximum) - log(total);
        }
        case 13u: {
            let offset = (index / parameters.divisor) * parameters.divisor;
            var total = 0.0;
            for (var col = 0u; col < parameters.divisor; col += 1u) {
                total += b[offset + col];
            }
            output[index] = b[index] - exp(a[index]) * total;
        }
        case 14u: {
            output[index] = select(0.0, 1.0, (bitcast<u32>(a[index]) & 0x7f800000u) == 0x7f800000u);
        }
        case 15u: {
            let value = a[index];
            if (value == 0.0) { output[index] = bitcast<f32>(0xff800000u); }
            else if (value < 0.0 || ((bitcast<u32>(value) & 0x7fffffffu) > 0x7f800000u)) { output[index] = bitcast<f32>(0x7fc00000u); }
            else { output[index] = log(value); }
        }
        case 16u: {
            let value = a[index];
            if ((bitcast<u32>(value) & 0x7fffffffu) > 0x7f800000u) { output[index] = bitcast<f32>(0x7fc00000u); }
            else {
                let e = exp(-2.0 * abs(value));
                let magnitude = (1.0 - e) / (1.0 + e);
                output[index] = select(magnitude, -magnitude, (bitcast<u32>(value) & 0x80000000u) != 0u);
            }
        }
        case 17u: {
            var total = 0.0;
            for (var col = 0u; col < parameters.divisor; col += 1u) { total += a[index * parameters.divisor + col]; }
            output[index] = total;
        }
        case 18u: { output[index] = a[index / parameters.divisor]; }
        case 19u: {
            let value = a[index];
            output[index] = clamp(value, bitcast<f32>(parameters.divisor), b[index]);
            if ((bitcast<u32>(value) & 0x7fffffffu) > 0x7f800000u) { output[index] = value; }
        }
        default: { output[index] = bitcast<f32>(0x7fc00000u); }
    }
}

var<workgroup> partials: array<f32, 256>;

@compute @workgroup_size(256)
fn reduce(@builtin(workgroup_id) group: vec3<u32>,
          @builtin(num_workgroups) grid: vec3<u32>,
          @builtin(local_invocation_index) lane: u32) {
    let group_index = group.y * grid.x + group.x;
    // This condition is uniform for the entire workgroup.
    if (group_index >= parameters.groups) {
        return;
    }
    let index = group_index * 256u + lane;
    partials[lane] = 0.0;
    if (index < parameters.length) {
        partials[lane] = a[index];
    }
    workgroupBarrier();
    for (var stride = 128u; stride > 0u; stride /= 2u) {
        if (lane < stride) {
            partials[lane] += partials[lane + stride];
        }
        workgroupBarrier();
    }
    if (lane == 0u) {
        var value = partials[0];
        if (parameters.operation == 4u) {
            value /= f32(parameters.divisor);
        }
        output[group_index] = value;
    }
}
