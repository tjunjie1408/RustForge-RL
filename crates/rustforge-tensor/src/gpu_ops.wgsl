struct Parameters {
    length: u32,
    operation: u32,
    groups: u32,
    divisor: u32,
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
        default: { output[index] = a[index] / b[index]; }
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
