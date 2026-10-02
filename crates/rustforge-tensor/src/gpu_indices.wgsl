struct Parameters { length: u32, operation: u32, groups: u32, columns: u32 }
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> indices: array<u32>;
// Typed output stores indices directly or the bit pattern of a floating result.
// No integer ever passes through floating-point arithmetic or conversions.
@group(0) @binding(2) var<storage, read_write> output: array<u32>;
@group(0) @binding(3) var<uniform> parameters: Parameters;
fn ordered_key(value: f32) -> i32 {
    let bits = bitcast<i32>(value);
    return bits ^ bitcast<i32>(bitcast<u32>(bits >> 31u) >> 1u);
}
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) group: vec3<u32>, @builtin(num_workgroups) grid: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    let group_index = group.y * grid.x + group.x;
    if (group_index >= parameters.groups) { return; }
    let index = group_index * 256u + lane;
    if (index >= parameters.length) { return; }
    if (parameters.operation == 0u) {
        var best = 0u;
        var key = ordered_key(input[index * parameters.columns]);
        for (var col = 0u; col < parameters.columns; col += 1u) {
            let value = input[index * parameters.columns + col];
            if ((bitcast<u32>(value) & 0x7f800000u) == 0x7f800000u) {
                output[index] = 0xffffffffu;
                return;
            }
            let candidate = ordered_key(value);
            if (candidate >= key) { best = col; key = candidate; }
        }
        output[index] = best;
    } else if (parameters.operation == 1u) {
        let col = indices[index];
        if (col >= parameters.columns) { output[index] = 0x7fc00000u; }
        else { output[index] = bitcast<u32>(input[index * parameters.columns + col]); }
    } else {
        let row = index / parameters.columns;
        let col = index % parameters.columns;
        output[index] = bitcast<u32>(select(0.0,input[row],col == indices[row]));
    }
}
