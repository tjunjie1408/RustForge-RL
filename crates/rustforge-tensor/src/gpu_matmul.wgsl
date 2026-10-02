struct Dimensions {
    m: u32,
    k: u32,
    n: u32,
    flags: u32,
}

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> dimensions: Dimensions;

fn left(row: u32, inner: u32) -> f32 {
    if ((dimensions.flags & 1u) != 0u) {
        return a[inner * dimensions.m + row];
    }
    return a[row * dimensions.k + inner];
}

fn right(inner: u32, column: u32) -> f32 {
    if ((dimensions.flags & 2u) != 0u) {
        return b[column * dimensions.k + inner];
    }
    return b[inner * dimensions.n + column];
}

// Every invocation participates in both barriers, including edge lanes.
var<workgroup> a_tile: array<f32, 64>;
var<workgroup> b_tile: array<f32, 64>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>,
        @builtin(local_invocation_id) local: vec3<u32>) {
    let row = id.y;
    let column = id.x;
    let lane = local.y * 8u + local.x;
    var sum = 0.0;
    for (var base = 0u; base < dimensions.k; base += 8u) {
        a_tile[lane] = 0.0;
        b_tile[lane] = 0.0;
        if (row < dimensions.m && base + local.x < dimensions.k) {
            a_tile[lane] = left(row, base + local.x);
        }
        if (column < dimensions.n && base + local.y < dimensions.k) {
            b_tile[lane] = right(base + local.y, column);
        }
        workgroupBarrier();
        let width = min(8u, dimensions.k - base);
        for (var inner = 0u; inner < width; inner += 1u) {
            sum += a_tile[local.y * 8u + inner] * b_tile[inner * 8u + local.x];
        }
        workgroupBarrier();
    }
    if (row < dimensions.m && column < dimensions.n) {
        output[row * dimensions.n + column] = sum;
    }
}

// Retained for numerical and performance comparisons on the same device.
@compute @workgroup_size(8, 8)
fn naive(@builtin(global_invocation_id) id: vec3<u32>) {
    let row = id.y;
    let column = id.x;
    if (row >= dimensions.m || column >= dimensions.n) {
        return;
    }
    var sum = 0.0;
    for (var inner = 0u; inner < dimensions.k; inner += 1u) {
        sum += left(row, inner) * right(inner, column);
    }
    output[row * dimensions.n + column] = sum;
}
