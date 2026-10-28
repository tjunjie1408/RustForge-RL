# Dependency maintenance assessment

Assessment for v0.3.0, 2026-10-28. Priorities below describe project work order;
they do not change the classifications assigned by RustSec.

| Priority | Dependency | Assessment and next step |
| --- | --- | --- |
| Release requirement | PyO3, LRU, Anyhow, Rand, Tokio | Upgrade the affected dependencies and retain minimum version constraints. Validate the Rust 1.88 workspace, Python stable ABI, terminal integration and GPU paths. |
| Medium, follow-up | Bincode 1.3.3 | Keep the existing codec for this release. Plan a maintained replacement with a versioned file format and an explicit legacy-reader compatibility policy. |
| Low, follow-up | Paste 1.0.15 | Reassess during a WGPU/Metal upgrade. It is a procedural macro pulled in through `wgpu-hal -> metal` for Apple targets. |

## Bincode

[RUSTSEC-2025-0141](https://rustsec.org/advisories/RUSTSEC-2025-0141.html)
is an `INFO Unmaintained` advisory with no patched version. It states that the
maintainers regard 1.3.3 as complete. This is not a guarantee against future
vulnerabilities.

The library uses Bincode for saved model parameters and training snapshots.
`rustforge-nn::serialization::load_parameters` explicitly reads a caller-selected
file, decodes it, and checks parameter counts and shapes before assigning model
parameters. It also accepts the historical headerless format. Keeping this codec
avoids breaking those files during the security dependency upgrade.

The generic parameter loader reads and decodes before checking model shapes;
those checks are not an allocation limit or a sandbox for hostile files. Use
trusted files. If accepting files across an untrusted boundary becomes a product
requirement, bounded decoding and input validation become a release requirement
for that use case. The maintenance advisory alone does not require changing the
current file format in v0.3.0.

## Paste

[RUSTSEC-2024-0436](https://rustsec.org/advisories/RUSTSEC-2024-0436.html)
is also an `INFO Unmaintained` advisory with no patched version. The all-target,
all-feature dependency tree resolves Paste through Metal 0.27 and WGPU HAL 0.19.
It expands Rust source during compilation; it is not a runtime deserializer or
part of the Windows Vulkan runtime. Replacing it requires coordinating the
upstream graphics dependency rather than substituting an unrelated runtime
package. Keep the lockfile, monitor advisories, and test Apple builds in CI.

## Release decision

Neither maintenance warning is a known vulnerability in the audit database used
for this release. They remain visible maintenance risks and do not block v0.3.0.
Future advisories, a new untrusted-input requirement, or platform build failures
can change that decision. Cross-platform release and wheel results are owned by
GitHub Actions; local Windows checks do not establish those results.
