# 0.2.0 release readiness

Local validation recorded on 2026-10-05. No release tag or publication was created.

## Fixed

- Workspace and seven internal lockfile packages use 0.2.0; external dependency
  versions are unchanged. Python distribution versions inherit Cargo.
- Release CLI builds enable `gpu` on Linux x86_64, macOS arm64 and Windows
  x86_64. Renamed artifacts must pass version, GPU plan discovery and CPU
  training checks. A mismatched version tag fails before artifact publication.
- GitHub Release publication waits for all three CLI builds and checks. PRs
  and manual CLI workflow runs only build, test and retain artifacts.
- Wheel checks install into a new virtual environment and run outside the
  checkout, verify installed version/import location, then execute Python tests.
- CHANGELOG includes the GPU softmax fix and physical benchmark evidence.

## Executed locally

| Check | Result |
|---|---|
| Windows release CLI build with `--locked --features gpu` | Passed |
| Renamed Windows artifact: version, GPU plan, two CPU episodes | Passed |
| Mismatched `v0.1.0` tag with 0.2.0 artifact | Correctly rejected |
| Windows abi3 wheel build and fresh installation | Passed |
| Installed-wheel Python tests, CPython 3.14 | 50 passed |
| Source distribution build | Passed |
| GPU-enabled CLI regression tests | 37 passed, 8 adapter tests ignored |
| Workspace default-feature strict Clippy | Passed |
| GPU-enabled CLI strict Clippy | Passed |
| Rust formatting, workflow YAML parsing, diff whitespace | Passed |

GPU plan discovery checks compiled capability without allocating an adapter.
Two CPU episodes can remain in replay warm-up: CSV v1 represents absent loss
with NaN; reward, epsilon and counters must remain finite. The artifact smoke
test does not establish GPU execution parity or training convergence.

## Before publication

Cross-platform artifact verification is **NOT RUN** at the time of this record.
Linux/macOS builds and wheel tests must
pass in GitHub Actions before claiming release readiness. macOS universal2
contains both architectures, but its current runner tests only Apple Silicon.

Run the Release workflow without a tag and the Python wheels workflow with
`publish-testpypi=false`, or run both through a PR. Review all three CLI and
four wheel matrix results before creating a matching `v0.2.0` tag. Publishing
requires separate authorization and the configured PyPI Trusted Publisher.

Local build outputs are ignored under `target/release-readiness/`; they are
not checked into the repository. Linux/macOS downloaded CLI files require
executable permission (`chmod +x`) before running.
