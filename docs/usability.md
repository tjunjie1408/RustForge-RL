# Python, CLI and console workflows

This usability update preserves existing training entry points and adds shorter
ways to create environments, run experiments and inspect plans. Training math
and GPU checkpoint formats are unchanged.

## CLI

```bash
rustforge train sac -n 10 -o sac.jsonl
rustforge fit dqn -e grid-world -n 10
rustforge live PPO -e Pendulum-v1
rustforge plan sac
```

`fit`, `live`, `watch` alias `train`, `run`, `monitor`. Short flags are `-e`
(environment), `-n` (episodes), `-o` (output), `-d` (device). Algorithm, device
and environment values ignore case. Documented environment aliases include
`cart-pole`, `CartPole-v1`, `grid-world` and `Pendulum-v1`.

The environment default is `auto`: Pendulum for SAC/TD3, CartPole otherwise.
Explicit environments retain their meaning and incompatible combinations fail
before output mutation. This fixes `train sac`/`train td3` selecting an invalid
CartPole default. Programmatic `TrainArgs`/`RunArgs` callers can also choose
`Environment::Auto`; resolution happens before configuration or backend creation.

`plan` prints a `rustforge-training-plan-v1` JSON object to stdout and exits.
It reports algorithm, resolved environment, episodes, device, configuration
source/display, checkpoint paths, metric schema/descriptors and control
capabilities. It constructs a trainer description, not an agent: no adapter,
TTY, checkpoint read or output write is required. Requested GPU execution still
requires a GPU-enabled build. Invalid combinations exit unsuccessfully without
JSON on stdout. With `--resume`, configuration is explicitly marked as restored;
this command does not validate the file or pretend to know its saved values.

For agents: parse the JSON plan and exit status, then run `train` and read its
CSV/JSONL metric file. `run` and `monitor` require an interactive terminal.
The monitor accepts CSV logs; generic JSONL is intended for script readers.

## Python

```python
import rustforge as rf

names = rf.available_envs()
env = rf.make_env("CartPole-v1")
agent = rf.train("cartpole", episodes=10, output="experiment.csv")
action = agent.predict(env.reset(42))

gym_env = rf.make_env("pendulum", api="gymnasium")
obs, info = gym_env.reset(seed=42)
```

`make_env` defaults to native classes and works without Gymnasium. Set
`api="gymnasium"` to request the optional bridge explicitly. The factory has
separate typed overloads for native and Gymnasium returns. Canonical names come
from `available_envs()`: cartpole, gridworld, mountaincar,
mountaincarcontinuous, pendulum. Case and hyphens/underscores are ignored;
supported version aliases are CartPole-v1, MountainCar-v0,
MountainCarContinuous-v0 and Pendulum-v1. Unknown versions fail rather than
silently selecting a different environment.

`train` returns native CPU DQN. It accepts typed hyperparameters and a string or
pathlib output path. Default episode limits are 500 for CartPole, 100 for
GridWorld and 200 for MountainCar. Invalid counts/rates/discounts are rejected
before opening output; continuous environments and other algorithms receive an
explanation directing users to the CLI. `DQN.train` still supports its existing
signature/defaults and now accepts discrete environment aliases. Native classes,
`predict`, `train_steps` and Gymnasium `make` remain available. Nonfinite
observations are invalid arguments (`ValueError`), while training/log I/O failures
remain `RuntimeError`.

Environment reset seeds work as before. This convenience API does not promise
fully seeded DQN training, nor expose Python GPU/other-algorithm agents. Install
from this checkout to use the new API; existing PyPI wheels are unchanged until
a new package release.

## Console shortcuts

| Key | Action |
| --- | --- |
| Space or p | Pause/resume live training |
| q or Ctrl-C | Graceful stop; a second press forces stop |
| Enter after completion | Exit |
| ? or F1 | Help |
| Tab / Shift-Tab | Next/previous view |
| Left / Right | Chart range |
| Up / Down, PgUp / PgDn | Scroll |
| Home / End | First/latest |
| f, g, t | Follow, alert settings, palette |
| Esc | Dismiss dialog |

Monitor and live navigation share one map. Existing shortcuts remain available.
Repeat events still scroll, but cannot toggle pause repeatedly or escalate stop
just because a key is held. Alt/Control combinations do not trigger plain-letter
shortcuts; Ctrl-Shift-C does not request a stop. Footer/help advertise Space and
respect trainer state/capabilities.

## Deviations from Plan

- Interpreted Python "key binds" as API/binding ergonomics, alongside actual TUI
  shortcuts. Added convenience entry points while preserving native interfaces.
- CLI introspection uses a dedicated JSON `plan` command instead of adding a
  second execution mode to `train`; it is read-only and needs no terminal.
- CLI Auto is resolved at the command boundary rather than silently replacing
  an explicit incompatible environment.
- A pre-existing default-feature test requested GPU routing unconditionally;
  it is now feature-gated to match the tested capability. The new default-feature
  plan and rejection tests exercise the CPU build.
- Added a direct CLI dependency on the already-locked `serde_json` package for
  valid structured output. No new dependency versions were downloaded.
- Root README shrank from 864 to 134 lines; detailed benchmark methodology and
  Rust examples moved into linked guides. Python README shrank from 57 to 39
  lines. Unsupported reproducibility/performance generalizations were removed.

## Issue Resolution Progress

| Work | Issue ID | Result |
| --- | --- | --- |
| Algorithm-aware defaults and concise command aliases/flags | Unassigned | Complete, parsing and resolution checks |
| JSON training-plan discovery for coding agents | Unassigned | Complete, process stdout/exit checks and read-only GPU resume description |
| Typed Python factory/training convenience | Unassigned | Complete, native/Gym aliases, pathlib output and optional-dependency isolation |
| Python invalid-input/file preservation | Unassigned | Complete, native validation and nonfinite observation errors |
| Shared console shortcuts and safe repeat/modifier handling | Unassigned | Complete, mapping and rendered footer checks |
| Short README and linked details | Unassigned | Complete, source-checkout guidance and documentation links |

## Verification

| Check | Result | Delta from stage 11c |
| --- | --- | --- |
| Native workspace/all features, excluding Python | 1,114 passed, 190 ignored, 0 failed | +4 CLI tests, +2 TUI tests |
| Default-feature CLI/TUI suites | 134 passed, 0 failed | Auto routing and read-only planning work without GPU/TTY |
| Final TUI suite | 101 passed, 0 failed | Includes shortcut repeats, modifiers, shared navigation and rendered state/capabilities |
| Rebuilt Python extension and regression suite | 50 passed, 0 failed | +18 parametrized workflow/factory/validation checks |
| Workspace/all-targets/all-features Clippy | Passed, warnings denied | Excluding Python |
| Python Rust wrapper/all-targets Clippy | Passed, warnings denied | Native validation and aliases checked |
| Rust 1.75 workspace/all-targets/all-features | Passed | Excluding Python |
| Python syntax/bytecode checks | Passed | Source parses using Python 3.9 grammar |
| Formatting/whitespace and documentation links | Passed | Rustfmt, git diff check, local links |

The four CLI additions cover aliases/defaults, real-process JSON output, invalid
plans and feature-gated GPU resume discovery without adapter allocation. The two
TUI additions cover shared navigation and safe pause/stop repeat/modifier behavior;
existing footer tests now assert the advertised Space shortcut. Python additions
are five native factories, three alias/training cases, one pathlib output, six
invalid-output-preservation cases, Gym/API/version validation, optional Gymnasium
isolation and nonfinite observations. No adapter-required tests were added and
training math did not change. Logs are under `/tmp/rustforge-usability-*.log`.
