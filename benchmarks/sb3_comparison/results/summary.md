# SB3 Benchmark — DQN on CartPole (10 runs, 50,000 env-step budget)

## Speed (training call, CPU)

| Framework | Train time (s) | Throughput (steps/sec) |
|---|---|---|
| rustforge | 6.29 ± 2.72 | 13,680 ± 5,949 |
| sb3 | 96.87 ± 35.93 | 616 ± 280 |

**RustForge throughput speedup:** 22.2x

## Learning (steps to reach CartPole-v1 solved threshold = 475)

| Framework | Mean steps to solved | Runs solved |
|---|---|---|
| rustforge | n/a | 0/10 |
| sb3 | n/a | 0/10 |
