# SIMTization and SIMDization of Verlet list construction for molecular dynamics simulations

## Benchmark results
CPUはsingleコア実行での実行時間

|              | Teska K40t          | Tesla P100          | E5-2680 v3 @ 2.50GHz | Phi 7250 @ 1.40GHz  |
| :----------: | :-----------------: | :-----------------: | :-----------------:  | :-----------------: |
| density 1.0  | 1464 [ms]           | 519 [ms]            | 12395 [ms]           | 30842 [ms]          |
| density 0.5  | 525 [ms]            | 223 [ms]            | 3531 [ms]            | 10199 [ms]          |
