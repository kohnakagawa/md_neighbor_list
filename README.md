# SIMTization and SIMDization of Verlet list construction for molecular dynamics simulations

## Compile and run

### @ Haswell Xeon CPU

``` bash
$ make hsw
$ ./make_list_cpu_4x1.out
```

### @ Skylake Xeon CPU

``` bash
$ make skl
$ ./make_list_cpu_8x1.out
```

### @ Knights Landing

``` bash
$ make knl
$ ./make_list_mic_8x1.out
```

### @ NVIDIA GPU

``` bash
$ make gpu
$ ./make_list_gpu_warp_unroll_smem.out
```

## Benchmark results
CPUはsingleコア実行での実行時間 (単位はms)

|              | Teska K40t          | Tesla P100          | E5-2680 v3 @ 2.50GHz | Phi 7250 @ 1.40GHz  | Gold 6148 @ 2.40GHz |
| :----------: | :-----------------: | :-----------------: | :-----------------:  | :-----------------: | :-----------------: |
| density 1.0  | 995             | 469             | 12395           | 28546           | 8587            |
| density 0.5  | 432             | 223             | 3531            | 10220           | 2466            |
