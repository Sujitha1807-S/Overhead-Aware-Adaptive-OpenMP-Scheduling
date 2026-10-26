# Overhead-Aware Adaptive OpenMP Scheduling

## Project Overview

This project implements an overhead-aware adaptive scheduling mechanism
within the LB4OMP OpenMP runtime.

The objective is to determine whether changing the scheduling policy
is worthwhile by comparing estimated performance benefits against
estimated adaptation overhead.

The implementation uses runtime measurements rather than machine learning.

## Base Framework

This project extends LB4OMP:

https://github.com/unibas-dmi-hpc/LB4OMP

The original LB4OMP framework and its associated license remain
attributed to the original authors.

## Scheduling Policies

The benchmark evaluates:

- STATIC
- DYNAMIC
- GUIDED
- Overhead-Aware Adaptive Scheduling

The adaptive implementation currently selects between STATIC and GUIDED.

## Adaptive Scheduling Mechanism

1. Begin execution with STATIC scheduling.
2. Collect baseline execution measurements.
3. Explore GUIDED scheduling.
4. Estimate potential performance savings.
5. Compare estimated savings with adaptation overhead.
6. Retain STATIC or switch to GUIDED based on the decision.

The current implementation uses a configurable reuse horizon.

## Implementation

Primary modified runtime file:

`runtime/src/kmp_dispatch.cpp`

Additional runtime modifications:

- `runtime/src/kmp.h`
- `runtime/src/kmp_sched.cpp`
- `runtime/src/kmp_stats_timing.h`
- `runtime/src/reinforcement-learning/utils/utils.cpp`
- `runtime/cmake/LibompCheckLinkerFlag.cmake`

Benchmark:

`final_cao_benchmark.cpp`

Results:

`final_cao_results.csv`

## Experimental Environment

- Operating System: Ubuntu 26.04 LTS
- Processor: AMD Ryzen 5 7500F
- CPU: 6 cores / 12 threads
- Compiler: Clang 21.1.8
- OpenMP threads: 4

## Experimental Results

Median execution time (milliseconds):

| Workload | STATIC | DYNAMIC | GUIDED | Adaptive |
|---|---:|---:|---:|---:|
| Uniform | 118.018 | 117.445 | 116.708 | 117.524 |
| Irregular | 118.585 | 118.467 | 117.394 | 118.378 |
| Increasing | 195.761 | 117.292 | 115.966 | 135.105 |

The adaptive implementation improves performance over STATIC for
the increasing workload but does not outperform the best fixed GUIDED
policy in the evaluated workloads.

All 60 recorded benchmark executions passed correctness verification.

## Limitations

- Exploration overhead is not fully accounted for.
- The adaptive decision currently considers STATIC and GUIDED.
- The reuse horizon is fixed.
- Evaluation uses synthetic workloads.
- Experiments were conducted on one hardware configuration.
- The adaptive implementation does not outperform the best fixed policy.

## Future Work

- Improve total adaptation overhead estimation.
- Reduce exploratory scheduling overhead.
- Evaluate additional scheduling policies.
- Introduce dynamic reuse-horizon selection.
- Evaluate real-world applications and additional hardware.
- Compare overhead-aware and overhead-unaware adaptation directly.

## Project Status

Working prototype implemented and experimentally evaluated.
