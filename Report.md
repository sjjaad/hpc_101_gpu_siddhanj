# Homework 6 - GPU Programming

## Part 1: Benchmarking the performance of a GEMM kernel

1. Benchmark the performance of the kernel on an Artemis GPU node for the following values of N: 64, 1024 and 4096. Paste your SLURM output from running the code on Artemis below.

```
SLURM output log here
```

2. What GPU type did you use to run the code on Artemis.
3. Of the devices theoretical peak performance (FLOPS) how many flops per second  did you get? Report for each N. What is the trend with respect to N?

## Part 2: Benchmarking the performance of a wave simulation
1. Benchmark the performance of the wave simulation (`simulation_kernel!` in `wave.jl`) on a GPU on Artemis.

```
SLURM output log here
```

## Part 3: Implementing a wave simulation kernel for GPUs
1. Complete the GPU kernel `simulation_kernel!` and benchmark it on Artemis.
    - Your kernel will be validated against the reference implementation and must match.
    - You may find `animate_wave` useful for visualizing the wave evolution and debugging any synchronize issues.
```
SLURM output log here
```
2. What was the speedup you got from using a single kernel launch vs. multiple kernel launches? Show your calculation of the speedup.
3. What is the largest simulation you can run on on Artemis in less than a minute?How did you find this?

## Part 4: What did you learn?
