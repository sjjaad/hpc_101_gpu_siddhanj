#!/usr/bin/env -S julia --color=yes --startup-file=no
# Helper script to run the GEMM benchmark

include("gemm.jl")
n = parse(Int, ARGS[1])
@info "Running benchmark for N = $n"
benchmark_gemm(gemm!, n)

