#!/usr/bin/env -S julia --color=yes --startup-file=no --project
# Helper script to run wave simulations
n = 512
sigma = 0.2
nt = 200

@info "Simulating wave on the GPU"
@time simulate_wave(n; sigma, nt)

@info "Simulating wave on the CPU"
@time simulate_wave(128; sigma, nt, gpu=false)

@info "My wave simulation on the GPU"
@time my_simulate_wave(n; sigma, nt)

