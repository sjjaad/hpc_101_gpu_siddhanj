# Package for writing GPU kernels
# The programming model is similar to CUDA's but with the goal of being
# device agnostic. Similar efforts within the C/C++ world are OpenCL, SCYCL,
# and Kokkos. PyOpenCL provides Python bindings for interacting with OpenCL.
# The alternative would be to write CUDA kernels in C/C++ directly, but for
# this exercise, that's not the point.
using KernelAbstractions

# Package for adapting converting various types to GPU-compatible types
# This like x.to(device) in PyTorch, but works for arbitrary data structures
using Adapt: adapt

# Backend for NVIDIA GPUs
using CUDA: CUDA, CuArray, CUDABackend

# NVTX for annotating HOST-side functions
using NVTX

# Helper functions for benchmarking
include("utils.jl")

# To run this on Artemis: `julia --project wave.jl`

# Helper function to wrap the torus indexing
@inline torus_index(i, o, n) = mod(i + o - 1, n) + 1

# This the stencil function that will be used to update the wave's state
# There's not much to it, but for the sake of the exercise the details
# don't matter, the key think is you'll use to update the wave's state
function update_pressure(p_center, p_prev, p_left, p_right, p_down, p_up)
    delta = p_left - 2 * p_center + p_right
    delta += p_down - 2 * p_center + p_up
    delta *= 0.1f0
    return 2 * p_center - p_prev + delta
end


"""
    wave_kernel!(p_next, p_prev)

For this example, we're going to model a 2D wave equation on a torus.
The stencil is:

    p_t+1[i, j] = 2 p_t-1[i, j] - p_t[i, j] + f(p_t-1)

Where f acts on the neighboring cells to the left, right, down, and up.

To save on memory, we'll do the following:
- Store the current value of the wave in `p_next` (p_t+1)
- Store the previous value of the wave in `p_prev` (p_t)

Each thread will compute the next value of the wave and store it in
local memory. We'll then wait for all threads to finish before
writing the new value of the wave to `p_next` and the previous value
to `p_prev`.

This may seem a bit contrived, but it's actually pretty common stragety
to reduce the memory footprint of a simulation. Maintaining seperate
compites for p_next, p_cur and p_prev would be a 50% increase in memory,
which would limited the size of the simulation that could fit onto a GPU.
Bigger simulations mean more science (larger domain or more detail) so it's
generally worth it.

This is not a very good wave simulation, but that's not really the point.
"""
@kernel function wave_kernel!(p_cur::AbstractArray{T}, p_prev::AbstractArray{T}) where {T <: AbstractFloat}
    i, j = @index(Global, NTuple)

    # Get the values of the neighboring cells
    n = size(p_prev, 1)
    m = size(p_prev, 2)
    p = p_cur[i, j]
    p_down = p_prev[i, torus_index(j, -1, m)]
    p_up = p_prev[i, torus_index(j, 1, m)]
    p_left = p_prev[torus_index(i, -1, n), j]
    p_right = p_prev[torus_index(i, 1, n), j]

    # Compute the next value of the wave
    p_next = update_pressure(p, p_prev[i, j], p_left, p_right, p_down, p_up)

    # Wait for all threads to finish before updating p_next and p_prev
    @synchronize()
    p_prev[i, j] = p_cur[i, j]
    p_cur[i, j] = p_next

end

NVTX.@annotate function step_wave!(p_next, p_prev)
    # Some abstraction stuff to make the kernel run on multiple backends
    # i.e. CUDA (NVIDIA), Metal (Apple), ROCm (AMD) or oneAPI (Intel)
    backend = KernelAbstractions.get_backend(p_next)
    kernel! = wave_kernel!(backend)

    # Launch the kernel
    kernel!(p_next, p_prev; ndrange=size(p_next))
    return p_next
end

# Helper function to initialize the wave as a gaussian centered in the domain
NVTX.@annotate function init_wave(n = 32, sigma = 0.1)
    p = Matrix{Float32}(undef, n, n)
    c = (n + 1)/2
    sigma *= n
    for idx in eachindex(IndexCartesian(), p)
        i, j = idx.I
        dist = hypot(i - c, j - c)
        p[idx] = exp(-dist^2 / (2*sigma)) / sqrt(2 * pi * sigma^2)
    end
    return p
end


"""
    simulate_wave(n = 32; sigma = 0.1, nt=200)

Simulates a wave by running `nt` steps as separate kernel launches

"""
function simulate_wave(n = 32; sigma = 0.1, nt=200, gpu=true)
    to, backend = get_backend(gpu)

    # Initialize the wave and move to the GPU
    p_cur = init_wave(n, sigma)
    NVTX.@range "Copying to GPU" begin
        p_prev = adapt(to, copy(p_cur))
        p_cur = adapt(to, p_cur)
    end

    # Update the wave's state over nt steps
    trace = zeros(Float32, n, n, nt)
    trace[:, :, 1] .= adapt(Array, p_cur)
    for i in axes(trace, 3)
        trace[:, :, i] .= adapt(Array, p_cur)
        step_wave!(p_cur, p_prev)
    end

    return trace
end


"""
    simulation_kernel!(p_cur, p_prev, nt)

Now it's your turn to implement the wave simulation

Given the initial conditions `p_cur` and `p_prev`, simulate `nt` steps

"""
@kernel function simulation_kernel!(p_cur, p_prev, nt)
    i, j = @index(Global, NTuple)

    # Get the dimensions of the grid
    n = size(p_prev, 1)
    m = size(p_prev, 2)

    # Initialize the current pressure value
    p = p_cur[i, j]

    # Advance the wave for `nt` steps
    for step in 1:nt
        # TODO: Complete operations in this loop 
    end
end

# Launcher for the simulation kernel
NVTX.@annotate function my_simulate_wave_launch(p_next::AbstractArray{Float32}, p_prev::AbstractArray{Float32}, nt::Int64)
    # Setup the kernel
    backend = KernelAbstractions.get_backend(p_next)
    kernel! = simulation_kernel!(backend)

    # Launch the kernel
    kernel!(p_next, p_prev, nt; ndrange=size(p_next))
    return p_next
end

function my_simulate_wave(n = 32; sigma = 0.1, nt=200, gpu=true)
    # Initialize the wave and move to the GPU
    to, backend = get_backend(gpu)
    p_cur = init_wave(n, sigma)
    
    NVTX.@range "Copying to GPU" begin
        p_prev = adapt(to, copy(p_cur))
        p_cur = adapt(to, p_cur)
    end

    # Call your kernel
    p_final = my_simulate_wave_launch(p_cur, p_prev, nt)
    NVTX.@range "Copying to CPU" begin
        p_final = adapt(Array, p_final)
    end

    # Validate the result
    NVTX.@range "Validating Simulation" begin
        p_expected = simulate_wave(n; sigma, nt)[:, :, end]
        p_expected = adapt(Array, p_expected)
        @assert p_final ≈ p_expected atol=1e-3
    end
end
