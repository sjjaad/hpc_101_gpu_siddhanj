"""
    benchmark_gemm(f, n; T = Float32)

Helper function to benchmark a GEMM function `f` on a matrix of size `n`
To use: `benchmark_gemm(gemm!, 64)` at the REPL
1. Warms up the function (ensures it is compiled)
2. Times the function
3. Validates the result

Unlike last time, we're using `T` to specify the element type of the matrices.
"""
function benchmark_gemm(f, n; T = Float32, repeats=5)
    # Construct our input matrices on the CPU
    A = rand(T, n, n)
    B = rand(T, n, n)
    C = zeros(T, n, n)

    to, backend = get_backend()

    # Move data to the device
    A = adapt(to, A)
    B = adapt(to, B)
    C = adapt(to, C)

    # We need to wait for the GPU to finish before moving on
    # this isn't normally necessary as subsequent kernel launches
    # would block until all inputs were ready, but because 
    # a) we don't actually use the outputs and b) are trying to
    # measure the time of a single kernel, we need to synchronize
    function timed_kernel()
        f(C, A, B) # Launch the kernel on the GPU
        KernelAbstractions.synchronize(backend) # Wait for GPU to finish
    end

    # First run to compile
    timed_kernel()

    # Run the kernel multiple times reporting the average time
    walltime = 0
    for i in 1:repeats
        stats = @timed timed_kernel()
        @info "Wall time for run $i: $(stats.time) seconds"
        walltime += stats.time
    end
    @info "Average wall time: $(walltime / repeats) seconds"

    return nothing
end

"""
    to, backend = get_backend(use_gpu=true)

Helper function to get the backend for the current device or not if
`use_gpu` is false
"""
function get_backend(use_gpu=true)
    if CUDA.functional()
        to = CuArray
        backend = CUDABackend()
    else
        to = Array
        backend = CPU()
        @warn "No GPU backend detected, falling back to CPU"
    end
    return to, backend
end
