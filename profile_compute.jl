# As we're profiling with NVIDIA's tools, we're restricted to CUDA
using CUDA

# Include the gemm.jl kernels
include("gemm.jl")

function profile_compute(fname::String, n::Int)
    if fname == "gemm!"
        f = gemm!
    elseif fname == "mul!"
        f = mul!
    else
        error("Function name ($fname) not recognized")
    end

    # Allocate the matrices
    A = rand(Float32, n, n)
    B = rand(Float32, n, n)
    C = zeros(Float32, n, n)

    # Move the matrices to the device
    to, backend = get_backend()
    A = adapt(to, A)
    B = adapt(to, B)
    C = adapt(to, C)

    f(C, A, B) # First run to compile
    CUDA.@profile(f(C, A, B)) |> display # Profile the kernel

    return nothing
end


# This will profile the kernel if julia is invoked on the script directly
if !isinteractive()
    fname = ARGS[1]
    n = length(ARGS) > 1 ? parse(Int, ARGS[2]) : 1024
    profile_compute(fname, n)
end
