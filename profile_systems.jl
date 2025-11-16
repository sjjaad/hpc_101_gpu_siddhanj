# As we're profiling with NVIDIA's tools, we're restricted to CUDA
using CUDA
using NVTX

# Include the wave simulation kernels
include("wave.jl")

function profile_system(sim_name::String)
    if sim_name == "single"
        f = simulate_wave
    elseif fname == "multi"
        f = my_simulate_wave
    else
        error("Simulation name ($sim_name) not recognized")
    end

    # Simulation parameters
    n = 4096
    sigma = 0.2
    nt = 200

    f(n; sigma, nt=2) # First run to compile
    NVTX.@range "Simulation" begin
        f(n; sigma, nt) # Profile the kernel
    end

    return nothing
end


# This will profile the kernel if Julia is invoked on the script directly
if !isinteractive()
    fname = ARGS[1]
    profile_system(fname)
end


