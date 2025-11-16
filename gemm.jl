# Package for writing GPU kernels
# The programming model is similar to CUDA's but with the goal of being
# device agnostic. Similar efforts within the C/C++ world are OpenCL, SCYCL,
# and Kokkos. PyOpenCL provides Python bindings for OpenCL.
# The alternative would be to write CUDA kernels in C/C++ directly, but for
# this exercise, that's not the point.
using KernelAbstractions

# Package for adapting converting various types to GPU-compatible types
# This is like x.to(device) in PyTorch, but works for arbitrary
# data structures
using Adapt: adapt

# Backend NVIDIA GPUs
using CUDA: CUDA, CuArray, CUDABackend

# Helper functions for benchmarking
include("utils.jl")

# How to run this on Artemis:
# 1. Clone the code to Artemis
# 2. Install julia on Artemis (again use juliaup) or use ``module load julia``.
# 3. Instantiate the project
# 4. Submit a job to run the following `julia --project gemm.jl N`
#    where N is the size of the matrix you want to benchmark

"""
    gemm_kernel!(C, A, B)

This is the GPU kernel that will be executed on device. Unlike CPU code where
a single process executes the code, GPU code is executed in parallel on multiple
threads with are grouped into grids, with grid then grouped into blocks.

Some key things to note:

1. Synchronization between threads is limited to the grid level.
   Blocks are launched independently of each other and can not be synchronized.
2. Shared Memory is visible to all threads in a grid, but only consistent after
   calling `@synchronize`. We'll see this in action later.

Once again, we'll start with a simple GEMM kernel, where each thread computes
a single element of the output matrix `C`.
"""
@kernel function gemm_kernel!(C, A, B)
    # We use the @index macro to the index of the current thread
    i, j = @index(Global, NTuple)

    # Standard loop over k
    s = zero(eltype(C))
    for k in 1:size(A, 2)
        s += A[i, k] * B[k, j]
    end

    # Store the result back to the output matrix
    C[i, j] = s

    # Note we're not returning anything. Kernel only act on preallocated global memory
    # and return nothing, their effects are limited to modifying the contents of
    # the device's global memory.
end

"""
    gemm!(C, A, B)

The next notable difference to writing CPU code is that GPU kernels need to be launched
from the host device (i.e. the CPU). So we also need to write the function to do this.

Thankfully the launch code tends to be pretty boilerplate. One thing to keep in mind,
launching a kernel does take some time so generally a single monolithic kernel will
be faster than launching many small kernels. We'll see this in action later.
"""
function gemm!(C, A, B)
    # Make sure the matrices are the right size
    @assert size(A, 2) == size(B, 1) "Dimension mismatch: A, B"
    @assert size(A, 1) == size(C, 1) "Dimension mismatch: A, C"
    @assert size(B, 2) == size(C, 2) "Dimension mismatch: B, C"

    # Some abstraction stuff to make the kernel run on multiple backends
    # i.e. CUDA (NVIDIA), Metal (Apple), ROCm (AMD) or oneAPI (Intel)
    backend = KernelAbstractions.get_backend(A)
    kernel! = gemm_kernel!(backend)

    # Launch the kernel
    # The ndrange specifics the size of the grid of threads to launch
    # One of the nice things about CUDA's programming model is that
    # grids be up to 3D, making operations like f(A[i, j,k]) easy, just get
    # launch a grid of size(a) and the index into the array with the grid index.
    #
    # Note: KernelAbstractions.jl calls this an ``ndrange`` and blocks a ``workgroup``.
    # Like everything, all GPU providers (NVIDIA, Intel, AMD) want to call it something else. 
    # See https://juliagpu.github.io/KernelAbstractions.jl/stable/quickstart/#Terminology
    # The key thing is that blocks don't have shared memory, but grids do.
    kernel!(C, A, B; ndrange=size(C))
end

