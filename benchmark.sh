#!/bin/bash
#SBATCH --job-name=gemm_benchmark
#SBATCH --output=gemm_benchmark_%j.out
#SBATCH --error=gemm_benchmark_%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=venkvis-a100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G

cd /home/siddhanj/hpc_101_gpu_siddhanj/
# Load Julia module (adjust version as needed)
module load julia

# Run benchmarks for each N value
echo "Starting benchmarks..."
echo "N=64"
julia --project=. benchmark_gemm.jl 64

echo "N=1024"
julia --project=. benchmark_gemm.jl 1024

echo "N=4096"
julia --project=. benchmark_gemm.jl 4096

echo "Benchmarks complete!"
