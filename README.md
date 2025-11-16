# HPC 101 GPU Programming

For this assignment, you'll be doing the following:

1. Benchmarking the performance of a GEMM kernel
2. Benchmarking the performance of a wave simulation
3. Implementing a wave simulation kernel for GPUs

You'll need to report your performance on a GPU on Artemis.

Detailed instructions are in the `gemm.jl` and `wave.jl` files.

## How to turn in your work
1. Use this repository as template to create your own, name it `hpc_101_gpu_uniquname` and clone it.
2. Work through the assignment, making and pushing at least one commit for each part (gemm, wave). Remember to include any addition files you create (for example SLURM job submission scripts).
3. Tabulate your results in `Report.md`.
4. Include your repository link in the report.
5. Submit the report to gradescope.


## Using Julia
To install [Julia](https://julialang.org/downloads/):
    `curl -fsSL https://install.julialang.org | sh`.

That's you need for this homework, but for a getting started guide see [Modern Julia Workflows](https://modernjuliaworkflows.org/writing/).

On Artemis, you can also use the `julia` provided via Lmod.

### Instantiate the project
1. Launch the REPL with `julia --project`
2. Press `]` to enter the package manager
3. Type `instantiate`
4. Press `Backspace` to exit the package manager


### Using System CUDA
This is critical when profiling on Artemis, as we need the CUDA drivers used by Julia to match what's being used to profile

1. Instantiate the project
2. Run `julia --project -e 'using CUDA; CUDA.set_runtime_version!(local_toolkit=true)'`


### Some useful commands:
1. If you don't know what a command does, type `?` followed by the command name and then press `Enter` and the julia REPL will print the documentation for that command.
2. To exit the REPL, type `exit()`.

While working on this assignment, you may find it helpful to be able to revise and run your code multiple times.
To do this:

1. Launch the REPL with `julia --project`
2. Start Revise with `using Revise`
3. Include the file you want to edit with `includet("wave.jl")`
4. Run a command, for example `my_simulate_wave(128; sigma=0.2, gpu=true, nt=200)`
5. Make changes to the code, the run `Revise.revise()`  to tell julia to recompile your code.
6. Run the command again to see the changes.

> I highly recommend using VSCode with the Remote SSH extension while working on this assignment.
> While you can edit code from the REPL with @edit, it'll launch your default editor which is almost certainly
> either vim or nano; this is unlikely to be what you want.

Do take this as an opportunity to learn how to code and debug on Artemis. If your workflow isn't working for you, ask for help.
