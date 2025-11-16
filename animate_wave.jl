using GLMakie  # You'll need to install this package (`pkg> add GLMakie`)
# On Artemis, set the env var JULIA_GLMAKIE_BACKEND to be headless
# or modify the function to generate static plots using CarioMakie instead

"""
    animate_wave(trace)
Give the trace of the wave simulation, animate it

This isn't required, but it's a nice way to see the wave evolve
and debug the simulation
"""
function animate_wave(trace)
    f = Figure()
    ax = Axis(f[1, 1])
    data = Observable(trace[:, :, 1])
    heatmap!(ax, data; colorrange=extrema(trace[:, :, 1]))
    nt = size(trace, 3)
    for i in 1:nt
        data[] = trace[:, :, i]
        display(f)
        sleep(3/nt)
    end
    return f
end

