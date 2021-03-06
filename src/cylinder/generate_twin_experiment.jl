export generate_data_cylinder_twin_experiment


"""
This routine advects the regularized point vortices stored in `source` (Lagrangian representation) from `t0` to `tf` using the Biot-Savart law,
and computes at each time the induced pressure at `config.ss`. An additional freestream of strength `config.U` can be applied i6f withfreestream = true.
The history of the state and pressure observations is stored in the folder `path` under the name `data_twin_experiment.jld`.
"""
function generate_data_cylinder_twin_experiment(source, t0, tf, config::VortexConfig, path::String)
    Δt = config.Δt
    source = deepcopy(source)

    x0 = cylinder_lagrange_to_state(source, config)

    ϵinfl = RecipeInflation([config.ϵX; config.ϵΓ])
    cachevels = allocate_velocity(source)

    n0 = ceil(Int64, t0/Δt) + 1
    J = ceil(Int64, (tf-t0)/Δt)
    Acycle = n0:n0+J-1

    Ny = length(config.ss)
    Nx = 3*config.Nv

    xt = zeros(size(x0, 1), J+1)
    x = deepcopy(x0)
    yt = zeros(Ny, J+1)
    tt = zeros(J+1)

    tt[1] = t0
    xt[:,1] .= deepcopy(x0)
    ϵinfl(view(xt,:,1), config)

    yt[:,1] .= pressure(config.ss, source; ϵ = config.δ)
    yt[:,1] .+= config.ϵY*randn(Ny)

    for i=1:length(Acycle)
        reset_velocity!(cachevels, source)
        
        cachevels = conj.(LowRankVortex.w(Elements.position(source),source; ϵ = config.δ))

        # Advect the set of vortices
        advect!(source, source, cachevels, config.Δt)
        xt[:,i+1] .= deepcopy(cylinder_lagrange_to_state(source, config))

        # Perturb the state
        ϵinfl(view(xt,:,i+1), config)
        source = deepcopy(cylinder_state_to_lagrange(xt[:,i+1], config))

        # Evaluate the observation operator at `config.ss`
        yt[:,i+1] .= pressure(config.ss, source; ϵ = config.δ)

        # Apply observation noise
        yt[:,i+1] .+= config.ϵY*randn(Ny)
        tt[i+1] = t0+i*config.Δt
    end

    # Store initial condition and the history of the state and observation variables into `data`.
    data = SyntheticData(tt, config.Δt, x0, xt, yt)

    # Save the fields of data into `data_twin_experiment.jld` in the folder `path`
    save(path*"data_cylinder_twin_experiment.jld", "tt", tt, "x0", x0, "xt", xt, "yt", yt)

    return data
end
