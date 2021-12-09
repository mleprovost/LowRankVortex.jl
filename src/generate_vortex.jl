export generate_vortex, generate_patch

# Routine to generate the truth data
function generate_vortex(source, t0, tf, config::VortexConfig, path::String)
    Δt = config.Δt
    source = deepcopy(source)

    x0 = lagrange_to_state(source, config)
    freestream = PotentialFlow.Freestream(config.U)
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

    yt[:,1] .= pressure(config.ss, source, freestream, t0)
    yt[:,1] .+= config.ϵY*randn(Ny)


    for i=1:length(Acycle)
        reset_velocity!(cachevels, source)
        self_induce_velocity!(cachevels, source, t0 + (i-1)*config.Δt)
        induce_velocity!(cachevels, source, freestream, t0 + (i-1)*config.Δt)

        # Advect the system
        advect!(source, source, cachevels, config.Δt)
        xt[:,i+1] .= deepcopy(lagrange_to_state(source, config))

        # Perturb the state
        ϵinfl(view(xt,:,i+1), config)
        source = deepcopy(state_to_lagrange(xt[:,i+1], config.zs, config))
        # Compute pressure field
        yt[:,i+1] .= pressure(config.ss, source, freestream, t0+i*config.Δt)
        yt[:,i+1] .+= config.ϵY*randn(Ny)
        tt[i+1] = t0+i*config.Δt
    end

    # Generate the observation at the same time
    data = SyntheticData(tt, x0, xt, yt)

    save(path*"data_test.jld", "tt", tt, "x0", x0, "xt", xt, "yt", yt)

    return data
end
