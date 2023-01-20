export localized_senkf_symmetric_vortexassim


function localized_senkf_symmetric_vortexassim(algo::StochEnKF, Lxy, Lyy, X, tspan::Tuple{S,S}, config::VortexConfig, data::SyntheticData; withfreestream::Bool = false, P::Parallel = serial) where {S<:Real}

    # Define the inflation parameters
    ϵX = config.ϵX
    ϵΓ = config.ϵΓ
    β = config.β
    ϵY = config.ϵY
    ϵx = RecipeInflation([ϵX; ϵΓ])
    ϵmul = MultiplicativeInflation(β)

    Ny = size(config.ss,1)

    # Set the time step between two assimilation steps
    Δtobs = algo.Δtobs
    # Set the time step for the time marching of the dynamical system
    Δtdyn = algo.Δtdyn
    t0, tf = tspan
    step = ceil(Int, Δtobs/Δtdyn)

    n0 = ceil(Int64, t0/Δtobs) + 1
    J = (tf-t0)/Δtobs
    Acycle = n0:n0+J-1

    # Array dimensions
    Nypx, Ne = size(X)
    Nx = Nypx - Ny
    ystar = zeros(Ny)

    # Cache variable for the velocities
    cachevels = allocate_velocity(state_to_lagrange(X[Ny+1:Ny+Nx,1], config))

    # Define the observation operator
    h(x, t) = measure_state_symmetric(x, t, config; withfreestream =  withfreestream)
    # Define an interpolation function in time and space of the true pressure field
    press_itp = CubicSplineInterpolation((LinRange(real(config.ss[1]), real(config.ss[end]), length(config.ss)),
                                   t0:data.Δt:tf), data.yt, extrapolation_bc =  Line())

    yt(t) = press_itp(real.(config.ss), t)
    Xf = Array{Float64,2}[]
    push!(Xf, copy(state(X, Ny, Nx)))

    Xa = Array{Float64,2}[]
    push!(Xa, copy(state(X, Ny, Nx)))

    ### specific to localization ###
    dyy = dobsobs(config)
    Gyy = gaspari.(dyy./Lyy)
    #######

    # Run the ensemble filter
    for i=1:length(Acycle)

        # Forecast step
        @inbounds for j=1:step
            tj = t0+(i-1)*Δtobs+(j-1)*Δtdyn
            X, _ = symmetric_vortex(X, tj, Ny, Nx, cachevels, config; withfreestream = withfreestream)
        end

        push!(Xf, deepcopy(state(X, Ny, Nx)))

        # Get the true observation ystar
        ystar .= yt(t0+i*Δtobs)

        # Perform state inflation
        ϵmul(X, Ny+1, Ny+Nx)
        ϵx(X, Ny, Nx, config)

        # Filter state
        if algo.isfiltered == true
            @inbounds for i=1:Ne
                x = view(X, Ny+1:Ny+Nx, i)
                x .= filter_state!(x, config)
            end
        end

        # Evaluate the observation operator for the different ensemble members
        observe(h, X, t0+i*Δtobs, Ny, Nx; P = P)

        # Generate samples from the observation noise
        ϵ = algo.ϵy.σ*randn(Ny, Ne) .+ algo.ϵy.m

        # Form the perturbation matrix for the state
        Xpert = (1/sqrt(Ne-1))*(X[Ny+1:Ny+Nx,:] .- mean(X[Ny+1:Ny+Nx,:]; dims = 2)[:,1])
        # Form the perturbation matrix for the observation
        HXpert = (1/sqrt(Ne-1))*(X[1:Ny,:] .- mean(X[1:Ny,:]; dims = 2)[:,1])
        # Form the perturbation matrix for the observation noise
        ϵpert = (1/sqrt(Ne-1))*(ϵ .- mean(ϵ; dims = 2)[:,1])

        # Kenkf = Xpert*HXpert'*inv(HXpert*HXpert'+ϵpert*ϵpert')

        # Apply the Kalman gain based on the representers
        # Burgers G, Jan van Leeuwen P, Evensen G. 1998 Analysis scheme in the ensemble Kalman
        # filter. Monthly weather review 126, 1719–1724. Solve the linear system for b ∈ R^{Ny × Ne}:

        Σy = (HXpert*HXpert' + ϵpert*ϵpert')
        localizedΣy =  Gyy .* Σy
        # Apply localization
        b = localizedΣy\(ystar .- (X[1:Ny,:] + ϵ))

        # Update the ensemble members according to:
        # x^{a,i} = x^i - Σ_{X,Y}b^i, with b^i =  Σ_Y^{-1}(h(x^i) + ϵ^i - ystar)
        dxy = dstateobs(X, Ny, Nx, config)
        Gxy = gaspari.(dxy./Lxy)
        localizedΣxy = (Xpert*HXpert')

        for J=1:config.Nv
            for i=-2:0
               localizedΣxy[3*J+i,:] .*= Gxy[J,:]
            end
        end
        view(X,Ny+1:Ny+Nx,:) .+= localizedΣxy*b

        # Filter state
        if algo.isfiltered == true
            @inbounds for i=1:Ne
                x = view(X, Ny+1:Ny+Nx, i)
                x .= filter_state!(x, config)
            end
        end

        push!(Xa, deepcopy(state(X, Ny, Nx)))
    end

    return Xf, Xa
end
