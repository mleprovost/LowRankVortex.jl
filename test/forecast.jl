@testset "Validate vortex and symmetric_vortex" begin
    atol = 1000*eps()

    Nv = 20
    Nx = 3*Nv
    Ne = 5

    U = complex(0.0)

    config = let Nv = Nv,
                 U = U,
                 Δt = 5e-3, δ = 1e-1
        VortexConfig(Nv, U, Δt, δ;body=LowRankVortex.OldFlatWall)
    end

    # Test vortex routine
    x = rand(Nx)
    X0 = BasicEnsembleMatrix(repeat(x, 1, Ne))
    t0 = 1.8
    sys = state_to_lagrange(x, config)
    sys0 = deepcopy(sys)
    sys₊ = deepcopy(sys)
    vels = self_induce_velocity(sys, t0)

    advect!(sys₊, sys0, vels, config.Δt)

    fdata = VortexForecast(config)
    X = deepcopy(X0)
    forecast!(X,t0,config.Δt,fdata)

    for i=1:Ne
        @test isapprox(X(i), lagrange_to_state(sys₊, config), atol = atol)
    end

    # Test that the symmetry is correclty enforced
    X = rand(Nx, Ne)
    X0 = BasicEnsembleMatrix(X)
    t0 = 1.5

    fsdata = SymmetricVortexForecast(config)
    X1 = deepcopy(X0)
    X2 = deepcopy(X0)
    forecast!(X1,t0,config.Δt,fdata)
    forecast!(X2,t0,config.Δt,fsdata)

    @test isapprox(X1, X2, atol = atol)

end

@testset "Validate vortex and symmetric_vortex with freestream" begin
    atol = 1000*eps()

    Nv = 20
    Nx = 3*Nv
    Ne = 5

    U = rand(ComplexF64)
    freestream = Freestream(U)

    config = let Nv = Nv,
                 U = U,
                 Δt = 5e-3, δ = 1e-1
        VortexConfig(Nv, U, Δt, δ;body=LowRankVortex.OldFlatWall)
    end
    config_nofs = let Nv = Nv,
                 U = complex(0.0),
                 Δt = 5e-3, δ = 1e-1
        VortexConfig(Nv, U, Δt, δ;body=LowRankVortex.OldFlatWall)
    end

    # Test vortex routine
    x = rand(Nx)
    X0 = BasicEnsembleMatrix(repeat(x, 1, Ne))
    t0 = 1.8
    sys = state_to_lagrange(x, config)
    sys0 = deepcopy(sys)
    sys₊ = deepcopy(sys)
    vels = self_induce_velocity(sys, t0)
    induce_velocity!(vels, sys, freestream, t0)

    advect!(sys₊, sys0, vels, config.Δt)

    fdata = VortexForecast(config)
    fdata_nofs = VortexForecast(config_nofs)

    X1 = deepcopy(X0)
    X2 = deepcopy(X0)

    forecast!(X1,t0,config.Δt,fdata)
    forecast!(X2,t0,config.Δt,fdata_nofs)

    for i = 1:Ne
      x = X2(i)
      x[config.state_id["vortex x"]] .+= real(config.U)*config.Δt
      x[config.state_id["vortex y"]] .+= imag(config.U)*config.Δt
    end


    for i=1:Ne
        @test isapprox(X1(i), lagrange_to_state(sys₊, config), atol = atol)
        @test isapprox(X2(i), lagrange_to_state(sys₊, config), atol = atol)
    end

    # Test that the symmetry is correclty enforced
    X = rand(Nx, Ne)
    X0 = BasicEnsembleMatrix(X)
    t0 = 1.5

    fdata = VortexForecast(config)
    fsdata = SymmetricVortexForecast(config)
    X1 = deepcopy(X0)
    X2 = deepcopy(X0)
    forecast!(X1,t0,config.Δt,fdata)
    forecast!(X2,t0,config.Δt,fsdata)

    @test isapprox(X1, X2, atol = atol)

end
