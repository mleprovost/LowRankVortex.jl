
@testset "Validate vortex and symmetric_vortex" begin
    atol = 1000*eps()

    Nv = 20
    Nx = 3*Nv
    Ne = 5

    sensors = -11.0:0.5:10.0

    config = let Nv = Nv,
                 ss = sensors, Δt = 5e-3, δ = 1e-1,
                 ϵX = 1e-3, ϵΓ = 1e-3,
                 β = 1.0,
                 ϵY = 1e-2
        VortexConfig(Nv, ss, Δt, δ, ϵX, ϵΓ, β, ϵY)
    end
    # Test vortex routine
    x = rand(Nx)
    X = repeat(x, 1, Ne)
    X0 = deepcopy(X)
    t0 = 1.8
    sys = state_to_lagrange(x, config)
    sys0 = deepcopy(sys)
    sys₊ = deepcopy(sys)
    vels = self_induce_velocity(sys, t0)

    advect!(sys₊, sys0, vels, config.Δt)

    cachevels = allocate_velocity(state_to_lagrange(zeros(Nx), config))
    X1, t1 = vortex(deepcopy(X0), t0, 0, Nx, cachevels, config)

    for i=1:Ne
        @test isapprox(X1[:,i], lagrange_to_state(sys₊, config), atol = atol)
    end

    # Test that the symmetry is correclty enforced
    X = rand(Nx, Ne)
    X0 = deepcopy(X)
    t0 = 1.5

    cachevels = allocate_velocity(state_to_lagrange(zeros(Nx), config))

    X1, t1 = vortex(deepcopy(X0), t0, 0, Nx, cachevels, config)

    X2, t2 = symmetric_vortex(deepcopy(X0), t0, 0, Nx, cachevels, config)

    @test isapprox(X1, X2, atol = atol)
    @test isapprox(t1, t0 + config.Δt, atol = atol)
    @test isapprox(t2, t0 + config.Δt, atol = atol)
end
