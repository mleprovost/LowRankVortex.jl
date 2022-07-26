
@testset "Test lagrange_to_state and state_to_lagrange" begin

    atol = 1000*eps()

    sensors = -11.0:0.5:10.0

    Nv = 10
    Nx = 3*Nv
    U = randn(ComplexF64)
    # Define the state
    x = rand(Nx)
    x0 = deepcopy(x)

    config = let Nv = Nv,
                 U = U,
                 ss = sensors, Δt = 5e-3, δ = 1e-1,
                 ϵX = 1e-3, ϵΓ = 1e-3,
                 β = 1.0,
                 ϵY = 1e-2
        VortexConfig(Nv, U, ss, Δt, δ, ϵX, ϵΓ, β, ϵY)
    end

    blobs = state_to_lagrange(x0, config)
    blobs0 = deepcopy(blobs)
    for i=1:Nv
        bi₊ = blobs[1][i]
        bi₋ = blobs[2][i]

        @test isapprox(bi₋.z, conj(bi₊.z), atol = atol)
        @test isapprox(bi₋.S, -bi₊.S, atol = atol)
    end

    xbis = lagrange_to_state(blobs0, config)

    @test isapprox(xbis, x, atol = atol)
end


@testset "Test cylinder_lagrange_to_state and cylinder_state_to_lagrange" begin

    atol = 1000*eps()

    sensors = complex.(-11.0:0.5:10.0)

    Nv = 10
    Nx = 3*Nv
    U = randn(ComplexF64)
    # Define the state

    config = let Nv = Nv,
                 U = U,
                 ss = sensors, Δt = 5e-3, δ = 1e-1,
                 ϵX = 1e-3, ϵΓ = 1e-3,
                 β = 1.0,
                 ϵY = 1e-2
        VortexConfig(Nv, U, ss, Δt, δ, ϵX, ϵΓ, β, ϵY)
    end

    blobs0 = create_random_vortices(Nv; σ = config.δ)
    xblobs = cylinder_lagrange_to_state(blobs0, config)
    blobs = cylinder_state_to_lagrange(xblobs, config)

    for i=1:config.Nv
        @test isapprox(blobs0[i].z, xblobs[3*i-2] + im*xblobs[3*i-1], atol = atol)
        @test isapprox(blobs0[i].z, blobs[i].z, atol = atol)

        @test isapprox(blobs0[i].S, xblobs[3*i], atol = atol)
        @test isapprox(blobs0[i].S, blobs[i].S, atol = atol)

        @test isapprox(blobs[i].δ, config.δ, atol = atol)
    end

end
