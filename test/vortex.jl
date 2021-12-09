
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
