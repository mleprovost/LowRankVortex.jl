
@testset "Test symmetric version of pressure for point vortices" begin
    atol = 1000*eps()
    Nv = 10
    zv = randn(Nv) + im *(0.05 .+ rand(Nv))
    Sv = randn(Nv)

    points₊ = Vortex.Point.(zv, Sv)
    points₋ = Vortex.Point.(conj(zv), -Sv)

    sys = deepcopy(vcat(points₊, points₋))

    xsensors = collect(-2.0:0.1:2.0)
    sensors = complex(xsensors)
    Ny = size(xsensors, 1)

    press_sym = symmetric_pressure(xsensors, points₊, 0.0)

    press = pressure_AD(sensors, deepcopy(sys), 0.0)
    @test isapprox(press_sym, press, atol = atol)
end

@testset "Test symmetric version of pressure for regularized point vortices" begin
    atol = 1000*eps()
    Nv = 10
    zv = randn(Nv) + im *rand(Nv)
    Sv = randn(Nv)
    δ = 0.5
    @show "dah"

    blobs₊ = Vortex.Blob.(zv, Sv, δ*ones(Nv))
    blobs₋ = Vortex.Blob.(conj(zv), -Sv, δ*ones(Nv))

    sys = deepcopy(vcat(blobs₊, blobs₋))


    xsensors = collect(-2.0:0.1:2.0)
    sensors = complex(xsensors)
    Ny = size(xsensors, 1)

    press_sym = symmetric_pressure(xsensors, blobs₊, 0.0)

    press = pressure_AD(sensors, deepcopy(sys), 0.0)
    @test isapprox(press_sym, press, atol = atol)
end

@testset "Test measure_state_symmetric and measure_state" begin
    Nv = 15
    Nx = 3*Nv

    atol = 1000*eps()

    xsensors = collect(-2.0:0.5:10)
    sensors = complex(xsensors)

    config = let Nv = Nv,
         ss = sensors, Δt = 1e-3, δ = 5e-2,
         ϵX = 1e-3, ϵΓ = 1e-4,
         β = 1.0,
         ϵY = 1e-16
         VortexConfig(Nv, ss, Δt, δ, ϵX, ϵΓ, β, ϵY)
    end

    sys = state_to_lagrange(x, config)
    press_truth = pressure(sensors, sys, 0.0)

    press = measure_state(x, 0.0, config)

    press_symmetric = measure_state_symmetric(x, 0.0, config)

    @test isapprox(press, press_truth, atol = atol)
    @test isapprox(press_symmetric, press_truth, atol = atol)
end
