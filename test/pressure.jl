@testset "Check pressure calculation" begin

    function compute_ẋ!(ẋ, x, t)
        # Zero the velocity
        reset_velocity!(ẋ)
        # Compute the self-induced velocity of the system
        self_induce_velocity!(ẋ, x, t)
    end

    atol_AD  = 100*eps()
    atol  = 1e-6

    Δt = sqrt(eps())

    δ = 1e-4

    Δs = 0.1
    ss = -3.0:Δs:3.0
    sensors = complex.(ss)
    Ny = length(ss)

    # Test with nothing
    testsys = vcat(Vortex.Blob(im, 0.0, δ),
                   Vortex.Blob(-im, 0.0, δ))

    press = pressure(sensors, deepcopy(testsys), 0.0)
    # Expected value 0.0 everywhere
    @test isapprox(press, zeros(Ny), atol = atol)



    # Test with one vortex (parallel translation)
    testsys = vcat(Vortex.Blob(im, 1.0, δ),
         Vortex.Blob(-im, -1.0, δ))
    press_FD = pressure_FD(sensors, deepcopy(testsys), 0.0, Δt)

    testϕ₋ = real.(complexpotential(sensors, testsys))
    testẋ = allocate_velocity(testsys)
    compute_ẋ!(testẋ, testsys, 0.0)
    testsys₊ = deepcopy(testsys)
    advect!(testsys₊, testsys₊, testẋ, Δt)
    testϕ₊ = real.(complexpotential(sensors, testsys₊))
    @test isapprox(press_FD, -0.5*abs2.(induce_velocity(sensors, testsys , 0.0)) - (testϕ₊-testϕ₋)/Δt,
                   atol = atol)

    # Test with vortices and sources
    testsys = (vcat(Vortex.Blob(im, 1.0, δ),
                    Vortex.Blob(-im, -1.0, δ)),
               vcat(Source.Blob(2*im,  1.0, δ),
                    Source.Blob(-2*im,  1.0, δ)))
    press_FD = pressure_FD(sensors, deepcopy(testsys), 0.0, Δt)

    testϕ₋ = real.(complexpotential(sensors, (testsys[1:2])))
    testẋ = allocate_velocity(testsys)
    compute_ẋ!(testẋ, testsys, 0.0)
    testsys₊ = deepcopy(testsys)
    advect!(testsys₊, testsys₊, testẋ, Δt)
    testϕ₊ = real.(complexpotential(sensors, (testsys₊[1:2])))
    @test isapprox(press_FD, (-0.5*abs2.(induce_velocity(sensors, testsys, 0.0)) - (testϕ₊-testϕ₋)/Δt), atol = atol)


    # Test with vortices and sources and freestream
    testsys = (vcat(Vortex.Blob(im, 1.0, δ),
               Vortex.Blob(-im, -1.0, δ)),
               vcat(Source.Blob(2*im,  1.0, δ),
               Source.Blob(-2*im,  1.0, δ)))
    press_FD = pressure_FD(sensors, deepcopy(testsys), 0.0, Δt)

    testϕ₋ = real.(complexpotential(sensors, (testsys[1:2])))
    testẋ = allocate_velocity(testsys)
    compute_ẋ!(testẋ, testsys, 0.0)
    testsys₊ = deepcopy(testsys)
    advect!(testsys₊, testsys₊, testẋ, Δt)
    testϕ₊ = real.(complexpotential(sensors, (testsys₊[1:2])))
    @test isapprox(press_FD, (-0.5*abs2.(induce_velocity(sensors, testsys, 0.0)) - (testϕ₊-testϕ₋)/Δt), atol = atol)

    # Frozen advected pressure field by a pair of two vortices
    testsys = (vcat(Vortex.Blob(im, 1.0, 0.0),
           Vortex.Blob(-im, -1.0, 0.0)),
           vcat(Source.Blob(2*im,  0.0, 0.0),
           Source.Blob(-2*im,  0.0, 0.0)))

    press_analytical(x) = -0.5*(1/(π*(x^2+1.0)))^2 + 1/(4*π^2*(x^2 + 1.0))
    press = pressure(complex(-10.0:0.01:10.0), deepcopy(testsys), 0.0)
    press_FD = pressure_FD(complex(-10.0:0.01:10.0), deepcopy(testsys), 0.0, Δt);

    @test isapprox(press, press_analytical.(-10.0:0.01:10.0), atol = atol_AD)
    @test isapprox(press_FD, press_analytical.(-10.0:0.01:10.0), atol = atol)
end
