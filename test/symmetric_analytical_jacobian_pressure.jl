

@testset "Test symmetric_analytical_jacobian_position for point vortices" begin

    atol = 1000*eps()

    Nv = 10
    Nx = 3*Nv

    x = rand(Nx)
    x0 = deepcopy(x)

    xsensors = collect(-2.0:0.5:10)
    sensors = complex(xsensors)
    Ny = length(sensors)

    U = complex(0.0)
    freestream = Freestream(U);

    config = let Nv = Nv,
             U = U,
             ss = sensors, Δt = 1e-3, δ = 5e-2,
             ϵX = 1e-3, ϵΓ = 1e-4,
             β = 1.0,
             ϵY = 1e-16
            VortexConfig(Nv, U, ss, Δt, δ, ϵX, ϵΓ, β, ϵY)
    end

    sys = state_to_lagrange(x, config; isblob = false)
    @test typeof(sys[1][1])<:Vortex.Point{Float64, Float64}

    sys = vcat(sys...)

    dpdz, dpdzstar = analytical_jacobian_position(sensors, sys, freestream, 0.0)

    dpdzsym, dpdzstarsym = symmetric_analytical_jacobian_position(xsensors, sys, 0.0)

    @test isapprox(dpdz, dpdzsym, atol = atol)
    @test isapprox(dpdzstar, dpdzstarsym, atol = atol)
end


@testset "Test symmetric_analytical_jacobian_position for regularized vortices" begin

    atol = 1000*eps()

    Nv = 10
    Nx = 3*Nv

    x = rand(Nx)
    x0 = deepcopy(x)

    xsensors = collect(-2.0:0.5:10)
    sensors = complex(xsensors)
    Ny = length(sensors)

    U = complex(0.0)
    freestream = Freestream(U);

    config = let Nv = Nv,
             U = U,
             ss = sensors, Δt = 1e-3, δ = 0.1,
             ϵX = 1e-3, ϵΓ = 1e-4,
             β = 1.0,
             ϵY = 1e-16
            VortexConfig(Nv, U, ss, Δt, δ, ϵX, ϵΓ, β, ϵY)
    end

    sys = state_to_lagrange(x, config; isblob = true)
    @test typeof(sys[1][1])<:Vortex.Blob{Float64, Float64}

    sys = vcat(sys...)

    dpdz, dpdzstar = analytical_jacobian_position(sensors, sys, freestream, 0.0)

    dpdzsym, dpdzstarsym = symmetric_analytical_jacobian_position(xsensors, sys, 0.0)

    @test isapprox(dpdz, dpdzsym, atol = atol)
    @test isapprox(dpdzstar, dpdzstarsym, atol = atol)
end


@testset "Test symmetric_analytical_jacobian_strength for point vortices" begin

    atol = 1000*eps()

    Nv = 10
    Nx = 3*Nv

    x = rand(Nx)
    x0 = deepcopy(x)

    xsensors = collect(-2.0:0.5:10)
    sensors = complex(xsensors)
    Ny = length(sensors)

    U = complex(0.0)
    freestream = Freestream(U);

    config = let Nv = Nv,
             U = U,
             ss = sensors, Δt = 1e-3, δ = 5e-2,
             ϵX = 1e-3, ϵΓ = 1e-4,
             β = 1.0,
             ϵY = 1e-16
            VortexConfig(Nv, U, ss, Δt, δ, ϵX, ϵΓ, β, ϵY)
    end

    sys = state_to_lagrange(x, config; isblob = false)
    @test typeof(sys[1][1])<:Vortex.Point{Float64, Float64}

    sys = vcat(sys...)

    dpdS, dpdSstar = analytical_jacobian_strength(sensors, sys, freestream, 0.0)

    dpdSsym, dpdSstarsym = symmetric_analytical_jacobian_strength(xsensors, sys, 0.0)

    @test isapprox(dpdS, dpdSsym, atol = atol)
    @test isapprox(dpdSstar, dpdSstarsym, atol = atol)
end

@testset "Test symmetric_analytical_jacobian_strength for regularized vortices" begin

    atol = 1000*eps()

    Nv = 10
    Nx = 3*Nv

    x = rand(Nx)
    x0 = deepcopy(x)

    xsensors = collect(-2.0:0.5:10)
    sensors = complex(xsensors)
    Ny = length(sensors)

    U = complex(0.0)
    freestream = Freestream(U);

    config = let Nv = Nv,
             U = U,
             ss = sensors, Δt = 1e-3, δ = 5e-2,
             ϵX = 1e-3, ϵΓ = 1e-4,
             β = 1.0,
             ϵY = 1e-16
            VortexConfig(Nv, U, ss, Δt, δ, ϵX, ϵΓ, β, ϵY)
    end

    sys = state_to_lagrange(x, config; isblob = true)
    @test typeof(sys[1][1])<:Vortex.Blob{Float64, Float64}

    sys = vcat(sys...)

    dpdS, dpdSstar = analytical_jacobian_strength(sensors, sys, freestream, 0.0)

    dpdSsym, dpdSstarsym = symmetric_analytical_jacobian_strength(xsensors, sys, 0.0)

    @test isapprox(dpdS, dpdSsym, atol = atol)
    @test isapprox(dpdSstar, dpdSstarsym, atol = atol)
end

@testset "Test routine symmetric_analytical_jacobian_pressure" begin
    atol = 1000*eps()

    Nv = 10
    Nx = 3*Nv

    x = rand(Nx)
    x0 = deepcopy(x)

    xsensors = collect(-2.0:0.5:10)
    sensors = complex(xsensors)
    Ny = length(sensors)

    U = complex(0.0)
    freestream = Freestream(U);

    config = let Nv = Nv,
             U = U,
             ss = sensors, Δt = 1e-3, δ = 5e-2,
             ϵX = 1e-3, ϵΓ = 1e-4,
             β = 1.0,
             ϵY = 1e-16
            VortexConfig(Nv, U, ss, Δt, δ, ϵX, ϵΓ, β, ϵY)
    end

    sys = state_to_lagrange(x, config; isblob = true)
    @test typeof(sys[1][1])<:Vortex.Blob{Float64, Float64}

    sys = vcat(sys...)

    Jfull = analytical_jacobian_pressure(sensors, sys, freestream, 0.0)
    Jsymfull = symmetric_analytical_jacobian_pressure(xsensors, sys, 0.0)

    @test isapprox(Jfull, Jsymfull, atol = atol)

    # Test only on a subset of the vortices

    J = analytical_jacobian_pressure(sensors, sys, freestream, 1:Nv, 0.0)
    Jsym = symmetric_analytical_jacobian_pressure(xsensors, sys, 1:Nv, 0.0)

    @test isapprox(J[:,1:3*Nv], Jfull[:,1:3*Nv], atol = atol)
    @test isapprox(Jsym[:,1:3*Nv], Jfull[:,1:3*Nv], atol = atol)

    @test norm(J[:,3*Nv+1:end]) < atol
    @test norm(Jsym[:,3*Nv+1:end]) < atol
end
