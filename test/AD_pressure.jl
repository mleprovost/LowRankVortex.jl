@testset "Test dpdx AD routine for Jacobian of the pressure field" begin
    atol = 1e-6
    atol_AD = 10*eps()
    ϵ = (eps())^(1/3)

    Nv = 20
    pos = randn(Nv) + im*rand(Nv)
    str = randn(length(pos))
    δ = 1e-4
    blobs = Vortex.Blob.(pos,str, δ)

    Ny = 15
    sensors = randn(ComplexF64, Ny)

    sys = deepcopy(blobs)
    sys0 = deepcopy(sys)

    Δs = 0.1
    ss = -3.0:Δs:3.0
    sensors = complex.(ss)
    Ny = length(ss)


    # Test jacobian_blobs_pressure

    idx = rand(1:Nv)
    # Test dpdx

    sys₋ = deepcopy(blobs)
    sys₊ = deepcopy(blobs)

    sys₋[idx] = Vortex.Blob(sys₋[idx].z - ϵ, sys₋[idx].S, sys₋[idx].δ)
    sys₊[idx] = Vortex.Blob(sys₊[idx].z + ϵ, sys₊[idx].S, sys₊[idx].δ)


    press₋ = pressure_AD(sensors, sys₋, 0.0)
    press₊ = pressure_AD(sensors, sys₊, 0.0)


    dpdz_AD, dpdzstar_AD = PotentialFlow.Elements.jacobian_position(x ->
              pressure_AD(sensors, x, 0.0), sys0)
    dpdx_FD = (press₊ - press₋)/2ϵ
    @test isapprox(dpdx_FD, 2*real.(dpdz_AD[:,idx]), atol = atol)
end

@testset "Test dpdy AD routine for Jacobian of the pressure field" begin
    atol = 1e-6
    atol_AD = 10*eps()
    ϵ = (eps())^(1/3)

    Nv = 20
    pos = randn(Nv) + im*rand(Nv)
    str = randn(length(pos))
    δ = 1e-4
    blobs = Vortex.Blob.(pos,str, δ)

    Ny = 15
    sensors = randn(ComplexF64, Ny)

    sys = deepcopy(blobs)
    sys0 = deepcopy(sys)

    Δs = 0.1
    ss = -3.0:Δs:3.0
    sensors = complex.(ss)
    Ny = length(ss)


    # Test jacobian_blobs_pressure

    idx = rand(1:Nv)
    # Test dpdx

    sys₋ = deepcopy(blobs)
    sys₊ = deepcopy(blobs)

    sys₋[idx] = Vortex.Blob(sys₋[idx].z - ϵ*im, sys₋[idx].S, sys₋[idx].δ)
    sys₊[idx] = Vortex.Blob(sys₊[idx].z + ϵ*im, sys₊[idx].S, sys₊[idx].δ)


    press₋ = pressure_AD(sensors, sys₋, 0.0)
    press₊ = pressure_AD(sensors, sys₊, 0.0)


    dpdz_AD, dpdzstar_AD = PotentialFlow.Elements.jacobian_position(x ->
              pressure_AD(sensors, x, 0.0), sys0)
    dpdy_FD = (press₊ - press₋)/2ϵ
    @test isapprox(dpdy_FD, (-2)*imag.(dpdz_AD[:,idx]), atol = atol)
end

@testset "Test dpdΓ AD routine for Jacobian of the pressure field" begin
    atol = 1e-6
    atol_AD = 10*eps()
    ϵ = (eps())^(1/3)

    Nv = 20
    pos = randn(Nv) + im*rand(Nv)
    str = randn(length(pos))
    δ = 1e-4
    blobs = Vortex.Blob.(pos,str, δ)

    Ny = 15
    sensors = randn(ComplexF64, Ny)

    sys = deepcopy(blobs)
    sys0 = deepcopy(sys)

    Δs = 0.1
    ss = -3.0:Δs:3.0
    sensors = complex.(ss)
    Ny = length(ss)


    # Test jacobian_blobs_pressure

    idx = rand(1:Nv)
    # Test dpdx

    sys₋ = deepcopy(blobs)
    sys₊ = deepcopy(blobs)

    sys₋[idx] = Vortex.Blob(sys₋[idx].z, sys₋[idx].S - ϵ, sys₋[idx].δ)
    sys₊[idx] = Vortex.Blob(sys₊[idx].z, sys₊[idx].S + ϵ, sys₊[idx].δ)


    press₋ = pressure_AD(sensors, sys₋, 0.0)
    press₊ = pressure_AD(sensors, sys₊, 0.0)


    dpdΓ_AD = PotentialFlow.Elements.jacobian_strength(x ->
              pressure_AD(sensors, x, 0.0), sys0)
    dpdΓ_FD = (press₊ - press₋)/2ϵ
    @test isapprox(dpdΓ_FD, dpdΓ_AD[:,idx], atol = atol)
end
