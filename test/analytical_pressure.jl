@testset "Test the Jacobian of the pressure with respect to the position for a set of point vortices" begin

	atol_AD  = 1000*eps()

	Ny = 15
	sensors = randn(ComplexF64, Ny)


	# Test for a set of vortices
	Ns = 10
	zs = randn(ComplexF64, Ns)
	Ss = randn(Ns)

	sys = Vortex.Point.(zs, Ss)
	U = 0.0*im
	freestream = Freestream(U)

	dpdzAD, dpdzstarAD = PotentialFlow.Elements.jacobian_position(x->pressure_AD(sensors, x, 0.0), sys)

	dpdz, dpdzstar = analytical_jacobian_position(sensors, sys, freestream, 0.0)

	@test isapprox(dpdz, dpdzAD, atol = atol_AD)
	@test isapprox(dpdzstar, dpdzstarAD, atol = atol_AD)
end

@testset "Test the Jacobian of the pressure with respect to the strength for a set of point vortices" begin

	atol_AD  = 1000*eps()

	Ny = 15
	sensors = randn(ComplexF64, Ny)


	# Test for a set of vortices
	Ns = 10
	zs = randn(ComplexF64, Ns)
	Ss = randn(Ns)

	sys = Vortex.Point.(zs, Ss)
	U = 0.0*im
	freestream = Freestream(U)

	dpdSAD = PotentialFlow.Elements.jacobian_strength(x->pressure_AD(sensors, x, 0.0), sys)

	dpdS, dpdSstar = analytical_jacobian_strength(sensors, sys, freestream, 0.0)

	# Here ∂p/∂ΓL = -i(∂p/∂SL - ∂p/∂S̄L) with SL = QL - iΓL, as S = Γ + iQ is used as the convention in PotentialFlow.jl
	@test isapprox(-im*(dpdS - dpdSstar), dpdSAD, atol = atol_AD)
end

@testset "Test the Jacobian of the pressure with respect to the position for a set of regularized vortices" begin

	atol_AD  = 100*eps()

	Ny = 15
	sensors = randn(ComplexF64, Ny)

	δ = 1e-1


	# Test for a set of vortices
	Ns = 10
	zs = randn(ComplexF64, Ns)
	Ss = randn(Ns)

	sys = Vortex.Blob.(zs, Ss, δ*ones(Ns))
	U = 0.0*im
	freestream = Freestream(U)

	dpdzAD, dpdzstarAD = PotentialFlow.Elements.jacobian_position(x->pressure_AD(sensors, x, 0.0), sys)

	dpdz, dpdzstar = analytical_jacobian_position(sensors, sys, freestream, 0.0)

	@test isapprox(dpdz, dpdzAD, atol = atol_AD)
	@test isapprox(dpdzstar, dpdzstarAD, atol = atol_AD)
end

@testset "Test the Jacobian of the pressure with respect to the strength for a set of regularized vortices" begin

	atol_AD  = 1000*eps()

	Ny = 15
	sensors = randn(ComplexF64, Ny)

	δ = 1e-1

	# Test for a set of vortices
	Ns = 10
	zs = randn(ComplexF64, Ns)
	Ss = randn(Ns)

	sys = Vortex.Blob.(zs, Ss, δ*ones(Ns))
	U = 0.0*im
	freestream = Freestream(U)

	dpdSAD = PotentialFlow.Elements.jacobian_strength(x->pressure_AD(sensors, x, 0.0), sys)

	dpdS, dpdSstar = analytical_jacobian_strength(sensors, sys, freestream, 0.0)

	# Here ∂p/∂ΓL = -i(∂p/∂SL - ∂p/∂S̄L) with SL = QL - iΓL, as S = Γ + iQ is used as the convention in PotentialFlow.jl
	@test isapprox(-im*(dpdS - dpdSstar), dpdSAD, atol = atol_AD)
end

@testset "Validate analytical_jacobian_pressure for a set of singularities" begin
    atol_AD = 1000*eps()
	Nv = 10
	zv = randn(Nv) + im *rand(Nv)
	Sv = randn(Nv)

	sensors = randn(ComplexF64, 15)

	sys = Vortex.Point.(zv, Sv)
	U = 0.0*im
	freestream = Freestream(U)

	∂p = analytical_jacobian_pressure(sensors, sys, freestream, 0.0)

	∂pAD = AD_symmetric_jacobian_pressure(sensors, sys, 0.0)

	@test isapprox(∂p, ∂pAD, atol = atol_AD)
end

@testset "Validate analytical_jacobian_pressure for a set of regularized singularities" begin
    atol_AD = 1000*eps()
	Nv = 10
	zv = randn(Nv) + im *rand(Nv)
	Sv = randn(Nv)
    δ = 1e-1

	sensors = randn(ComplexF64, 15)

	sys = Vortex.Blob.(zv, Sv, δ*ones(Nv))
	U = 0.0*im
	freestream = Freestream(U)

	∂p = analytical_jacobian_pressure(sensors, sys, freestream, 0.0)

	∂pAD = AD_symmetric_jacobian_pressure(sensors, sys, 0.0)

	@test isapprox(∂p, ∂pAD, atol = atol_AD)
end
