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

@testset "Dispatch on pressure routines" begin
	sens = range(-2.0,2.0,length=11) .- 0.0*im

	Nv = 3
	δ = 0.01
	config_data = VortexConfig(Nv, δ)

	z0 = complex(0.0)
	Γ0 = 1.0
	σx, σΓ = 1.0, 0.5

	zv, Γv = pointcluster(Nv,z0,Γ0,σx,σΓ)
	vort = Vortex.Blob.(zv,Γv,δ)

	p = analytical_pressure(sens,vort,config_data)
	p2 = analytical_pressure(sens,vort)
	@test p == p2

	x = lagrange_to_state_reordered(vort,config_data)

	H = zeros(Float64,length(sens),length(x))
	analytical_pressure_jacobian!(H,sens,vort,config_data)

	H2 = zeros(Float64,length(sens),length(x))
	analytical_pressure_jacobian!(H2,sens,vort)

	@test H == H2

	a1 = 0.5; b1 = 0.1; ccoeff = ComplexF64[0.5(a1+b1),0,0.5(a1-b1)]
  b = Bodies.ConformalBody(ccoeff,ComplexF64(0.0),0.0)

	Nsens = 10
	θsens = range(0,2π,length=Nsens+1)
	sens = exp.(im*θsens[1:end-1])

  config_data = VortexConfig(Nv, δ,body=b)

	zv = LowRankVortex.random_points_unit_circle(Nv,(1.05,1.4),(0,2π))
  vort = Vortex.Blob.(zv,Γv,δ)

	p = analytical_pressure(sens,vort,config_data)
	p2 = analytical_pressure(sens,vort,b)

	@test p == p2

	x = lagrange_to_state_reordered(vort,config_data)

	H = zeros(Float64,length(sens),length(x))
	analytical_pressure_jacobian!(H,sens,vort,config_data)

	H2 = zeros(Float64,length(sens),length(x))
	analytical_pressure_jacobian!(H2,sens,vort,b)

	@test H == H2

end
