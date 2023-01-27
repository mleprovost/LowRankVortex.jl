using Statistics

@testset "Test lagrange_to_state and state_to_lagrange" begin

    atol = 1000*eps()

    Nv = 10
    Nx = 3*Nv
    U = randn(ComplexF64)
    # Define the state
    x = rand(Nx)
    x0 = deepcopy(x)

    config = let Nv = Nv,
                 U = U,
                 Δt = 5e-3, δ = 1e-1
        VortexConfig(Nv, U, Δt, δ,body=LowRankVortex.OldFlatWall)
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

    Nv = 10
    Nx = 3*Nv
    U = randn(ComplexF64)
    # Define the state

    config = let Nv = Nv,
                 U = U,
                 Δt = 5e-3, δ = 1e-1
        VortexConfig(Nv, U, Δt, δ)
    end

    blobs0 = create_random_vortices(Nv; σ = config.δ)
    xblobs = lagrange_to_state(blobs0, config)
    blobs = state_to_lagrange(xblobs, config)

    for i=1:config.Nv
        @test isapprox(blobs0[i].z, xblobs[3*i-2] + im*xblobs[3*i-1], atol = atol)
        @test isapprox(blobs0[i].z, blobs[i].z, atol = atol)

        @test isapprox(blobs0[i].S, xblobs[3*i], atol = atol)
        @test isapprox(blobs0[i].S, blobs[i].S, atol = atol)

        @test isapprox(blobs[i].δ, config.δ, atol = atol)
    end

end

@testset "Randomizer routines" begin
    rmin, rmax = 1.05, 1.4
    zv = LowRankVortex.random_points_unit_circle(1000,(rmin,rmax),(0,2π));
    rmedian = 1.0 + sqrt((rmin-1)*(rmax-1))

    @test isapprox(median(abs.(zv)),rmedian,atol=5e-3)
    @test minimum(abs.(zv)) > 1
    @test maximum(abs.(zv)) < 1.5

    zv = LowRankVortex.random_points_plane(1000,-2,2,-4,5)
    @test maximum(real(zv)) <= 2 && minimum(real(zv)) >= -2
    @test maximum(real(zv)) <= 5 && minimum(real(zv)) >= -4
end

@testset "Lagrange-state maps" begin
  Nv = 3
  δ = 0.01
  config_data = VortexConfig(Nv, δ)

  z0 = complex(0.0)
  Γ0 = 1.0

  zv = z0 .+ LowRankVortex.random_points_plane(Nv,-2.0,2.0,-2.0,2.0)
  Γv = -Γ0 .+ 2*Γ0*rand(Nv)
  vort = Vortex.Blob.(zv,Γv,δ)

  x = lagrange_to_state(vort,config_data)
  @test x[1] == real(vort[1].z) && x[2] == imag(vort[1].z)
  @test x[4] == real(vort[2].z) && x[5] == imag(vort[2].z)
  @test x[3] == vort[1].S
  @test x[6] == vort[2].S

  vort_2 = state_to_lagrange(x,config_data)
  @test Elements.position(vort_2) ≈ Elements.position(vort)
  @test LowRankVortex.strength(vort_2) ≈ LowRankVortex.strength(vort)

  a1 = 0.5; b1 = 0.1; ccoeff = ComplexF64[0.5(a1+b1),0,0.5(a1-b1)]
  b = Bodies.ConformalBody(ccoeff,ComplexF64(0.0),0.0)

  config_data = VortexConfig(Nv, δ,body=b)

  zv = LowRankVortex.random_points_unit_circle(Nv,(1.05,1.4),(0,2π))
  vort = Vortex.Blob.(zv,Γv,δ)

  x = lagrange_to_state(vort,config_data)
  @test x[1] == log(abs(vort[1].z)-1.0) && x[2] == angle(vort[1].z)*abs(vort[1].z)
  @test x[4] == log(abs(vort[2].z)-1.0) && x[5] == angle(vort[2].z)*abs(vort[2].z)
  @test x[3] == vort[1].S
  @test x[6] == vort[2].S

  vort_2 = state_to_lagrange(x,config_data)
  @test Elements.position(vort_2) ≈ Elements.position(vort)
  @test LowRankVortex.strength(vort_2) ≈ LowRankVortex.strength(vort)


end
