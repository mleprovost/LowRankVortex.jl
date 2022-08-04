

@testset "Test Jacobian of pressure induced by point vortices and blobs outside a cylinder" begin
    atolAD = 1000*eps()
    atolFD = 1e-5

    # Create evaluation points about the unit circle
    θcircle = range(0,2π,length=51)
    zcircle = 1.1*exp.(im*θcircle[1:end-1]);

    # Create a ring of point vortices
    Nv = 5
    θv = range(0,2π,length=Nv+1)[1:end-1]

    δ = 2.0

    rv = 2.0
    zv = rv*exp.(im*θv)
    Γv = 5.0*ones(Nv)

    points = Vortex.Point.(zv,Γv)
    blobs = Vortex.Blob.(zv,Γv,δ*ones(Nv))

    dx =  sqrt(eps())
    points_x1 = Vortex.Point.(zv .+ vcat(dx, zeros(Nv-1)),Γv)
    points_y1 = Vortex.Point.(zv .+ vcat(im*dx, zeros(Nv-1)),Γv)
    points_Γ1 = Vortex.Point.(zv, Γv .+ vcat(dx, zeros(Nv-1)))

    blobs_x1 = Vortex.Blob.(zv .+ vcat(dx, zeros(Nv-1)), Γv, δ*ones(Nv))
    blobs_y1 = Vortex.Blob.(zv .+ vcat(im*dx, zeros(Nv-1)), Γv, δ*ones(Nv))
    blobs_Γ1 = Vortex.Blob.(zv, Γv .+ vcat(dx, zeros(Nv-1)), δ*ones(Nv))

    ### Error results for dpdz from analytical, AD and FD differentiation for point vortices

    dpdz_AD, dpdzstar_AD = PotentialFlow.Elements.jacobian_position(x -> pressure(zcircle, x; ϵ = 0.0, walltype=LowRankVortex.Cylinder), points)

    dpdz_analytical = dpdzv(zcircle,1,points; ϵ = 0.0, walltype=LowRankVortex.Cylinder)


    # Comparison analytical vs AD
    @test isapprox(dpdz_AD[:,1], dpdz_analytical; atol = atolAD)


    # Comparison FD vs AD
    pc = pressure(zcircle, points; ϵ = 0.0, walltype=LowRankVortex.Cylinder);

    pc_x1 = pressure(zcircle, points_x1; ϵ = 0.0, walltype=LowRankVortex.Cylinder)
    pc_y1 = pressure(zcircle, points_y1; ϵ = 0.0, walltype=LowRankVortex.Cylinder)

    dpdx1_FD = (1 / dx) * (pc_x1 - pc)
    dpdy1_FD = (1 / dx) * (pc_y1 - pc)

    @test isapprox(dpdx1_FD, 2 * real.(dpdz_AD[:,1]); atol = atolFD)
    @test isapprox(dpdy1_FD, -2 * imag.(dpdz_AD[:,1]); atol = atolFD)


    ### Error results for dpdΓ from analytical, AD and FD differentiation for point vortices

    dpdΓ_AD = PotentialFlow.Elements.jacobian_strength(x -> pressure(zcircle, x; ϵ = 0.0, walltype=LowRankVortex.Cylinder), points)

    dpdΓ_analytical = dpdΓv(zcircle,1,points; ϵ = 0.0, walltype=LowRankVortex.Cylinder)


    # Comparison analytical vs AD
    @test isapprox(dpdΓ_AD[:,1], dpdΓ_analytical; atol = atolAD)


    # Comparison FD vs AD
    pc = pressure(zcircle, points; ϵ = 0.0, walltype=LowRankVortex.Cylinder);
    pc_Γ1 = pressure(zcircle, points_Γ1; ϵ = 0.0, walltype=LowRankVortex.Cylinder);


    dpdΓ1_FD = (1 / dx) * (pc_Γ1 - pc)

    @test isapprox(dpdΓ1_FD, dpdΓ_analytical[:,1]; atol = atolFD)

    ### Error results for dpdz from analytical, AD and FD differentiation for blobs

    dpdz_AD, dpdzstar_AD = PotentialFlow.Elements.jacobian_position(x -> pressure(zcircle, x; ϵ = δ, walltype=LowRankVortex.Cylinder), blobs)

    dpdz_analytical = dpdzv(zcircle,1,blobs; ϵ = δ, walltype=LowRankVortex.Cylinder)

    # Comparison analytical vs AD
    @test isapprox(dpdz_AD[:,1], dpdz_analytical; atol = atolAD)

    # Comparison FD vs AD
    pc = pressure(zcircle, blobs; ϵ = δ, walltype=LowRankVortex.Cylinder);

    pc_x1 = pressure(zcircle, blobs_x1; ϵ = δ, walltype=LowRankVortex.Cylinder)
    pc_y1 = pressure(zcircle, blobs_y1; ϵ = δ, walltype=LowRankVortex.Cylinder)

    dpdx1_FD = (1 / dx) * (pc_x1 - pc)
    dpdy1_FD = (1 / dx) * (pc_y1 - pc)

    @test isapprox(dpdx1_FD, 2 * real.(dpdz_AD[:,1]); atol = atolFD)
    @test isapprox(dpdy1_FD, -2 * imag.(dpdz_AD[:,1]); atol = atolFD)

    ### Error results for dpdΓ from analytical, AD and FD differentiation for blobs

    dpdΓ_AD = PotentialFlow.Elements.jacobian_strength(x -> pressure(zcircle, x; ϵ = δ, walltype=LowRankVortex.Cylinder), blobs)

    dpdΓ_analytical = dpdΓv(zcircle,1,blobs; ϵ = δ, walltype=LowRankVortex.Cylinder)

    # Comparison analytical vs AD
    @test isapprox(dpdΓ_AD[:,1], dpdΓ_analytical; atol = atolAD)

    # Comparison FD vs AD
    pc = pressure(zcircle, blobs; ϵ = δ, walltype=LowRankVortex.Cylinder);
    pc_Γ1 = pressure(zcircle, blobs_Γ1; ϵ = δ, walltype=LowRankVortex.Cylinder);

    dpdΓ1_FD = (1 / dx) * (pc_Γ1 - pc)

    @test isapprox(dpdΓ1_FD, dpdΓ_analytical[:,1]; atol = atolFD)

end
