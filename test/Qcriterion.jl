
@testset "Test velocity_gradient for point vortices" begin

	atol = 1000*eps()

	Nv = 10
	ρv = 0.5
	zv = 1.0*im .+ ρv*randn(ComplexF64, Nv)
	Γv = randn(Nv)

	points = Vortex.Point.(zv, Γv)

	target = rand(ComplexF64)

	function velocity_point(target, zvortex::Vector{ComplexF64}, Γvortex::Vector{Float64})
        w = 0.0*im
        Nx = size(zvortex,1)
        for J = 1:Nx
            zJ = zvortex[J]
            ΓJ = Γvortex[J]
            w += (-im*ΓJ)/(2*π*(target - zJ))
        end
        return w
    end

	@test isapprox(conj(velocity_point(target, zv, Γv)), induce_velocity(target, points, 0.0), atol = atol)

	# True velocity gradient from AD
	dvdz, dvdzstar = ForwardDiff.derivative(x->conj(velocity_point(x, zv, Γv)), target)
	∇v = zeros(2,2)
	# ∂u/∂x
	∇v[1,1] = real(dvdz + dvdzstar)
	#∂u/∂y
	∇v[1,2] = real(im*(dvdz - dvdzstar))
	# ∂v/∂x
	∇v[2,1] = imag(dvdz + dvdzstar)
	#∂v/∂y
	∇v[2,2] = imag(im*(dvdz - dvdzstar))

	@test isapprox(velocity_gradient(target, points), ∇v, atol = atol)

	# Flow is incompressible so ∇⋅v = 0
	@test isapprox(tr(∇v), 0.0, atol = atol)

	# Flow is incompressible so ∇⋅v = 0
	@test isapprox(tr(velocity_gradient(target, points)), 0.0, atol = atol)
end

@testset "Test velocity_gradient for regularized vortices" begin

	atol = 1000*eps()

	Nv = 10
	ρv = 0.5
	zv = 1.0*im .+ ρv*randn(ComplexF64, Nv)
	Γv = randn(Nv)

	δ = 0.1

	blobs = Vortex.Blob.(zv, Γv, δ*ones(Nv))

	target = rand(ComplexF64)

	function velocity_blob(target, zvortex::Vector{ComplexF64}, Γvortex::Vector{Float64}, δ)
	    w = 0.0*im
	    Nx = size(zvortex,1)
	    for J = 1:Nx
	        zJ = zvortex[J]
	        ΓJ = Γvortex[J]
	        w += (-im*ΓJ*(conj(target) - conj(zJ)))/(2*π*((target - zJ)*(conj(target) - conj(zJ)) + δ^2))
	    end
	    return w
	end

	@test isapprox(conj(velocity_blob(target, zv, Γv, δ)), induce_velocity(target, blobs, 0.0), atol = atol)

	# True velocity gradient from AD
	dvdz, dvdzstar = ForwardDiff.derivative(x->conj(velocity_blob(x, zv, Γv, δ)), target)

	∇v = zeros(2,2)
	# ∂u/∂x
	∇v[1,1] = real(dvdz + dvdzstar)
	#∂u/∂y
	∇v[1,2] = real(im*(dvdz - dvdzstar))
	# ∂v/∂x
	∇v[2,1] = imag(dvdz + dvdzstar)
	#∂v/∂y
	∇v[2,2] = imag(im*(dvdz - dvdzstar))

	@test isapprox(velocity_gradient(target, blobs), ∇v, atol = atol)

	# Flow is incompressible so ∇⋅v = 0
	@test isapprox(tr(∇v), 0.0, atol = atol)

	# Flow is incompressible so ∇⋅v = 0
	@test isapprox(tr(velocity_gradient(target, blobs)), 0.0, atol = atol)

end
