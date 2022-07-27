export cylinder_analytical_jacobian_pressure!, cylinder_analytical_jacobian_pressure



# Version for point singularities
"""
Returns the Jacobian of the pressure computed from the unsteady Bernoulli equation with respect to the position and conjugate position of the singularities.
Note that we multiply the strength `point.S` of a singularity by -i to move from the convention Γ+iQ (used in PotentialFlow.jl) to Q-iΓ.
"""
function cylinder_analytical_jacobian_pressure!(J, target::Vector{ComplexF64}, source::T) where T <: Vector{PotentialFlow.Blobs.Blob{Float64, Float64}}
	Nv = size(source, 1)
	Nx = 3*Nv
	Ny = size(target, 1)
	@assert size(J) == (Ny, Nx)

	δ = source[1].δ

	# dpdzi = zeros(ComplexF64, Ny)
	# dpdΓi = zeros(Ny)

	for i=1:Nv
		dpdzi = dpdzv(target, i, source; ϵ = δ)
		J[:,3*i-2] .= 2*real(dpdzi)
		J[:,3*i-1] .= -2*imag(dpdzi)

		dpdΓi = dpdΓv(target, i, source; ϵ = δ)
		J[:,3*i] .= dpdΓi
	end

	return J
end

cylinder_analytical_jacobian_pressure(target::Vector{ComplexF64}, source::T) where T <: Vector{PotentialFlow.Blobs.Blob{Float64, Float64}} =
cylinder_analytical_jacobian_pressure!(zeros(size(target,1), 3*size(source,1)), target, source)
