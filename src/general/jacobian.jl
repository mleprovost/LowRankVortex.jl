function analytical_pressure_jacobian!(J,target::AbstractVector{<:ComplexF64}, source::Vector{T}; kwargs...) where T <: Union{PotentialFlow.Points.Point,PotentialFlow.Blobs.Blob}
	Nv = length(source)
	Nx = 3*Nv
	Ny = size(target, 1)
  #J = zeros(Float64,Ny,Nx)
  @assert size(J) == (Ny, Nx)

	# dpdzi = zeros(ComplexF64, Ny)
	# dpdΓi = zeros(Ny)

	for i=1:Nv
		dpdzi = dpdzv(target, i, source; kwargs...)
		J[:,2*i-1] .= 2*real.(dpdzi)
		J[:,2*i] .= -2*imag.(dpdzi)

		dpdΓi = dpdΓv(target, i, source; kwargs...)
		J[:,2*Nv+i] .= dpdΓi
	end

	return J
end
