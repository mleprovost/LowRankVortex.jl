export analytical_pressure_jacobian!

analytical_pressure_jacobian!(J,target::AbstractVector{<:ComplexF64}, source::Vector{T},config::VortexConfig{Bodies.ConformalBody};kwargs...) where T <: PotentialFlow.Element =
			analytical_pressure_jacobian!(J,target,source,config.body;kwargs...)

analytical_pressure_jacobian!(J,target,source,config::VortexConfig{DataType}) where {WT} =
    analytical_pressure_jacobian!(J,target,source;ϵ=config.δ,walltype=config.body)

analytical_pressure_jacobian!(J,target::AbstractVector{<:ComplexF64}, source::Vector{T},config::VortexConfig;kwargs...) where T <: PotentialFlow.Element =
			analytical_pressure_jacobian!(J,target,source;kwargs...)



function analytical_pressure_jacobian!(J,target::AbstractVector{<:ComplexF64}, source::Vector{T}, b::Bodies.ConformalBody; kwargs...) where T <: PotentialFlow.Element
	Nv = length(source)
	Nx = 3*Nv
	Ny = size(target, 1)
  #J = zeros(Float64,Ny,Nx)
  @assert size(J) == (Ny, Nx)

	# dpdzi = zeros(ComplexF64, Ny)
	# dpdΓi = zeros(Ny)

	for i=1:Nv
		#dpdzi = analytical_dpdzv(target, i, source, b; kwargs...)
		#J[:,2*i-1] .= 2*real.(dpdzi)
		#J[:,2*i] .= -2*imag.(dpdzi)
		dpdζi = analytical_dpdζv(target, i, source, b; kwargs...)
		ζi = Elements.position(source[i])
		J[:,i] .= 2*(abs(ζi)-1.0)*real.(dpdζi*ζi/abs(ζi))
		J[:,Nv+i] .= -2*imag.(dpdζi*ζi/abs(ζi))

		dpdΓi = analytical_dpdΓv(target, i, source, b; kwargs...)
		J[:,2*Nv+i] .= dpdΓi
	end

	return J
end


function analytical_pressure_jacobian!(J,target::AbstractVector{<:ComplexF64}, source::Vector{T}; kwargs...) where T <: PotentialFlow.Element
	Nv = length(source)
	Nx = 3*Nv
	Ny = size(target, 1)
  #J = zeros(Float64,Ny,Nx)
  @assert size(J) == (Ny, Nx)

	# dpdzi = zeros(ComplexF64, Ny)
	# dpdΓi = zeros(Ny)

	for i=1:Nv
		dpdzi = analytical_dpdzv(target, i, source; kwargs...)
		J[:,2*i-1] .= 2*real.(dpdzi)
		J[:,2*i] .= -2*imag.(dpdzi)

		dpdΓi = analytical_dpdΓv(target, i, source; kwargs...)
		J[:,2*Nv+i] .= dpdΓi
	end

	return J
end

analytical_pressure_jacobian!(J,target::AbstractVector{<:ComplexF64}, source::Vector{T}, ::Nothing; kwargs...) where T <: PotentialFlow.Element =
		analytical_pressure_jacobian!(J,target,source;kwargs...)
