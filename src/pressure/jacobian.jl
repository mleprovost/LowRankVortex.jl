export analytical_pressure_jacobian!, analytical_force_jacobian!

analytical_pressure_jacobian!(J,target::AbstractVector{<:ComplexF64}, source::Vector{T},config::VortexConfig{Body};kwargs...) where T <: PotentialFlow.Element =
			analytical_pressure_jacobian!(J,target,source,config.body,config.state_id;kwargs...)

analytical_pressure_jacobian!(J,target::AbstractVector{<:ComplexF64},source::Vector{T},config::VortexConfig{WT}) where {T <: PotentialFlow.Element, WT <: ImageType} =
    analytical_pressure_jacobian!(J,target,source,config.state_id;ϵ=config.δ,walltype=config.body)

analytical_pressure_jacobian!(J,target::AbstractVector{<:ComplexF64}, source::Vector{T},config::VortexConfig;kwargs...) where T <: PotentialFlow.Element =
			analytical_pressure_jacobian!(J,target,source,config.state_id;kwargs...)

analytical_force_jacobian!(J, source::Vector{T},config::VortexConfig{Body};kwargs...) where T <: PotentialFlow.Element =
			analytical_force_jacobian!(J,source,config.body,config.state_id;kwargs...)

function analytical_pressure_jacobian!(J,target::AbstractVector{<:ComplexF64}, source::Vector{T}, b::Bodies.ConformalBody, state_id::Dict; kwargs...) where T <: PotentialFlow.Element
	Nv = length(source)
	Nx = state_length(state_id)
	Ny = size(target, 1)
  #J = zeros(Float64,Ny,Nx)
  @assert size(J) == (Ny, Nx)

	logr_ids = state_id["vortex logr"]
  rϴ_ids = state_id["vortex rΘ"]
  Γ_ids = state_id["vortex Γ"]

	for i=1:Nv
		dpdζi = analytical_dpdζv(target, i, source, b; kwargs...)
		ζi = Elements.position(source[i])
		J[:,logr_ids[i]] .= 2*(abs(ζi)-1.0)*real.(dpdζi*ζi/abs(ζi))
		J[:,rϴ_ids[i]] .= -2*imag.(dpdζi*ζi/abs(ζi))

		dpdΓi = analytical_dpdΓv(target, i, source, b; kwargs...)
		J[:,Γ_ids[i]] .= dpdΓi
	end

	return J
end

function analytical_pressure_jacobian!(J,target::AbstractVector{<:ComplexF64}, source::Vector{T}, state_id::Dict; kwargs...) where T <: PotentialFlow.Element
	Nv = length(source)
	Nx = state_length(state_id)
	Ny = size(target, 1)
  #J = zeros(Float64,Ny,Nx)
  @assert size(J) == (Ny, Nx)

	x_ids = state_id["vortex x"]
  y_ids = state_id["vortex y"]
  Γ_ids = state_id["vortex Γ"]

	# dpdzi = zeros(ComplexF64, Ny)
	# dpdΓi = zeros(Ny)

	for i=1:Nv
		dpdzi = analytical_dpdzv(target, i, source; kwargs...)
		J[:,x_ids[i]] .= 2*real.(dpdzi)
		J[:,y_ids[i]] .= -2*imag.(dpdzi)

		dpdΓi = analytical_dpdΓv(target, i, source; kwargs...)
		J[:,Γ_ids[i]] .= dpdΓi
	end

	return J
end

analytical_pressure_jacobian!(J,target::AbstractVector{<:ComplexF64}, source::Vector{T}, ::Nothing, state_id; kwargs...) where T <: PotentialFlow.Element =
		analytical_pressure_jacobian!(J,target,source,state_id;kwargs...)

function analytical_force_jacobian!(J,source::Vector{T}, b::Bodies.ConformalBody, state_id::Dict; kwargs...) where T <: PotentialFlow.Element
	Nv = length(source)
	Nx = state_length(state_id)
	Ny = 3
  #J = zeros(Float64,Ny,Nx)
  @assert size(J) == (Ny, Nx)

	logr_ids = state_id["vortex logr"]
  rϴ_ids = state_id["vortex rΘ"]
  Γ_ids = state_id["vortex Γ"]

	for i=1:Nv
		dfdζi = analytical_dfdζv(i, source, b; kwargs...)
		ζi = Elements.position(source[i])
		J[:,logr_ids[i]] .= 2*(abs(ζi)-1.0)*real.(dfdζi*ζi/abs(ζi))
		J[:,rϴ_ids[i]] .= -2*imag.(dfdζi*ζi/abs(ζi))

		dfdΓi = analytical_dfdΓv(i, source, b; kwargs...)
		J[:,Γ_ids[i]] .= dfdΓi
	end

	return J
end
