export AD_symmetric_jacobian_pressure

"""
This routine commputes the Jacobian of the pressure field by automatic differentiation.
For improved performance, the symmetry of the point vortices with the x-axis is embedded in the calculation
"""
function AD_symmetric_jacobian_pressure(target, source, t)

	state_id = construct_state_mapping(length(source))

	Nv = length(source)
	Nx = state_length(state_id)
	Ny = length(target)
	J = zeros(Ny, Nx)

	x_ids = state_id["vortex x"]
	y_ids = state_id["vortex y"]
	Γ_ids = state_id["vortex Γ"]


	dpdz, _ = PotentialFlow.Elements.jacobian_position(x ->
	       pressure_AD(target, x, t), source)

	dpdΓ = PotentialFlow.Elements.jacobian_strength(x ->
	      pressure_AD(target, x, t), source)

	# Fill dpdpx and dpdy
	J[:, x_ids] .= 2*real.(dpdz)
	J[:, y_ids] .= -2*imag.(dpdz)
	J[:, Γ_ids] .= real.(dpdΓ)

	return J
end
