export AD_symmetric_jacobian_pressure


"""
This routine commputes the Jacobian of the pressure field by automatic differentiation.
For improved performance, the symmetry of the point vortices with the x-axis is embedded in the calculation
"""
function AD_symmetric_jacobian_pressure(target, source, t)

	Nv = length(source)
	Nx = 3*Nv
	Ny = length(target)
	J = zeros(Ny, Nx)

	dpdz, _ = PotentialFlow.Elements.jacobian_position(x ->
	       pressure_AD(target, x, t), source)

	dpdΓ = PotentialFlow.Elements.jacobian_strength(x ->
	      pressure_AD(target, x, t), source)

	# Fill dpdpx and dpdy
	J[:, 1:3:3*(Nv-1)+1] .= 2*real.(dpdz)
	J[:, 2:3:3*(Nv-1)+2] .= -2*imag.(dpdz)
	J[:, 3:3:3*(Nv-1)+3] .= real.(dpdΓ)

	return J
end
