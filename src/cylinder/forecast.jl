export cylinder_vortex


"""
This routine advects the point vortices of the different ensemble members stored in X by a time step config.Δt.
Note that this version assumes that for each vortex with strength Γ_J located at z_J,
there is a mirror point vortex with strength -Γ_J located at conj(z_J).
"""
function cylinder_vortex(X, t::Float64, Ny, Nx, cachevels, config)
	Nypx, Ne = size(X)
	@assert Nypx == Ny + Nx "Wrong value of Ny or Nx"
	# freestream = Freestream(config.U)
	@inbounds for i = 1:Ne
		col = view(X, Ny+1:Nypx, i)
		source = cylinder_state_to_lagrange(col, config)

		# Compute the induced velocity of the point vortices in the presence of a cylinder. We use the circle theorem
		reset_velocity!(cachevels, source)
		cachevels = conj.(LowRankVortex.w(Elements.position(source),source; ϵ = config.δ))

		# Advect the set of vortices
		if config.advect_flag
			advect!(source, source, cachevels, config.Δt)
		end
		X[Ny+1:Nypx, i] .= cylinder_lagrange_to_state(source, config)
	end

	return X, t + config.Δt
end
