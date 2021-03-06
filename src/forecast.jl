export vortex, symmetric_vortex

import TransportBasedInference: Parallel, Serial, Thread


vortex(X, t::Float64, Ny, Nx, cachevels, config; withfreestream::Bool = false) = vortex(X, t, Ny, Nx, cachevels, config, serial, withfreestream = withfreestream)

"""
This routine advects the regularized point vortices of the different ensemble members stored in X by a time step config.Δt.
"""
function vortex(X, t::Float64, Ny, Nx, cachevels, config, P::Serial; withfreestream::Bool=false)
	Nypx, Ne = size(X)
	@assert Nypx == Ny + Nx "Wrong value of Ny or Nx"

	freestream = Freestream(config.U)

	@inbounds for i = 1:Ne
		col = view(X, Ny+1:Nypx, i)
		source = state_to_lagrange(col, config)

		# Compute the induced velocity (by exploiting the symmetry of the problem)
		reset_velocity!(cachevels, source)
		self_induce_velocity!(cachevels, source, t)

		if withfreestream == true
			induce_velocity!(cachevels, source, freestream, t)
		end

		# Advect the system
		advect!(source, source, cachevels, config.Δt)

		X[Ny+1:Nypx, i] .= lagrange_to_state(source, config)
	end

	return X, t + config.Δt
end

"""
This routine advects the point vortices of the different ensemble members stored in X by a time step config.Δt.
Note that this version assumes that for each vortex with strength Γ_J located at z_J,
there is a mirror point vortex with strength -Γ_J located at conj(z_J).
"""
symmetric_vortex(X, t::Float64, Ny, Nx, cachevels, config; withfreestream::Bool=false) = symmetric_vortex(X, t, Ny, Nx, cachevels, config, serial; withfreestream = withfreestream)

function symmetric_vortex(X, t::Float64, Ny, Nx, cachevels, config, P::Serial; withfreestream::Bool=false)
	Nypx, Ne = size(X)
	@assert Nypx == Ny + Nx "Wrong value of Ny or Nx"
	freestream = Freestream(config.U)
	@inbounds for i = 1:Ne
		col = view(X, Ny+1:Nypx, i)
		source = state_to_lagrange(col, config)

		# Compute the induced velocity (by exploiting the symmetry of the problem)
		reset_velocity!(cachevels, source)
		self_induce_velocity!(cachevels[1], source[1], t)
		induce_velocity!(cachevels[1], source[1], source[2], t)

		if withfreestream == true
			induce_velocity!(cachevels[1], source[1], freestream, t)
		end
		# @. cachevels[2] = conj(cachevels[1])

		# We only care about the transport of the top vortices
		# Advect the system
		advect!(source[1:1], source[1:1], cachevels[1:1], config.Δt)

		X[Ny+1:Nypx, i] .= lagrange_to_state(source, config)
	end

	return X, t + config.Δt
end
