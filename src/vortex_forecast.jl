### Forecasting operators for vortex problems ####

export VortexForecastOperator, SymmetricVortexForecastOperator

export vortex, symmetric_vortex # These should be removed

import TransportBasedInference: Parallel, Serial, Thread # These should not be necessary


struct VortexForecastOperator{Nx,withfreestream,CVT} <: AbstractForecastOperator{Nx}
		config :: VortexConfig
		cachevels :: CVT
end

"""
		VortexForecastOperator(config::VortexConfig)

Allocate the structure for forecasting of vortex dynamics
"""
function VortexForecastOperator(config::VortexConfig)
	withfreestream = config.U == 0.0 ? false : true
	Nx = 3*config.Nv
	cachevels = allocate_velocity(state_to_lagrange(zeros(Nx), config))
	SymmetricVortexForecastOperator{Nx,withfreestream,typeof(cachevels)}(config,cachevels)
end


function forecast(x::AbstractVector,t,Δt,fdata::VortexForecastOperator{Nx,withfreestream}) where {Nx,withfreestream}
	@unpack config, cachevels = fdata
	@unpack U, advect_flag = config

	freestream = Freestream(U)
	source = state_to_lagrange(col, config)

	# Compute the induced velocity (by exploiting the symmetry of the problem)
	reset_velocity!(cachevels, source)
	self_induce_velocity!(cachevels, source, t)

	if withfreestream
		induce_velocity!(cachevels, source, freestream, t)
	end

	# Advect the system
	if advect_flag
		advect!(source, source, cachevels, Δt)
	end

	return lagrange_to_state(source, config)

end

struct SymmetricVortexForecastOperator{Nx,withfreestream,CVT} <: AbstractForecastOperator{Nx}
		config :: VortexConfig
		cachevels :: CVT
end


"""
		SymmetricVortexForecastOperator(config::VortexConfig)

Allocate the structure for forecasting of vortex dynamics with symmetry
about the x axis.
"""
function SymmetricVortexForecastOperator(config::VortexConfig)
	withfreestream = config.U == 0.0 ? false : true
	Nx = 3*config.Nv
	cachevels = allocate_velocity(state_to_lagrange(zeros(Nx), config))
	SymmetricVortexForecastOperator{Nx,withfreestream,typeof(cachevels)}(config,cachevels)
end

function forecast(x::AbstractVector,t,Δt,fdata::SymmetricVortexForecastOperator{Nx,withfreestream}) where {Nx,withfreestream}
	@unpack config, cachevels = fdata
	@unpack U, advect_flag = config

	freestream = Freestream(U)
	source = state_to_lagrange(x, config)

	reset_velocity!(cachevels, source)
	self_induce_velocity!(cachevels[1], source[1], t)
	induce_velocity!(cachevels[1], source[1], source[2], t)

	if withfreestream
		induce_velocity!(cachevels[1], source[1], freestream, t)
	end
	# @. cachevels[2] = conj(cachevels[1])

	# We only care about the transport of the top vortices
	# Advect the system
	if advect_flag
		advect!(source[1:1], source[1:1], cachevels[1:1], Δt)
	end

	return lagrange_to_state(source, config)

end


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
		if config.advect_flag
			advect!(source, source, cachevels, config.Δt)
		end

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
		if config.advect_flag
			advect!(source[1:1], source[1:1], cachevels[1:1], config.Δt)
		end

		X[Ny+1:Nypx, i] .= lagrange_to_state(source, config)
	end

	return X, t + config.Δt
end
