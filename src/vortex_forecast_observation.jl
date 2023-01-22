### Forecasting and observation operators for vortex problems ####

export VortexForecast, SymmetricVortexForecast, SymmetricVortexPressureObservations

export vortex, symmetric_vortex # These should be removed

import TransportBasedInference: Parallel, Serial, Thread # These should not be necessary


struct VortexForecast{Nx,withfreestream,CVT} <: AbstractForecastOperator{Nx}
		config :: VortexConfig
		cachevels :: CVT
end

"""
		VortexForecast(config::VortexConfig)

Allocate the structure for forecasting of vortex dynamics
"""
function VortexForecast(config::VortexConfig)
	withfreestream = config.U == 0.0 ? false : true
	Nx = 3*config.Nv
	cachevels = allocate_velocity(state_to_lagrange(zeros(Nx), config))
	VortexForecast{Nx,withfreestream,typeof(cachevels)}(config,cachevels)
end


function forecast(x::AbstractVector,t,Δt,fdata::VortexForecast{Nx,withfreestream}) where {Nx,withfreestream}
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

struct SymmetricVortexForecast{Nx,withfreestream,CVT} <: AbstractForecastOperator{Nx}
		config :: VortexConfig
		cachevels :: CVT
end

"""
		SymmetricVortexForecast(config::VortexConfig)

Allocate the structure for forecasting of vortex dynamics with symmetry
about the x axis.
"""
function SymmetricVortexForecast(config::VortexConfig)
	withfreestream = config.U == 0.0 ? false : true
	Nx = 3*config.Nv
	cachevels = allocate_velocity(state_to_lagrange(zeros(Nx), config))
	SymmetricVortexForecast{Nx,withfreestream,typeof(cachevels)}(config,cachevels)
end

function forecast(x::AbstractVector,t,Δt,fdata::SymmetricVortexForecast{Nx,withfreestream}) where {Nx,withfreestream}
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

# This one is meant to replace the legacy pressure functions
struct SymmetricVortexPressureObservations{Nx,Ny,withfreestream,ST} <: AbstractObservationOperator{Nx,Ny}
    sens::ST
    config::VortexConfig
		wtarget :: Vector{ComplexF64}
		dpd :: Matrix{ComplexF64}
		dpdstar :: Matrix{ComplexF64}
		Css :: Matrix{ComplexF64}
		Cts :: Matrix{ComplexF64}
		∂Css :: Matrix{Float64}
		Ctsblob :: Matrix{ComplexF64}
		∂Ctsblob :: Matrix{Float64}
end

function SymmetricVortexPressureObservations(sens::AbstractVector,config::VortexConfig)
	withfreestream = config.U == 0.0 ? false : true

	Nv = config.Nv
	Nx = 3*Nv
	Ny = length(sens)

	wtarget = zeros(ComplexF64, Ny)
	dpd = zeros(ComplexF64, Ny, 2*Nv)
	dpdstar = zeros(ComplexF64, Ny, 2*Nv)

	Css = zeros(ComplexF64, 2*Nv, 2*Nv)
	Cts = zeros(ComplexF64, Ny, 2*Nv)

	∂Css = zeros(2*Nv, 2*Nv)
	Ctsblob = zeros(ComplexF64, Ny, 2*Nv)
	∂Ctsblob = zeros(Ny, 2*Nv)

	return SymmetricVortexPressureObservations{Nx,length(sens),withfreestream,typeof(sens)}(sens,config,wtarget,dpd,dpdstar,Css,Cts,∂Css,Ctsblob,∂Ctsblob)
end

function observations(x::AbstractVector,t,obs::SymmetricVortexPressureObservations{Nx,Ny,false}) where {Nx,Ny}
  @unpack config, sens = obs
	return pressure(sens, state_to_lagrange(state, config), t)
end

function observations(x::AbstractVector,t,obs::SymmetricVortexPressureObservations{Nx,Ny,true}) where {Nx,Ny}
  @unpack config, sens = obs
	@unpack U = config
	freestream = Freestream(U)
	return pressure(sens, state_to_lagrange(state, config), freestream, t)
end


function jacob!(J,x,obs::SymmetricVortexPressureObservations{Nx,Ny,false}) where {Nx,Ny}
	@unpack sens,config,wtarget,dpd,dpdstar,Css,Cts,∂Css,Ctsblob,∂Ctsblob = obs

	return symmetric_analytical_jacobian_pressure!(J, wtarget, dpd, dpdstar, Css, Cts, ∂Css, Ctsblob, ∂Ctsblob,
														sens, vcat(state_to_lagrange(x, config)...), 1:config.Nv, t)
end

function jacob!(J,x,obs::SymmetricVortexPressureObservations{Nx,Ny,true}) where {Nx,Ny}
	@unpack sens,config,wtarget,dpd,dpdstar,Css,Cts,∂Css,Ctsblob,∂Ctsblob = obs
	@unpack U = config
	freestream = Freestream(U)
	return symmetric_analytical_jacobian_pressure!(J, wtarget, dpd, dpdstar, Css, Cts, ∂Css, Ctsblob, ∂Ctsblob,
										sens, vcat(state_to_lagrange(x, config)...), freestream, 1:config.Nv, t)
end



"""
		state_filter!(x,obs::SymmetricVortexPressureObservations)

A filter function to ensure that the point vortices stay above the x-axis, and retain a positive circulation.
This function would typically be used before and after the analysis step to enforce those constraints.
"""
state_filter!(x, obs::SymmetricVortexPressureObservations) = symmetry_state_filter!(x, obs.config)

function symmetry_state_filter!(x, config::VortexConfig)
	@inbounds for j=1:config.Nv
		# Ensure that vortices stay above the x axis
		x[3*j-1] = clamp(x[3*j-1], 1e-2, Inf)
		# Ensure that the circulation remains positive
    	x[3*j] = clamp(x[3*j], 0.0, Inf)
	end
    return x
end



#### OLD STUFF ####

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
