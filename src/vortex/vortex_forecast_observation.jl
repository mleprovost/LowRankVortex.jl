### Forecasting, observation, filtering, and localization operators for vortex problems ####

export VortexForecast, SymmetricVortexForecast, VortexPressureObservations,
					SymmetricVortexPressureObservations,
					PressureObservations, ForceObservations, physical_space_sensors

#import TransportBasedInference: Parallel, Serial, Thread # These should not be necessary


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
	source = state_to_lagrange(x, config)

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

struct CylinderVortexForecast{Nx,CVT} <: AbstractForecastOperator{Nx}
		config :: VortexConfig
		cachevels :: CVT
end

"""
		CylinderVortexForecast(config::VortexConfig)

Allocate the structure for forecasting of vortex dynamics about a cylinder.
"""
function CylinderVortexForecast(config::VortexConfig)
	Nx = 3*config.Nv
	cachevels = allocate_velocity(state_to_lagrange(zeros(Nx), config))
	CylinderVortexForecast{Nx,typeof(cachevels)}(config,cachevels)
end

function forecast(x::AbstractVector,t,Δt,fdata::CylinderVortexForecast{Nx}) where {Nx}
	@unpack config, cachevels = fdata
	@unpack advect_flag, Δt = config

	source = state_to_lagrange(x, config)

	reset_velocity!(cachevels, source)
	cachevels .= conj.(LowRankVortex.w(Elements.position(source),source; ϵ = config.δ))

	# Advect the system
	if advect_flag
		advect!(source, source, cachevels, Δt)
	end

	return lagrange_to_state(source, config)

end


abstract type AbstractCartesianVortexObservations{Nx,Ny} <: AbstractObservationOperator{Nx,Ny,true} end

# This one is meant to replace the legacy pressure functions
struct SymmetricVortexPressureObservations{Nx,Ny,withfreestream,ST} <: AbstractCartesianVortexObservations{Nx,Ny}
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
	return symmetric_pressure(real(sens), state_to_lagrange(x, config), t)
end

function observations(x::AbstractVector,t,obs::SymmetricVortexPressureObservations{Nx,Ny,true}) where {Nx,Ny}
  @unpack config, sens = obs
	@unpack U = config
	freestream = Freestream(U)
	return symmetric_pressure(real(sens), state_to_lagrange(x, config), freestream, t)
end


function jacob!(J,x::AbstractVector,t,obs::SymmetricVortexPressureObservations{Nx,Ny,false}) where {Nx,Ny}
	@unpack sens,config,wtarget,dpd,dpdstar,Css,Cts,∂Css,Ctsblob,∂Ctsblob = obs

	return symmetric_analytical_jacobian_pressure!(J, wtarget, dpd, dpdstar, Css, Cts, ∂Css, Ctsblob, ∂Ctsblob,
														sens, vcat(state_to_lagrange(x, config)...), 1:config.Nv, config.state_id, t)
end

function jacob!(J,x::AbstractVector,t,obs::SymmetricVortexPressureObservations{Nx,Ny,true}) where {Nx,Ny}
	@unpack sens,config,wtarget,dpd,dpdstar,Css,Cts,∂Css,Ctsblob,∂Ctsblob = obs
	@unpack U = config
	freestream = Freestream(U)
	return symmetric_analytical_jacobian_pressure!(J, wtarget, dpd, dpdstar, Css, Cts, ∂Css, Ctsblob, ∂Ctsblob,
										sens, vcat(state_to_lagrange(x, config)...), freestream, 1:config.Nv, config.state_id, t)
end


struct VortexPressureObservations{Nx,Ny,withfreestream,ST} <: AbstractCartesianVortexObservations{Nx,Ny}
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

function VortexPressureObservations(sens::AbstractVector,config::VortexConfig)
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

	return VortexPressureObservations{Nx,length(sens),withfreestream,typeof(sens)}(sens,config,wtarget,dpd,dpdstar,Css,Cts,∂Css,Ctsblob,∂Ctsblob)
end

function observations(x::AbstractVector,t,obs::VortexPressureObservations{Nx,Ny,false}) where {Nx,Ny}
  @unpack config, sens = obs
	return pressure(sens, state_to_lagrange(x, config), t)
end

function observations(x::AbstractVector,t,obs::VortexPressureObservations{Nx,Ny,true}) where {Nx,Ny}
  @unpack config, sens = obs
	@unpack U = config
	freestream = Freestream(U)
	return pressure(sens, state_to_lagrange(x, config), freestream, t)
end


function jacob!(J,x::AbstractVector,t,obs::VortexPressureObservations{Nx,Ny,false}) where {Nx,Ny}
	@unpack sens,config,wtarget,dpd,dpdstar,Css,Cts,∂Css,Ctsblob,∂Ctsblob = obs

	return analytical_jacobian_pressure!(J, wtarget, dpd, dpdstar, Css, Cts, ∂Css, Ctsblob, ∂Ctsblob,
														sens, vcat(state_to_lagrange(x, config)...), 1:config.Nv, config.state_id, t)
end

function jacob!(J,x::AbstractVector,t,obs::VortexPressureObservations{Nx,Ny,true}) where {Nx,Ny}
	@unpack sens,config,wtarget,dpd,dpdstar,Css,Cts,∂Css,Ctsblob,∂Ctsblob = obs
	@unpack U = config
	freestream = Freestream(U)
	return analytical_jacobian_pressure!(J, wtarget, dpd, dpdstar, Css, Cts, ∂Css, Ctsblob, ∂Ctsblob,
										sens, vcat(state_to_lagrange(x, config)...), freestream, 1:config.Nv, config.state_id, t)
end


#### END OF LEGACY PRESSURE OBSERVATION OPS #####

# Pressure


struct PressureObservations{Nx,Ny,ST,CT} <: AbstractObservationOperator{Nx,Ny,true}
    sens::ST
    config::CT
end

"""
    PressureObservations(sens::AbstractVector,config::VortexConfig)

Constructor to create an instance of pressure sensors. The locations of the
sensors are specified by `sens`, which should be given as a vector of
complex coordinates.
"""
function PressureObservations(sens::AbstractVector,config::VortexConfig)
    return PressureObservations{3*config.Nv,length(sens),typeof(sens),typeof(config)}(sens,config)
end

function observations(x::AbstractVector,t,obs::PressureObservations)
  @unpack config, sens = obs
  v = state_to_lagrange(x,config)
  return _pressure(sens,v,config)
end

function jacob!(J,x::AbstractVector,t,obs::PressureObservations)
    @unpack config, sens = obs
    v = state_to_lagrange(x,config)
    return _pressure_jacobian!(J,sens,v,config)
end

_pressure(sens,v,config::VortexConfig{Body}) = analytical_pressure(sens,v,config;preserve_circ=false)
_pressure(sens,v,config::VortexConfig) = analytical_pressure(sens,v,config)

_pressure_jacobian!(J,sens,v,config::VortexConfig{Body}) = analytical_pressure_jacobian!(J,sens,v,config;preserve_circ=false)
_pressure_jacobian!(J,sens,v,config::VortexConfig) = analytical_pressure_jacobian!(J,sens,v,config)


physical_space_sensors(obs::PressureObservations) = physical_space_sensors(obs.sens,obs.config)
physical_space_sensors(sens,config::VortexConfig) = sens
physical_space_sensors(sens,config::VortexConfig{Body}) = physical_space_sensors(sens,config.body)
physical_space_sensors(sens,body::Bodies.ConformalBody) = Bodies.conftransform(sens,body)



# Force

struct ForceObservations{Nx,Ny,ST,CT} <: AbstractObservationOperator{Nx,Ny,false}
    sens::ST
    config::CT
end

"""
    ForceObservations(config::VortexConfig)

Constructor to create an instance of force sensing.
"""
function ForceObservations(config::VortexConfig)
    return ForceObservations{3*config.Nv,3,Nothing,typeof(config)}(nothing,config)
end

function observations(x::AbstractVector,t,obs::ForceObservations)
  @unpack config = obs
  v = state_to_lagrange(x,config)
  return analytical_force(v,config)
end

function jacob!(J,x::AbstractVector,t,obs::ForceObservations)
    @unpack config = obs
    v = state_to_lagrange(x,config)
    analytical_force_jacobian!(J,v,config)
end



"""
		state_filter!(x,obs::SymmetricVortexPressureObservations)

A filter function to ensure that the point vortices stay above the x-axis, and retain a positive circulation.
This function would typically be used before and after the analysis step to enforce those constraints.
"""
state_filter!(x, obs::SymmetricVortexPressureObservations) = symmetry_state_filter!(x, obs.config)

function symmetry_state_filter!(x, config::VortexConfig)
	@unpack Nv, state_id = config

	y_ids = state_id["vortex y"]
	Γ_ids = state_id["vortex Γ"]

	@inbounds for j=1:Nv
		# Ensure that vortices stay above the x axis
		x[y_ids[j]] = clamp(x[y_ids[j]], 1e-2, Inf)
		# Ensure that the circulation remains positive
    x[Γ_ids[j]] = clamp(x[Γ_ids[j]], 0.0, Inf)
	end
  return x
end

"""
		state_filter!(x,obs::PressureObservations)

A filter function to ensure that the point vortices stay above the x-axis, and retain a positive circulation.
This function would typically be used before and after the analysis step to enforce those constraints.
"""
state_filter!(x, obs::PressureObservations) = flip_symmetry_state_filter!(x, obs.config)


function flip_symmetry_state_filter!(x, config::VortexConfig)
	@unpack Nv, state_id = config

	x_ids = state_id["vortex x"]
	y_ids = state_id["vortex y"]
	Γ_ids = state_id["vortex Γ"]

	# Flip the sign of vorticity if it is negative on average
	Γtot = sum(x[Γ_ids])
	x[Γ_ids] = Γtot < 0 ? -x[Γ_ids] : x[Γ_ids]

	# Sort the vortices by strength to try to ensure they don't take each other's role
	id_sort = sortperm(x[Γ_ids])
	x[x_ids] = x[x_ids[id_sort]]
	x[y_ids] = x[y_ids[id_sort]]
	x[Γ_ids] = x[Γ_ids[id_sort]]

	# Make all y locations positive
	#x[y_ids] = abs.(x[y_ids])

  return x
end


# Localization
function dobsobs(obs::AbstractCartesianVortexObservations{Nx,Ny}) where {Nx,Ny}
		@unpack sens, config = obs
    dYY = zeros(Ny, Ny)
    # Exploit symmetry of the distance matrix dYY
    for i=1:Ny
        for j=1:i-1
            dij = abs(sens[i] - sens[j])
            dYY[i,j] = dij
            dYY[j,i] = dij
        end
    end
    return dYY
end

function dstateobs(X::BasicEnsembleMatrix{Nx,Ne}, obs::AbstractCartesianVortexObservations{Nx,Ny}) where {Nx,Ny,Ne}
#function dstateobs(X, obs::AbstractCartesianVortexObservations{Nx,Ny}) where {Nx,Ny}

	@unpack config, sens = obs
	@unpack Nv, state_id = config

	x_ids = state_id["vortex x"]
	y_ids = state_id["vortex y"]
	Γ_ids = state_id["vortex Γ"]

	dXY = zeros(Nx, Ny)
	for i in 1:Ne
		#zi, Γi = state_to_positions_and_strengths(X(i),config)
		xi = X(i)
		zi = map(l->xi[x_ids[l]] + im*xi[y_ids[l]], 1:Nv)

		for J in 1:Nv
				for k in 1:Ny
						dXY[J,k] += abs(zi[J] - sens[k])
				end
		end
	end
	dXY ./= Ne
	return dXY
end

function apply_state_localization!(Σxy,X,Lxy,obs::AbstractCartesianVortexObservations)
	@unpack config = obs
	@unpack Nv, state_id = config
  dxy = dstateobs(X, obs)
  Gxy = gaspari.(dxy./Lxy)

	x_ids = state_id["vortex x"]
	y_ids = state_id["vortex y"]
	Γ_ids = state_id["vortex Γ"]

  for J=1:Nv
		Σxy[x_ids[J],:] .*= Gxy[J,:]
		Σxy[y_ids[J],:] .*= Gxy[J,:]
		Σxy[Γ_ids[J],:] .*= Gxy[J,:]
  end
  return nothing
end
