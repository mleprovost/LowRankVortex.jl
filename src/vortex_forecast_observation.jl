### Forecasting and observation operators for vortex problems ####

export VortexForecast, SymmetricVortexForecast, SymmetricVortexPressureObservations

export vortex, symmetric_vortex # These should be removed

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
	return pressure(sens, state_to_lagrange(x, config), t)
end

function observations(x::AbstractVector,t,obs::SymmetricVortexPressureObservations{Nx,Ny,true}) where {Nx,Ny}
  @unpack config, sens = obs
	@unpack U = config
	freestream = Freestream(U)
	return pressure(sens, state_to_lagrange(x, config), freestream, t)
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
