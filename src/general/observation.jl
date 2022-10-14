# subtypes of AbstractObservationOperator should provide
#  - an extension of the observation and observation! operators
#  - an extension of the jacob! operator
# They should keep sensor locations and vortex configuration internal
# Nx is number of states, Ny is number of observations

export observations!, observations, AbstractObservationOperator, jacob!,
        PressureObservations, ForceObservations

abstract type AbstractObservationOperator{Nx,Ny} end

"""
    observations(x::AbstractVector,obs::AbstractObservationOperator) -> X

Compute the observation function `h` for state `x`.
The function `h` should take as inputs a vector of measurement points (`sens`), a vector of vortices,
and the configuration data `config`.
"""
function observations(x::AbstractVector,obs::AbstractObservationOperator) end

"""
    observations!(Y::EnsembleMatrix,X::EnsembleMatrix,obs::AbstractObservationOperator)

Compute the observation function `h` for each of the states in `X` and place them in `Y`.
The function `h` should take as inputs a Ny-dimensional vector of measurement points (`sens`), a vector of vortices,
and the configuration data `config`.
"""
function observations!(Y::EnsembleMatrix{Ny,Ne},X::EnsembleMatrix{Nx,Ne},obs::AbstractObservationOperator{Nx,Ny}) where {Nx,Ny,Ne}
  for j in 1:Ne
      Y(j) .= observations(X(j),obs)
  end
  return Y
end

"""
    jacob!(J,x::AbstractVector,obs::AbstractObservationOperator)

Compute the Jacobian of the observation function at state `x` and return it in `J`.
"""
function jacob!(J,x::AbstractVector,obs::AbstractObservationOperator) end

# Pressure

struct PressureObservations{Nx,Ny,ST,CT} <: LowRankVortex.AbstractObservationOperator{Nx,Ny}
    sens::ST
    config::CT
end

function PressureObservations(sens::AbstractVector,config::VortexConfig)
    return PressureObservations{3*config.Nv,length(sens),typeof(sens),typeof(config)}(sens,config)
end

function observations(x::AbstractVector,obs::PressureObservations)
  @unpack config, sens = obs
  v = state_to_lagrange_reordered(x,config)
  return analytical_pressure(sens,v,config)
end

function jacob!(J,x::AbstractVector,obs::PressureObservations)
    @unpack config, sens = obs
    v = state_to_lagrange_reordered(x,config)
    analytical_pressure_jacobian!(J,sens,v,config)
end

# Force

struct ForceObservations{Nx,Ny,CT} <: LowRankVortex.AbstractObservationOperator{Nx,Ny}
    config::CT
end

function ForceObservations(config::VortexConfig)
    return ForceObservations{3*config.Nv,3,typeof(config)}(config)
end

function observations(x::AbstractVector,obs::ForceObservations)
  @unpack config = obs
  v = state_to_lagrange_reordered(x,config)
  return analytical_force(v,config)
end

function jacob!(J,x::AbstractVector,obs::ForceObservations)
    @unpack config = obs
    v = state_to_lagrange_reordered(x,config)
    analytical_force_jacobian!(J,v,config)
end
