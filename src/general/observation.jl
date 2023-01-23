# subtypes of AbstractObservationOperator should provide
#  - an extension of the observation operator
#  - an extension of the jacob! operator
# They should keep sensor locations and vortex configuration internal
# Nx is number of states, Ny is number of observations

export observations!, observations, AbstractObservationOperator, jacob!,
       state_filter!,
        PressureObservations, ForceObservations, physical_space_sensors,
        loglikelihood

abstract type AbstractObservationOperator{Nx,Ny} end

measurement_length(::AbstractObservationOperator{Nx,Ny}) where {Nx,Ny} = Ny
state_length(::AbstractObservationOperator{Nx,Ny}) where {Nx,Ny} = Nx


"""
    loglikelihood(x,ystar,Σϵ,obs) -> Float64

For a given state `x`, return the log of the likelihood function,
given observations `ystar`, noise covariance `Σϵ`, and observation structure `obs`.
"""
function loglikelihood(x,ystar,Σϵ,obs::AbstractObservationOperator)
    y = observations(x,obs)
    #loss = norm(ystar-y .- mean(ystar-y),Σϵ)
    loss = norm(ystar-y,Σϵ)
    return -loss^2/2
end

"""
    observations(x::AbstractVector,t::Float64,obs::AbstractObservationOperator) -> X

Compute the observation function `h` for state `x` at time `t`.
The function `h` should take as inputs a vector of measurement points (`sens`), a vector of vortices,
and the configuration data `config`.
"""
function observations(x::AbstractVector,t,obs::AbstractObservationOperator) end

"""
    observations!(Y::EnsembleMatrix,X::EnsembleMatrix,t::Float64,obs::AbstractObservationOperator)

Compute the observation function `h` for each of the states in `X` and place them in `Y`.
The function `h` should take as inputs a Ny-dimensional vector of measurement points (`sens`), a vector of vortices,
and the configuration data `config`.
"""
function observations!(Y::EnsembleMatrix{Ny,Ne},X::EnsembleMatrix{Nx,Ne},t,obs::AbstractObservationOperator{Nx,Ny}) where {Nx,Ny,Ne}
  for j in 1:Ne
      Y(j) .= observations(X(j),t,obs)
  end
  return Y
end

"""
    jacob!(J,x::AbstractVector,t::Float64,obs::AbstractObservationOperator)

Compute the Jacobian of the observation function at state `x` and return it in `J`.
"""
function jacob!(J,x::AbstractVector,t,obs::AbstractObservationOperator) end


state_filter!(x,obs::AbstractObservationOperator) = x

# Pressure


struct PressureObservations{Nx,Ny,ST,CT} <: AbstractObservationOperator{Nx,Ny}
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
  v = state_to_lagrange_reordered(x,config)
  return _pressure(sens,v,config)
end

function jacob!(J,x::AbstractVector,t,obs::PressureObservations)
    @unpack config, sens = obs
    v = state_to_lagrange_reordered(x,config)
    return _pressure_jacobian!(J,sens,v,config)
end

_pressure(sens,v,config::VortexConfig{Bodies.ConformalBody}) = analytical_pressure(sens,v,config;preserve_circ=false)
_pressure(sens,v,config::VortexConfig) = analytical_pressure(sens,v,config)

_pressure_jacobian!(J,sens,v,config::VortexConfig{Bodies.ConformalBody}) = analytical_pressure_jacobian!(J,sens,v,config;preserve_circ=false)
_pressure_jacobian!(J,sens,v,config::VortexConfig) = analytical_pressure_jacobian!(J,sens,v,config)


physical_space_sensors(obs::PressureObservations) = physical_space_sensors(obs.sens,obs.config)
physical_space_sensors(sens,config::VortexConfig) = sens
physical_space_sensors(sens,config::VortexConfig{Bodies.ConformalBody}) = physical_space_sensors(sens,config.body)
physical_space_sensors(sens,body::Bodies.ConformalBody) = Bodies.conftransform(sens,body)



# Force

struct ForceObservations{Nx,Ny,CT} <: AbstractObservationOperator{Nx,Ny}
    config::CT
end

"""
    ForceObservations(config::VortexConfig)

Constructor to create an instance of force sensing.
"""
function ForceObservations(config::VortexConfig)
    return ForceObservations{3*config.Nv,3,typeof(config)}(config)
end

function observations(x::AbstractVector,t,obs::ForceObservations)
  @unpack config = obs
  v = state_to_lagrange_reordered(x,config)
  return analytical_force(v,config)
end

function jacob!(J,x::AbstractVector,t,obs::ForceObservations)
    @unpack config = obs
    v = state_to_lagrange_reordered(x,config)
    analytical_force_jacobian!(J,v,config)
end
