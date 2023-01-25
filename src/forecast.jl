# subtypes of AbstractForecastOperator should provide
#  - an extension of the forecast operator
#  - an internal cache in their structure

export forecast, forecast!, AbstractForecastOperator, IdentityForecastOperator

abstract type AbstractForecastOperator{Nx} end

"""
    forecast(x::AbstractVector,t::Float64,Δt::Float64,fdata::AbstractForecastOperator) -> x

Forecast the state `x` at time `t` to its value at the next time `t+Δt`.
The default forecast function is simply the identity.
"""
function forecast(x,t,Δt,::AbstractForecastOperator) end

struct IdentityForecastOperator{Nx} <: AbstractForecastOperator{Nx} end

forecast(x,t,Δt,::IdentityForecastOperator) = x


"""
    forecast!(X::BasicEnsembleMatrix,t,Δt,fdata::AbstractForecastOperator)

In-place forecast updating of ensemble matrix `X`, using the specific `forecast` function
defined for `fdata`.
"""
function forecast!(X::BasicEnsembleMatrix{Nx,Ne},t,Δt,fdata::AbstractForecastOperator) where {Nx,Ne}
  for j in 1:Ne
    X(j) .= forecast(X(j),t,Δt,fdata)
  end
  return X
end
