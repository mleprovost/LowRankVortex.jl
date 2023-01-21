# subtypes of AbstractForecastOperator should provide
#  - an extension of the forecast operator
#  - an internal cache

export forecast, AbstractForecastOperator

abstract type AbstractForecastOperator{Nx} end

"""
    forecast(x::AbstractVector,t::Float64,Δt::Float64,fdata::AbstractForecastOperator) -> x

Forecast the state `x` at time `t` to its value at the next time `t+Δt`.
"""
function forecast(x::AbstractVector,t::Float64,Δt::Float64,fdata::AbstractForecastOperator) end
