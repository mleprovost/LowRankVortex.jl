using Test

using LinearAlgebra, Statistics
using LowRankVortex
#using TransportBasedInference
using ForwardDiff
using PotentialFlow
import PotentialFlow.Plates: Plate, Points, Blobs
import PotentialFlow.Motions: reset_velocity!
import PotentialFlow.Points
import PotentialFlow.Elements
import PotentialFlow.Properties: @property
import PotentialFlow.Elements: jacobian_position, jacobian_strength, jacobian_param

const GROUP = get(ENV, "GROUP", "All")



if GROUP == "All" || GROUP == "Ensemble"
  include("ensemble.jl")
end
if GROUP == "All" || GROUP == "Vortex"
  include("vortex.jl")
end
if GROUP == "All" || GROUP == "Forecast"
  include("forecast.jl")
end
if GROUP == "All" || GROUP == "Pressure"
  include("pressure.jl")
  include("analytical_pressure.jl")
  include("convective_complexpotential.jl")
  include("symmetric_pressure.jl")
  include("symmetric_analytical_jacobian_pressure_freestream.jl")
  include("symmetric_analytical_jacobian_pressure.jl")
  include("AD_pressure.jl")
  include("cylinder_analytical_jacobian_pressure.jl")
  include("GaussianMixtureModel.jl")
end
