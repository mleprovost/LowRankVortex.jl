module LowRankVortex

using Interpolations
using JLD
using LinearAlgebra
using PotentialFlow
import PotentialFlow.Motions: reset_velocity!
import PotentialFlow.Points
import PotentialFlow.Blobs
import PotentialFlow.Elements
using ProgressMeter
using Statistics
using NamedColors
using Distributions
using UnPack


include("ensemble.jl")
include("forecast.jl")
include("observation.jl")

include("DA/types.jl")
include("DA/generate_twin_experiment.jl")
include("DA/enkf.jl")

include("vortex.jl")
include("vortex_clusters.jl")
include("vortex_forecast_observation.jl")


include("pressure/convective_complexpotential.jl")
include("pressure/pressure.jl")
include("pressure/AD_pressure.jl")
include("pressure/analytical_jacobian_pressure.jl")
include("pressure/symmetric_pressure.jl")
include("pressure/symmetric_analytical_jacobian_pressure.jl")
include("pressure/symmetric_analytical_jacobian_pressure_freestream.jl")
include("pressure/jacobian.jl")
include("pressure/pressure_utilities.jl")
include("pressure/cylinder_analytical_jacobian_pressure.jl")

include("general/state_utilities.jl")
include("general/api.jl")
include("general/classification.jl")


include("plot_recipes.jl")






end # module
