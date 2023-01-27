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
include("DA/state_utilities.jl")
include("DA/classification.jl")
include("DA/MCMC.jl")

include("vortex/vortex.jl")
include("vortex/vortex_clusters.jl")
include("vortex/vortex_forecast_observation.jl")
include("vortex/vortex_inference.jl")

include("pressure/convective_complexpotential.jl")
include("pressure/pressure.jl")
include("pressure/AD_pressure.jl")
include("pressure/analytical_jacobian_pressure.jl")
include("pressure/symmetric_pressure.jl")
include("pressure/symmetric_analytical_jacobian_pressure.jl")
include("pressure/symmetric_analytical_jacobian_pressure_freestream.jl")
include("pressure/jacobian.jl")
include("pressure/pressure_decomposed.jl")
include("pressure/cylinder_analytical_jacobian_pressure.jl")



include("plot_recipes.jl")






end # module
