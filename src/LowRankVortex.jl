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



include("vortex.jl")
include("convective_complexpotential.jl")
include("pressure.jl")
include("AD_pressure.jl")
include("analytical_jacobian_pressure.jl")
include("symmetric_pressure.jl")
include("symmetric_analytical_jacobian_pressure.jl")
include("symmetric_analytical_jacobian_pressure_freestream.jl")

include("general/types.jl")
include("general/ensemble.jl")
include("general/observation.jl")
include("general/forecast.jl")
include("general/pressure_utilities.jl")
include("general/state_utilities.jl")
include("general/jacobian.jl")
include("general/vortex.jl")
include("general/api.jl")
include("general/classification.jl")

include("vortex_forecast_observation.jl")
include("generate_twin_experiment.jl")
include("enkf.jl")





# Point vortices about a circular cylinder
include("cylinder/cylindervortex.jl")
include("cylinder/tools.jl")
include("cylinder/forecast.jl")
include("cylinder/generate_twin_experiment.jl")
#include("cylinder/cylinder_pressure.jl")
include("cylinder/cylinder_analytical_jacobian_pressure.jl")


include("plot_recipes.jl")






end # module
