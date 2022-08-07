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
import Statistics: cov, mean, std, var
using TransportBasedInference
using NamedColors
using StaticArrays
using Distributions



include("vortex.jl")
include("convective_complexpotential.jl")
include("pressure.jl")
include("AD_pressure.jl")
include("analytical_jacobian_pressure.jl")
include("symmetric_pressure.jl")
include("symmetric_analytical_jacobian_pressure.jl")
include("symmetric_analytical_jacobian_pressure_freestream.jl")
include("forecast.jl")
include("generate_twin_experiment.jl")
include("senkf_vortexassim.jl")
include("senkf_symmetric_vortexassim.jl")
include("localized_senkf_symmetric_vortexassim.jl")
include("lowrankenkf_vortexassim.jl")
include("lowrankenkf_symmetric_vortexassim.jl")
include("adaptive_lowrankenkf_symmetric_vortexassim.jl")
include("plot_recipes.jl")

include("general/pressure_utilities.jl")
include("general/state_utilities.jl")
include("general/jacobian.jl")
include("general/ensemble.jl")
include("general/api.jl")


# Point vortices about a circular cylinder
include("cylinder/cylindervortex.jl")
include("cylinder/tools.jl")
include("cylinder/forecast.jl")
include("cylinder/generate_twin_experiment.jl")
include("cylinder/cylinder_pressure.jl")
include("cylinder/cylinder_analytical_jacobian_pressure.jl")
include("cylinder/senkf_cylinder_vortexassim.jl")
include("cylinder/localized_senkf_cylinder_vortexassim.jl")
include("cylinder/adaptive_lowrankenkf_cylinder_vortexassim.jl")








end # module
