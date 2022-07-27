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
using TransportBasedInference
using NamedColors



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

# Point vortices about a circular cylinder
include("cylinder/cylindervortex.jl")
include("cylinder/tools.jl")
include("cylinder/forecast.jl")
include("cylinder/generate_twin_experiment.jl")
include("cylinder/cylinder_pressure.jl")
include("cylinder/senkf_cylinder_vortexassim.jl")
include("cylinder/localized_senkf_cylinder_vortexassim.jl")






end # module
