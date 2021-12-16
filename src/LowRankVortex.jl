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
include("generate_vortex.jl")
include("generate_patch.jl")
include("assimilation.jl")
include("lowrankassimilation.jl")
include("symmetric_assimilation.jl")
include("symmetric_lowrankassimilation.jl")
include("adaptive_symmetric_lowrankassimilation.jl")
include("plot_recipes.jl")


end # module
