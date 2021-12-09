using Test

using LinearAlgebra, Statistics
using VortexPatch
using TransportBasedInference
using ForwardDiff
using PotentialFlow
import PotentialFlow.Plates: Plate, Points, Blobs
import PotentialFlow.Motions: reset_velocity!
import PotentialFlow.Points
import PotentialFlow.Elements
import PotentialFlow.Properties: @property
import PotentialFlow.Elements: jacobian_position, jacobian_strength, jacobian_param


include("symmetric_analytical_jacobian_pressure.jl")
include("Qcriterion.jl")
# include("symmetric_pressure.jl")
# include("vortex.jl")
# include("forecast.jl")
# include("convective_complexpotential.jl")
# include("pressure.jl")
# include("AD_pressure.jl")
# include("analytical_pressure.jl")
