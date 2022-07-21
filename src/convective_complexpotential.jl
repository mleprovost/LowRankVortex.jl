export convective_complexpotential

import PotentialFlow.Properties: @property

# These routines compute the time rate of change of the complex potential of a moving point vortex or regularized point vortex
# For a point vortex with strength Γ located at z_J,
# i.e. dF(z)/dt = 1/(z - z_J) (-ż_J) where ż_J denotes the velocity of the point vortex.
@property begin
    signature = convective_complexpotential(targ::Target, src::Source, srcvel::Source)
    preallocator = allocate_convective_complexpotential
    stype = ComplexF64
end

function convective_complexpotential(z::Complex{T},
                    p::Vortex.Point, point_vel::Complex{S}) where {T,S}
    circulation(p)*conj(PotentialFlow.Points.cauchy_kernel(z - p.z))*(-point_vel)
end

function convective_complexpotential(z::Complex{T},
                           b::Vortex.Blob, blob_vel::Complex{S}) where {T,S}
    convective_complexpotential(z, PotentialFlow.Vortex.Point(Elements.position(b), circulation(b)), blob_vel)
end

function convective_complexpotential(z::Complex{T},
                    p::PotentialFlow.Source.Point, point_vel::Complex{S}) where {T,S}
    PotentialFlow.flux(p)*(im)*conj(PotentialFlow.Points.cauchy_kernel(z - p.z))*(-point_vel)
end

function convective_complexpotential(z::Complex{T},
                           b::PotentialFlow.Source.Blob, blob_vel::Complex{S}) where {T,S}
    convective_complexpotential(z, PotentialFlow.Source.Point(PotentialFlow.Elements.position(b), flux(b)), blob_vel)
end
