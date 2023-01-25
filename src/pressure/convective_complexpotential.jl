export convective_complexpotential

import PotentialFlow.Properties: @property

@property begin
    signature = convective_complexpotential(targ::Target, src::Source, srcvel::Source)
    preallocator = allocate_convective_complexpotential
    stype = ComplexF64
end

"""
Computes the time rate of change of the complex potential of a moving point vortex
For a point vortex with circulation Γ_J located at z_J, it returns dF(z)/dt = Γ_J/(z - z_J) (-ż_J),
where ż_J denotes the velocity of the point vortex.
"""
function convective_complexpotential(z::Complex{T},
                    p::Vortex.Point, point_vel::Complex{S}) where {T,S}
    circulation(p)*conj(PotentialFlow.Points.cauchy_kernel(z - p.z))*(-point_vel)
end

"""
Computes the time rate of change of the complex potential of a moving point regularized vortex
For a regularized point vortex with circulation Γ_J located at z_J, it returns dF(z)/dt = Γ_J/(z - z_J) (-ż_J),
where ż_J denotes the velocity of the regularizedpoint vortex.
"""
function convective_complexpotential(z::Complex{T},
                           b::Vortex.Blob, blob_vel::Complex{S}) where {T,S}
    convective_complexpotential(z, PotentialFlow.Vortex.Point(Elements.position(b), circulation(b)), blob_vel)
end

"""
Computes the time rate of change of the complex potential of a moving point source
For a point source with flux S_J located at z_J, it returns dF(z)/dt = iS_J/(z - z_J) (-ż_J),
where ż_J denotes the velocity of the point source.
"""
function convective_complexpotential(z::Complex{T},
                    p::PotentialFlow.Source.Point, point_vel::Complex{S}) where {T,S}
    PotentialFlow.flux(p)*(im)*conj(PotentialFlow.Points.cauchy_kernel(z - p.z))*(-point_vel)
end

"""
Computes the time rate of change of the complex potential of a moving regularized point source
For a regularized point source with flux S_J located at z_J, it returns dF(z)/dt = iS_J/(z - z_J) (-ż_J),
where ż_J denotes the velocity of the regularized point source.
"""
function convective_complexpotential(z::Complex{T},
                           b::PotentialFlow.Source.Blob, blob_vel::Complex{S}) where {T,S}
    convective_complexpotential(z, PotentialFlow.Source.Point(PotentialFlow.Elements.position(b), flux(b)), blob_vel)
end
