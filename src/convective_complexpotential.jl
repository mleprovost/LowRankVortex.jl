export convective_complexpotential

import PotentialFlow.Properties: @property

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
