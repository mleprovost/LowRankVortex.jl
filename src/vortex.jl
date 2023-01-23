export VortexConfig, state_to_lagrange, lagrange_to_state, statelength


abstract type ImageType end
abstract type Cylinder <: ImageType end
abstract type FlatWall <: ImageType end
abstract type NoWall <: ImageType end


"""
    VortexConfig

A structure to hold the parameters of the vortex simulations

## Fields
- `Nv::Int64`: number of vortices
- `body::ConformalBody` : body (if present)
- `U::ComplexF64`: freestream velocity
- `Δt::Float64`: time step
- `δ::Float64` : blob radius
- `advect_flag::Bool`: true if the vortex system should be advected
"""
struct VortexConfig{BT,ST}

    "Number of vortices"
    Nv::Int64

    "Body"
    body::BT

    "Freestream velocity"
    U::ComplexF64

    "Time step"
    Δt::Float64

    "Blob radius"
    δ::Float64

    "Advect flag"
    advect_flag::Bool

end

VortexConfig(Nv,U,Δt,δ;advect_flag=true,body=nothing) = VortexConfig(Nv,body,U,Δt,δ,advect_flag)
VortexConfig(Nv,δ;body=nothing) = VortexConfig(Nv,body,complex(0.0),0.0,δ,false)

Base.length(config::VortexConfig) = config.Nv
state_length(config::VortexConfig) = 3*length(config)


"A routine to convert the state from a vector representation (State representation) to a set of vortex blobs and their mirrored images (Lagrangian representation).
 Use `lagrange_to_state` for the inverse transformation."
function state_to_lagrange(state::AbstractVector{Float64}, config::VortexConfig; isblob::Bool=true)
    Nv = config.Nv

    zv = (state[3i-2] + im*state[3i-1] for i in 1:Nv)
    Γv = (state[3i] for i in 1:Nv)

    if isblob == true #return collection of regularized point vortices
        δ = config.δ
        δv = (δ for i in 1:Nv)
        blobs₊ = Nv > 0 ? map(Vortex.Blob, zv, Γv, δv) : Vortex.Blob{Float64, Float64}[]
        blobs₋ = Nv > 0 ? map((z, Γ, δ) -> Vortex.Blob(conj(z), -Γ, δ), zv, Γv, δv) : Vortex.Blob{Float64, Float64}[]

        return (blobs₊, blobs₋)
    else #return collection of point vortices
        points₊ = Nv > 0 ? map(Vortex.Point, zv, Γv) : Vortex.Point{Float64, Float64}[]
        points₋ = Nv > 0 ? map((z, Γ) -> Vortex.Point(conj(z), -Γ), zv, Γv) : Vortex.Point{Float64, Float64}[]

        return (points₊, points₋)
    end
end

"A routine to convert a set of vortex blobs (Lagrangian representation) to a vector representation (State representation).
 Use `state_to_lagrange` for the inverse transformation."
function lagrange_to_state(source, config::VortexConfig; withcylinder::Bool=false)
    Nv = length(source[1])
    @assert Nv == config.Nv

    states = Vector{Float64}(undef, 3*Nv)

    for i=1:Nv
        bi = source[1][i]
        states[3i-2] = real(bi.z)
        states[3i-1] = imag(bi.z)
        states[3i] =  circulation(bi)
    end

    return states
end
