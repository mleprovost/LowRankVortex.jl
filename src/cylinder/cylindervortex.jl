export cylinder_state_to_lagrange, cylinder_lagrange_to_state

"A routine to convert the state from a vector representation (State representation) to a set of vortex blobs and their mirrored images (Lagrangian representation).
 Use `cylinder_lagrange_to_state` for the inverse transformation."
function cylinder_state_to_lagrange(state::AbstractVector{Float64}, config::VortexConfig; isblob::Bool=true)
    Nv = config.Nv

    zv = (state[3i-2] + im*state[3i-1] for i in 1:Nv)
    Γv = (state[3i] for i in 1:Nv)

    if isblob == true #return collection of regularized point vortices
        δ = config.δ
        δv = (δ for i in 1:Nv)
        blobs = Nv > 0 ? map(Vortex.Blob, zv, Γv, δv) : Vortex.Blob{Float64, Float64}[]

        return blobs
    else #return collection of point vortices
        points = Nv > 0 ? map(Vortex.Point, zv, Γv) : Vortex.Point{Float64, Float64}[]

        return points
    end
end

"A routine to convert a set of vortex blobs (Lagrangian representation) to a vector representation (State representation).
 Use `cy;inder_state_to_lagrange` for the inverse transformation."
function cylinder_lagrange_to_state(source, config::VortexConfig)
    Nv = length(source)
    @assert Nv == config.Nv

    states = Vector{Float64}(undef, 3*Nv)

    for i=1:Nv
        bi = source[i]
        states[3i-2] = real(bi.z)
        states[3i-1] = imag(bi.z)
        states[3i] =  circulation(bi)
    end

    return states
end
