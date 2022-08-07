export lagrange_to_state_reordered, state_to_lagrange_reordered

function lagrange_to_state_reordered(source::Vector{T}, config::VortexConfig) where T <: Union{PotentialFlow.Points.Point,PotentialFlow.Blobs.Blob}
    Nv = length(source)

    states = Vector{Float64}(undef, 3*Nv)

    for i=1:Nv
        bi = source[i]
        states[2i-1] = real(bi.z)
        states[2i] = imag(bi.z)
        states[2*Nv+i] =  strength(bi)
    end

    return states
end

function state_to_lagrange_reordered(state::AbstractVector{Float64}, config::VortexConfig; isblob::Bool=true)
    Nv = length(state) ÷ 3

    zv = [state[2i-1] + im*state[2i] for i in 1:Nv]
    Γv = [state[2Nv+i] for i in 1:Nv]

    if isblob #return collection of regularized point vortices
        blobs = Nv > 0 ? Vortex.Blob.(zv, Γv, config.δ) : Vortex.Blob[]

        return blobs
    else #return collection of point vortices
        points = Nv > 0 ? Vortex.Point.(zv, Γv) : Vortex.Point[]

        return points
    end
end
