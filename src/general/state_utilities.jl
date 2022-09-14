export lagrange_to_state_reordered, state_to_lagrange_reordered, state_to_positions_and_strengths

function lagrange_to_state_reordered(source::Vector{T}, config::VortexConfig) where T <: PotentialFlow.Element
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

function lagrange_to_state_reordered(source::Vector{T}, config::VortexConfig{Bodies.ConformalBody}) where T <: PotentialFlow.Element
    Nv = length(source)

    states = Vector{Float64}(undef, 3*Nv)

    for i=1:Nv
        bi = source[i]
        zi = Elements.position(bi)
        ri, Θi = abs(zi), angle(zi)
        #states[i] = ri
        states[i] = log(ri-1.0)
        states[Nv+i] = ri*Θi
        states[2*Nv+i] =  strength(bi)
    end

    return states
end


function state_to_lagrange_reordered(state::AbstractVector{Float64}, config::VortexConfig; isblob::Bool=true)

    zv, Γv = state_to_positions_and_strengths(state,config)

    if isblob #return collection of regularized point vortices
        blobs = length(zv) > 0 ? Vortex.Blob.(zv, Γv, config.δ) : Vortex.Blob[]

        return blobs
    else #return collection of point vortices
        points = length(zv) > 0 ? Vortex.Point.(zv, Γv) : Vortex.Point[]

        return points
    end
end

function state_to_positions_and_strengths(state::AbstractVector{Float64}, config::VortexConfig{Bodies.ConformalBody})
  Nv = length(state) ÷ 3

  ζv = zeros(ComplexF64,Nv)
  Γv = zeros(Float64,Nv)
  for i in 1:Nv
    #rv = state[i]
    rv = 1.0 + exp(state[i])
    Θv = state[Nv+i]/rv
    ζv[i] = rv*exp(im*Θv)
    Γv[i] = state[2Nv+i]
  end
  return ζv, Γv
end

function state_to_positions_and_strengths(state::AbstractVector{Float64}, config::VortexConfig)
  Nv = length(state) ÷ 3

  zv = [state[2i-1] + im*state[2i] for i in 1:Nv]
  Γv = [state[2Nv+i] for i in 1:Nv]

  return zv, Γv
end
