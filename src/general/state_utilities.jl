export lagrange_to_state_reordered, state_to_lagrange_reordered, state_to_positions_and_strengths,
        state_to_vortex_states, states_to_vortex_states, collect_estimated_states

function collect_estimated_states(collection::Vector{Vector{T}},config::VortexConfig) where {T <: LowRankENKFSolution}
    Nv = config.Nv
    #xarray = zeros(3,config.Nv*length(collection))
    xarray = zeros(3*config.Nv,length(collection))

    for j in eachindex(collection)
        solhist = collection[j]
        laststate = solhist[end]
        #xarray[:,(j-1)*Nv+1:j*Nv] = state_to_vortex_states(mean(laststate.X),config)
        xarray[:,j] = mean(laststate.X)
    end
    return xarray
end

"""
    states_to_vortex_states(state_array::Matrix,config::VortexConfig)

Take an array of states (3Nv x nstates) and convert it to a (3 x Nv*nstates) array
of individual vortex states.
"""
function states_to_vortex_states(state_array::Matrix{Float64}, config::VortexConfig)
   Nv = config.Nv
   ndim, nstates = size(state_array)
   vortex_array = zeros(3,Nv*nstates)
   for j in 1:nstates
     vortexstatej = state_to_vortex_states(state_array[:,j],config)
     vortex_array[:,(j-1)*Nv+1:j*Nv] = vortexstatej
   end
   return vortex_array
end

index_of_vortex_state(v::Int,config::VortexConfig) = (v-1)÷config.Nv+1

function state_to_vortex_states(state::AbstractVector{Float64}, config::VortexConfig)
    zv, Γv = state_to_positions_and_strengths(state,config)
    Nv = length(state) ÷ 3
    xarray = zeros(3,Nv)
    for v in 1:Nv
        xarray[:,v] .= [real(zv[v]),imag(zv[v]),Γv[v]]
        #push!(xarray,[real(zv[v]),imag(zv[v]),Γv[v]])
    end
    return xarray
end

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
