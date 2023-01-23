export lagrange_to_state_reordered, state_to_lagrange_reordered, state_to_positions_and_strengths,
        state_to_vortex_states, states_to_vortex_states, collect_estimated_states,
        generate_random_state,generate_random_states, positions_and_strengths_to_state

"""
    collect_estimated_states(collection::Vector{Vector{AbstractENKFSolution}},config::VortexConfig) -> Array

Takes a vector of ENKF solution histories and returns the ensemble mean
of the final solution in each history. The output is a matrix of dimension
length(state) x length(collection).
"""
function collect_estimated_states(collection::Vector{Vector{T}},config::VortexConfig) where {T <: AbstractENKFSolution}
    Nv = config.Nv
    #xarray = zeros(3,config.Nv*length(collection))
    xarray = zeros(statelength(config),length(collection))

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

Take an array of states (length(state) x nstates) and convert it to a (3 x Nv*nstates) array
of individual vortex states.
"""
function states_to_vortex_states(state_array::AbstractMatrix{Float64}, config::VortexConfig)
   Nv = length(config)
   ndim, nstates = size(state_array)
   vortex_array = zeros(3,Nv*nstates)
   for j in 1:nstates
     vortexstatej = state_to_vortex_states(state_array[:,j],config)
     vortex_array[:,(j-1)*Nv+1:j*Nv] = vortexstatej
   end
   return vortex_array
end

states_to_vortex_states(state_array::BasicEnsembleMatrix, config::VortexConfig) =
    states_to_vortex_states(state_array.X,config)

"""
    index_of_vortex_state(v::Integer,config::VortexConfig)

Return the column of a (length(state) x nstates) array of states
that a single vortex of index `v` in a (3 x Nv*nstates) vortex state array belongs to.
"""
index_of_vortex_state(v::Int,config::VortexConfig) = (v-1)÷length(config)+1

"""
    state_to_vortex_states(state::AbstractVector,config::VortexConfig)

Given a state `state`, return a 3 x Nv array of the vortex states.
"""
function state_to_vortex_states(state::AbstractVector{Float64}, config::VortexConfig)
    zv, Γv = state_to_positions_and_strengths(state,config)
    Nv = length(state) ÷ 3
    xarray = zeros(3,Nv)
    for v in 1:Nv
        zv_v = vortex_position_to_phys_space(zv[v],config)
        xarray[:,v] .= [real(zv_v),imag(zv_v),Γv[v]]
        #push!(xarray,[real(zv[v]),imag(zv[v]),Γv[v]])
    end
    return xarray
end

vortex_position_to_phys_space(zj,config::VortexConfig) = zj
vortex_position_to_phys_space(zj,config::VortexConfig{Bodies.ConformalBody}) = Elements.conftransform(zj,config.body)

"""
    generate_random_state(xr::Tuple,yr::Tuple,Γr::Tuple,config::VortexConfig) -> Vector{Float64}

Generate a random state, taking the parameters for the vortices from the ranges
`xr`, `yr`, `Γr`.
"""
function generate_random_state(xr::Tuple,yr::Tuple,Γr::Tuple,config::VortexConfig)
    zv_prior, Γv_prior = createclusters(config.Nv,1,xr,yr,Γr,0.0,0.0;body=config.body)
    vort_prior = Vortex.Blob.(zv_prior,Γv_prior,config.δ)
    return lagrange_to_state_reordered(vort_prior,config)
end

generate_random_states(nstates,xr::Tuple,yr::Tuple,Γr::Tuple,config::VortexConfig) =
    [generate_random_state(xr,yr,Γr,config) for i in 1:nstates]


# DEPENDS ON STATE ARRANGEMENT

"""
    lagrange_to_state_reordered(source::Vector{Element},config::VortexConfig) -> Vector{Float64}

Convert a set of point or blob vortex elements `source` to a state vector.
If this is a problem with a `ConformalBody`, then it assumes that the
element positions in `source` are in the circle plane.
"""
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

# DEPENDS ON STATE ARRANGEMENT

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

"""
    state_to_lagrange_reordered(state::AbstractVector,config::VortexConfig[;isblob=true]) -> Vector{Element}

Convert a state vector `state` to a vector of vortex elements. If this is a problem with
a `ConformalBody`, then the element positions are given in the circle plane.
"""
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

# DEPENDS ON STATE ARRANGEMENT

"""
    state_to_positions_and_strengths(state::AbstractVector,config::VortexConfig) -> Vector{ComplexF64}, Vector{Float64}

Convert the state vector `state` to vectors of positions and strengths. If
this is a problem with a `ConformalBody`, then it returns the positions in
the circle plane.
"""
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

# DEPENDS ON STATE ARRANGEMENT

function state_to_positions_and_strengths(state::AbstractVector{Float64}, config::VortexConfig)
  Nv = length(state) ÷ 3

  zv = [state[2i-1] + im*state[2i] for i in 1:Nv]
  Γv = [state[2Nv+i] for i in 1:Nv]

  return zv, Γv
end

# DEPENDS ON STATE ARRANGEMENT

function positions_and_strengths_to_state(zv::AbstractVector{ComplexF64},Γv::AbstractVector{Float64},config::VortexConfig)
  Nv = length(zv)
  state = zeros(3Nv)
  for i = 1:Nv
    state[2i-1] = real(zv[i])
    state[2i] = imag(zv[i])
    state[2Nv+i] = Γv[i]
  end
  return state
end

# DEPENDS ON STATE ARRANGEMENT

function positions_and_strengths_to_state(ζv::AbstractVector{ComplexF64},Γv::AbstractVector{Float64},config::VortexConfig{Bodies.ConformalBody})
  Nv = length(ζv)
  state = zeros(3Nv)
  for i = 1:Nv
    ri = abs(ζv[i])
    Θi = angle(ζv[i])
    state[i] = log(ri-1.0)
    state[Nv+i] = ri*Θi
    state[2*Nv+i] =  Γv[i]
  end
  return state
end
