export VortexConfig, state_to_lagrange, lagrange_to_state, state_length, construct_state_mapping,
          state_to_positions_and_strengths, positions_and_strengths_to_state,
          state_to_vortex_states, states_to_vortex_states, state_covariance


abstract type ImageType end
abstract type Body <: ImageType end
abstract type Cylinder <: ImageType end
abstract type NoWall <: ImageType end
abstract type FlatWall <: ImageType end # For case in which images are simply included as extra terms
abstract type OldFlatWall <: ImageType end # Meant for case in which images are explicitly created



"""
    VortexConfig

A structure to hold the parameters of the vortex simulations

## Fields
- `Nv::Int64`: number of vortices
- `state_id`: Look-up for state component indices
- `body::ConformalBody` : body (if present)
- `U::ComplexF64`: freestream velocity
- `Δt::Float64`: time step
- `δ::Float64` : blob radius
- `advect_flag::Bool`: true if the vortex system should be advected
"""
struct VortexConfig{WT,BT,SID}

    "Number of vortices"
    Nv::Int64

    "State IDs"
    state_id::SID

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

function VortexConfig(Nv,U,Δt,δ;advect_flag=true,body=nothing)
  state_id = construct_state_mapping(Nv,body)
  VortexConfig{_walltype(body),typeof(body),typeof(state_id)}(Nv,state_id,body,U,Δt,δ,advect_flag)
end

function VortexConfig(Nv,δ;body=nothing)
  state_id = construct_state_mapping(Nv,body)
  VortexConfig{_walltype(body),typeof(body),typeof(state_id)}(Nv,state_id,body,complex(0.0),0.0,δ,false)
end

_walltype(body::Bodies.ConformalBody) = Body
_walltype(body) = body


Base.length(config::VortexConfig) = config.Nv

# Mapping from vortex elements and other degrees of freedom to the state components
construct_state_mapping(Nv,body) = construct_state_mapping(Nv)
construct_state_mapping(Nv,::Bodies.ConformalBody) = construct_state_mapping_conformal(Nv)


function construct_state_mapping(Nv::Int64)
  state_id = Dict()
  vortex_x_ids = zeros(Int,Nv)
  vortex_y_ids = zeros(Int,Nv)
  vortex_Γ_ids = zeros(Int,Nv)
  for j in 1:Nv
    vortex_x_ids[j] = 3j-2
    vortex_y_ids[j] = 3j-1
    vortex_Γ_ids[j] = 3j
  end
  state_id["vortex x"] = vortex_x_ids
  state_id["vortex y"] = vortex_y_ids
  state_id["vortex Γ"] = vortex_Γ_ids

  return state_id
end

function construct_state_mapping_conformal(Nv::Int64)
  state_id = Dict()
  vortex_logr_ids = zeros(Int,Nv)
  vortex_rΘ_ids = zeros(Int,Nv)
  vortex_Γ_ids = zeros(Int,Nv)
  for j in 1:Nv
    vortex_logr_ids[j] = 3j-2
    vortex_rΘ_ids[j] = 3j-1
    vortex_Γ_ids[j] = 3j
  end
  state_id["vortex logr"] = vortex_logr_ids
  state_id["vortex rΘ"] = vortex_rΘ_ids
  state_id["vortex Γ"] = vortex_Γ_ids

  return state_id
end

"""
    state_length(config::VortexConfig)

Return the number of components of the state vector
"""
state_length(config::VortexConfig) =  state_length(config.state_id)

state_length(a::Dict) = mapreduce(key -> state_length(a[key]),+,keys(a))
state_length(a::Vector) = length(a)


"A routine to convert the state from a vector representation (State representation) to a set of vortex blobs and their mirrored images (Lagrangian representation).
 Use `lagrange_to_state` for the inverse transformation."
function state_to_lagrange(state::AbstractVector{Float64}, config::VortexConfig{OldFlatWall}; isblob::Bool=true)
    @unpack Nv = config

    zv, Γv = state_to_positions_and_strengths(state,config)

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

"""
    state_to_lagrange(state::AbstractVector,config::VortexConfig[;isblob=true]) -> Vector{Element}

Convert a state vector `state` to a vector of vortex elements. If this is a problem with
a `ConformalBody`, then the element positions are given in the circle plane.
"""
function state_to_lagrange(state::AbstractVector{Float64}, config::VortexConfig; isblob::Bool=true)

    zv, Γv = state_to_positions_and_strengths(state,config)

    if isblob #return collection of regularized point vortices
        blobs = length(zv) > 0 ? Vortex.Blob.(zv, Γv, config.δ) : Vortex.Blob[]

        return blobs
    else #return collection of point vortices
        points = length(zv) > 0 ? Vortex.Point.(zv, Γv) : Vortex.Point[]

        return points
    end
end



lagrange_to_state(source::Tuple, config::VortexConfig;kwargs...) = lagrange_to_state(source[1],config;kwargs...)

"A routine to convert a set of vortex blobs (Lagrangian representation) to a vector representation (State representation).
 Use `state_to_lagrange` for the inverse transformation."
function lagrange_to_state(source::Vector{T}, config::VortexConfig; withcylinder::Bool=false) where T <: PotentialFlow.Element
    @unpack Nv, state_id = config
    @assert length(source) == Nv

    states = Vector{Float64}(undef, state_length(config))

    x_ids = state_id["vortex x"]
    y_ids = state_id["vortex y"]
    Γ_ids = state_id["vortex Γ"]

    for i=1:Nv
        bi = source[i]
        states[x_ids[i]] = real(bi.z)
        states[y_ids[i]] = imag(bi.z)
        states[Γ_ids[i]] =  circulation(bi)
    end

    return states
end


function lagrange_to_state(source::Vector{T}, config::VortexConfig{Body}) where T <: PotentialFlow.Element
    @unpack Nv, state_id = config
    @assert length(source) == Nv

    states = Vector{Float64}(undef, state_length(config))

    logr_ids = state_id["vortex logr"]
    rϴ_ids = state_id["vortex rΘ"]
    Γ_ids = state_id["vortex Γ"]

    for i=1:Nv
        bi = source[i]
        zi = Elements.position(bi)
        ri, Θi = abs(zi), angle(zi)
        states[logr_ids[i]] = log(ri-1.0)
        states[rϴ_ids[i]] = ri*Θi
        states[Γ_ids[i]] =  strength(bi)
    end

    return states
end


"""
    state_to_positions_and_strengths(state::AbstractVector,config::VortexConfig) -> Vector{ComplexF64}, Vector{Float64}

Convert the state vector `state` to vectors of positions and strengths. If
this is a problem with a `ConformalBody`, then it returns the positions in
the circle plane.
"""
function state_to_positions_and_strengths(state::AbstractVector{Float64}, config::VortexConfig{Body})
  @unpack Nv, state_id = config

  logr_ids = state_id["vortex logr"]
  rϴ_ids = state_id["vortex rΘ"]
  Γ_ids = state_id["vortex Γ"]

  ζv = zeros(ComplexF64,Nv)
  Γv = zeros(Float64,Nv)
  for i in 1:Nv
    rv = 1.0 + exp(state[logr_ids[i]])
    Θv = state[rϴ_ids[i]]/rv
    ζv[i] = rv*exp(im*Θv)
    Γv[i] = state[Γ_ids[i]]
  end
  return ζv, Γv
end

function state_to_positions_and_strengths(state::AbstractVector{Float64}, config::VortexConfig)
  @unpack Nv, state_id = config

  x_ids = state_id["vortex x"]
  y_ids = state_id["vortex y"]
  Γ_ids = state_id["vortex Γ"]

  zv = [state[x_ids[i]] + im*state[y_ids[i]] for i in 1:Nv]
  Γv = [state[Γ_ids[i]] for i in 1:Nv]

  return zv, Γv
end

function positions_and_strengths_to_state(zv::AbstractVector{ComplexF64},Γv::AbstractVector{Float64},config::VortexConfig)
  @unpack Nv, state_id = config
  state = zeros(state_length(state_id))

  x_ids = state_id["vortex x"]
  y_ids = state_id["vortex y"]
  Γ_ids = state_id["vortex Γ"]

  for i = 1:Nv
    state[x_ids[i]] = real(zv[i])
    state[y_ids[i]] = imag(zv[i])
    state[Γ_ids[i]] = Γv[i]
  end
  return state
end


function positions_and_strengths_to_state(ζv::AbstractVector{ComplexF64},Γv::AbstractVector{Float64},config::VortexConfig{Body})
  @unpack Nv, state_id = config
  state = zeros(state_length(state_id))

  logr_ids = state_id["vortex logr"]
  rϴ_ids = state_id["vortex rΘ"]
  Γ_ids = state_id["vortex Γ"]

  for i = 1:Nv
    ri = abs(ζv[i])
    Θi = angle(ζv[i])
    state[logr_ids[i]] = log(ri-1.0)
    state[rϴ_ids[i]] = ri*Θi
    state[Γ_ids[i]] =  Γv[i]
  end
  return state
end

function state_covariance(varx, vary, varΓ, config::VortexConfig)
  @unpack Nv, state_id = config

  x_ids = state_id["vortex x"]
  y_ids = state_id["vortex y"]
  Γ_ids = state_id["vortex Γ"]

  Σx_diag = zeros(Float64,state_length(config))
  for j = 1:Nv
    Σx_diag[x_ids[j]] = varx
    Σx_diag[y_ids[j]] = vary
    Σx_diag[Γ_ids[j]] = varΓ
  end
  return Diagonal(Σx_diag)
end

### OTHER WAYS OF DECOMPOSING STATES ###


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
    @unpack Nv = config
    zv, Γv = state_to_positions_and_strengths(state,config)
    xarray = zeros(3,Nv)
    for v in 1:Nv
        zv_v = vortex_position_to_phys_space(zv[v],config)
        xarray[:,v] .= [real(zv_v),imag(zv_v),Γv[v]]
        #push!(xarray,[real(zv[v]),imag(zv[v]),Γv[v]])
    end
    return xarray
end

states_to_vortex_states(state_array::BasicEnsembleMatrix, config::VortexConfig) =
    states_to_vortex_states(state_array.X,config)

vortex_position_to_phys_space(zj,config::VortexConfig) = zj
vortex_position_to_phys_space(zj,config::VortexConfig{Body}) = Elements.conftransform(zj,config.body)
