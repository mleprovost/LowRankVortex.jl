export VortexConfig, state_to_lagrange, lagrange_to_state, state_length, construct_vortex_state_mapping,
          state_to_positions_and_strengths, positions_and_strengths_to_state, get_config_and_state,
          state_to_singularity_states, states_to_singularity_states, state_covariance,create_state_bounds,
          number_of_singularities, vorticity, get_singularity_ids


abstract type ImageType end
abstract type Body <: ImageType end
abstract type Cylinder <: ImageType end
abstract type NoWall <: ImageType end
abstract type FlatWall <: ImageType end # For case in which images are simply included as extra terms
abstract type OldFlatWall <: ImageType end # Meant for case in which images are explicitly created

# A few helper routines
strength(v::Union{PotentialFlow.Points.Point,PotentialFlow.Blobs.Blob}) = v.S
strength(v::Vector{T}) where {T<:Union{PotentialFlow.Points.Point,PotentialFlow.Blobs.Blob}} = map(vj -> strength(vj),v)



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
struct VortexConfig{WT,BT,SID} <: SingularityConfig

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

function VortexConfig(Nv,U,Δt,δ;advect_flag=true,body=nothing,blobstate=false)
  state_id = construct_vortex_state_mapping(Nv,body,Val(blobstate))
  VortexConfig{_walltype(body),typeof(body),typeof(state_id)}(Nv,state_id,body,U,Δt,δ,advect_flag)
end

function VortexConfig(Nv,δ;body=nothing,blobstate=false)
  state_id = construct_vortex_state_mapping(Nv,body,Val(blobstate))
  VortexConfig{_walltype(body),typeof(body),typeof(state_id)}(Nv,state_id,body,complex(0.0),0.0,δ,false)
end

_walltype(body::Bodies.ConformalBody) = Body
_walltype(body) = body


# Mapping from vortex elements and other degrees of freedom to the state components
construct_vortex_state_mapping(Nv,body,::Val{false}) = construct_vortex_state_mapping(Nv)
construct_vortex_state_mapping(Nv,::Bodies.ConformalBody,::Val{false}) = construct_vortex_state_mapping_conformal(Nv)
construct_vortex_state_mapping(Nv,body,::Val{true}) = construct_vortex_state_mapping_with_blobs(Nv)


function construct_vortex_state_mapping(Nv::Int64)
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

  state_id["vortex Γ total"] = vortex_Γ_ids[1]

  # Create the strength transform matrix
  #  T maps actual strengths to states
  #  inv T maps states to actual strengths
  Tmat = _strength_transform_matrix_identity(Nv)
  #Tmat = _strength_transform_matrix_sum(Nv)


  state_id["vortex Γ transform"] = Tmat
  state_id["vortex Γ inverse transform"] = inv(Tmat)

  return state_id
end

function construct_vortex_state_mapping_with_blobs(Nv::Int64)
  state_id = Dict()
  vortex_x_ids = zeros(Int,Nv)
  vortex_y_ids = zeros(Int,Nv)
  vortex_Γ_ids = zeros(Int,Nv)
  vortex_ϵ_ids = zeros(Int,Nv)
  for j in 1:Nv
    vortex_x_ids[j] = 4j-3
    vortex_y_ids[j] = 4j-2
    vortex_Γ_ids[j] = 4j-1
    vortex_ϵ_ids[j] = 4j
  end
  state_id["vortex x"] = vortex_x_ids
  state_id["vortex y"] = vortex_y_ids
  state_id["vortex Γ"] = vortex_Γ_ids
  state_id["vortex ϵ"] = vortex_ϵ_ids

  state_id["vortex Γ total"] = vortex_Γ_ids[1]

  # Create the strength transform matrix
  #  T maps actual strengths to states
  #  inv T maps states to actual strengths
  Tmat = _strength_transform_matrix_identity(Nv)
  #Tmat = _strength_transform_matrix_sum(Nv)


  state_id["vortex Γ transform"] = Tmat
  state_id["vortex Γ inverse transform"] = inv(Tmat)

  return state_id
end

function construct_vortex_state_mapping_conformal(Nv::Int64)
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

  state_id["vortex Γ total"] = vortex_Γ_ids[1]

  Tmat = _strength_transform_matrix_identity(Nv)
  #Tmat = _strength_transform_matrix_sum(Nv)


  state_id["vortex Γ transform"] = Tmat
  state_id["vortex Γ inverse transform"] = inv(T)

  return state_id
end

function _strength_transform_matrix_sum(Nv)
  T = zeros(Float64,Nv,Nv)
  T[1,:] .= 1.0
  for j = 2:Nv
    T[j,j-1] = 1.0
    T[j,j] = -1.0
  end
  return T
end

function _strength_transform_matrix_identity(Nv)
  T = zeros(Float64,Nv,Nv)
  for j = 1:Nv
    T[j,j] = 1.0
  end
  return T
end

"""
    state_length(config::AbstractConfig)

Return the number of components of the state vector
"""
state_length(config::AbstractConfig) =  state_length(config.state_id)

state_length(a::Dict) = mapreduce(key -> state_length(a[key]),+,keys(a))
state_length(a::Vector) = length(a)
state_length(a::Matrix) = 0
state_length(a::Int) = 0


number_of_singularities(config::VortexConfig) = config.Nv

function get_config_and_state(zv::AbstractVector,Γv::AbstractVector;δ = 0.01,body=LowRankVortex.NoWall,blobstate=false)
    Nv = length(zv)
    config = VortexConfig(Nv, δ, body=body, blobstate=blobstate)
    x = positions_and_strengths_to_state(zv,Γv,config)

    return config, x
end



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

    Tmat = state_id["vortex Γ transform"]

    for i=1:Nv
        bi = source[i]
        states[x_ids[i]] = real(bi.z)
        states[y_ids[i]] = imag(bi.z)
        states[Γ_ids[i]] =  circulation(bi)
    end

    states[Γ_ids] = Tmat*states[Γ_ids]

    return states
end


function lagrange_to_state(source::Vector{T}, config::VortexConfig{Body}) where T <: PotentialFlow.Element
    @unpack Nv, state_id = config
    @assert length(source) == Nv

    states = Vector{Float64}(undef, state_length(config))

    logr_ids = state_id["vortex logr"]
    rϴ_ids = state_id["vortex rΘ"]
    Γ_ids = state_id["vortex Γ"]

    Tmat =state_id["vortex Γ transform"]

    for i=1:Nv
        bi = source[i]
        zi = Elements.position(bi)
        ri, Θi = abs(zi), angle(zi)
        states[logr_ids[i]] = log(ri-1.0)
        states[rϴ_ids[i]] = ri*Θi
        states[Γ_ids[i]] =  strength(bi)
    end

    states[Γ_ids] = Tmat*states[Γ_ids]

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

  inv_Tmat =state_id["vortex Γ inverse transform"]

  ζv = zeros(ComplexF64,Nv)
  Γv = zeros(Float64,Nv)
  for i in 1:Nv
    rv = 1.0 + exp(state[logr_ids[i]])
    Θv = state[rϴ_ids[i]]/rv
    ζv[i] = rv*exp(im*Θv)
    Γv[i] = state[Γ_ids[i]]
  end
  Γv = inv_Tmat*Γv

  return ζv, Γv
end

function state_to_positions_and_strengths(state::AbstractVector{Float64}, config::VortexConfig)
  @unpack Nv, state_id = config

  x_ids = state_id["vortex x"]
  y_ids = state_id["vortex y"]
  Γ_ids = state_id["vortex Γ"]
  inv_Tmat = state_id["vortex Γ inverse transform"]

  zv = [state[x_ids[i]] + im*state[y_ids[i]] for i in 1:Nv]
  Γv = [state[Γ_ids[i]] for i in 1:Nv]

  Γv = inv_Tmat*Γv

  return zv, Γv
end

function positions_and_strengths_to_state(zv::AbstractVector{ComplexF64},Γv::AbstractVector{Float64},config::VortexConfig)
  @unpack Nv, state_id, δ = config
  state = zeros(state_length(state_id))

  x_ids = state_id["vortex x"]
  y_ids = state_id["vortex y"]
  Γ_ids = state_id["vortex Γ"]


  Tmat = state_id["vortex Γ transform"]

  Γv_transform = Tmat*Γv

  for i = 1:Nv
    state[x_ids[i]] = real(zv[i])
    state[y_ids[i]] = imag(zv[i])
    state[Γ_ids[i]] = Γv_transform[i]
  end

  if haskey(state_id,"vortex ϵ")
    ϵ_ids = state_id["vortex ϵ"]
    state[ϵ_ids] .= δ
  end

  return state
end


function positions_and_strengths_to_state(ζv::AbstractVector{ComplexF64},Γv::AbstractVector{Float64},config::VortexConfig{Body})
  @unpack Nv, state_id = config
  state = zeros(state_length(state_id))

  logr_ids = state_id["vortex logr"]
  rϴ_ids = state_id["vortex rΘ"]
  Γ_ids = state_id["vortex Γ"]
  Tmat = state_id["vortex Γ transform"]

  Γv_transform = Tmat*Γv

  for i = 1:Nv
    ri = abs(ζv[i])
    Θi = angle(ζv[i])
    state[logr_ids[i]] = log(ri-1.0)
    state[rϴ_ids[i]] = ri*Θi
    state[Γ_ids[i]] =  Γv_transform[i]
  end
  return state
end

"""
    state_covariance(varx, vary, varΓ, config::VortexConfig; varΓtot=varΓ)

Create a state covariance matrix with variances `varx`, `vary` and `varΓ`
for the x, y, and strength entries for every vortex.
"""
function state_covariance(varx, vary, varΓ, config::VortexConfig; varΓtot=varΓ)
  @unpack Nv, state_id = config

  x_ids = state_id["vortex x"]
  y_ids = state_id["vortex y"]
  Γ_ids = state_id["vortex Γ"]
  Γtot_id = state_id["vortex Γ total"]

  Σx_diag = zeros(Float64,state_length(config))
  for j = 1:Nv
    Σx_diag[x_ids[j]] = varx
    Σx_diag[y_ids[j]] = vary
    Σx_diag[Γ_ids[j]] = varΓ
  end
  Σx_diag[Γtot_id] = varΓtot

  return Diagonal(Σx_diag)
end

"""
    create_state_bounds(xr::Tuple,yr::Tuple,strengthr::Tuple,config::SingularityConfig[;Γtotr = Γr])

Create a vector of tuples (of length equal to the state vector) containing the
bounds of each type of vector component.
"""
function create_state_bounds(xr,yr,strengthr,config::VortexConfig;strtotr = strengthr)
    @unpack state_id = config
    N = number_of_singularities(config)

    bounds = [(-Inf,Inf) for i in 1:state_length(config)]

    x_ids = state_id["vortex x"]
    y_ids = state_id["vortex y"]
    Γ_ids = state_id["vortex Γ"]

    Γtot_id = state_id["vortex Γ total"]

    for j = 1:N
      bounds[x_ids[j]] = xr
      bounds[y_ids[j]] = yr
      bounds[Γ_ids[j]] = strengthr
    end
    bounds[Γtot_id] = strtotr

    return bounds
end

### OTHER WAYS OF DECOMPOSING STATES ###


"""
    states_to_singularity_states(state_array::Matrix,config::AbstractConfig)

Take an array of states (length(state) x nstates) and convert it to a (3 x Nv*nstates) array
of individual singularity states.
"""
function states_to_singularity_states(state_array::AbstractMatrix{Float64}, config::SingularityConfig)
   Ns = number_of_singularities(config)
   ndim, nstates = size(state_array)
   sing_array = zeros(3,Ns*nstates)
   for j in 1:nstates
     singstatej = state_to_singularity_states(state_array[:,j],config)
     sing_array[:,(j-1)*Ns+1:j*Ns] = singstatej
   end
   return sing_array
end


"""
    index_of_singularity_state(v::Integer,config::VortexConfig)

Return the column of a (length(state) x nstates) array of states
that a single singularity of index `v` in a (3 x Nv*nstates) singularity state array belongs to.
"""
index_of_singularity_state(v::Int,config::AbstractConfig) = (v-1)÷number_of_singularities(config)+1

"""
    state_to_singularity_states(state::AbstractVector,config::VortexConfig)

Given a state `state`, return a 3 x Nv array of the singularity states.
"""
function state_to_singularity_states(state::AbstractVector{Float64}, config::SingularityConfig)
    Nv = number_of_singularities(config)
    zv, strengthv = state_to_positions_and_strengths(state,config)
    xarray = zeros(3,Nv)
    for v in 1:Nv
        zv_v = singularity_position_to_phys_space(zv[v],config)
        xarray[:,v] .= [real(zv_v),imag(zv_v),strengthv[v]]
        #push!(xarray,[real(zv[v]),imag(zv[v]),Γv[v]])
    end
    return xarray
end

state_to_singularity_states(state_array::BasicEnsembleMatrix, config::VortexConfig) =
    state_to_singularity_states(state_array.X,config)

singularity_position_to_phys_space(zj,config::SingularityConfig) = zj
singularity_position_to_phys_space(zj,config::VortexConfig{Body}) = Elements.conftransform(zj,config.body)

"""
    get_singularity_ids(config) -> Tuple{Int}

Return the global position and strength IDs in the state vector as
a tuple of 3 integers (e.g. xid, yid, Γid)
"""
function get_singularity_ids(config::VortexConfig)
  @unpack state_id = config

  x_ids = state_id["vortex x"]
  y_ids = state_id["vortex y"]
  Γ_ids = state_id["vortex Γ"]

  return x_ids, y_ids, Γ_ids
end

function get_singularity_ids(config::VortexConfig{Body})
  @unpack state_id = config

  logr_ids = state_id["vortex logr"]
  rϴ_ids = state_id["vortex rΘ"]
  Γ_ids = state_id["vortex Γ"]

  return logr_ids, rϴ_ids, Γ_ids
end

"""
    get_singularity_ids(v,config) -> Tuple{Int}

Return the global position and strength IDs in the state vector as
a tuple of 3 integers (e.g. xid, yid, Γid)
"""
function get_singularity_ids(v::Integer,config::VortexConfig)
  @unpack state_id = config
  Nv = number_of_singularities(config)

  @assert v <= Nv && v > 0

  x_ids = state_id["vortex x"]
  y_ids = state_id["vortex y"]
  Γ_ids = state_id["vortex Γ"]

  return x_ids[v], y_ids[v], Γ_ids[v]
end

function get_singularity_ids(v::Integer,config::VortexConfig{Body})
  @unpack state_id = config
  Nv = number_of_singularities(config)

  @assert v <= Nv && v > 0

  logr_ids = state_id["vortex logr"]
  rϴ_ids = state_id["vortex rΘ"]
  Γ_ids = state_id["vortex Γ"]

  return logr_ids[v], rϴ_ids[v], Γ_ids[v]
end

get_inverse_strength_transform(config::VortexConfig) = config.state_id["vortex Γ inverse transform"]


"""
    blobfield(x,y,μ::AbstactVector,Σ::AbstractMatrix,config::VortexConfig)

Evaluate the blobfield at point x,y, given the mean state vector `μ` and uncertainty matrix `Σ`
"""
function blobfield(x,y,μ::Vector,Σ::AbstractMatrix,config::SingularityConfig)
    @unpack state_id = config
    Nv = number_of_singularities(config)

    x_ids, y_ids, strength_ids = get_singularity_ids(config)

    inv_Tmat = get_inverse_strength_transform(config)
    μ_trans = copy(μ)
    μ_trans[strength_ids] =  inv_Tmat*μ_trans[strength_ids]

    Σ_trans = copy(Σ)
    Σ_trans[x_ids,strength_ids] = Σ_trans[x_ids,strength_ids]*inv_Tmat
    Σ_trans[y_ids,strength_ids] = Σ_trans[y_ids,strength_ids]*inv_Tmat
    Σ_trans[strength_ids,x_ids] = inv_Tmat'*Σ_trans[strength_ids,x_ids]
    Σ_trans[strength_ids,y_ids] = inv_Tmat'*Σ_trans[strength_ids,y_ids]
    Σ_trans[strength_ids,strength_ids] = inv_Tmat'*Σ_trans[strength_ids,strength_ids]*inv_Tmat



    xvec = [x,y]
    w = 0.0
    for j = 1:Nv
        xidj, yidj, strengthidj = get_singularity_ids(j,config)
        μxj = μ_trans[[xidj,yidj]]
        Σxxj = Σ_trans[xidj:yidj,xidj:yidj]
        Σstrengthxj = Σ_trans[xidj:yidj,strengthidj]
        μstrengthj = μ_trans[strengthidj]
        wj = _blobfield(xvec,μxj,μstrengthj,Σxxj,Σstrengthxj)
        w += wj
    end
    return w
end


function _blobfield(xvec, μx, μstrength, Σxx, Σstrengthx)
    xvec_rel = xvec .- μx
    w = exp(-0.5*xvec_rel'*inv(Σxx)*xvec_rel)
    w *= (μstrength + Σstrengthx'*inv(Σxx)*xvec_rel)
    return w
end
