# This contains observation routines for potential fields,
#    related to solutions of ∇²ϕ = Q

export SourceConfig, PotentialObservations

struct SourceConfig{SID} <: SingularityConfig
  "Number of sources"
  Ns::Int64

  "State IDs"
  state_id::SID

  "Blob radius"
  δ::Float64

end

function SourceConfig(Ns,δ)
  state_id = construct_source_state_mapping(Ns)
  SourceConfig{typeof(state_id)}(Ns,state_id,δ)
end

number_of_singularities(config::SourceConfig) = config.Ns


function construct_source_state_mapping(Ns::Int64)
  state_id = Dict()
  source_x_ids = zeros(Int,Ns)
  source_y_ids = zeros(Int,Ns)
  source_Q_ids = zeros(Int,Ns)
  for j in 1:Ns
    source_x_ids[j] = 3j-2
    source_y_ids[j] = 3j-1
    source_Q_ids[j] = 3j
  end
  state_id["source x"] = source_x_ids
  state_id["source y"] = source_y_ids
  state_id["source Q"] = source_Q_ids

  state_id["source Q total"] = source_Q_ids[1]

  # Create the strength transform matrix
  #  T maps actual strengths to states
  #  inv T maps states to actual strengths
  Tmat = _strength_transform_matrix_identity(Ns)
  #Tmat = _strength_transform_matrix_sum(Ns)


  state_id["source Q transform"] = Tmat
  state_id["source Q inverse transform"] = inv(Tmat)

  return state_id
end

"""
    get_singularity_ids(v,config) -> Tuple{Int}

Return the global position and strength IDs in the state vector as
a tuple of 3 integers (e.g. xid, yid, Γid)
"""
function get_singularity_ids(v::Integer,config::SourceConfig)
  @unpack state_id = config
  Nv = number_of_singularities(config)

  @assert v <= Nv && v > 0

  x_ids = state_id["source x"]
  y_ids = state_id["source y"]
  Q_ids = state_id["source Q"]

  return x_ids[v], y_ids[v], Q_ids[v]
end



"""
    get_singularity_ids(config) -> Tuple{Tuple{Int}}

Return the global position and strength IDs in the state vector as
a tuple of 3 integers (e.g. xid, yid, Γid)
"""
function get_singularity_ids(config::SourceConfig)
  @unpack state_id = config

  x_ids = state_id["source x"]
  y_ids = state_id["source y"]
  Q_ids = state_id["source Q"]

  return x_ids, y_ids, Q_ids
end

get_inverse_strength_transform(config::SourceConfig) = config.state_id["source Q inverse transform"]



struct PotentialObservations{Nx,Ny,ST,CT} <: AbstractObservationOperator{Nx,Ny,true}
    sens::ST
    config::CT
end

"""
    PotentialObservations(sens::AbstractVector,config::SourceConfig)

Constructor to create an instance of potential field sensors. The locations of the
sensors are specified by `sens`, which should be given as a vector of
complex coordinates.
"""
function PotentialObservations(sens::AbstractVector,config::SourceConfig)
    return PotentialObservations{3*config.Ns,length(sens),typeof(sens),typeof(config)}(sens,config)
end

function observations(x::AbstractVector,t,obs::PotentialObservations)
  @unpack config, sens = obs
  s = state_to_lagrange(x,config)
  return _potential(sens,s,config)
end

function jacob!(J,x::AbstractVector,t,obs::PotentialObservations)
    @unpack config, sens = obs
    s = state_to_lagrange(x,config)
    return _potential_jacobian!(J,sens,s,config)
end

_potential(sens,s,config::SourceConfig) = analytical_potential(sens,s,config)


"""
    state_to_lagrange(state::AbstractVector,config::SourceConfig[;isblob=true]) -> Vector{Element}

Convert a state vector `state` to a vector of source elements.
"""
function state_to_lagrange(state::AbstractVector{Float64}, config::SourceConfig; isblob::Bool=true)

    zs, Qs = state_to_positions_and_strengths(state,config)

    if isblob #return collection of regularized point sources
        blobs = length(zs) > 0 ? Source.Blob.(zs, Qs, config.δ) : Source.Blob[]

        return blobs
    else #return collection of point sources
        points = length(zs) > 0 ? Source.Point.(zs, Qs) : Source.Point[]

        return points
    end
end



lagrange_to_state(source::Tuple, config::SourceConfig;kwargs...) = lagrange_to_state(source[1],config;kwargs...)

"A routine to convert a set of source blobs (Lagrangian representation) to a vector representation (State representation).
 Use `state_to_lagrange` for the inverse transformation."
function lagrange_to_state(source::Vector{T}, config::SourceConfig) where T <: PotentialFlow.Element
    @unpack Ns, state_id = config
    @assert length(source) == Ns

    states = Vector{Float64}(undef, state_length(config))

    x_ids = state_id["source x"]
    y_ids = state_id["source y"]
    Q_ids = state_id["source Q"]
    Tmat = state_id["source Q transform"]


    for i=1:Ns
        bi = source[i]
        states[x_ids[i]] = real(bi.z)
        states[y_ids[i]] = imag(bi.z)
        states[Q_ids[i]] =  flux(bi)
    end

    states[Q_ids] = Tmat*states[Q_ids]

    return states
end


function state_to_positions_and_strengths(state::AbstractVector{Float64}, config::SourceConfig)
  @unpack Ns, state_id = config

  x_ids = state_id["source x"]
  y_ids = state_id["source y"]
  Q_ids = state_id["source Q"]
  inv_Tmat = state_id["source Q inverse transform"]

  zs = [state[x_ids[i]] + im*state[y_ids[i]] for i in 1:Ns]
  Qs = [state[Q_ids[i]] for i in 1:Ns]

  Qs = inv_Tmat*Qs

  return zs, Qs
end

function positions_and_strengths_to_state(zs::AbstractVector{ComplexF64},Qs::AbstractVector{Float64},config::SourceConfig)
  @unpack Ns, state_id = config
  state = zeros(state_length(state_id))

  x_ids = state_id["source x"]
  y_ids = state_id["source y"]
  Q_ids = state_id["source Q"]
  Tmat = state_id["source Q transform"]

  Qs_transform = Tmat*Qs

  for i = 1:Ns
    state[x_ids[i]] = real(zs[i])
    state[y_ids[i]] = imag(zs[i])
    state[Q_ids[i]] = Qs_transform[i]
  end
  return state
end

"""
    state_covariance(varx, vary, varstrength, config::SourceConfig; varstrengthtot=varstrength)

Create a state covariance matrix with variances `varx`, `vary` and `varstrength`
for the x, y, and strength entries for every singularity.
"""
function state_covariance(varx, vary, varstrength, config::SourceConfig; varstrengthtot=varstrength)
  @unpack state_id = config
  N = number_of_singularities(config)

  x_ids = state_id["source x"]
  y_ids = state_id["source y"]
  Q_ids = state_id["source Q"]
  Qtot_id = state_id["source Q total"]

  Σx_diag = zeros(Float64,state_length(config))
  for j = 1:N
    Σx_diag[x_ids[j]] = varx
    Σx_diag[y_ids[j]] = vary
    Σx_diag[Q_ids[j]] = varstrength
  end
  Σx_diag[Qtot_id] = varstrengthtot

  return Diagonal(Σx_diag)
end


"""
    create_state_bounds(xr::Tuple,yr::Tuple,strengthr::Tuple,config::SingularityConfig[;Γtotr = Γr])

Create a vector of tuples (of length equal to the state vector) containing the
bounds of each type of vector component.
"""
function create_state_bounds(xr,yr,strengthr,config::SourceConfig;strtotr = strengthr)
    @unpack state_id = config
    N = number_of_singularities(config)

    bounds = [(-Inf,Inf) for i in 1:state_length(config)]

    x_ids = state_id["source x"]
    y_ids = state_id["source y"]
    Q_ids = state_id["source Q"]

    Qtot_id = state_id["source Q total"]

    for j = 1:N
      bounds[x_ids[j]] = xr
      bounds[y_ids[j]] = yr
      bounds[Q_ids[j]] = strengthr
    end
    bounds[Qtot_id] = strtotr

    return bounds
end




analytical_potential(ζ,s::Vector{T},config::SourceConfig;kwargs...) where {T<:Element} = analytical_potential(ζ,s;ϵ=config.δ)

analytical_potential(z::AbstractArray,s::Vector{T};kwargs...) where {T<:Union{PotentialFlow.Points.Point,PotentialFlow.Blobs.Blob}} = map(zj -> analytical_potential(zj,s;kwargs...),z)

function analytical_potential(z,s::Vector{T};kwargs...) where {T<:Union{PotentialFlow.Points.Point,PotentialFlow.Blobs.Blob}}
        out = 0.0
        for (j,sj) in enumerate(s)
            zj,Qj  = Elements.position(sj), flux(sj)
            out -= Qj*srcpot(z,zj;kwargs...)
        end
        return out
end


srcpot(z::Number,zs;ϵ=EPSILON_DEFAULT) = -0.25/π*log(abs2(z-zs)+ϵ^2)

srcpot(z::Vector{T},zs;kwargs...) where {T<:Number} = map(zi -> srcpot(zi,zs;kwargs...),z)
