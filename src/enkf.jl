export StochEnKFParameters, LREnKFParameters, RecipeInflation, LREnKF, filter_state!, dstateobs, dobsobs, enkf

const RXDEFAULT = 100
const RYDEFAULT = 100
const DEFAULT_ADAPTIVE_RATIO = 0.95

default_state_filter!(x, config) = x


const DEFAULT_STATE_FILTER = default_state_filter!

#### OLD ####

"""
A structure for the low-rank ensemble Kalman filter (LREnKF)

References:
Le Provost, Baptista, Marzouk, and Eldredge, "A low-rank ensemble Kalman filter for elliptic observations", arXiv preprint, 2203.05120, 2022.

## Fields
- `G::Function`: "Filter function"
- `ϵy::AdditiveInflation`: "Standard deviations of the measurement noise distribution"
- `Δtdyn::Float64`: "Time step dynamic"
- `Δtobs::Float64`: "Time step observation"
- `isfiltered::Bool`: "Boolean: is state vector filtered"
"""
struct LREnKF<:SeqFilter
    "Filter function"
    G::Function

    "Standard deviations of the measurement noise distribution"
    ϵy::AdditiveInflation

    "Time step dynamic"
    Δtdyn::Float64

    "Time step observation"
    Δtobs::Float64

    "Boolean: is state vector filtered"
    isfiltered::Bool
end

function LREnKF(G::Function, ϵy::AdditiveInflation,
    Δtdyn, Δtobs; isfiltered = false)
    @assert norm(mod(Δtobs, Δtdyn))<1e-12 "Δtobs should be an integer multiple of Δtdyn"
    return LREnKF(G, ϵy, Δtdyn, Δtobs, isfiltered)
end

# If no filtering function is provided, use the identity in the constructor.
function LREnKF(ϵy::AdditiveInflation,
    Δtdyn, Δtobs, Δtshuff; islocal = false)
    @assert norm(mod(Δtobs, Δtdyn))<1e-12 "Δtobs should be an integer multiple of Δtdyn"

    return LREnKF(x-> x, ϵy, Δtdyn, Δtobs, false)
end

function Base.show(io::IO, lrenkf::LREnKF)
	println(io,"LREnKF  with filtered = $(lrenkf.isfiltered)")
end


#### NEW ####

abstract type AbstractSeqFilter end

"""
A structure for parameters of the the stochastic ensemble Kalman filter (LREnKF)

## Fields
- `fdata::AbstractForecastOperator`: "Forecast data"
- `odata::AbstractObservationOperator`: "Observation data"
- `ϵy::AdditiveInflation`: "Standard deviations of the measurement noise distribution"
- `Δtdyn::Float64`: "Time step dynamic"
- `Δtobs::Float64`: "Time step observation"
Optional arguments:
- `islocal=false`: true if localization is desired
- `Lyy::Float64=1e10`: Distance above which to truncate sensor-to-sensor interactions
- `Lxy::Float64=1e10`: Distance above which to truncate state-to-sensor interactions
"""
struct StochEnKFParameters{islocal,FT,OT}<:AbstractSeqFilter
    "Forecast data"
    fdata::FT

    "Observation data"
    odata::OT

    "Standard deviations of the measurement noise distribution"
    ϵy::AdditiveInflation

    "Time step dynamic"
    Δtdyn::Float64

    "Time step observation"
    Δtobs::Float64

    "Sensor localization distance"
    Lyy::Float64

    "State-to-sensor localization distance"
    Lxy::Float64
end

function StochEnKFParameters(fdata::AbstractForecastOperator,odata::AbstractObservationOperator,ϵy::AdditiveInflation,
    Δtdyn, Δtobs; islocal = false, Lyy = 1.0e10, Lxy = 1.0e10)
    @assert norm(mod(Δtobs, Δtdyn))<1e-12 "Δtobs should be an integer multiple of Δtdyn"

    return StochEnKFParameters{islocal,typeof(fdata),typeof(odata)}(fdata,odata, ϵy, Δtdyn, Δtobs, Lyy, Lxy)
end

function Base.show(io::IO, ::StochEnKFParameters{islocal}) where {islocal}
	println(io,"Stochastic EnKF with localization = $(islocal)")
end

"""
A structure for the low-rank ensemble Kalman filter (LREnKF)

References:
Le Provost, Baptista, Marzouk, and Eldredge, "A low-rank ensemble Kalman filter for elliptic observations", arXiv preprint, 2203.05120, 2022.

## Fields
- `fdata::AbstractForecastOperator`: "Forecast data"
- `odata::AbstractObservationOperator`: "Observation data"
- `ϵy::AdditiveInflation`: "Standard deviations of the measurement noise distribution"
- `Δtdyn::Float64`: "Time step dynamic"
- `Δtobs::Float64`: "Time step observation"
Optional arguments:
- `rxdefault::Union{Nothing, Int64} = 100`: the truncated dimension of the informative subspace of the state space
- `rydefault::Union{Nothing, Int64} = 100`: the truncated dimension of the informative subspace of the observations space
- `isadaptive::Bool=true`: equals `true` if the ranks are not fixed but determined to capture at least `ratio` of
   the cumulative energy of the state and observation Gramians, see Le Provost et al., 2022 for further details.
- `ratio::Float64=0.95`: the ratio of cumulative energy of the state and observation Gramians to retain.

"""
struct LREnKFParameters{isadaptive,FT,OT}<:AbstractSeqFilter
    "Forecast data"
    fdata::FT

    "Observation data"
    odata::OT

    "Standard deviations of the measurement noise distribution"
    ϵy::AdditiveInflation

    "Time step dynamic"
    Δtdyn::Float64

    "Time step observation"
    Δtobs::Float64

    "Truncated dimension of the informative subspace of the state space"
    rxdefault::Int

    "Truncated dimension of the informative subspace of the observations space"
    rydefault::Int

    "Ratio of cumulative energy of the state and observation Gramians to retain"
    ratio::Float64
end


function LREnKFParameters(fdata::AbstractForecastOperator,odata::AbstractObservationOperator, ϵy::AdditiveInflation,
    Δtdyn, Δtobs; isadaptive = true, rxdefault::Int = RXDEFAULT, rydefault::Int = RYDEFAULT, ratio::Float64 = DEFAULT_ADAPTIVE_RATIO)
    @assert norm(mod(Δtobs, Δtdyn))<1e-12 "Δtobs should be an integer multiple of Δtdyn"
    return LREnKFParameters{isadaptive,typeof(fdata),typeof(odata)}(fdata,odata, ϵy, Δtdyn, Δtobs, rxdefault, rydefault, ratio)
end


function Base.show(io::IO, ::LREnKFParameters{isadaptive}) where isadaptive
	println(io,"LREnKF with adaptive = $(isadaptive)")
end




struct RecipeInflation <: InflationType
    "Parameters"
    p::Array{Float64,1}
end


# Legacy purposes
filter_state!(a...) = symmetry_state_filter!(a...)


# This function apply additive inflation to the state components only,
# not the measurements, X is an Array{Float64,2} or a view of it
"""
Applies the additive inflation `ϵX`, `ϵΓ` to the positions, strengths of the point vortices, respectively.
The rows Ny+1 to Ny+Nx of `X` contain the state representation for the different ensemble members.
`X` contains Ne columns (one per each ensemble member) of Ny+Nx lines. The associated observations are stored in the first Ny rows.
"""
function (ϵ::RecipeInflation)(X, Ny, Nx, config::VortexConfig)
	ϵX, ϵΓ = ϵ.p
	Nv = config.Nv
	@assert Nx == 3*Nv
	for col in eachcol(X)
		if config.Nv > 0
			for i in 1:Nv
				col[Ny + 3*(i-1) + 1: Ny + 3*(i-1) + 2] .+= ϵX*randn(2)
				col[Ny + 3*i] += ϵΓ*randn()
			end
		end
	end
end

"""
Applies the additive inflation `ϵX`, `ϵΓ` to the positions, strengths of the point vortices, respectively.
The vector `x` contains the state representation of the collection of point vortices.
"""
function (ϵ::RecipeInflation)(x::AbstractVector{Float64}, config::VortexConfig)
	ϵX, ϵΓ = ϵ.p
	Nv = config.Nv
	Nx = size(x, 1)

	@assert Nx == 3*Nv
	for i=1:Nv
		x[3*(i-1) + 1:3*(i-1) + 2] .+= ϵX*randn(2)
		x[3*(i-1) + 3] += ϵΓ*randn()
	end
end


"""
This routine sequentially assimilates pressure observations collected at locations `config.ss` into the ensemble matrix `X`.
The assimilation is performed with the ensemble Kalman filter (EnKF).
The user should provide the following arguments:
- `algo::SeqFilter`: A variable with the parameters of the sEnKF
- `X::Matrix{Float64}`: `X` is an ensemble matrix whose columns hold the `Ne` samples of the joint distribution π_{h(X),X}.
   For each column, the first Ny rows store the observation sample h(x^i), and the remaining rows (row Ny+1 to row Ny+Nx) store the state sample x^i.
- `tspan::Tuple{S,S} where S <: Real`: a tuple that holds the start and final time of the simulation
- `config::VortexConfig`: A configuration file for the vortex simulation
- `data::SyntheticData`: A structure that holds the history of the state and observation variables
The type of `algo` determines the type of EnKF to be used.

Optional arguments:
- `withfreestream::Bool`: equals `true` if a freestream is applied
- `P::Parallel = serial`: Determine whether some steps of the routine can be runned in parallel.
"""
function enkf(algo::AbstractSeqFilter, X, tspan::Tuple{S,S}, config::VortexConfig, data::SyntheticData; withfreestream::Bool = false, P::Parallel = serial) where {S<:Real}

  @unpack fdata, odata = algo

	# Define the inflation parameters
	ϵX = config.ϵX
	ϵΓ = config.ϵΓ
	β = config.β
	ϵY = config.ϵY
	ϵx = RecipeInflation([ϵX; ϵΓ])
	ϵmul = MultiplicativeInflation(β)

	Ny = size(config.ss,1)

	# Set the time step between two assimilation steps
	Δtobs = algo.Δtobs
	# Set the time step for the time marching of the dynamical system
	Δtdyn = algo.Δtdyn
	t0, tf = tspan
	step = ceil(Int, Δtobs/Δtdyn)

	n0 = ceil(Int64, t0/Δtobs) + 1
	J = (tf-t0)/Δtobs
	Acycle = n0:n0+J-1

	# Array dimensions
	Nypx, Ne = size(X)
	Nx = Nypx - Ny
  Nv = config.Nv
	ystar = zeros(Ny)

	# Define the observation operator
	h(x, t) = measure_state(x, t, config; withfreestream = withfreestream)
	# Define an interpolation function in time and space of the true pressure field
	press_itp = CubicSplineInterpolation((LinRange(real(config.ss[1]), real(config.ss[end]), length(config.ss)),
	                                   t0:data.Δt:tf), data.yt, extrapolation_bc =  Line())

	yt(t) = press_itp(real.(config.ss), t)
	Xf = Array{Float64,2}[]
	push!(Xf, copy(state(X, Ny, Nx)))

	Xa = Array{Float64,2}[]
	push!(Xa, copy(state(X, Ny, Nx)))

  Jac = zeros(Ny, Nx)

  # Cache variable for the velocities
  cachevels = allocate_forecast_cache(X,Nx,Ny,config,algo)
  Jac_data = allocate_jacobian_cache(Nv,Ny,algo)

  # Pre-allocate the state and observation Gramians
  Cx = allocate_state_gramian(Nx,algo)
  Cy = allocate_observation_gramian(Ny,algo)

  rxhist = Int64[]
	ryhist = Int64[]
  Cx_history = Array{Float64,2}[]
  Cy_history = Array{Float64,2}[]

  Gyy = sensor_localization_operator(algo)

	# Run the ensemble filter
	@showprogress for i=1:length(Acycle)

		# Forecast step
		@inbounds for j=1:step
		   tj = t0+(i-1)*Δtobs+(j-1)*Δtdyn
		   X, _ = symmetric_vortex(X, tj, Ny, Nx, cachevels, config, withfreestream = withfreestream)
       #forecast!(X,t,Δtdyn,fdata) # For this, X needs to be BasicEnsembleMatrix
		end

	   push!(Xf, deepcopy(state(X, Ny, Nx)))

     tnext = t0+i*Δtobs

	   # Get the true observation ystar
	   ystar .= yt(tnext)

     ## These steps belong as a general analysis step ##

	   # Perform state inflation
	   ϵmul(X, Ny+1, Ny+Nx)
	   ϵx(X, Ny, Nx, config)
     #inflate && multiplicative_inflation!(X,β)
     #inflate && additive_inflation!(X,Σx)

	   # Filter state
     apply_filter!(X,Ne,Nx,Ny,odata)


	   # Evaluate the observation operator for the different ensemble members
	   observe(h, X, tnext, Ny, Nx; P = P)
     #observations!(Y,X,tnext,odata)

	   # Generate samples from the observation noise
	   ϵ = algo.ϵy.σ*randn(Ny, Ne) .+ algo.ϵy.m
     #ϵ = create_ensemble(Ne,zeros(Ny),Σϵ)


     enkf_kalman_update!(algo,X,Cx_history,Cy_history,rxhist,ryhist,tnext,ϵ,ystar,Ne,Nx,Ny,Cx,Cy,Gyy,Jac,Jac_data,config,withfreestream)

	   # Filter state
     apply_filter!(X,Ne,Nx,Ny,odata)

     ######

	   push!(Xa, deepcopy(state(X, Ny, Nx)))
	end

	return Xf, Xa, rxhist, ryhist, Cx_history, Cy_history
end

#=
Generically, the user should supply
- An overloaded version of forecast! function for specific AbstractForecastOperator type (with internal cache)
- An overloaded version of observations! function for specific AbstractObservationOperator type (with internal cache)
- An overloaded version of jacob! function for specific AbstractObservationOperator
- A state filter function
- A function `true_observations(t,obsdata)` of supplying the truth data as a function of time.

The signature of the forecast operator should be
forecast!(x,t,foredata::AbstractForecastOperator)

The signature of the forecast operator should be
observations!(x,t,obsdata::AbstractObservationOperator)

The signature of the Jacobian operator should be
jacob!(J,x,t,obsdata::AbstractObservationOperator)

The signature of the state filter function should be
state_filter!(x,obsdata::AbstractObservationOperator)
=#

##### ANALYSIS STEPS VIA DIFFERENT FLAVORS OF ENKF #####

### Stochastic ENKF ####

enkf_kalman_update!(algo::StochEnKFParameters,args...) = _senkf_kalman_update!(algo,args...)
enkf_kalman_update!(algo::LREnKFParameters,args...) = _lrenkf_kalman_update!(algo,args...)


function _senkf_kalman_update!(algo,X,Cx_history,Cy_history,rxhist,ryhist,t,ϵ,ystar,Ne,Nx,Ny,Cx,Cy,Gyy,Jac,Jac_data,config,withfreestream)

  # The implementation of the stochastic EnKF follows
  # Form the perturbation matrix for the state
  Xpert = (1/sqrt(Ne-1))*(X[Ny+1:Ny+Nx,:] .- mean(X[Ny+1:Ny+Nx,:]; dims = 2)[:,1])
  # Form the perturbation matrix for the observation
  HXpert = (1/sqrt(Ne-1))*(X[1:Ny,:] .- mean(X[1:Ny,:]; dims = 2)[:,1])
  # Form the perturbation matrix for the observation noise
  ϵpert = (1/sqrt(Ne-1))*(ϵ .- mean(ϵ; dims = 2)[:,1])
  # Kenkf = Xpert*HXpert'*inv(HXpert*HXpert'+ϵpert*ϵpert')

  # Apply the Kalman gain based on the representers
  # Burgers G, Jan van Leeuwen P, Evensen G. 1998 Analysis scheme in the ensemble Kalman
  # filter. Monthly weather review 126, 1719–1724. Solve the linear system for b ∈ R^{Ny × Ne}:

  # Update the ensemble members according to:
  # x^{a,i} = x^i - Σ_{X,Y}b^i, with b^i =  Σ_Y^{-1}(h(x^i) + ϵ^i - ystar)

  Σy = HXpert*HXpert' + ϵpert*ϵpert'
  apply_sensor_localization!(Σy,Gyy,algo)
  b = Σy\(ystar .- (X[1:Ny,:] + ϵ))

  Σxy = Xpert*HXpert'
  apply_state_localization!(Σxy,X,algo)

  view(X,Ny+1:Ny+Nx,:) .+= Σxy*b

  return X
end

### Low-rank ENKF ####

function _lrenkf_kalman_update!(algo::LREnKFParameters{isadaptive},X,Cx_history,Cy_history,rxhist,ryhist,t,ϵ,ystar,Ne,Nx,Ny,Cx,Cy,Gyy,Jac,Jac_data,config,withfreestream) where isadaptive

  wtarget, dpd, dpdstar, Css, Cts, ∂Css, Ctsblob, ∂Ctsblob = Jac_data

  @unpack rxdefault, rydefault, ratio, odata = algo


  # Compute marginally the standard deviation of the state ensemble
  Dx = Diagonal(std(X[Ny+1:Ny+Nx, :]; dims = 2)[:,1])
  Dϵ = config.ϵY*I

  fill!(Cx,0.0)
  fill!(Cy,0.0)

  # Compute the state and observation Gramians. The Jacobian of the pressure field is computed
  # analytically by exploiting the symmetry of the problem about the x-axis
  @inbounds Threads.@threads for j=1:Ne
    # @time Jac_AD = AD_symmetric_jacobian_pressure(config.ss, vcat(state_to_lagrange(X[Ny+1:Ny+Nx,j], config)...), t0+i*Δtobs)
    # Jac = analytical_jacobian_pressure(config.ss, vcat(state_to_lagrange(X[Ny+1:Ny+Nx,j], config)...), freestream, 1:config.Nv, t0+i*Δtobs)
    # analytical_jacobian_pressure!(Jac, wtarget, dpd, dpdstar, Css, Cts, ∂Css, Ctsblob, ∂Ctsblob,
    #                               config.ss, vcat(state_to_lagrange(X[Ny+1:Ny+Nx,j], config)...), freestream, 1:config.Nv, t0+i*Δtobs)
    if withfreestream == false
        symmetric_analytical_jacobian_pressure!(Jac, wtarget, dpd, dpdstar, Css, Cts, ∂Css, Ctsblob, ∂Ctsblob,
                                  config.ss, vcat(state_to_lagrange(X[Ny+1:Ny+Nx,j], config)...), 1:config.Nv, t)
    else
      symmetric_analytical_jacobian_pressure!(Jac, wtarget, dpd, dpdstar, Css, Cts, ∂Css, Ctsblob, ∂Ctsblob,
                      config.ss, vcat(state_to_lagrange(X[Ny+1:Ny+Nx,j], config)...), Freestream(config.U), 1:config.Nv, t)
    end
    #jacob!(J,X(j))

    Jacj = view(Jac,:,1:3*config.Nv)
    Cx .+= 1/(Ne-1)*(inv(Dϵ)*Jacj*Dx)'*(inv(Dϵ)*Jacj*Dx)
    Cy .+= 1/(Ne-1)*(inv(Dϵ)*Jacj*Dx)*(inv(Dϵ)*Jacj*Dx)'
  end

  if typeof(rydefault)<:Int64
    ry = min(Ny, rydefault)
  end
  if typeof(rxdefault)<:Int64
    rx = min(Nx, rxdefault)
  end

  # Compute the eigenspectrum of Cx. For improved robustness, we use a SVD decomposition
  V, Λx, _ = svd(Symmetric(Cx))

  # Determine the rank rx to capture at least `ratio` of the cumulative energy of Cx
  if isadaptive == true
    tmpx = findfirst(x-> x >= ratio, cumsum(Λx)./sum(Λx))
    if typeof(tmpx) <: Nothing
      rx = 1
    else
      rx = copy(tmpx)
    end
    push!(rxhist, copy(rx))
  end

  # Extract the top (i.e. most energetic) rx eigenvectors of Cx
  V = V[:,1:rx]

  # Compute the eigenspectrum of Cy. For improved robustness, we use a SVD decomposition
  U, Λy, _ = svd(Symmetric(Cy))

  # Determine the rank ry to capture at least `ratio` of the cumulative energy of Cy
  if isadaptive == true
    tmpy = findfirst(x-> x >= ratio, cumsum(Λy)./sum(Λy))
    if typeof(tmpy) <: Nothing
      ry = 1
    else
      ry = copy(tmpy)
    end
    push!(ryhist, copy(ry))
  end

  # Extract the top ry eigenvectors of Cy
  U = U[:,1:ry]

  # Whiten and project the prior samples x^i by substracitng the empirical mean and rotating the samples by V^⊤ Σ_x^{-1/2}
  Xbreve = V'*(inv(Dx)*(X[Ny+1:Ny+Nx,:] .- mean(X[Ny+1:Ny+Nx,:]; dims = 2)[:,1]))
  # Form the perturbation matrix for the whitened state
  Xbrevepert = (1/sqrt(Ne-1))*(Xbreve .- mean(Xbreve; dims = 2)[:,1])

  # Whiten and project the observation samples h(x^i) by substracitng the empirical mean and rotating the samples by U^⊤ Σ_ϵ^{-1/2}
  HXbreve = U'*(inv(Dϵ)*(X[1:Ny,:] .- mean(X[1:Ny,:]; dims = 2)[:,1]))
  # Form the perturbation matrix for the whitened observation
  HXbrevepert = (1/sqrt(Ne-1))*(HXbreve .- mean(HXbreve; dims = 2)[:,1])

  # Whiten and project the observation noise samples ϵ^i by substracitng the empirical mean and rotating the samples by U^⊤ Σ_ϵ^{-1/2}
  ϵbreve = U'*(inv(Dϵ)*(ϵ .- mean(ϵ; dims =2)[:,1]))
  # Form the perturbation matrix for the whitened observation noise
  ϵbrevepert = (1/sqrt(Ne-1))*(ϵbreve .- mean(ϵbreve; dims = 2)[:,1])

  # Apply the Kalman gain in the projected space based on the representers
  # (Burgers G, Jan van Leeuwen P, Evensen G. 1998 Analysis scheme in the ensemble Kalman
      # filter. Monthly weather review 126, 1719–1724.) Solve the linear system for b̆ ∈ R^{ry × Ne}:
  b̆ = (HXbrevepert*HXbrevepert' + ϵbrevepert*ϵbrevepert')\(U'*(Dϵ\(ystar .- (X[1:Ny,:] + ϵ))))
  # Lift result to the original space
  view(X,Ny+1:Ny+Nx,:) .+= Dx*V*(Xbrevepert*HXbrevepert')*b̆

  push!(Cx_history,copy(Cx))
  push!(Cy_history,copy(Cy))

  return nothing

end

### Utilities ####

# Legacy version
function apply_filter!(X,Ne,Nx,Ny,odata::AbstractObservationOperator)
  @inbounds for i=1:Ne
    x = view(X, Ny+1:Ny+Nx, i)
    state_filter!(x, odata)
  end
  return X
end

function apply_filter!(X::BasicEnsembleMatrix{Nx,Ne},odata::AbstractObservationOperator) where {Nx,Ne}
  @inbounds for i=1:Ne
    state_filter!(X(i), odata)
  end
  return X
end

#### THESE WILL GET REPLACED BY forecast and observation operators ###
allocate_forecast_cache(X,Nx,Ny,config,::AbstractSeqFilter) = allocate_velocity(state_to_lagrange(X[Ny+1:Ny+Nx,1], config))

allocate_jacobian_cache(Nv,Ny,::StochEnKFParameters) = nothing

function allocate_jacobian_cache(Nv,Ny,::LREnKFParameters)
	wtarget = zeros(ComplexF64, Ny)

	dpd = zeros(ComplexF64, Ny, 2*Nv)
	dpdstar = zeros(ComplexF64, Ny, 2*Nv)

	Css = zeros(ComplexF64, 2*Nv, 2*Nv)
	Cts = zeros(ComplexF64, Ny, 2*Nv)

	∂Css = zeros(2*Nv, 2*Nv)
	Ctsblob = zeros(ComplexF64, Ny, 2*Nv)
	∂Ctsblob = zeros(Ny, 2*Nv)

  return wtarget, dpd, dpdstar, Css, Cts, ∂Css, Ctsblob, ∂Ctsblob
end
#########


allocate_state_gramian(Nx,::StochEnKFParameters) = nothing
allocate_observation_gramian(Ny,::StochEnKFParameters) = nothing

allocate_state_gramian(Nx,::LREnKFParameters) = zeros(Nx,Nx)
allocate_observation_gramian(Ny,::LREnKFParameters) = zeros(Ny,Ny)

function sensor_localization_operator(algo::StochEnKFParameters{true})
  @unpack odata, Lyy = algo
  dyy = dobsobs(odata)
  return gaspari.(dyy./Lyy)
end

sensor_localization_operator(::AbstractSeqFilter) = nothing

function apply_sensor_localization!(Σy,Gyy,::StochEnKFParameters{true})
  Σy .*= Gyy
  return nothing
end

apply_sensor_localization!(Σy,Gyy,::AbstractSeqFilter) = nothing

apply_state_localization!(Σxy,X,algo::StochEnKFParameters{true}) = apply_state_localization!(Σxy,X,algo.Lxy,algo.odata)

apply_state_localization!(Σxy,X,::AbstractSeqFilter) = nothing


###### THINGS THAT ARE SPECIFIC TO VORTEX PROBLEM #######
