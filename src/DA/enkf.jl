#=
Generically, the user should define subtypes of AbstractForecastOperator
and AbstractObservationOperator, and then define
- An overloaded version of `forecast` function for specific AbstractForecastOperator subtype (with internal cache)
- An overloaded version of `observations` function for specific AbstractObservationOperator subtype (with internal cache)
- An overloaded version of `jacob!`` function for specific AbstractObservationOperator subtype
- A state filter function for specific AbstractObservationOperator subtype
- A function `ytrue(t)` for supplying the truth data as a function of time.

The instances of the forecast and observation operator data are given as arguments
to an overarching EnKF parameters type, a subtype of AbstractSeqFilter. This
parameters type is also used to set algorithm metaparameters and, by dispatch, to
set the specific flavor of EnKF (e.g., StochEnKFParameters <: AbstractSeqFilter will automatically
choose a stochastic EnKF, whereas LREnKFParameters will specify low-rank EnKF.

The signature of the forecast operator should be
forecast(x,t,Δt,fdata::AbstractForecastOperator)

The signature of the forecast operator should be
observations(x,t,odata::AbstractObservationOperator)

The signature of the Jacobian operator should be
jacob!(J,x,t,odata::AbstractObservationOperator)

The signature of the state filter function should be
state_filter!(x,odata::AbstractObservationOperator)
=#


export StochEnKFParameters, LREnKFParameters, enkf

const RXDEFAULT = 100
const RYDEFAULT = 100
const DEFAULT_ADAPTIVE_RATIO = 0.95


abstract type AbstractSeqFilter end

"""
A structure for parameters of the the stochastic ensemble Kalman filter (sEnKF)

## Fields
- `fdata::AbstractForecastOperator`: "Forecast data"
- `odata::AbstractObservationOperator`: "Observation data"
- `ytrue::Function` : "Function that evaluates the truth data at any given time"
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

    "Truth data function"
    ytrue::Function

    "Time step dynamic"
    Δtdyn::Float64

    "Time step observation"
    Δtobs::Float64

    "Sensor localization distance"
    Lyy::Float64

    "State-to-sensor localization distance"
    Lxy::Float64
end

function StochEnKFParameters(fdata::AbstractForecastOperator,odata::AbstractObservationOperator,ytrue::Function,
    Δtdyn, Δtobs; islocal = false, Lyy = 1.0e10, Lxy = 1.0e10)
    @assert norm(mod(Δtobs, Δtdyn))<1e-12 "Δtobs should be an integer multiple of Δtdyn"

    return StochEnKFParameters{islocal,typeof(fdata),typeof(odata)}(fdata, odata, ytrue, Δtdyn, Δtobs, Lyy, Lxy)
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
- `ytrue::Function` : "Function that evaluates the truth data at any given time"
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

    "Truth data function"
    ytrue::Function

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


function LREnKFParameters(fdata::AbstractForecastOperator,odata::AbstractObservationOperator, ytrue::Function,
    Δtdyn, Δtobs; isadaptive = true, rxdefault::Int = RXDEFAULT, rydefault::Int = RYDEFAULT, ratio::Float64 = DEFAULT_ADAPTIVE_RATIO)
    @assert norm(mod(Δtobs, Δtdyn))<1e-12 "Δtobs should be an integer multiple of Δtdyn"
    return LREnKFParameters{isadaptive,typeof(fdata),typeof(odata)}(fdata, odata, ytrue, Δtdyn, Δtobs, rxdefault, rydefault, ratio)
end


function Base.show(io::IO, ::LREnKFParameters{isadaptive}) where isadaptive
	println(io,"LREnKF with adaptive = $(isadaptive)")
end


###########################
######## NEW ENKF #########
###########################




"""
This routine sequentially assimilates pressure observations collected at locations `config.ss` into the ensemble matrix `X`.
The assimilation is performed with the ensemble Kalman filter (EnKF).
The user should provide the following arguments:
- `algo::SeqFilter`: A variable with the parameters of the sEnKF
- `X::BasicEnsembleMatrix`: `X` is an ensemble matrix whose columns hold the `Ne` samples of the state distribution.
   For each column, the first Ny rows store the observation sample h(x^i), and the remaining rows (row Ny+1 to row Ny+Nx) store the state sample x^i.
- `Σx`: State covariance matrix
- `Σϵ`: Measurement noise covariance matrix
- `tspan::Tuple{S,S} where S <: Real`: a tuple that holds the start and final time of the simulation
The type of `algo` determines the type of EnKF to be used.

Optional arguments:
- `inflate::Bool = true`: True if additive/multiplicative inflation is desired
- `β::Float64 = 1.0`: Multiplicative inflation parameter
"""
function enkf(algo::AbstractSeqFilter, X::BasicEnsembleMatrix{Nx,Ne}, Σx, Σϵ, tspan::Tuple{S,S}; inflate::Bool = true, β = 1.0) where {Nx,Ne,S<:Real}
  @unpack fdata, odata, ytrue, Δtobs, Δtdyn = algo

  Ny = measurement_length(odata)

  Y = similar(X,dims=(Ny,Ne))
  Ny = measurement_length(odata)

	t0, tf = tspan
	step = ceil(Int, Δtobs/Δtdyn)

	n0 = ceil(Int64, t0/Δtobs) + 1
	J = (tf-t0)/Δtobs
	Acycle = n0:n0+J-1

	ystar = zeros(Ny)

	Xf = typeof(X)[]
	push!(Xf, deepcopy(X))

  Xa = typeof(X)[]
  push!(Xa, deepcopy(X))

  # Pre-allocate the Jacobian, state and observation Gramians
  Jac = allocate_jacobian(Nx,Ny,algo)
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
       forecast!(X,tj,Δtdyn,fdata)
		end
    push!(Xf, deepcopy(X))

     tnext = t0+i*Δtobs

	   # Get the true observation ystar
	   ystar .= ytrue(tnext)

     ## These steps belong as a general analysis step ##

	   # Perform state inflation
     inflate && additive_inflation!(X,Σx)
     inflate && multiplicative_inflation!(X,β)

	   # Filter state
     apply_filter!(X,odata)


	   # Evaluate the observation operator for the different ensemble members
     observations!(Y,X,tnext,odata)

	   # Generate samples from the observation noise
     ϵ = create_ensemble(Ne,zeros(Ny),Σϵ)

     enkf_kalman_update!(algo,X,Y,Σx,Σϵ,Cx_history,Cy_history,rxhist,ryhist,tnext,ϵ,ystar,Cx,Cy,Gyy,Jac)

	   # Filter state
     apply_filter!(X,odata)

     ######

	   push!(Xa, deepcopy(X))
	end

	return Xf, Xa, rxhist, ryhist, Cx_history, Cy_history
end

##### ANALYSIS STEPS VIA DIFFERENT FLAVORS OF ENKF #####

enkf_kalman_update!(algo::StochEnKFParameters,args...) = _senkf_kalman_update!(algo,args...)
enkf_kalman_update!(algo::LREnKFParameters,args...) = _lrenkf_kalman_update!(algo,args...)

### Stochastic ENKF ####

function _senkf_kalman_update!(algo,X::BasicEnsembleMatrix{Nx,Ne},Y::BasicEnsembleMatrix{Ny,Ne},Σx,Σϵ,Cx_history,Cy_history,rxhist,ryhist,t,ϵ,ystar,Cx,Cy,Gyy,Jac) where {Nx,Ny,Ne}

  yerr = norm(ystar-mean(Y),Σϵ)

  innov = ystar - Y + ϵ
  Y̆ = innov

  X̆p = whiten(X,Σx)
  HX̆p = whiten(Y,Σϵ)
  ϵ̆p = whiten(ϵ,Σϵ)

  CHH = cov(HX̆p)
  Cee = cov(ϵ̆p)

  ΣY̆ = CHH + Cee
  apply_sensor_localization!(ΣY̆,Gyy,algo)

  ΣX̆Y̆ = cov(X̆p,HX̆p) # should be analogous to Λ
  apply_state_localization!(ΣX̆Y̆,X,algo)

  X .+= ΣX̆Y̆*(ΣY̆\Y̆)

  return X
end

### Low-rank ENKF ####

function _lrenkf_kalman_update!(algo::LREnKFParameters{isadaptive},X::BasicEnsembleMatrix{Nx,Ne},Y::BasicEnsembleMatrix{Ny,Ne},Σx,Σϵ,Cx_history,Cy_history,rxhist,ryhist,t,ϵ,ystar,Cx,Cy,Gyy,Jac) where {isadaptive,Nx,Ny,Ne}

  @unpack rxdefault, rydefault, ratio, odata = algo

  yerr = norm(ystar-mean(Y),Σϵ)
  #yerr = norm(ystar-Y+ϵ,Σϵ)

  gramians!(Cx,Cy,Jac,odata, Σϵ, X, Σx, t)
  #gramians_approx!(Cx,Cy, Jac, odata, Σϵ, X, Σx, t)

  V, Λx, _ = svd(Symmetric(Cx))  # Λx = Λ^2
  U, Λy, _ = svd(Symmetric(Cy))  # Λy = Λ^2

  ry = min(Ny, rydefault)
  rx = min(Nx, rxdefault)

  # find reduced rank
  if isadaptive
    rx = ratio < 1.0 ? findfirst(x-> x >= ratio, cumsum(Λx)./sum(Λx)) : rxdefault
    ry = ratio < 1.0 ? findfirst(x-> x >= ratio, cumsum(Λy)./sum(Λy)) : rydefault
    rx = isnothing(rx) ? 1 : rx
    ry = isnothing(ry) ? 1 : ry
    push!(rxhist, rx)
    push!(ryhist, ry)
  end

  # rank reduction
  Vr = V[:,1:rx]
  Ur = U[:,1:ry]

  # calculate the innovation, plus measurement noise
  innov = ystar - Y + ϵ
  Y̆ = Ur'*inv(sqrt(Σϵ))*innov # should show which modes are still active
  #Y̆ = Ur'*whiten(innov, Σϵ)

  X̆p = Vr'*whiten(X,Σx)
  HX̆p = Ur'*whiten(Y,Σϵ)
  ϵ̆p = Ur'*whiten(ϵ,Σϵ)

  CHH = cov(HX̆p)
  Cee = cov(ϵ̆p)

  ΣY̆ = CHH + Cee
  ΣX̆Y̆ = cov(X̆p,HX̆p) # should be analogous to Λ

  sqrt_Σx = sqrt(Σx)
  X .+= sqrt_Σx*Vr*ΣX̆Y̆*(ΣY̆\Y̆)

  push!(Cx_history,copy(Cx))
  push!(Cy_history,copy(Cy))

  return nothing

end

"""
    gramians!(Cx,Cy,J,obs,Σϵ,X,Σx,t) -> Matrix, Matrix

Compute the state and observation gramians Cx and Cy.
"""
function gramians!(Cx,Cy,H,obs::AbstractObservationOperator{Nx,Ny},Σϵ,X::EnsembleMatrix{Nx,Ne},Σx,t) where {Nx,Ny,Ne}

    fill!(Cx,0.0)
    fill!(Cy,0.0)

    invDϵ = inv(sqrt(Σϵ))
    Dx = sqrt(Σx)

    fact = min(1.0,1.0/(Ne-1)) # why isn't this just 1/Ne? it's a first-order statistic
    #fact = min(1.0,1.0/Ne)

    for j in 1:Ne

        jacob!(H,X(j),t,obs)

        H .= invDϵ*H*Dx

        Cx .+= H'*H
        Cy .+= H*H'
    end
    Cx .*= fact
    Cy .*= fact
    return Cx, Cy
end

"""
    gramians_approx!(Cx,Cy,J,obs,Σϵ,X,Σx,t) -> Matrix, Matrix

Compute the state and observation gramians Cx and Cy using an approximation
in which we evaluate the jacobian at the mean of the ensemble `X`.
"""
function gramians_approx!(Cx,Cy,H,obs::AbstractObservationOperator{Nx,Ny},Σϵ,X::EnsembleMatrix{Nx,Ne},Σx,t) where {Nx,Ny,Ne}

    fill!(Cx,0.0)
    fill!(Cy,0.0)

    invDϵ = inv(sqrt(Σϵ))
    Dx = sqrt(Σx)

    jacob!(H,mean(X),t,obs)

    H .= invDϵ*H*Dx

    Cx .= H'*H
    Cy .= H*H'

    return Cx, Cy
end





### Utilities ####

function apply_filter!(X::BasicEnsembleMatrix{Nx,Ne},odata::AbstractObservationOperator) where {Nx,Ne}
  @inbounds for i=1:Ne
    state_filter!(X(i), odata)
  end
  return X
end


allocate_jacobian(Nx,Ny,::StochEnKFParameters) = nothing
allocate_state_gramian(Nx,::StochEnKFParameters) = nothing
allocate_observation_gramian(Ny,::StochEnKFParameters) = nothing

allocate_jacobian(Nx,Ny,::LREnKFParameters) = zeros(Ny,Nx)
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

# Calls a specialized version of apply_state_localization! based on observation operator
apply_state_localization!(Σxy,X,algo::StochEnKFParameters{true}) = apply_state_localization!(Σxy,X,algo.Lxy,algo.odata)

apply_state_localization!(Σxy,X,::AbstractSeqFilter) = nothing

# Caspari-Cohn kernel, Formula found in Data assimilation in the geosciences:
# An overview of methods, issues, and perspectives
g1(r) = 1 - (5/3)*r^2 +(5/8)*r^3 +(1/2)*r^4 -0.25*r^5
g2(r) = 4 - 5r +(5/3)*r^2 + (5/8)*r^3 -(1/2)*r^4 +(1/12)*r^5 -(2/3)*r^(-1)
gaspari(r) = abs(r)>2.0 ? 0.0 : abs(r)<1.0 ? g1(abs(r)) : g2(abs(r))
