export RecipeInflation, filter_state!, senkf_vortexassim

struct RecipeInflation <: InflationType
    "Parameters"
    p::Array{Float64,1}
end

"""
A filter function to ensure that the point vortices stay above the x-axis, and retain a positive circulation.
This function would typically be used before and after the analysis step to enforce those constraints.
"""
function filter_state!(x, config::VortexConfig)
	@inbounds for j=1:config.Nv
		# Ensure that vortices stay above the x axis
		x[3*j-1] = clamp(x[3*j-1], 1e-2, Inf)
		# Ensure that the circulation remains positive
    	x[3*j] = clamp(x[3*j], 0.0, Inf)
	end
    return x
end

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
The assimilation is performed with the stochastic ensemble Kalman filter (sEnKF), see
Asch, Bocquet, and Nodet, "Data assimilation: methods, algorithms, and applications", Society for Industrial and Applied Mathematics, 2016.
The user should provide the following arguments:
- `algo::StochEnKF`: A variable with the parameters of the sEnKF
- `X::Matrix{Float64}`: `X` is an ensemble matrix whose columns hold the `Ne` samples of the joint distribution π_{h(X),X}.
   For each column, the first Ny rows store the observation sample h(x^i), and the remaining rows (row Ny+1 to row Ny+Nx) store the state sample x^i.
- `tspan::Tuple{S,S} where S <: Real`: a tuple that holds the start and final time of the simulation
- `config::VortexConfig`: A configuration file for the vortex simulation
- `data::SyntheticData`: A structure that holds the history of the state and observation variables
Optional arguments:
- `withfreestream::Bool`: equals `true` if a freestream is applied
- `P::Parallel = serial`: Determine whether some steps of the routine can be runned in parallel.
"""
function senkf_vortexassim(algo::StochEnKF, X, tspan::Tuple{S,S}, config::VortexConfig, data::SyntheticData; withfreestream::Bool = false, P::Parallel = serial) where {S<:Real}

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
	ystar = zeros(Ny)

	# Cache variable for the velocities
	cachevels = allocate_velocity(state_to_lagrange(X[Ny+1:Ny+Nx,1], config))

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

	# Run the ensemble filter
	@showprogress for i=1:length(Acycle)

		# Forecast step
		@inbounds for j=1:step
		   tj = t0+(i-1)*Δtobs+(j-1)*Δtdyn
		   X, _ = vortex(X, tj, Ny, Nx, cachevels, config, withfreestream = withfreestream)
		end

	   push!(Xf, deepcopy(state(X, Ny, Nx)))

	   # Get the true observation ystar
	   ystar .= yt(t0+i*Δtobs)

	   # Perform state inflation
	   ϵmul(X, Ny+1, Ny+Nx)
	   ϵx(X, Ny, Nx, config)

	   # Filter state
	   if algo.isfiltered == true
		   @inbounds for i=1:Ne
			   x = view(X, Ny+1:Ny+Nx, i)
			   x .= filter_state!(x, config)
		   end
	   end

	   # Evaluate the observation operator for the different ensemble members
	   observe(h, X, t0+i*Δtobs, Ny, Nx; P = P)

	   # Generate samples from the observation noise
	   ϵ = algo.ϵy.σ*randn(Ny, Ne) .+ algo.ϵy.m

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
	   b = (HXpert*HXpert' + ϵpert*ϵpert')\(ystar .- (X[1:Ny,:] + ϵ))

	   # Update the ensemble members according to:
	   # x^{a,i} = x^i - Σ_{X,Y}b^i, with b^i =  Σ_Y^{-1}(h(x^i) + ϵ^i - ystar)
	   view(X,Ny+1:Ny+Nx,:) .+= (Xpert*HXpert')*b

	   # Filter state
	   if algo.isfiltered == true
		   @inbounds for i=1:Ne
			   x = view(X, Ny+1:Ny+Nx, i)
			   x .= filter_state!(x, config)
		   end
	   end

	   push!(Xa, deepcopy(state(X, Ny, Nx)))
	end

	return Xf, Xa
end
