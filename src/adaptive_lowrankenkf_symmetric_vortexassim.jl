export  adaptive_lowrankenkf_symmetric_vortexassim


"""
This routine sequentially assimilates pressure observations collected along the x-axis at locations `config.ss` into the ensemble matrix `X`.
In this version, the no-flow through is enforced along the x axis, by using the method of images.
We augment the collection of `config.Nv`vortices with another set of `config.Nv` vortices at the conjugate positions with opposite circulation.
The assimilation is performed with the adaptive version of the low-rank ensemble Kalman filter (LREnKF) introduced in
Le Provost et al. "A low-rank ensemble Kalman filter for elliptic observations" (arXiv:2203.05120), 2022.
The user should provide the following arguments:
- `algo::LREnKF`: A variable with the parameters of the LREnKF
- `X::Matrix{Float64}`: `X` is an ensemble matrix whose columns hold the `Ne` samples of the joint distribution π_{h(X),X}.
   For each column, the first Ny rows store the observation sample h(x^i), and the remaining rows (row Ny+1 to row Ny+Nx) store the state sample x^i.
- `tspan::Tuple{S,S} where S <: Real`: a tuple that holds the start and final time of the simulation
- `config::VortexConfig`: A configuration file for the vortex simulation
- `data::SyntheticData`: A structure that holds the history of the state and observation variables
Optional arguments:
- `withfreestream::Bool`: equals `true` if a freestream is applied
- `rxdefault::Union{Nothing, Int64} = 100`: the truncated dimension of the informative subspace of the state space
- `rydefault::Union{Nothing, Int64} = 100`: the truncated dimension of the informative subspace of the observations space
- `isadaptive::Bool=false`: equals `true` if the ranks are not fixed but determined to capture at least `ratio` of
   the cumulative energy of the state and observation Gramians, see Le Provost et al., 2022 for further details.
- `ratio::Float64=0.95`: the ratio of cumulative energy of the state and observation Gramians to retain.
- `P::Parallel = serial`: Determine whether some steps of the routine can be runned in parallel.
                          In the current version of the code, only the serial version is validated.
"""
function adaptive_lowrankenkf_symmetric_vortexassim(algo::LREnKF, X, tspan::Tuple{S,S}, config::VortexConfig, data::SyntheticData;
	                        withfreestream::Bool = false, rxdefault::Union{Nothing, Int64} = 100, rydefault::Union{Nothing, Int64} = 100,
							isadaptive::Bool=false, ratio::Float64=0.95, P::Parallel = serial) where {S<:Real}

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

	# Cache variable for the velocities
	cachevels = allocate_velocity(state_to_lagrange(X[Ny+1:Ny+Nx,1], config))

	# Define the observation operator
	h(x, t) = measure_state_symmetric(x, t, config; withfreestream =  withfreestream)
	# Define an interpolation function in time and space of the true pressure field
	press_itp = CubicSplineInterpolation((LinRange(real(config.ss[1]), real(config.ss[end]), length(config.ss)),
	                                   t0:data.Δt:tf), data.yt, extrapolation_bc =  Line())

	# Pre allocate arrays for the sensitivity analysis
	Jac = zeros(Ny, 6*Nv)
	wtarget = zeros(ComplexF64, Ny)

	dpd = zeros(ComplexF64, Ny, 2*Nv)
	dpdstar = zeros(ComplexF64, Ny, 2*Nv)

	Css = zeros(ComplexF64, 2*Nv, 2*Nv)
	Cts = zeros(ComplexF64, Ny, 2*Nv)

	∂Css = zeros(2*Nv, 2*Nv)
	Ctsblob = zeros(ComplexF64, Ny, 2*Nv)
	∂Ctsblob = zeros(Ny, 2*Nv)

	yt(t) = press_itp(real.(config.ss), t)
	rxhist = Int64[]
	ryhist = Int64[]


	Xf = Array{Float64,2}[]
	push!(Xf, copy(state(X, Ny, Nx)))

	Xa = Array{Float64,2}[]
	push!(Xa, copy(state(X, Ny, Nx)))

	# Run the ensemble filter
	for i=1:length(Acycle)

		# Forecast step
		@inbounds for j=1:step
			tj = t0+(i-1)*Δtobs+(j-1)*Δtdyn
			X, _ = symmetric_vortex(X, tj, Ny, Nx, cachevels, config, withfreestream = withfreestream)
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

		# Pre-allocate the state and observation Gramians
		Cx = zeros(Nx, Nx)
		Cy  = zeros(Ny, Ny)

		# Compute marginally the standard deviation of the state ensemble
		Dx = Diagonal(std(X[Ny+1:Ny+Nx, :]; dims = 2)[:,1])
		Dϵ = config.ϵY*I

		# Compute the state and observation Gramians. The Jacobian of the pressure field is computed
		# analytically by exploiting the symmetry of the problem about the x-axis
		@inbounds Threads.@threads for j=1:Ne
			# @time Jac_AD = AD_symmetric_jacobian_pressure(config.ss, vcat(state_to_lagrange(X[Ny+1:Ny+Nx,j], config)...), t0+i*Δtobs)
			# Jac = analytical_jacobian_pressure(config.ss, vcat(state_to_lagrange(X[Ny+1:Ny+Nx,j], config)...), freestream, 1:config.Nv, t0+i*Δtobs)
			# analytical_jacobian_pressure!(Jac, wtarget, dpd, dpdstar, Css, Cts, ∂Css, Ctsblob, ∂Ctsblob,
			#                               config.ss, vcat(state_to_lagrange(X[Ny+1:Ny+Nx,j], config)...), freestream, 1:config.Nv, t0+i*Δtobs)
			if withfreestream == false
		    	symmetric_analytical_jacobian_pressure!(Jac, wtarget, dpd, dpdstar, Css, Cts, ∂Css, Ctsblob, ∂Ctsblob,
			                              config.ss, vcat(state_to_lagrange(X[Ny+1:Ny+Nx,j], config)...), 1:config.Nv, t0+i*Δtobs)
			else
				symmetric_analytical_jacobian_pressure!(Jac, wtarget, dpd, dpdstar, Css, Cts, ∂Css, Ctsblob, ∂Ctsblob,
											  config.ss, vcat(state_to_lagrange(X[Ny+1:Ny+Nx,j], config)...), freestream, 1:config.Nv, t0+i*Δtobs)
			end

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
		# Burgers G, Jan van Leeuwen P, Evensen G. 1998 Analysis scheme in the ensemble Kalman
        # filter. Monthly weather review 126, 1719–1724. Solve the linear system for b̆ ∈ R^{ry × Ne}:
		b̆ = (HXbrevepert*HXbrevepert' + ϵbrevepert*ϵbrevepert')\(U'*(Dϵ\(ystar .- (X[1:Ny,:] + ϵ))))
		# Lift result to the original space
		view(X,Ny+1:Ny+Nx,:) .+= Dx*V*(Xbrevepert*HXbrevepert')*b̆

		# Filter state
		if algo.isfiltered == true
			@inbounds for i=1:Ne
				x = view(X, Ny+1:Ny+Nx, i)
				x .= filter_state!(x, config)
			end
		end

		push!(Xa, deepcopy(state(X, Ny, Nx)))
	end

	return Xf, Xa, rxhist, ryhist
end
