export symmetric_lowrankvortexassim


# Create a function to perform the sequential assimilation for any sequential filter SeqFilter
function symmetric_lowrankvortexassim(algo::SeqFilter, X, tspan::Tuple{S,S}, config::VortexConfig, data::SyntheticData;
	                        withfreestream::Bool=false, rxdefault::Int64 = 100, rydefault::Int64 = 100,
							israndomized::Bool=false, P::Parallel = serial) where {S<:Real}

	# Define the additive Inflation
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
	h(x, t) = measure_state_symmetric(x, t, config; withfreestream = withfreestream)
	# Define an interpolation function in time and space of the true pressure field
	press_itp = CubicSplineInterpolation((LinRange(real(config.ss[1]), real(config.ss[end]), length(config.ss)),
	                                   t0:data.Δt:tf), data.yt, extrapolation_bc =  Line())

	# Pre-allocate arrays for the sensitivity analysis
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
	Xf = Array{Float64,2}[]
	push!(Xf, copy(state(X, Ny, Nx)))

	Xa = Array{Float64,2}[]
	push!(Xa, copy(state(X, Ny, Nx)))

	# Run the ensemble filter
	@showprogress for i=1:length(Acycle)

		# Forecast step
		@inbounds for j=1:step
			tj = t0+(i-1)*Δtobs+(j-1)*Δtdyn
			X, _ = symmetric_vortex(X, tj, Ny, Nx, cachevels, config; withfreestream = withfreestream)
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
				                              config.ss, vcat(state_to_lagrange(X[Ny+1:Ny+Nx,j], config)...), freestream,
											  1:config.Nv, t0+i*Δtobs)
			end

			Jacj = view(Jac,:,1:3*config.Nv)
			Cx .+= 1/(Ne-1)*(inv(Dϵ)*Jacj*Dx)'*(inv(Dϵ)*Jacj*Dx)
			Cy .+= 1/(Ne-1)*(inv(Dϵ)*Jacj*Dx)*(inv(Dϵ)*Jacj*Dx)'
		end

		# Set the ranks of the informative subspaces in the observation and state spaces
		ry = min(Ny, rydefault)
		rx = min(Nx, rxdefault)

		# Compute the eigenspectrum of Cx. For improved robustness, we use a SVD decomposition
		V, Λx, _ = svd(Symmetric(Cx))
		# Extract the top (i.e. most energetic) rx eigenvectors of Cx
		V = V[:,1:rx]

		# Compute the eigenspectrum of Cy. For improved robustness, we use a SVD decomposition
		U, Λy, _ = svd(Symmetric(Cy))
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

	return Xf, Xa
end
