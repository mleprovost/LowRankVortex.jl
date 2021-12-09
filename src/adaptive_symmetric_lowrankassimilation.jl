export adaptive_symmetric_lowrankvortexassim


# Create a function to perform the sequential assimilation for any sequential filter SeqFilter
function adaptive_symmetric_lowrankvortexassim(algo::SeqFilter, X, tspan::Tuple{S,S}, config::VortexConfig, data::SyntheticData;
							rxdefault::Union{Nothing, Int64} = 100, rydefault::Union{Nothing, Int64} = 100, isadaptive::Bool=false, ratio::Float64=0.95, israndomized::Bool=false, P::Parallel = serial) where {S<:Real}

	# Define the additive Inflation
	ϵX = config.ϵX
	ϵΓ = config.ϵΓ
	β = config.β
	ϵY = config.ϵY

	Ny = size(config.ss,1)

	ϵx = RecipeInflation([ϵX; ϵΓ])
	ϵmul = MultiplicativeInflation(β)
	# Set different times
	Δtobs = algo.Δtobs
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

	freestream = Freestream(0.0*im)

	cachevels = allocate_velocity(state_to_lagrange(X[Ny+1:Ny+Nx,1], config))

	h(x, t) = measure_state_symmetric(x, t, config)
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
	rxhist = Int64[]
	ryhist = Int64[]


	Xf = Array{Float64,2}[]
	push!(Xf, copy(state(X, Ny, Nx)))

	Xa = Array{Float64,2}[]
	push!(Xa, copy(state(X, Ny, Nx)))

	# Run particle filter
	for i=1:length(Acycle)
		# Forecast step
		@inbounds for j=1:step
			tj = t0+(i-1)*Δtobs+(j-1)*Δtdyn
			X, _ = symmetric_vortex(X, tj, Ny, Nx, cachevels, config)
		end

		push!(Xf, deepcopy(state(X, Ny, Nx)))

		# Get real measurement
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

		observe(h, X, t0+i*Δtobs, Ny, Nx; P = P)

		ϵ = algo.ϵy.σ*randn(Ny, Ne) .+ algo.ϵy.m

		# Low-rank basis
		Cx = zeros(Nx, Nx)
		Cy  = zeros(Ny, Ny)


		Dx = Diagonal(std(X[Ny+1:Ny+Nx, :]; dims = 2)[:,1])
		# Dx = I
		# Dϵ = Diagonal(std(ϵ; dims = 2)[:,1])
		Dϵ = config.ϵY*I
		# Dϵ = I

		@inbounds Threads.@threads for j=1:Ne
			# @time Jac_AD = AD_symmetric_jacobian_pressure(config.ss, vcat(state_to_lagrange(X[Ny+1:Ny+Nx,j], config)...), t0+i*Δtobs)
			# Jac = analytical_jacobian_pressure(config.ss, vcat(state_to_lagrange(X[Ny+1:Ny+Nx,j], config)...), freestream, 1:config.Nv, t0+i*Δtobs)
			# analytical_jacobian_pressure!(Jac, wtarget, dpd, dpdstar, Css, Cts, ∂Css, Ctsblob, ∂Ctsblob,
			#                               config.ss, vcat(state_to_lagrange(X[Ny+1:Ny+Nx,j], config)...), freestream, 1:config.Nv, t0+i*Δtobs)

		    symmetric_analytical_jacobian_pressure!(Jac, wtarget, dpd, dpdstar, Css, Cts, ∂Css, Ctsblob, ∂Ctsblob,
			                              config.ss, vcat(state_to_lagrange(X[Ny+1:Ny+Nx,j], config)...), 1:config.Nv, t0+i*Δtobs)

			# @show Jac_AD[1:3,1:3]
			# @show Jac[1:3,1:3]

			# @show norm(Jac_AD[:,1:3*config.Nv]-Jac[:,1:3*config.Nv])
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

		# Lbx, V = pheig(Symmetric(Cx), rank = rx)
		# V = reverse(V, dims = 2)
		# Λx, V = eigen(Symmetric(Cx); sortby = λ -> -λ)
		V, Λx, _ = svd(Symmetric(Cx))

		if isadaptive == true
			tmpx = findfirst(x-> x >= ratio, cumsum(Λx)./sum(Λx))
			if typeof(tmpx) <: Nothing
				rx = 1
			else
				rx = copy(tmpx)
			end
			push!(rxhist, copy(rx))
		end

		V = V[:,1:rx]
		# @show norm(V*V'-I), norm(V'*V-I)

		# Lby, U = pheig(Symmetric(Cy), rank = ry)
		# U = reverse(U, dims = 2)
		# Λy, U = eigen(Symmetric(Cy); sortby = λ -> -λ)
		U, Λy, _ = svd(Symmetric(Cy))

		if isadaptive == true
			tmpy = findfirst(x-> x >= ratio, cumsum(Λy)./sum(Λy))
			if typeof(tmpy) <: Nothing
				ry = 1
			else
				ry = copy(tmpy)
			end
			push!(ryhist, copy(ry))
		end

		U = U[:,1:ry]
		# U = I

		Xbreve = V'*(inv(Dx)*(X[Ny+1:Ny+Nx,:] .- mean(X[Ny+1:Ny+Nx,:]; dims = 2)[:,1]))
		Xbrevepert = (1/sqrt(Ne-1))*(Xbreve .- mean(Xbreve; dims = 2)[:,1])

		HXbreve = U'*(inv(Dϵ)*(X[1:Ny,:] .- mean(X[1:Ny,:]; dims = 2)[:,1]))
		HXbrevepert = (1/sqrt(Ne-1))*(HXbreve .- mean(HXbreve; dims = 2)[:,1])

		ϵbreve = U'*(inv(Dϵ)*(ϵ .- mean(ϵ; dims =2)[:,1]))
		ϵbrevepert = (1/sqrt(Ne-1))*(ϵbreve .- mean(ϵbreve; dims = 2)[:,1])

		"Low-rank analysis step with representers, Evensen, Leeuwen et al. 1998"

		# K̆ = Xbrevepert*HXbrevepert'*inv(HXbrevepert*HXbrevepert' + ϵbrevepert*ϵbrevepert')
		# Kcxcy =  Dx*V*K̆*U'*inv(Dϵ)
		# Xpert = (1/sqrt(Ne-1))*(X[Ny+1:Ny+Nx,:] .- mean(X[Ny+1:Ny+Nx,:]; dims = 2)[:,1])
		# HXpert = (1/sqrt(Ne-1))*(X[1:Ny,:] .- mean(X[1:Ny,:]; dims = 2)[:,1])
		# ϵpert = (1/sqrt(Ne-1))*(ϵ .- mean(ϵ; dims = 2)[:,1])
		# Kenkf = Xpert*HXpert'*inv(HXpert*HXpert'+ϵpert*ϵpert')
		# K̆enkf = V'*inv(Dx)*Kenkf*Dϵ*U
		# @show norm(Kcxcy - Kenkf), norm(Kcxcy - Kenkf)/norm(Kenkf)
		# @show cumsum(svd(Kenkf).S) ./ sum(svd(Kenkf).S)
		# @show norm(K̆ - K̆enkf), norm(K̆ - K̆enkf)/norm(K̆enkf)

		b̆ = (HXbrevepert*HXbrevepert' + ϵbrevepert*ϵbrevepert')\(U'*(Dϵ\(ystar .- (X[1:Ny,:] + ϵ))))
		view(X,Ny+1:Ny+Nx,:) .+= Dx*V*(Xbrevepert*HXbrevepert')*b̆

		# X = algo(X, ystar, t0+i*Δtobs)

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
