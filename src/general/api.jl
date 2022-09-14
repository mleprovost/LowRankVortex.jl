export vortexinference, gramians, observations!, observations, adaptive_lowrank_enkf!, LowRankENKFSolution


struct LowRankENKFSolution{XT,YT,YYT,SIGXT,SIGYT,SYT,SXYT}
   X :: XT
   Xf :: XT
   Y :: YYT
   crit_ratio :: Float64
   V :: AbstractMatrix{Float64}
   U :: AbstractMatrix{Float64}
   Λx :: Vector{Float64}
   Λy :: Vector{Float64}
   rx :: Int64
   ry :: Int64
   Σx :: SIGXT
   Σy :: SIGYT
   Y̆ :: YT
   ΣY̆ :: SYT
   ΣX̆Y̆ :: SXYT
   yerr :: Float64
end

# Need to set this up differently so that it does not need `sens`, but
# has some other way to be informed of length and type of measurement vector
"""
    vortexinference(Nv,sens,ystar,xr::Tuple,yr::Tuple,Γr::Tuple,ϵmeas,ϵx,ϵy,ϵΓ,config::VortexConfig
                    [,Ne=50][,maxiter=50][,numsample=5],kwargs...)

Generate `numsample` trajectories of the vortex inference problem, using initial
states that are drawn from a uniform distribution in the ranges `xr`, `yr`, `Γr` for
the `Nv` vortices. The state covariance is set by the standard deviations `ϵx`,
`ϵy`, `ϵΓ`, and the measurement covariance by `ϵmeas`. Each of the trajectories will use an ensemble of `Ne` members
and will run for `maxiter` iterations. For `ConformalBody` problems,
`xr`, `yr` and `ϵx`, `ϵy` should be interpreted as describing the generalized state components
in the circle plane (``r`` and ``\\theta``).
"""
function vortexinference(Nv,sens,ystar,xr,yr,Γr,ϵmeas,ϵX,ϵY,ϵΓ,config::VortexConfig;linear_flag=true,Ne=50,maxiter=50,numsample=5,crit_ratio=1.0,β=1.02,inflate=true)
    Ny = length(sens)
    sol_collection = []

    #h = analytical_pressure
    #jacob! = analytical_pressure_jacobian!

    h = analytical_force
    jacob! = analytical_force_jacobian!

    Σϵ = Diagonal(ϵmeas^2*ones(Ny))
    Σx = Diagonal(vcat(ϵX^2*ones(Nv),ϵY^2*ones(Nv),ϵΓ^2*ones(Nv)))

    lk = ReentrantLock()

    Threads.@threads for i in 1:numsample
        # things to do for every sample from prior

        # generate new sample
        zv_prior, Γv_prior = createclusters(Nv,1,xr,yr,Γr,0.0,0.0;body=config.body)
        vort_prior = Vortex.Blob.(zv_prior,Γv_prior,config.δ)
        x_prior = lagrange_to_state_reordered(vort_prior,config)
        X0 = create_ensemble(Ne,x_prior,Σx)
        X = deepcopy(X0)
        Y = similar(X,dims=(Ny,Ne))

        rx_set = min(size(X0,1),Ny)
        ry_set = min(size(X0,1),Ny)

        # Initialize history
        solhist = []

        for i = 1:maxiter
            sol = adaptive_lowrank_enkf!(X,Σx,Y,Σϵ,ystar,h,jacob!,sens,config; linear_flag=linear_flag,crit_ratio=crit_ratio, rxdefault = rx_set, rydefault = ry_set, inflate=inflate, β = β)
            #ϵX = min(0.01*sol.yerr,0.05)
            #ϵΓ = min(0.01*sol.yerr,0.05)
            #Σx = Diagonal(vcat(ϵX^2*ones(Nv),ϵX^2*ones(Nv),ϵΓ^2*ones(Nv)));
            push!(solhist,deepcopy(sol))
        end
        lock(lk) do
          push!(sol_collection,deepcopy(solhist))
        end
    end
    return sol_collection

end


"""
    gramians(jacob!,sens,Σϵ,X,Σx,config) -> Matrix, Matrix

Compute the state and observation gramians Cx and Cy. The function `jacob!` should
take as inputs a matrix J (of size Ny x Nx), a Ny vector of measurement points (`sens`),
a vector of vortices, and the configuration data `config`.
"""
function gramians(jacob!,sens::AbstractVector,Σϵ,X::EnsembleMatrix{Nx,Ne},Σx,config::VortexConfig) where {Nx,Ne}

    Ny = length(sens)
    H = zeros(Float64,Ny,Nx)
    Cx = zeros(Float64,Nx,Nx)
    Cy = zeros(Float64,Ny,Ny)
    invDϵ = inv(sqrt(Σϵ))
    Dx = sqrt(Σx)

    fact = min(1.0,1.0/(Ne-1)) # why isn't this just 1/Ne? it's a first-order statistic
    #fact = min(1.0,1.0/Ne) # why isn't this just 1/Ne? it's a first-order statistic

    for j in 1:Ne

        # the next two lines should be combined into one step
        # that takes in X(j) and returns Hj
        vj = state_to_lagrange_reordered(X(j),config)
        jacob!(H,sens,vj,config)

        H̃ = invDϵ*H*Dx

        Cx .+= H̃'*H̃
        Cy .+= H̃*H̃'
    end
    return fact*Cx, fact*Cy
end

"""
    gramians_approx(jacob!,sens,Σϵ,X,Σx,config) -> Matrix, Matrix

Compute the state and observation gramians Cx and Cy using an approximation
in which we evaluate the jacobian at the mean of the ensemble `X`. The function `jacob!` should
take as inputs a matrix J (of size Ny x Nx), a Ny vector of measurement points (`sens`),
a vector of vortices, and the configuration data `config`.
"""
function gramians_approx(jacob!,sens::AbstractVector,Σϵ,X::EnsembleMatrix{Nx,Ne},Σx,config::VortexConfig) where {Nx,Ne}

    Ny = length(sens)
    H = zeros(Float64,Ny,Nx)
    Cx = zeros(Float64,Nx,Nx)
    Cy = zeros(Float64,Ny,Ny)
    invDϵ = inv(sqrt(Σϵ))
    Dx = sqrt(Σx)

    vmean = state_to_lagrange_reordered(mean(X),config)
    jacob!(H,sens,vmean,config)

    H̃ = invDϵ*H*Dx

    Cx .= H̃'*H̃
    Cy .= H̃*H̃'

    return Cx, Cy
end

"""
    observations!(Y::EnsembleMatrix,X::EnsembleMatrix,h::Function,sens,config)

Compute the observation function `h` for each of the states in `X` and place them in `Y`.
The function `h` should take as inputs a Ny-dimensional vector of measurement points (`sens`), a vector of vortices,
and the configuration data `config`.
"""
function observations!(Y::EnsembleMatrix{Ny,Ne},X::EnsembleMatrix{Nx,Ne},h,sens::AbstractVector,config::VortexConfig) where {Ny,Nx,Ne}

    @assert length(sens) == Ny "Invalid length of sensor point vector"

    for j in 1:Ne
        Y(j) .= observations(X(j),h,sens,config)
    end
    return Y
end

"""
    observations(x::AbstractVector,h::Function,sens,config) -> X

Compute the observation function `h` for state `x`.
The function `h` should take as inputs a vector of measurement points (`sens`), a vector of vortices,
and the configuration data `config`.
"""
function observations(x::AbstractVector,h,sens::AbstractVector,config::VortexConfig)
  vj = state_to_lagrange_reordered(x,config)
  return h(sens,vj,config)
end


"""
        adaptive_lowrank_enkf!(X,Σx,sens,Σϵ,ystar,h,jacob!)
"""
function adaptive_lowrank_enkf!(X::BasicEnsembleMatrix{Nx,Ne},Σx,Y::BasicEnsembleMatrix{Ny,Ne},Σϵ,ystar,h,jacob!,sens,config::VortexConfig;
                                  rxdefault = Nx, rydefault = Ny, crit_ratio = 1.0,
                                  inflate=true,β=1.0,linear_flag=true) where {Nx,Ny,Ne}
    inflate && additive_inflation!(X,Σx)
    inflate && multiplicative_inflation!(X,β)
    observations!(Y,X,h,sens,config)
    ϵ = create_ensemble(Ne,zeros(Ny),Σϵ)
    Xf = deepcopy(X)

    # Calculate error
    yerr = norm(ystar-mean(Y),Σϵ)
    #yerr = norm(ystar-Y+ϵ,Σϵ)

    if Ne == 1 || linear_flag
        H = zeros(Float64,Ny,Nx)

        # the next two lines should be combined into one step
        # that takes in X(j) and returns Hj
        vmean = state_to_lagrange_reordered(mean(X),config)

        # Remove `sens` from this.
        jacob!(H,sens,vmean,config)

        H̃ = inv(sqrt(Σϵ))*H*sqrt(Σx)

        sqrt_Σ̃x = sqrt(cov(whiten(X,Σx)))
        H̃ .= H̃*sqrt_Σ̃x

        F = svd(H̃)

        V = F.V
        U = F.U

        #rx = size(X,1)
        #ry = rx
        rx = crit_ratio < 1.0 ? findfirst(x-> x >= crit_ratio, cumsum(F.S.^2)./sum(F.S.^2)) : rxdefault
        ry = crit_ratio < 1.0 ? findfirst(x-> x >= crit_ratio, cumsum(F.S.^2)./sum(F.S.^2)) : rydefault

        Vr = V[:,1:rx]
        Ur = U[:,1:ry]
        Λ = Diagonal(F.S[1:rx])
        Λx = F.S[1:rx].^2
        Λy = copy(Λx)
    else
        # calculate Jacobian and its transformed version
        # These use `sens` to determine length of measurement vector
        #Cx, Cy = gramians(jacob!, sens, Σϵ, X, Σx, config)
        Cx, Cy = gramians_approx(jacob!, sens, Σϵ, X, Σx, config)


        V, Λx, _ = svd(Symmetric(Cx))  # Λx = Λ^2
        U, Λy, _ = svd(Symmetric(Cy))  # Λy = Λ^2

        # find reduced rank
        rx = crit_ratio < 1.0 ? findfirst(x-> x >= crit_ratio, cumsum(Λx)./sum(Λx)) : rxdefault
        ry = crit_ratio < 1.0 ? findfirst(x-> x >= crit_ratio, cumsum(Λy)./sum(Λy)) : rydefault

        # rank reduction
        Vr = V[:,1:rx]
        Ur = U[:,1:ry]

    end

    # calculate the innovation, plus measurement noise
    innov = ystar - Y + ϵ
    Y̆ = Ur'*inv(sqrt(Σϵ))*innov # should show which modes are still active
    #Y̆ = Ur'*whiten(innov, Σϵ)

    # perform the update
    if Ne == 1 || linear_flag
        ΣY̆ = Λ^2+I
        ΣX̆Y̆ = Λ
        sqrt_Σx = sqrt(Σx)*sqrt_Σ̃x
    else
        #X̆p = ensemble_perturb(Vr'*whiten(X,Σx))
        #HX̆p = ensemble_perturb(Ur'*whiten(Y,Σϵ))
        #ϵ̆p = ensemble_perturb(Ur'*whiten(ϵ,Σϵ))
        X̆p = Vr'*whiten(X,Σx)
        HX̆p = Ur'*whiten(Y,Σϵ)
        ϵ̆p = Ur'*whiten(ϵ,Σϵ)
        #X̆p = Vr'*inv(sqrt(Σx))*X
        #HX̆p = Ur'*inv(sqrt(Σϵ))*Y
        #ϵ̆p = Ur'*inv(sqrt(Σϵ))*ϵ
        CHH = cov(HX̆p)
        Cee = cov(ϵ̆p)

        ΣY̆ = CHH + Cee
        ΣX̆Y̆ = cov(X̆p,HX̆p) # should be analogous to Λ

        sqrt_Σx = sqrt(Σx)
    end
    X .+= sqrt_Σx*Vr*ΣX̆Y̆*(ΣY̆\Y̆)


    soln = LowRankENKFSolution(X,Xf,Y,crit_ratio,V,U,Λx,Λy,rx,ry,Σx,Σϵ,Y̆,ΣY̆,ΣX̆Y̆,yerr)

    return soln
end
