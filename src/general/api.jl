export vortexinference, gramians, adaptive_lowrank_enkf!, senkf!



"""
    vortexinference(ystar,xr::Tuple,yr::Tuple,Γr::Tuple,ϵmeas,ϵx,ϵy,ϵΓ,obs
                    [,Ne=50][,maxiter=50][,numsample=5],kwargs...)

Generate `numsample` trajectories of the vortex inference problem, using initial
states that are drawn from a uniform distribution in the ranges `xr`, `yr`, `Γr`. The state
covariance is set by the standard deviations `ϵx`,
`ϵy`, `ϵΓ`, and the measurement covariance by `ϵmeas`. Each of the trajectories will use an ensemble of `Ne` members
and will run for `maxiter` iterations. For `ConformalBody` problems,
`xr`, `yr` and `ϵx`, `ϵy` should be interpreted as describing the generalized state components
in the circle plane (``r`` and ``\\theta``).
"""
function vortexinference(ystar,xr,yr,Γr,ϵmeas,ϵX,ϵY,ϵΓ,obs::AbstractObservationOperator{Nx,Ny};linear_flag=true,Ne=50,maxiter=50,numsample=5,
                                                                                               lowrank=true,crit_ratio=1.0,β=1.02,inflate=true,errtol=1.0) where {Nx,Ny}
    @unpack config = obs
    @unpack Nv = config

    soltype = lowrank ? LowRankENKFSolution : ENKFSolution

    sol_collection = Vector{soltype}[]

    Σϵ = Diagonal(ϵmeas^2*ones(Ny))
    Σx = Diagonal(vcat(ϵX^2*ones(Nv),ϵY^2*ones(Nv),ϵΓ^2*ones(Nv)))

    #lk = ReentrantLock()

    #Threads.@threads for i in 1:numsample
    for i in 1:numsample

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
        solhist = soltype[]

        for i = 1:maxiter
            if lowrank
              sol = adaptive_lowrank_enkf!(X,Σx,Y,Σϵ,ystar,obs; linear_flag=linear_flag,crit_ratio=crit_ratio, rxdefault = rx_set, rydefault = ry_set, inflate=inflate, β = β)
            else
              sol = senkf!(X,Σx,Y,Σϵ,ystar,obs; inflate=inflate, β = β)
            end
            #ϵX = min(0.01*sol.yerr,0.05)
            #ϵΓ = min(0.01*sol.yerr,0.05)
            #Σx = Diagonal(vcat(ϵX^2*ones(Nv),ϵX^2*ones(Nv),ϵΓ^2*ones(Nv)));
            push!(solhist,deepcopy(sol))
            sol.yerr < errtol && break
        end
        #lock(lk) do
        #  push!(sol_collection,deepcopy(solhist))
        #end
        push!(sol_collection,deepcopy(solhist))

    end
    return sol_collection

end





"""
      adaptive_lowrank_enkf!(X,Σx,Y,Σϵ,ystar,obs)
"""
function adaptive_lowrank_enkf!(X::BasicEnsembleMatrix{Nx,Ne},Σx,Y::BasicEnsembleMatrix{Ny,Ne},Σϵ,ystar,obs::AbstractObservationOperator{Nx,Ny};
                                  rxdefault = Nx, rydefault = Ny, crit_ratio = 1.0,
                                  inflate=true,β=1.0,linear_flag=true) where {Nx,Ny,Ne}
    inflate && additive_inflation!(X,Σx)
    inflate && multiplicative_inflation!(X,β)
    observations!(Y,X,obs)
    ϵ = create_ensemble(Ne,zeros(Ny),Σϵ)
    Xf = deepcopy(X)

    # Calculate error
    yerr = norm(ystar-mean(Y),Σϵ)
    #yerr = norm(ystar-Y+ϵ,Σϵ)

    if Ne == 1 || linear_flag
        H = zeros(Float64,Ny,Nx)

        jacob!(H,mean(X),obs)

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
        #Cx, Cy = gramians(obs, Σϵ, X, Σx)
        Cx, Cy = gramians_approx(obs, Σϵ, X, Σx)


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

"""
      senkf!(X,Σx,Y,Σϵ,ystar,obs)
"""
function senkf!(X::BasicEnsembleMatrix{Nx,Ne},Σx,Y::BasicEnsembleMatrix{Ny,Ne},Σϵ,ystar,obs::AbstractObservationOperator{Nx,Ny};
                   inflate=true,β=1.0) where {Nx,Ny,Ne}
    inflate && additive_inflation!(X,Σx)
    inflate && multiplicative_inflation!(X,β)
    observations!(Y,X,obs)
    ϵ = create_ensemble(Ne,zeros(Ny),Σϵ)
    Xf = deepcopy(X)

    # Calculate error
    yerr = norm(ystar-mean(Y),Σϵ)
    #yerr = norm(ystar-Y+ϵ,Σϵ)

    # calculate the innovation, plus measurement noise
    innov = ystar - Y + ϵ
    Y̆ = innov

    X̆p = whiten(X,Σx)
    HX̆p = whiten(Y,Σϵ)
    ϵ̆p = whiten(ϵ,Σϵ)

    CHH = cov(HX̆p)
    Cee = cov(ϵ̆p)

    ΣY̆ = CHH + Cee
    ΣX̆Y̆ = cov(X̆p,HX̆p) # should be analogous to Λ

    sqrt_Σx = sqrt(Σx)
    X .+= ΣX̆Y̆*(ΣY̆\Y̆)

    soln = ENKFSolution(X,Xf,Y,Σx,Σϵ,Y̆,ΣY̆,ΣX̆Y̆,yerr)

    return soln
end
