export gramians, observations!, adaptive_lowrank_enkf!

pressure(z,v,config::VortexConfig{WT}) where {WT} =
    pressure(z,v;ϵ=config.δ,walltype=WT)

analytical_pressure_jacobian!(J,target,source,config::VortexConfig{WT}) where {WT} =
    analytical_pressure_jacobian!(J,target,source;ϵ=config.δ,walltype=WT)


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
    for j in 1:Ne

        # the next two lines should be combined into one step
        # that takes in X(j) and returns Hj
        vj = state_to_lagrange_reordered(X(j),config)
        jacob!(H,sens,vj,config)

        H̃ = invDϵ*H*Dx

        Cx .+= fact*H̃'*H̃
        Cy .+= fact*H̃*H̃'
    end
    return Cx, Cy
end

"""
    observations!(Y::EnsembleMatrix,X::EnsembleMatrix,h,sens,config) -> X

Compute the observations `h` for each of the states in `X` and place them in `Y`.
The function `h` should take as inputs a Ny-dimensional vector of measurement points (`sens`), a vector of vortices,
and the configuration data `config`.
"""
function observations!(Y::EnsembleMatrix{Ny,Ne},X::EnsembleMatrix{Nx,Ne},h,sens::AbstractVector,config::VortexConfig) where {Ny,Nx,Ne}

    @assert length(sens) == Ny "Invalid length of sensor point vector"

    for j in 1:Ne

        # The next two lines should be combined into one
        # step that just takes in X(j) and returns Y(j)
        vj = state_to_lagrange_reordered(X(j),config)
        Y(j) .= h(sens,vj,config)
    end
    return X
end


"""
        adaptive_lowrank_enkf!(X,Σx,sens,Σϵ,ystar,h,jacob!)
"""
function adaptive_lowrank_enkf!(X::BasicEnsembleMatrix{Nx,Ne},Σx,Y,Σϵ,ystar,h,jacob!,sens,config::VortexConfig; linear_flag=true) where {Nx,Ne}
    additive_inflation!(X,Σx)
    observations!(Y,X,h,sens,config)
    ϵ = create_ensemble(Ne,zeros(length(sens)),Σϵ)

    if Ne == 1 && linear_flag
        H = zeros(Float64,length(sens),size(X,1))

        # the next two lines should be combined into one step
        # that takes in X(j) and returns Hj
        vj = state_to_lagrange_reordered(X(1),config)
        jacob!(H,sens,vj,config)

        H̃ = inv(sqrt(Σϵ))*H*sqrt(Σx)
        F = svd(H̃)

        #r = findfirst(x-> x >= crit_ratio, cumsum(F.S)./sum(F.S))
        rx = size(X,1)
        ry = rx

        Vr = F.V[:,1:rx]
        Ur = F.U[:,1:ry]
        Λ = Diagonal(F.S[1:rx])
    else
        # calculate Jacobian and its transformed version
        Cx, Cy = gramians(jacob!, sens, Σϵ, X, Σx, config)
        push!(Cxhist,copy(Cx))
        push!(Cyhist,copy(Cy))
        V, Λx, _ = svd(Symmetric(Cx))  # Λx = Λ^2
        U, Λy, _ = svd(Symmetric(Cy))  # Λy = Λ^2

        # find reduced rank
        rx = size(Cx,1)
        #rx = findfirst(x-> x >= crit_ratio, cumsum(Λx)./sum(Λx))

        ry = size(Cy,1)
        #ry = findfirst(x-> x >= crit_ratio, cumsum(Λy)./sum(Λy))

        # rank reduction
        Vr = V[:,1:rx]
        Ur = U[:,1:ry]

    end

    # calculate the innovation, plus measurement noise
    innov = ystar - Y + ϵ
    Y̆ = Ur'*inv(sqrt(Σϵ))*innov # should show which modes are still active
    #Y̆ = Ur'*whiten(innov, Σϵ)

    # perform the update
    if Ne == 1
        ΣY̆ = Λ^2+I
        ΣX̆Y̆ = Λ
        #K̃ = Λ*inv(Λ^2+I)
        #K̃ = ΣX̆Y̆*inv(ΣY̆)
    else
        X̆p = ensemble_perturb(Vr'*whiten(X,Σx))
        HX̆p = ensemble_perturb(Ur'*whiten(Y,Σϵ))
        ϵ̆p = ensemble_perturb(Ur'*whiten(ϵ,Σϵ))

        ΣY̆ = cov(HX̆p) + cov(ϵ̆p)  # should be analogous to Λ^2 + I
        ΣX̆Y̆ = cov(X̆p,HX̆p) # should be analogous to Λ
    end
    X .+= sqrt(Σx)*Vr*ΣX̆Y̆*(ΣY̆\Y̆)

    yerr = norm(mean(innov))/norm(mean(ystar+ϵ))

    return Vr, Ur, rx, ry, Y̆, yerr
end