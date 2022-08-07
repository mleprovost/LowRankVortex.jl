export gramians, observations!

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
        vj = state_to_lagrange_reordered(X(j),config)
        Y(j) .= h(sens,vj,config)
    end
    return X
end
