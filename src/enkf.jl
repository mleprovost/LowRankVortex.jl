export RecipeInflation, LREnKF, filter_state!, dstateobs, dobsobs

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

### Localization routines ###
function dstateobs(X, Ny, Nx, config::VortexConfig)
    Nypx, Ne = size(X)
    @assert Nypx == Ny + Nx
    @assert Ny == length(config.ss)
    Nv = config.Nv
    dXY = zeros(Nv, Ny, Ne)

    for i=1:Ne
        xi = X[Ny+1:Ny+Nx, i]
        zi = map(l->xi[3*l-2] + im*xi[3*l-1], 1:Nv)

        for J=1:Nv
            for k=1:Ny
                dXY[J,k,i] = abs(zi[J] - config.ss[k])
            end
        end
    end
    return mean(dXY, dims = 3)[:,:,1]
end

function dobsobs(config::VortexConfig)
    Ny = length(config.ss)
    dYY = zeros(Ny, Ny)
    # Exploit symmetry of the distance matrix dYY
    for i=1:Ny
        for j=1:i-1
            dij = abs(config.ss[i] - config.ss[j])
            dYY[i,j] = dij
            dYY[j,i] = dij
        end
    end
    return dYY
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
function enkf()
end
