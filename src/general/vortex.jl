export createclusters, pointcluster

"""
    createclusters(Nclusters,Npercluster,z0,Γ0,σcx,σcΓ,σx,σΓ;cluster_circle_radius=0.0,each_cluster_radius=0.0) -> Vector{ComplexF64}, Vector{Float64}

Return `Nclusters` clusters of points and their strengths. The centers of the clusters are distributed
randomly about center `z0` according to the standard deviation `σcx`, and their total strengths
deviate randomly by `σcΓ` from `Γ0` divided equally about the clusters. Within each cluster,
the points are distributed randomly about the cluster center accordingly to `σx`, with strengths
randomly distributed about the cluster strength (divided equally), with standard deviation `σΓ`.
If `cluster_circle_radius` is set to a value larger than 0, then the cluster centers are perturbed from
a circle of that radius. If `each_cluster_radius` is set larger than 0, then each cluster is
first distributed on a circle of that radius before being perturbed.
"""
function createclusters(Nclusters,Npercluster,z0,Γ0,σcx,σcΓ,σx,σΓ;cluster_circle_radius=0.0,each_cluster_radius=0.0)
    zc, Γc = pointcluster(Nclusters,z0,Γ0,σcx,σcΓ;circle_radius=cluster_circle_radius)

    zp = ComplexF64[]
    Γp = Float64[]
    for j in 1:Nclusters
        zpj, Γpj = pointcluster(Npercluster,zc[j],Γc[j],σx,σΓ;circle_radius=each_cluster_radius)
        append!(zp,zpj)
        append!(Γp,Γpj)
    end
    return zp, Γp
end

"""
    pointcluster(N,z0,Γ0,σx,σΓ;circle_radius=0.0) -> Vector{ComplexF64}, Vector{Float64}

Create a cluster of `N` points and their strengths, distributed randomly about the center `z0` according to standard
deviation `σx`, each with strengths deviating randomly by `σΓ` from `Γ0` divided equally about the points.
If `circle_radius` is set to a value larger than 0, then the points are perturbed from a circle of that radius.
"""
function pointcluster(N,z0,Γ0,σx,σΓ;circle_radius=0.0)
    @assert circle_radius >= 0.0 "circle_radius must be non-negative"
    dx = Normal(0.0,σx)
    dΓ = Normal(0.0,σΓ)

    θv = range(0,2π,length=N+1)[1:end-1] .+ π/2*mod(N,2)
    zcirc = N > 1 ? circle_radius*exp.(im*θv) : zeros(ComplexF64,1)
    zp = z0 .+ zcirc
    Γp = Γ0/N*ones(N)
    zp .+= rand(dx,N) .+ im*rand(dx,N)
    Γp .+= rand(dΓ,N)
    return zp, Γp
end
