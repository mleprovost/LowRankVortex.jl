export createclusters, pointcluster, vortexmoment

function vortexmoment(mom::Int,zv::AbstractVector,Γv::AbstractVector)
    @assert length(zv) == length(Γv) "Inconsistent lengths of vectors"
    sum = complex(0.0)
    for j in 1:length(zv)
      sum += Γv[j]*zv[j]^mom
    end
    return sum
end

"""
    createclusters(Nclusters,Npercluster,xr::Tuple,yr::Tuple,Γr::Tuple,σx,σΓ;body=nothing,each_cluster_radius=0.0) -> Vector{ComplexF64}, Vector{Float64}

Return `Nclusters` clusters of points and their strengths. The centers of the clusters are distributed
randomly in the box `xr` x `yr`. Their total strengths
are uniformly distributed in the range `Γr`. . Within each cluster,
the points are distributed randomly about the cluster center accordingly to `σx`, with strengths
randomly distributed about the cluster strength (divided equally), with standard deviation `σΓ`.
If `each_cluster_radius` is set larger than 0, then each cluster is
first distributed on a circle of that radius before being perturbed. If `body` is provided
with a `ConformalBody`, then points are uniformly
distributed circumferentially in the range `yr` and log-normally in the radial direction,
with a median (peak) at radial distance ``1 + \\sqrt{(r_{min}-1)(r_{max}-1)}``, where
`xr = (rmin,rmax)`.
"""
createclusters(Nclusters,Npercluster,xr::Tuple,yr::Tuple,Γr::Tuple,σx,σΓ;body=nothing,each_cluster_radius=0.0) =
      _createclusters(Nclusters,Npercluster,xr,yr,Γr,σx,σΓ,each_cluster_radius,body)


function _createclusters(Nclusters,Npercluster,xr::Tuple,yr::Tuple,Γr::Tuple,σx,σΓ,each_cluster_radius,b)
    #zc, Γc = pointcluster(Nclusters,z0,Γ0,σcx,σcΓ;circle_radius=cluster_circle_radius)
    zc = random_points_plane(Nclusters,xr...,yr...)
    Γmin, Γmax = Γr
    Γc = Γmin .+ (Γmax-Γmin)*rand(Nclusters)

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
    createclusters(Nclusters,Npercluster,rr::Tuple,Θr::Tuple,Γr::Tuple,σx,σΓ,b::ConformalBody;each_cluster_radius=0.0) -> Vector{ComplexF64}, Vector{Float64}

Return `Nclusters` clusters of points and their strengths about a unit circle, uniformly
distributed circumferentially in the range `Θr` and log-normally in the radial direction,
with a median (peak) at radial distance
``1 + \\sqrt{(r_{min}-1)(r_{max}-1)}``. Their total strengths
are uniformly distributed in the range `Γr`. Within each cluster,
the points are distributed randomly about the cluster center accordingly to `σx`, with strengths
randomly distributed about the cluster strength (divided equally), with standard deviation `σΓ`.
If `each_cluster_radius` is set larger than 0, then each cluster is
first distributed on a circle of that radius before being perturbed.
"""
function _createclusters(Nclusters,Npercluster,rr::Tuple,Θr::Tuple,Γr::Tuple,σx,σΓ,each_cluster_radius,b::Bodies.ConformalBody)
    #zc, Γc = pointcluster(Nclusters,z0,Γ0,σcx,σcΓ;circle_radius=cluster_circle_radius)
    zc = random_points_unit_circle(Nclusters,rr,Θr)
    Γmin, Γmax = Γr
    Γc = Γmin .+ (Γmax-Γmin)*rand(Nclusters)

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

"""
    random_points_plane(N,xmin,xmax,ymin,ymax) -> Vector{ComplexF64}

Create a set of `N` random points in the plane, uniformly
distributed in the complex plane inside the region [xmin,xmax] x [ymin,ymax]
"""
function random_points_plane(N,xmin,xmax,ymin,ymax)
    @assert xmax > xmin "xmax must be larger than xmin"
    @assert ymax > ymin "ymax must be larger than ymin"
    x = xmin .+ (xmax-xmin)*rand(N)
    y = ymin .+ (ymax-ymin)*rand(N)
    return x .+ im*y
end


"""
    random_points_unit_circle(N,rr::Tuple,Θr::Tuple) -> Vector{ComplexF64}

Create a set of `N` random points outside the unit circle, uniformly
distributed circumferentially in the range `Θr` and log-normally in the radial direction,
with a median (peak) at radial distance
``1 + \\sqrt{(r_{min}-1)(r_{max}-1)}`` (which must be larger than 1), and with almost all of the
points within ``r_{max}`` (which must be larger than ``r_{min}``)
"""
function random_points_unit_circle(N,rr::Tuple,Θr::Tuple)
    rmin, rmax = rr
    Θmin, Θmax = Θr
    @assert rmin > 1 "rmin must be larger than 1"
    @assert rmax > rmin "rmax must be larger than rmin"

    rmedian = 1.0 + sqrt((rmin-1)*(rmax-1))

    Θ = Θmin .+ (Θmax-Θmin).*rand(N)
    r = _random_radial_points_unit_circle(N,rmedian,rmax)

    return r.*exp.(im*Θ)
end

function _random_radial_points_unit_circle(N,rmedian::Float64,r4sig::Float64)
  z = randn(N)
  rscatter = ((r4sig-1)/(rmedian-1))^(1/4)
  σy = log(rscatter)
  r = 1.0 .+ (rmedian.-1.0).*exp.(σy*z)
  return r
end

_random_radial_points_unit_circle(rmedian,r4sig) = first(_random_points_unit_circle(1,rmedian,r4sig))
