using Clustering

export classify_by_data_mismatch, classify_by_density, metropolis

"""
    classify_by_data_mismatch(collection::Vector{Vector{AbstractENKFSolution}},obs::AbstractObservationOperator[;kluster=5,min_cluster_size=1,cluster_choice=nothing])

From a given collection of iterative ENKF solution trajectories, inspect
the final sqrt(data mismatch) of each trajectory and use this to perform K-means clustering
to classify them into different clusters. Return the cluster with the
lowest data mismatch. The routine returns a tuple of the indices of `collection` that
belong to this lowest-mismatch cluster, the sqrt(data mismatch) of this cluster,
and a vector of the final sqrt(data mismatch) for the entire collection.
The number of clusters is `kcluster`. If `cluster_choice` is set to an integer <= `kcluster`,
then it chooses this cluster to return indices for. If the lowest-mismatch cluster
has fewer than `min_cluster_size` members, then it throws an error.
"""
function classify_by_data_mismatch(sol_collection::Vector{Vector{T}},obs::AbstractObservationOperator;
                            kcluster::Int = 5,
                            min_cluster_size::Int = 1,
                            cluster_choice = nothing) where T <: AbstractENKFSolution

  # get the final y errors of each trajectory
  collection_yerr = map(x -> x[end].yerr,sol_collection)

  R = Clustering.kmeans(reshape(collection_yerr,1,length(collection_yerr)),kcluster)

  if isnothing(cluster_choice)
    minyerr, I = findmin(R.centers)
    idex = getindex(I,2)
  else
    I = sortperm(vec(R.centers))
    cluster_choice <= kcluster || error("Cluster choice not available")
    idex = I[cluster_choice]
    minyerr = R.centers[idex]
  end
  cnts = counts(R)
  cnts[idex] >= min_cluster_size || error("Lowest-error group is not large enough")

  # get the indices of the members of the lowest-error group (goodones)
  member_indices = findall(x -> x == idex,assignments(R))

  return member_indices, minyerr, collection_yerr

end

"""
    classify_by_density(collection::Vector{Vector{AbstractENKFSolution}},obs::AbstractObservationOperator[;cluster_size=0.05,min_cluster_size=1])

From a given collection of iterative ENKF solution trajectories, collect the
final mean states of each trajectory, re-arrange these into a list of 3-dimensional vortex
element states, and classify these by density using the dbscan algorithm. Find
the largest cluster and return the indices of the collection members
that each vortex in the largest cluster belongs to.
"""
function classify_by_density(sol_collection::Vector{Vector{T}},obs::AbstractObservationOperator;
                             cluster_size::Float64 = 0.05, min_cluster_size = 1, kwargs...) where T <: AbstractENKFSolution

  # form the state array from the provided trajectories
  xarray = collect_estimated_states(sol_collection,obs.config)

  # rearrange the state array into vortex states
  x_vlist = states_to_vortex_states(xarray,obs.config)

  # find cluster by density
  db = Clustering.dbscan(x_vlist,cluster_size; min_cluster_size = min_cluster_size, kwargs...)

  # find the largest cluster
  cnt, idx = findmax(x -> x.size,db)
  biggest_vortex_cluster = db[idx].core_indices
  member_indices = Int[]
  for v in biggest_vortex_cluster
    state_index = index_of_vortex_state(v,obs.config)
    push!(member_indices,state_index)
  end

  return member_indices

end

"""
    mhstep(zn) -> Vector{Number}

Given initial sample, generates new sample znp1, according to Metropolis-Hastings
algorithm
"""
function mhstep(zn::Vector{Float64},logp̃::Function,propvar,β)
    #q = MvNormal(zn,sqrt(propvar))
    #z_trial = rand(q)
    z_trial = zn .+ sqrt(propvar)*randn(length(zn))
    logp_zn = β*logp̃(zn)
    logp_trial = β*logp̃(z_trial)
    a = min(0.0,logp_trial-logp_zn)
    accept = a > log(rand())
    znp1 = copy(zn)
    if accept
        znp1 .= z_trial
    end
    return znp1, accept
end

"""
    metropolis(zi::Vector,nsamp,p̃::Distribution,propvar) -> BasicEnsembleMatrix

Given an initial trial `zi`, carry out `nsamp` Metropolis-Hastings steps with
the unscaled distribution `p̃`. The proposal distribution is Gaussian, centered
at each step's starting value, with a variance of `propvar`.
"""
function metropolis(zi::Vector{Float64},nsamp::Integer,logp̃::Function,propvar;burnin=max(1,floor(Int,nsamp/2)),β=1.0)
    z = copy(zi)
    z_chain = zeros(length(z),nsamp-burnin+1)
    accept_chain = zeros(Bool,nsamp-burnin+1)
    logp_chain = zeros(Float64,nsamp-burnin+1)
    for i in 1:burnin
        z, accept = mhstep(z,logp̃,propvar,β)
    end
    for i in 1:nsamp-burnin+1
        z, accept = mhstep(z,logp̃,propvar,β)
        z_chain[:,i] = z
        accept_chain[i] = accept
        logp_chain[i] = β*logp̃(z)
    end
    return BasicEnsembleMatrix(z_chain), accept_chain, logp_chain
end

function metropolis(zi::Vector{Vector{T}},nsamp::Integer,logp̃::Function,propvars::Vector{S};burnin=max(1,floor(Int,nsamp/2)),
                                                                                                  β = ones(length(zi)),
                                                                                                  process_state = x -> x) where {T <: Float64, S}
    nchain = length(zi)
    n = length(first(zi))
    z_data = [zeros(n,nsamp-burnin+1) for j in 1:nchain]
    accept_data = [zeros(Bool,nsamp-burnin+1) for j in 1:nchain]
    logp_data = [zeros(Float64,nsamp-burnin+1) for j in 1:nchain]
    swaps = zeros(nchain-1)
    swapaccepts = zeros(nchain-1)

    for j = 1:nchain
        z = copy(zi[j])
        for i in 1:burnin-1
            z, accept = mhstep(z,logp̃,propvars[j],β[j])
        end
        newz = process_state(copy(z))
        z_data[j][:,1] = newz
        accept_data[j][1] = true
        logp_data[j][1] = β[j]*logp̃(z)
    end
    for i in 2:nsamp-burnin+1
      for j = 1:nchain
        z, accept = mhstep(z_data[j][:,i-1],logp̃,propvars[j],β[j])
        newz = process_state(copy(z))
        z_data[j][:,i] = newz
        accept_data[j][i] = accept
        logp_data[j][i] = β[j]*logp̃(z)
      end

      # Look for swaps
      j = rand(1:nchain-1)
      swaps[j] += 1
      jp1 = mod(j,nchain)+1
      zj = copy(z_data[j][:,i])
      zjp1 = copy(z_data[jp1][:,i])
      logjzj = logp_data[j][i]
      logjp1zjp1 = logp_data[jp1][i]
      logjzjp1 = β[j]*logp̃(zjp1)
      logjp1zj = β[jp1]*logp̃(zj)
      a = logjzjp1 + logjp1zj - logjzj - logjp1zjp1
      swapaccept = a > log(rand())
      if swapaccept
        z_data[j][:,i] .= zjp1
        z_data[jp1][:,i] .= zj
        logp_data[j][i] = logjp1zjp1
        logp_data[jp1][i] = logjzj
        swapaccepts[j] += 1
      end
    end
    return z_data, accept_data, logp_data, swaps, swapaccepts
end
