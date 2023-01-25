using Clustering

export classify_by_data_mismatch, classify_by_density

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
