using Clustering

export classify_by_yerr, classify_by_density


function classify_by_yerr(sol_collection::Vector{Vector{T}},obs::AbstractObservationOperator;
                            kcluster::Int = 5,
                            mincount::Int = 1,
                            maxerr::Float64 = 1.0e2,
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
  cnts[idex] >= mincount || error("Lowest-error group is not large enough")


  # get the indices of the members of the lowest-error group (goodones)
  goodones = findall(x -> x == idex,assignments(R))

  return sol_collection[goodones], minyerr, collection_yerr, goodones

end

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
  clustermembers = Int[]
  for v in biggest_vortex_cluster
    state_index = index_of_vortex_state(v,obs.config)
    push!(clustermembers,state_index)
  end
  sol_goodones = sol_collection[clustermembers]

  return sol_goodones, length(biggest_vortex_cluster)

end
