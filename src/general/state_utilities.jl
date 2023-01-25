export collect_estimated_states


"""
    collect_estimated_states(collection::Vector{Vector{AbstractENKFSolution}},config::VortexConfig) -> Array

Takes a vector of ENKF solution histories and returns the ensemble mean
of the final solution in each history. The output is a matrix of dimension
length(state) x length(collection).
"""
function collect_estimated_states(collection::Vector{Vector{T}},config::VortexConfig) where {T <: AbstractENKFSolution}
    Nv = config.Nv
    #xarray = zeros(3,config.Nv*length(collection))
    xarray = zeros(state_length(config),length(collection))

    for j in eachindex(collection)
        solhist = collection[j]
        laststate = solhist[end]
        #xarray[:,(j-1)*Nv+1:j*Nv] = state_to_vortex_states(mean(laststate.X),config)
        xarray[:,j] = mean(laststate.X)
    end
    return xarray
end

states_to_vortex_states(state_array::BasicEnsembleMatrix, config::VortexConfig) =
    states_to_vortex_states(state_array.X,config)
