export metropolis, MetropolisSolution, acceptance_rates, swapacceptance_rates, numchains

struct MetropolisSolution{CT,AT,LPT}
  data :: CT
  accepts :: AT
  logp :: LPT
  swaps :: Vector{Int}
  swap_accepts :: Vector{Int}

end

numchains(chains::MetropolisSolution) = length(chains.data)

acceptance_rates(chains::MetropolisSolution) = count.(chains.accepts)/length(chains.accepts[1])

swapacceptance_rates(chains::MetropolisSolution) = chains.swap_accepts./chains.swaps


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


function metropolis(zi::Vector{Vector{T}},nsamp::Integer,logp̃::Function,propvars::Vector{S};burnin=max(1,floor(Int,nsamp/2)),
                                                                                            β = ones(length(zi)),
                                                                                            process_state = x -> x) where {T <: Float64, S}

    x_data, accept_data, logp_data, swaps, swapaccepts = _metropolis(zi,nsamp,logp̃,propvars,burnin,β,process_state)
    x_ens = [BasicEnsembleMatrix(x) for x in x_data]

    return MetropolisSolution(x_ens,accept_data,logp_data,swaps,swapaccepts)

end

"""
    metropolis(zi::Vector,nsamp,p̃::Function,propvar) -> MetropolisSolution

Given an initial trial `zi`, carry out `nsamp` Metropolis-Hastings steps with
the unscaled distribution `p̃`. The proposal distribution is Gaussian, centered
at each step's starting value, with a variance of `propvar`.
"""
metropolis(zi::Vector{Float64},nsamp::Integer,logp̃::Function,propvar;burnin=max(1,floor(Int,nsamp/2)),process_state = x -> x) =
      metropolis([zi],nsamp,logp̃,[propvar];burnin=burnin,β = [1.0],process_state=process_state)



"""
    metropolis(zi::Vector,nsamp,p̃::Function,propvar,nchain[;burnin,process_state,β]) -> Vector{MetropolisSolution}

Given an initial trial `zi`, carry out `nsamp` Metropolis-Hastings steps with
the unscaled distribution `p̃`. The proposal distribution is Gaussian, centered
at each step's starting value, with a variance of `propvar`. The burn-in period
defaults to the first half of the samples. The state is processed (modified) at each step
in the chain with `process_state`, which defaults to the identity.

This version allows multiple chains with parallel tempering. The inverse temperatures
of the tempering are set by optional vector `β`, which defaults to 5^0 through 5^(1-nchain).
The variance `propvar` is scaled by the corresponding `β` for each chain.
"""
metropolis(zi::Vector{Float64},nsamp::Integer,logp̃::Function,propvar,nchain;burnin=max(1,floor(Int,nsamp/2)),
                                                                            process_state = x -> x,
                                                                            β = 5.0.^(range(0,1-nchain,length=nchain))) =
      metropolis([zi for n = 1:nchain],nsamp,logp̃,[propvar/β[i] for i = 1:nchain];burnin=burnin,β=β,process_state=process_state)


metropolis(chains::MetropolisSolution,nsamp::Integer,logp̃,propvar;β = 5.0.^(range(0,1-numchains(chains),length=numchains(chains))),kwargs...) =
      metropolis([copy(x[:,end]) for x in chains.data],nsamp,logp̃,[propvar/β[i] for i = 1:numchains(chains)];β = β,kwargs...)


function _metropolis(zi::Vector{Vector{T}},nsamp::Integer,logp̃::Function,propvars::Vector{S},burnin,β,process_state) where {T <: Float64, S}
    nchain = length(zi)
    n = length(first(zi))
    z_data = [zeros(n,nsamp-burnin+1) for j in 1:nchain]
    accept_data = [zeros(Bool,nsamp-burnin+1) for j in 1:nchain]
    logp_data = [zeros(Float64,nsamp-burnin+1) for j in 1:nchain]
    swaps = zeros(Int,nchain-1)
    swapaccepts = zeros(Int,nchain-1)

    # Burn-in period. Don't collect data
    for j = 1:nchain
        z = copy(zi[j])
        accept = false
        for i in 1:burnin-1
            z, accept = mhstep(z,logp̃,propvars[j],β[j])
            z = process_state(z)
        end
        z_data[j][:,1] = copy(z)
        accept_data[j][1] = accept
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

      # Look for swaps, choosing two random ones
      if nchain > 1
        chainj = rand(1:nchain-1)
        swaps[chainj] += 1
        k = rand(1:nchain-1)
        chaink = mod(chainj+k-1,nchain)+1
        zj = copy(z_data[chainj][:,i])
        zk = copy(z_data[chaink][:,i])
        logjzj = logp_data[chainj][i]
        logkzk = logp_data[chaink][i]
        logjzk = β[chainj]*logp̃(zk)
        logkzj = β[chaink]*logp̃(zj)
        a = logjzk + logkzj - logjzj - logkzk
        swapaccept = a > log(rand())
        if swapaccept
          z_data[chainj][:,i] .= zk
          z_data[chaink][:,i] .= zj
          logp_data[chainj][i] = logkzj
          logp_data[chaink][i] = logjzj
          swapaccepts[chainj] += 1
        end
      end
    end
    return z_data, accept_data, logp_data, swaps, swapaccepts
end
