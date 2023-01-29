export metropolis

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

    # Burn-in period. Don't collect data
    for j = 1:nchain
        z = copy(zi[j])
        for i in 1:burnin-1
            z, accept = mhstep(z,logp̃,propvars[j],β[j])
        end
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
      j = rand(1:nchain-1)
      k = rand(1:nchain-2)
      swaps[j] += 1
      jp1 = mod(j+k-1,nchain)+1
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
