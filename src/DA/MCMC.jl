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
                                                                                            process_state = x -> x,k=1,sparsity=100,tolerance=0.0001,update_mean=1000) where {T <: Float64, S}

    x_data, accept_data, logp_data, swaps, swapaccepts = _metropolis(zi,nsamp,logp̃,propvars,burnin,β,process_state,k,sparsity,tolerance,update_mean)
    x_ens = [BasicEnsembleMatrix(x) for x in x_data]


    return MetropolisSolution(x_ens,accept_data,logp_data,swaps,swapaccepts)

end

"""
    metropolis(zi::Vector,nsamp,p̃::Function,propvar) -> MetropolisSolution

Given an initial trial `zi`, carry out `nsamp` Metropolis-Hastings steps with
the unscaled distribution `p̃`. The proposal distribution is Gaussian, centered
at each step's starting value, with a variance of `propvar`.
"""
metropolis(zi::Vector{Float64},nsamp::Integer,logp̃::Function,propvar;burnin=max(1,floor(Int,nsamp/2)),process_state = x -> x,kwargs...) =
      metropolis([zi],nsamp,logp̃,[propvar];burnin=burnin,β = [1.0],process_state=process_state,kwargs...)



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
                                                                            β = 5.0.^(range(0,1-nchain,length=nchain)),kwargs...) =
      metropolis([zi for n = 1:nchain],nsamp,logp̃,[propvar/β[i] for i = 1:nchain];burnin=burnin,β=β,process_state=process_state,kwargs...)


metropolis(chains::MetropolisSolution,nsamp::Integer,logp̃,propvar;β = 5.0.^(range(0,1-numchains(chains),length=numchains(chains))),kwargs...) =
      metropolis([copy(x[:,end]) for x in chains.data],nsamp,logp̃,[propvar/β[i] for i = 1:numchains(chains)];β = β,kwargs...)


 function _metropolis(zi::Vector{Vector{T}},nsamp::Integer,logp̃::Function,propvars::Vector{S},burnin,β,process_state,k,sparsity,tolerance,update_mean) where {T <: Float64, S}
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

      for i in 2:update_mean
          for j = 1:nchain
              z, accept = mhstep(z_data[j][:,i-1],logp̃,propvars[j],β[j])
              newz = process_state(copy(z))
              z_data[j][:,i] = newz
              accept_data[j][i] = accept
              logp_data[j][i] = β[j]*logp̃(z)
          end
      end

      i=update_mean

      old_clusters=new_GMM(k,z_data[1][:,1:update_mean])
      new_clusters=init_gmm(k,ones(2,100).*10^6)

      convergence_check=true


      while(i<(nsamp-burnin+1) && convergence_check)

        if(i%update_mean==0)

          new_clusters = deepcopy(old_clusters)
          new_clusters = new_GMM2(new_clusters,z_data[1][:,1:sparsity:i])

          convergence_check=convergence_checks(old_clusters,new_clusters,tolerance)

          println("Total Iterations: $i")
          #println("Norm : $(norm(new_mean-old_mean))")
          println("KL DIV : $(KL_div_clusters(old_clusters,new_clusters))")

          old_clusters .= new_clusters

        end


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
            k = rand(1:nchain-2)
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
      i=i+1
      end

   #Removing the zeros from the chain
     for j = 1:nchain
         z_data[j] = z_data[j][:,1:i-1]
     end

      return z_data, accept_data, logp_data, swaps, swapaccepts
  end



  function convergence_checks(old_clusters,new_clusters,tolerance)

      if(KL_div_clusters(old_clusters,new_clusters)<tolerance)
          return false
      else
          return true
      end
  end

"""
Function to initialize the Gaussian Mixture Model. The Gaussians are stored as entries in a dictionary and the parameters can be accessed with the keys pi_k, mu_k and cov_k.
The center of each gaussian is initialized using the K-Means algorithm seeded with the Centrality Algorithm.

"""
  function init_gmm(z,data)
    clusters = []
    # Use the KMeans centroids to initialise the GMM
    km = Clustering.kmeans(data,z;init=initseeds!(Array{Integer}(undef, z),Clustering.KmCentralityAlg(),data))
    mu = km.centers


    for i in range(1,z)
        pi_k=1.0 /z
        mu_k=mu[:,i]
        cov_k=Matrix(1.0I,size(data)[1],size(data)[1])
        cluster=Dict("pi_k" => pi_k, "mu_k" => mu_k, "cov_k" => cov_k)
        push!(clusters,cluster)
    end
    return clusters
end

"""
Expectation step of the EM algorithm
"""

function expectation_step(data, clusters)
    #global gamma_nk, totals

    N = size(data)[2]
    K = length(clusters)
    totals = zeros((N, 1))
    gamma_nk = zeros((N, K))


    for k in range(1,K)
        pi_k = clusters[k]["pi_k"]
        mu_k = clusters[k]["mu_k"]
        cov_k = clusters[k]["cov_k"]
        dist= MvNormal(mu_k,cov_k)

        gamma_nk[:, k] = (pi_k *pdf(dist, data))
    end

    totals = sum(gamma_nk, dims=2)

    gamma_nk ./= totals

    return gamma_nk,totals
end

"""
Maximization step of the EM Algorithm
"""
function maximization_step(gamma_nk,data,clusters)
    #global gamma_nk
    N = float(size(data)[2])
    K = length(clusters)


    for k in range(1,K)
        gamma_k = gamma_nk[:, k]
        N_k = sum(gamma_k, dims=1)[1]

        pi_k = N_k/ N
        mu_k=gamma_k'.*data
        mu_k=sum(mu_k, dims=2)/N_k
        cov_k = gamma_k'.*(data.-mu_k)*(data.-mu_k)'./N_k
        cov_k=0.5.*(cov_k.+cov_k')

        clusters[k]["pi_k"]=pi_k
        clusters[k]["mu_k"]=dropdims(mu_k,dims=2)
        clusters[k]["cov_k"]=cov_k  #+Diagonal([0.000001,0.000001])
    end
    return clusters
end

function get_likelihood(totals)
    return sum(log.(totals))
end


"""
This GMM function takes data from the metropolis chain and initializes a GMM with 'z' gaussians.
"""
function new_GMM(z,data)
    # z is the number of models
    # should z always be equal to the first dimension of the data?

    likelihood=0
    new_likelihood=-10
    step=1

    # Step 1: Initialize the means,covariances and mixing coeffs
    # Use K-means to initialize the means
    clusters=init_gmm(z,data)

    while(abs(new_likelihood-likelihood)>1)
        likelihood=new_likelihood

        #Step 2: E Step - Evaluate the responsibilites using the current parameter values
        gamma_nk,totals= expectation_step(data, clusters)

        #Step 3: M Step - Re-estimate the parameters using the current responsibilities
        maximization_step(gamma_nk,data,clusters)

        #Step 4: Evaluate the log likelihood
        new_likelihood=get_likelihood(totals)

        #println("Step: $step Likelihood: $new_likelihood")
        step=step+1
    end
    return clusters
end

"""
This GMM function updates an existing GMM cluster with new data. It skips the initialization step and preserves the order of the clusters.
"""
function new_GMM2(clusters,data)

    likelihood=0
    new_likelihood=-10
    step=1

    #Step 1: Initialize the means,covariances and mixing coeffs - This is already done and we are just updating the current clusters


    while(abs(new_likelihood-likelihood)>1)
        likelihood=new_likelihood

        #Step 2: E Step - Evaluate the responsibilites using the current parameter values
        gamma_nk,totals= expectation_step(data, clusters)

        #Step 3: M Step - Re-estimate the parameters using the current responsibilities
        maximization_step(gamma_nk,data,clusters)

        #Step 4: Evaluate the log likelihood
        new_likelihood=get_likelihood(totals)

        #println("Step: $step Likelihood: $new_likelihood")
        step=step+1
    end
    return clusters
end


function KL_div_clusters(old_clusters,new_clusters)
    K = length(old_clusters)
    max_kldiv=0
    for k in range(1,K)

        term1= tr(inv(new_clusters[k]["cov_k"])*old_clusters[k]["cov_k"])- 2
        term2= log(det(new_clusters[k]["cov_k"])/det(old_clusters[k]["cov_k"]))

        mean_diff=new_clusters[k]["mu_k"]-old_clusters[k]["mu_k"]
        term3= mean_diff'*inv(new_clusters[k]["cov_k"])*mean_diff

        kldiv=term1+term2+term3

        if(kldiv>max_kldiv)
            max_kldiv=kldiv
        end

    end
    return 0.5*max_kldiv
end


function get_means(clusters)
    # extract all the means from the GMM dictionary
    K = length(clusters)
    means= zeros(0)

    for k in range(1,K)
        mu_k = clusters[k]["mu_k"]
        append!(means, mu_k)
    end
   return means
end

function get_covs(clusters)
    # extract all the covariance matrices from the GMM dictionary
    K = length(clusters)
    covs = Matrix{Float64}[]

    for k in range(1,K)
        cov_k = clusters[k]["cov_k"]
        push!(covs, cov_k)
    end
   return covs
end
