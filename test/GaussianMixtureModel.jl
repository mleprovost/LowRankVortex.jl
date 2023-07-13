@testset "Testing the init_gmm function used to initialize the GMM" begin

    atol = 1000*eps()

    testdata=ones(2,100)
    testdata[:,1:50].= 2

    clusters=LowRankVortex.init_gmm(2,testdata)

    pi_k1=clusters[1]["pi_k"]
    pi_k2=clusters[2]["pi_k"]
    mu_k1=clusters[1]["mu_k"]
    mu_k2=clusters[2]["mu_k"]
    cov_k1=clusters[1]["cov_k"]
    cov_k2=clusters[2]["cov_k"]

    @test isapprox(pi_k1,0.5, atol = atol)
    @test isapprox(pi_k2,0.5, atol = atol)
    @test isapprox(mu_k1,[2.0,2.0], atol = atol)
    @test isapprox(mu_k2,[1.0,1.0], atol = atol)
    @test isapprox(cov_k1,[1.0 0.0; 0.0 1.0], atol = atol)
    @test isapprox(cov_k2,[1.0 0.0; 0.0 1.0], atol = atol)
end

@testset "Testing the expectation_step function of the GMM" begin

    atol = 1000*eps()

    testdata=ones(2,100)
    testdata[:,1:50].=2

    clusters=LowRankVortex.init_gmm(2,testdata)
    gamma_nk,totals=LowRankVortex.expectation_step(testdata, clusters)

    @test typeof(gamma_nk) <: Matrix && typeof(totals) <: Matrix
    @test size(gamma_nk,1) == 100 && size(gamma_nk,2) == 2
    @test size(totals,1) == 100 && size(totals,2) == 1

end

@testset "Testing the maximization_step function of the GMM" begin

    atol = 1000*eps()

    testdata=ones(2,100)
    testdata[:,1:50].=2

    clusters=LowRankVortex.init_gmm(2,testdata)
    gamma_nk,totals=LowRankVortex.expectation_step(testdata, clusters)
    clusters=LowRankVortex.maximization_step(gamma_nk,testdata,clusters)

    pi_k1=clusters[1]["pi_k"]
    pi_k2=clusters[2]["pi_k"]
    mu_k1=clusters[1]["mu_k"]
    mu_k2=clusters[2]["mu_k"]
    cov_k1=clusters[1]["cov_k"]
    cov_k2=clusters[2]["cov_k"]

    @test isapprox(pi_k1,0.5, atol = atol) && isapprox(pi_k2,0.5, atol = atol)
    @test typeof(mu_k1) <: Vector &&  typeof(mu_k2) <: Vector
    @test size(mu_k1) == (2,) && size(mu_k2) == (2,)
    @test typeof(cov_k1) <: Matrix &&  typeof(cov_k2) <: Matrix
    @test size(cov_k1) == (2,2) && size(cov_k2) == (2,2)

end

using Distributions
@testset "Testing the complete GMM function" begin

    atol = 0.1
    coeffs = [1.0]
    d   = MvNormal[]
    μ1 = [1.0,1.0]

    var11, var12 = 1.0,1.0
    α1 = 0.0
    X = [cos(α1) sin(α1); -sin(α1) cos(α1)]
    Σ1 = X'*Diagonal([var11,var12])*X
    push!(d,MvNormal(μ1,Σ1))


    # Create the mixture model
    dm = MixtureModel(d, coeffs);
    data=rand(dm,10000)
    clusters=LowRankVortex.new_GMM(1,data)

    pi_k1=clusters[1]["pi_k"]
    mu_k1=clusters[1]["mu_k"]
    cov_k1=clusters[1]["cov_k"]


    @test isapprox(pi_k1,1.0, atol = atol)
    @test typeof(mu_k1) <: Vector && size(mu_k1) == (2,) && isapprox(mu_k1,[1.0,1.0], atol = atol)
    @test typeof(cov_k1) <: Matrix && size(cov_k1) == (2,2) && isapprox(cov_k1,[1.0 0.0; 0.0 1.0], atol = atol)

end

@testset "Testing the KL Divergence function" begin
    atol = 1000*eps()

    coeffs = [1.0]
    d   = MvNormal[]
    μ1 = [1.0,1.0]

    var11, var12 = 1.0,1.0
    α1 = 0.0
    X = [cos(α1) sin(α1); -sin(α1) cos(α1)]
    Σ1 = X'*Diagonal([var11,var12])*X
    push!(d,MvNormal(μ1,Σ1))


    # Create the mixture model
    dm = MixtureModel(d, coeffs);
    data=rand(dm,10000)
    old_clusters=LowRankVortex.init_gmm(1,data)
    new_clusters=LowRankVortex.init_gmm(1,data)
    kl_div=LowRankVortex.KL_div_clusters(old_clusters,new_clusters)
    @test isapprox(kl_div,0.0, atol = atol)


    d   = MvNormal[]
    μ1 = [2.0,2.0]

    var11, var12 = 2.0,2.0
    α1 = 0.0
    X = [cos(α1) sin(α1); -sin(α1) cos(α1)]
    Σ1 = X'*Diagonal([var11,var12])*X
    push!(d,MvNormal(μ1,Σ1))


    # Create the mixture model
    dm = MixtureModel(d, coeffs);
    data=rand(dm,10000)
    new_clusters=LowRankVortex.init_gmm(1,data)
    kl_div=LowRankVortex.KL_div_clusters(old_clusters,new_clusters)
    @test kl_div!=0
end
