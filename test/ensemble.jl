@testset "Ensemble operations" begin

  atol = 1000*eps()

  μx = [1.0,2.0,3.0,4.0]
  Σx = 0.2*I

  Ne = 5000
  X = create_ensemble(Ne,μx,Σx)

  @test typeof(X) <: BasicEnsembleMatrix
  @test size(X.X,1) == 4 && size(X.X,2) == Ne
  @test cov(X) == cov(X.X,dims=2,corrected=true)
  @test typeof(mean(X)) <: Vector


  Xp = ensemble_perturb(X)
  @test isapprox(cov(Xp),cov(X),atol=atol)

  @test isapprox(norm(mean(Xp)),0.0,atol=atol)

  @test typeof(X(1)) <: AbstractVector

  μy = zeros(Float64,10)
  Σy = 0.01*I
  Y = create_ensemble(Ne,μy,Σy)

  YX = vcat(Y,X)
  @test typeof(YX) <: YXEnsembleMatrix
  @test size(YX,1) = size(Y,1) + size(X,1)

  @test typeof(X+X) <: BasicEnsembleMatrix

  v = [-2.0;-3.0;-4.0;-5.0]
  X1 = v + X
  X2 = X + v
  @test isapprox(X1.X,X2.X,atol=atol)

  X1 = 2*X
  X2 = X*2
  @test isapprox(X1.X,X2.X,atol=atol)

  @test_throws DimensionMismatch [-2.0;-3.0;-4.0;-5.0;-6.0]+X

  ϵ = create_ensemble(Ne,zero(μx),Σϵ)

  u = v - X + ϵ
  u2 = BasicEnsembleMatrix(v .- X.X .+ ϵ.X)
  @test isapprox(u.X,u2.X,atol=atol)

   

end
