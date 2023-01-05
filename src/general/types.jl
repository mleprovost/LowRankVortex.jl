export LowRankENKFSolution, ENKFSolution, BasicEnsembleMatrix

abstract type EnsembleMatrix{Nx,Ne,T} <: AbstractMatrix{T} end

abstract type AbstractENKFSolution end


"""
    BasicEnsembleMatrix{Nx,Ne}

A type which holds data ensembles of size `Nx` x `Ne`, where `Nx`
is the dimension of the data vectors and `Ne` the size of the
ensemble.
"""
struct BasicEnsembleMatrix{Nx,Ne,T,XT} <: EnsembleMatrix{Nx,Ne,T}
    X :: XT
    burnin :: Integer
    BasicEnsembleMatrix(X::XT;burnin=1) where {XT<:AbstractMatrix} =
        new{size(X,1),size(X,2),eltype(X),XT}(X,burnin)
end

similar(X::BasicEnsembleMatrix;element_type=eltype(X),dims=size(X)) = BasicEnsembleMatrix(Array{element_type}(undef, dims...))

BasicEnsembleMatrix(X::BasicEnsembleMatrix,a...) = X

(X::BasicEnsembleMatrix)(i::Int) = view(X.X,:,i)

function Base.show(io::IO,m::MIME"text/plain",X::BasicEnsembleMatrix{Nx,Ne}) where {Nx,Ne}
  println(io,"Ensemble with $Ne members of $Nx-dimensional data")
  show(io,m,X.X)
end


struct LowRankENKFSolution{XT,YT,YYT,SIGXT,SIGYT,SYT,SXYT} <: AbstractENKFSolution
   X :: XT
   Xf :: XT
   Y :: YYT
   crit_ratio :: Float64
   V :: AbstractMatrix{Float64}
   U :: AbstractMatrix{Float64}
   Λx :: Vector{Float64}
   Λy :: Vector{Float64}
   rx :: Int64
   ry :: Int64
   Σx :: SIGXT
   Σy :: SIGYT
   Y̆ :: YT
   ΣY̆ :: SYT
   ΣX̆Y̆ :: SXYT
   yerr :: Float64
end

struct ENKFSolution{XT,YT,YYT,SIGXT,SIGYT,SYT,SXYT} <: AbstractENKFSolution
   X :: XT
   Xf :: XT
   Y :: YYT
   Σx :: SIGXT
   Σy :: SIGYT
   Y̆ :: YT
   ΣY̆ :: SYT
   ΣX̆Y̆ :: SXYT
   yerr :: Float64
end
