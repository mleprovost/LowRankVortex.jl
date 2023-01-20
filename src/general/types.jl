export LowRankENKFSolution, ENKFSolution

abstract type AbstractENKFSolution end


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
