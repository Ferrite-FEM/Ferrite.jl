module AMR

using .. Ferrite
using SparseArrays

include("BWG.jl")
include("ncgrid.jl")
include("constraints.jl")

export ForestBWG, 
       refine!,
       coarsen!,
       balanceforest!,
       creategrid,
       ConformityConstraint,
       NonConformingGrid

end
