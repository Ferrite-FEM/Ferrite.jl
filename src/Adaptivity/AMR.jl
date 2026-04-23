module AMR

using .. Ferrite
import Ferrite: @debug
using SparseArrays
using OrderedCollections

include("BWG.jl")
include("ncgrid.jl")
include("constraints.jl")

export ForestBWG,
       refine!,
       refine_all!,
       coarsen!,
       balanceforest!,
       creategrid,
       ConformityConstraint,
       NonConformingGrid

end
