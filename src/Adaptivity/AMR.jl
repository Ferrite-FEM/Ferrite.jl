module AMR

using .. Ferrite
import Ferrite: @debug
using SparseArrays: SparseMatrixCSC, spzeros
using OrderedCollections: OrderedSet

include("ncgrid.jl")
include("BWG.jl")
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
