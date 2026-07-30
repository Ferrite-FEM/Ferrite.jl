module AMR

using .. Ferrite
import Ferrite: @debug
using OrderedCollections: OrderedSet

# `octree.jl` defines the octant/octree types the rest builds signatures on, so it has to
# come first; the remaining files only call into what precedes them.
include("octree.jl")
include("forest.jl")
include("ncgrid.jl")
include("constraints.jl")

export ForestBWG,
    refine!,
    refine_all!,
    coarsen!,
    refine_and_coarsen!,
    balanceforest!,
    creategrid,
    ConformityConstraint,
    NonConformingGrid

end
