module AMR

using .. Ferrite
import Ferrite: @debug
using OrderedCollections: OrderedSet

# `octree.jl` defines the abstract element/tree types and the octant/octree types the rest
# builds signatures on, so it has to come first; `simplex.jl` adds the simplex element/tree;
# `forest.jl` holds the shared forest algorithms plus the hypercube-specific inter-tree and
# materialization code, `simplex_forest.jl` the simplex counterparts of the latter.
include("octree.jl")
include("simplex.jl")
include("forest.jl")
include("simplex_forest.jl")
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
