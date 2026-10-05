module AMR

using .. Ferrite
import Ferrite: @debug
using OrderedCollections: OrderedSet
using PureP4est: PureP4est, AbstractForest, Connectivity, OctantBWG, OctreeBWG, DEFAULT_MAXLEVEL,
    forest_leaves, foreach_root_facet_leaf, lnodes, nleaves, node_map₂, node_map₃,
    transform_corner, transform_corner_remote, transform_edge, transform_edge_remote,
    transform_facet, transform_facet_remote, _element_offsets
# The forest operations are PureP4est's; Ferrite re-exports them for `ForestBWG`.
using PureP4est: refine!, refine_all!, coarsen!, refine_and_coarsen!, balanceforest!

# `forest.jl` translates between Ferrite's grids and PureP4est's forests; `ncgrid.jl` and
# `constraints.jl` are the Ferrite-side consumers of the materialized grid.
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
