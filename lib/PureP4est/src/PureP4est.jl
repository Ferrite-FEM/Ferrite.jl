"""
    PureP4est

A native Julia implementation of the forest-of-octrees adaptive mesh refinement algorithms of
p4est ([BWG2011](@citet), [IBWG2015](@citet)) for quadrilateral and hexahedral coarse meshes,
independent of any finite element framework: refinement, coarsening, 2:1 balancing, inter-tree
transformations, the point iterator, node numbering with hanging-node detection
([`lnodes`](@ref)) and the facet skeleton ([`facetskeleton`](@ref)).
"""
module PureP4est

# Debug-only code (`@debug ex`) is compiled out unless this is flipped to `true`.
const DEBUG = false
macro debug(ex)
    return DEBUG ? esc(ex) : nothing
end

include("octree.jl")
include("connectivity.jl")
include("forest.jl")

export OctantBWG, OctreeBWG, AbstractForest, Forest, Connectivity, NeighborTable, LNodes,
    trees, connectivity, nleaves, forest_leaves,
    refine!, refine_all!, coarsen!, refine_and_coarsen!, balanceforest!,
    lnodes, foreach_root_facet_leaf
# `facetskeleton` is not exported: it would clash with Ferrite's function of that name.

end
