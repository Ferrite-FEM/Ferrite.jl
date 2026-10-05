# PureP4est

A native Julia implementation of the forest-of-octrees adaptive mesh refinement algorithms of
[p4est](https://www.p4est.org/) for quadrilateral (2D) and hexahedral (3D) coarse meshes:

- refinement and coarsening of marked leaves,
- 2:1 balancing across faces, edges and corners, also across tree boundaries,
- inter-tree coordinate transformations for arbitrarily rotated neighbouring trees,
- node numbering with hanging-node detection (`lnodes`) and the facet skeleton
  (`facetskeleton`) of the refined mesh.

It is independent of any finite element framework. The algorithms follow

- C. Burstedde, L. C. Wilcox, O. Ghattas, *p4est: Scalable Algorithms for Parallel Adaptive
  Mesh Refinement on Forests of Octrees*, SIAM J. Sci. Comput. 33 (2011).
- T. Isaac, C. Burstedde, L. C. Wilcox, O. Ghattas, *Recursive Algorithms for Distributed
  Forests of Octrees*, SIAM J. Sci. Comput. 37 (2015).

This package is developed as part of [Ferrite.jl](https://github.com/Ferrite-FEM/Ferrite.jl),
whose adaptive mesh refinement is built on it.

## Example

```julia
using PureP4est

# Two quadrilaterals sharing an edge, given by their vertex ids in counter-clockwise order
forest = Forest([(1, 2, 5, 4), (2, 3, 6, 5)], #= maximum level =# 6)

refine_all!(forest, 1)    # refine every leaf once
refine!(forest, [2, 3])   # refine leaves 2 and 3
balanceforest!(forest)    # restore the 2:1 balance

ln = lnodes(forest)
ln.E          # element-node matrix: node ids of each leaf's corners (z-order)
ln.noderefs   # per node: (tree, integer coordinate in the tree)
ln.hanging2   # (hanging node, master 1, master 2)
```

To attach geometry or other data to the forest, implement the `AbstractForest` interface
(`trees(forest)` and `connectivity(forest)`) for a type of your own; Ferrite's `ForestBWG` does
this to carry the coarse mesh's nodes and named sets.
