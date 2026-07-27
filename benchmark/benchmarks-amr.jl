#----------------------------------------------------------------------#
# AMR: adaptive refinement, 2:1 balancing, grid extraction and
# hanging-node (conformity) constraints for the forest-of-octrees grid
#----------------------------------------------------------------------#
SUITE["amr"] = BenchmarkGroup()

# The forests are built by `FerriteBenchmarkHelpers.adaptive_forest`: a uniform refinement
# followed by one more level for every fourth cell, so that 2:1 violations and hanging
# nodes are spread over all trees and tree boundaries. 2D and 3D are both covered where the
# code paths differ structurally: balancing propagates over faces and corners in 2D but
# faces, edges and corners in 3D, and `creategrid` emits edge-midpoint hanging nodes in 2D
# but face-center and edge hanging nodes in 3D.
#
# `ForestBWG` construction is not benchmarked: it is dominated by `ExclusiveTopology`,
# which the mesh group covers. `coarsen!`/`refine_and_coarsen!` share the marked-leaf-list
# rebuild machinery with `refine!`.
let g = SUITE["amr"]
    # 8×8 quadrilateral base grid (64 trees) refined to 1024 cells uniformly + 256 marks;
    # 4×4×4 hexahedron base grid (64 trees) refined to 512 cells uniformly + 128 marks.
    quadforest = FerriteBenchmarkHelpers.adaptive_forest(Quadrilateral, (8, 8), 2)
    hexforest = FerriteBenchmarkHelpers.adaptive_forest(Hexahedron, (4, 4, 4), 1)

    # Adaptive refinement of scattered marked cells (every other leaf). Mutates the
    # forest, so every sample refines a fresh copy from `setup`.
    g["refine! (Quadrilateral 8×8 base)"] = @benchmarkable(
        refine!(forest, marked),
        setup = (forest = deepcopy($quadforest); marked = 1:2:getncells(forest)),
        evals = 1,
        seconds = 1.0,
    )

    # Restore the 2:1 balance invariant that creategrid requires.
    g["balanceforest! (Quadrilateral 8×8 base)"] = @benchmarkable(
        balanceforest!(forest),
        setup = (forest = deepcopy($quadforest)),
        evals = 1,
        seconds = 1.0,
    )
    g["balanceforest! (Hexahedron 4×4×4 base)"] = @benchmarkable(
        balanceforest!(forest),
        setup = (forest = deepcopy($hexforest)),
        evals = 1,
        seconds = 1.0,
    )

    # Extract the NonConformingGrid from a balanced forest: global node numbering across
    # tree boundaries plus hanging-node (conformity) detection. Does not mutate the forest.
    quadbalanced = FerriteBenchmarkHelpers.balanced_forest(Quadrilateral, (8, 8), 2)
    hexbalanced = FerriteBenchmarkHelpers.balanced_forest(Hexahedron, (4, 4, 4), 1)
    g["creategrid (Quadrilateral 8×8 base)"] = @benchmarkable creategrid($quadbalanced) evals = 1
    g["creategrid (Hexahedron 4×4×4 base)"] = @benchmarkable creategrid($hexbalanced) evals = 1

    # Turn the conformity info into affine constraints and close the handler. 3D covers
    # both constraint kinds (2-master edge and 4-master face hanging nodes).
    hexgrid = creategrid(hexbalanced)
    dh = DofHandler(hexgrid)
    add!(dh, :u, Lagrange{RefHexahedron, 1}())
    close!(dh)
    g["ConformityConstraint close! (Hexahedron 4×4×4 base)"] = @benchmarkable(
        FerriteBenchmarkHelpers.conformity_constraints($dh),
        evals = 1,
    )
end
