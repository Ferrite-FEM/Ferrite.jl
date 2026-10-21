#----------------------------------------------------------------------#
# Mesh: grid generation, topology construction and queries, sets, coloring
#----------------------------------------------------------------------#
SUITE["mesh"] = BenchmarkGroup()

# Repeated traversal must not recompute grid-wide reference dimensions when the
# cell vector has an abstract element type. Construction is outside the timing.
SUITE["mesh"]["InterfaceIterator"] = BenchmarkGroup()
let g = SUITE["mesh"]["InterfaceIterator"]
    grid = generate_grid(Quadrilateral, (40, 40))
    for (name, cells) in (("concrete", grid.cells), ("abstract", Ferrite.AbstractCell[grid.cells...]))
        iterator = InterfaceIterator(Grid(cells, grid.nodes))
        g[name] = @benchmarkable FerriteBenchmarkHelpers.interface_sweep($iterator) evals = 1
    end
end

# Grid generation for one geometry per structurally different generator: 2D quadrilateral
# (tensor product), 3D hexahedron (tensor product with face sets in 3D) and 3D tetrahedron
# (subdivision of hexahedra). Sizes are picked to land well above the noise floor.
SUITE["mesh"]["generate_grid"] = BenchmarkGroup()
let g = SUITE["mesh"]["generate_grid"]
    g["Quadrilateral (100×100)"] = @benchmarkable generate_grid(Quadrilateral, (100, 100)) evals = 1
    g["Hexahedron (20×20×20)"] = @benchmarkable generate_grid(Hexahedron, (20, 20, 20)) evals = 1
    g["Tetrahedron (12×12×12)"] = @benchmarkable generate_grid(Tetrahedron, (12, 12, 12)) evals = 1
end

# Topology: construction and neighborhood queries. Hexahedra exercise the 3D path with
# faces and edges, triangles and tetrahedra the simplex paths where cells share only
# vertices diagonally (with many more neighbors per cell in 3D).
SUITE["mesh"]["topology"] = BenchmarkGroup()
let g = SUITE["mesh"]["topology"]
    hexgrid = generate_grid(Hexahedron, (10, 10, 10))
    quadgrid = generate_grid(Quadrilateral, (25, 25))
    trigrid = generate_grid(Triangle, (25, 25))
    tetgrid = generate_grid(Tetrahedron, (8, 8, 8))
    hextopo = ExclusiveTopology(hexgrid)
    g["ExclusiveTopology (Hexahedron 10×10×10)"] = @benchmarkable ExclusiveTopology($hexgrid) evals = 1
    g["ExclusiveTopology (Quadrilateral 25×25)"] = @benchmarkable ExclusiveTopology($quadgrid) evals = 1
    g["ExclusiveTopology (Triangle 25×25)"] = @benchmarkable ExclusiveTopology($trigrid) evals = 1
    g["ExclusiveTopology (Tetrahedron 8×8×8)"] = @benchmarkable ExclusiveTopology($tetgrid) evals = 1
    g["getneighborhood all cells (Hexahedron 10×10×10)"] = @benchmarkable FerriteBenchmarkHelpers.neighborhood_sweep($hextopo, $hexgrid) evals = 1
    # getneighborhood through the EdgeIndex path, which recomputes the full neighborhood
    # for each query since only the exclusive neighbors are stored for 3D cells.
    g["getneighborhood all edges (Hexahedron 10×10×10)"] = @benchmarkable FerriteBenchmarkHelpers.edge_neighborhood_sweep($hextopo, $hexgrid) evals = 1
    # Iteration over the raw vertex/edge/face adjacency storage (issue #1019).
    g["neighbor iteration (Hexahedron 10×10×10)"] = @benchmarkable FerriteBenchmarkHelpers.neighbor_index_sum($hextopo) evals = 1
    # getneighborhood through the FacetIndex path for every skeleton facet (issue #1019).
    # The skeleton is precomputed; its construction is measured by the benchmark below.
    hexskeleton = facetskeleton(hextopo, hexgrid)
    g["getneighborhood skeleton facets (Hexahedron 10×10×10)"] = @benchmarkable FerriteBenchmarkHelpers.skeleton_neighborhood_sweep($hextopo, $hexgrid, $hexskeleton) evals = 1
    # facetskeleton memoizes its result in the topology, so the cache must be cleared in
    # setup or every sample after the first measures a field read.
    g["facetskeleton (Hexahedron 10×10×10)"] = @benchmarkable(
        facetskeleton($hextopo, $hexgrid),
        setup = ($hextopo.facet_skeleton = nothing),
        evals = 1,
    )
    # Smaller grid than the rest of the group: the stencil construction runs at ~15 us per
    # vertex and would otherwise eat a disproportionate share of the time budget.
    stencilgrid = generate_grid(Hexahedron, (6, 6, 6))
    stenciltopo = ExclusiveTopology(stencilgrid)
    g["vertex_star_stencils (Hexahedron 6×6×6)"] = @benchmarkable vertex_star_stencils($stenciltopo, $stencilgrid) evals = 1 seconds = 1.0
end

# Predicate-based set construction: every boundary facet is visited and the predicate
# evaluated on its node coordinates. `addfacetset!` refuses to overwrite an existing set,
# so each sample gets a fresh grid from `setup`.
SUITE["mesh"]["sets"] = BenchmarkGroup()
let g = SUITE["mesh"]["sets"]
    g["addfacetset! predicate (Quadrilateral 50×50)"] = @benchmarkable(
        addfacetset!(grid, "right side", x -> x[1] ≈ 1.0),
        setup = (grid = generate_grid(Quadrilateral, (50, 50))),
        evals = 1,
    )
end

# Grid coloring, used to set up threaded assembly. The 3D grids gather many more
# node-sharing neighbor candidates per cell than the 2D grid (quadrilateral ~12,
# hexahedron ~56, tetrahedron ~92), so they weight the deduplication in the construction
# of the (node based) cell conflict graph differently.
SUITE["mesh"]["coloring"] = BenchmarkGroup()
let g = SUITE["mesh"]["coloring"]
    quadgrid = generate_grid(Quadrilateral, (50, 50))
    g["workstream (Quadrilateral 50×50)"] = @benchmarkable create_coloring($quadgrid; alg = ColoringAlgorithm.WorkStream) evals = 1
    g["greedy (Quadrilateral 50×50)"] = @benchmarkable create_coloring($quadgrid; alg = ColoringAlgorithm.Greedy) evals = 1
    hexgrid = generate_grid(Hexahedron, (15, 15, 15))
    g["workstream (Hexahedron 15×15×15)"] = @benchmarkable create_coloring($hexgrid; alg = ColoringAlgorithm.WorkStream) evals = 1
    g["greedy (Hexahedron 15×15×15)"] = @benchmarkable create_coloring($hexgrid; alg = ColoringAlgorithm.Greedy) evals = 1
    tetgrid = generate_grid(Tetrahedron, (8, 8, 8))
    g["workstream (Tetrahedron 8×8×8)"] = @benchmarkable create_coloring($tetgrid; alg = ColoringAlgorithm.WorkStream) evals = 1
    g["greedy (Tetrahedron 8×8×8)"] = @benchmarkable create_coloring($tetgrid; alg = ColoringAlgorithm.Greedy) evals = 1

    # Dof based (exact) cell conflicts: a vector field has many dofs per shared node
    # (grouped by their set of writing cells), a discontinuous field has no shared dofs
    # at all, and periodic constraints add (remote) master dofs to the write sets.
    dh_vector = DofHandler(hexgrid)
    add!(dh_vector, :u, Lagrange{RefHexahedron, 1}()^3)
    close!(dh_vector)
    g["AssemblyConflicts(dh) (Hexahedron 15×15×15, Lagrange^3)"] = @benchmarkable AssemblyConflicts($dh_vector) evals = 1
    dh_dg = DofHandler(hexgrid)
    add!(dh_dg, :u, DiscontinuousLagrange{RefHexahedron, 1}())
    close!(dh_dg)
    g["AssemblyConflicts(dh) (Hexahedron 15×15×15, DiscontinuousLagrange)"] = @benchmarkable AssemblyConflicts($dh_dg) evals = 1
    dh_periodic = DofHandler(quadgrid)
    add!(dh_periodic, :u, Lagrange{RefQuadrilateral, 1}())
    close!(dh_periodic)
    ch = ConstraintHandler(dh_periodic)
    add!(ch, PeriodicDirichlet(:u, collect_periodic_facets(quadgrid, "left", "right")))
    close!(ch)
    g["AssemblyConflicts(dh, ch) (Quadrilateral 50×50, PeriodicDirichlet)"] = @benchmarkable AssemblyConflicts($dh_periodic, $ch) evals = 1

    # Coloring of facets and interfaces from precomputed conflicts, which gathers the
    # conflicts per item through the cells (unlike the cell coloring which uses the cell
    # conflict graph directly). The interface conflict graph is much denser than the cell
    # one, which weights the two algorithms differently.
    hexgrid12 = generate_grid(Hexahedron, (12, 12, 12))
    conflicts12 = AssemblyConflicts(hexgrid12)
    skeleton = interfaceskeleton(ExclusiveTopology(hexgrid12), hexgrid12)
    g["interfaces workstream (Hexahedron 12×12×12)"] = @benchmarkable create_coloring($conflicts12, $skeleton; alg = ColoringAlgorithm.WorkStream) evals = 1
    g["interfaces greedy (Hexahedron 12×12×12)"] = @benchmarkable create_coloring($conflicts12, $skeleton; alg = ColoringAlgorithm.Greedy) evals = 1
    conflicts15 = AssemblyConflicts(hexgrid)
    boundary = union(values(hexgrid.facetsets)...)
    g["facets workstream (Hexahedron 15×15×15, boundary)"] = @benchmarkable create_coloring($conflicts15, $boundary; alg = ColoringAlgorithm.WorkStream) evals = 1
end
