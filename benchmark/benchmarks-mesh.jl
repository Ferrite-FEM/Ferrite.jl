#----------------------------------------------------------------------#
# Mesh: grid generation, topology construction and queries, sets, coloring
#----------------------------------------------------------------------#
SUITE["mesh"] = BenchmarkGroup()

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
# faces and edges, triangles the simplex path where cells share only vertices diagonally.
SUITE["mesh"]["topology"] = BenchmarkGroup()
let g = SUITE["mesh"]["topology"]
    hexgrid = generate_grid(Hexahedron, (10, 10, 10))
    trigrid = generate_grid(Triangle, (25, 25))
    hextopo = ExclusiveTopology(hexgrid)
    g["ExclusiveTopology (Hexahedron 10×10×10)"] = @benchmarkable ExclusiveTopology($hexgrid) evals = 1
    g["ExclusiveTopology (Triangle 25×25)"] = @benchmarkable ExclusiveTopology($trigrid) evals = 1
    g["getneighborhood all cells (Hexahedron 10×10×10)"] = @benchmarkable FerriteBenchmarkHelpers.neighborhood_sweep($hextopo, $hexgrid) evals = 1
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

# Grid coloring, used to set up threaded assembly.
SUITE["mesh"]["coloring"] = BenchmarkGroup()
let g = SUITE["mesh"]["coloring"]
    grid = generate_grid(Quadrilateral, (50, 50))
    g["workstream (Quadrilateral 50×50)"] = @benchmarkable create_coloring($grid; alg = ColoringAlgorithm.WorkStream) evals = 1
    g["greedy (Quadrilateral 50×50)"] = @benchmarkable create_coloring($grid; alg = ColoringAlgorithm.Greedy) evals = 1
end
