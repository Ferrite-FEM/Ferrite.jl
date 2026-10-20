using Ferrite, Test

@testset "grid coloring" begin
    function test_coloring(grid, cellset = 1:getncells(grid))
        for alg in (ColoringAlgorithm.Greedy, ColoringAlgorithm.WorkStream)
            color_vectors = create_coloring(grid, cellset; alg = alg)
            @test sum(length, color_vectors, init = 0) == length(cellset)
            @test union!(Set{Int}(), color_vectors...) == Set(cellset)
            # No two cells of the same color share a node
            for color in color_vectors, c1 in color, c2 in color
                c1 == c2 && continue
                @test isdisjoint(Ferrite.get_node_ids(getcells(grid, c1)), Ferrite.get_node_ids(getcells(grid, c2)))
            end
        end
    end
    test_coloring(generate_grid(Line, (5,)))
    test_coloring(generate_grid(QuadraticLine, (5,)))
    test_coloring(generate_grid(Triangle, (5, 5)))
    test_coloring(generate_grid(QuadraticTriangle, (5, 5)))
    test_coloring(generate_grid(Quadrilateral, (5, 5)))
    test_coloring(generate_grid(QuadraticQuadrilateral, (5, 5)))
    test_coloring(generate_grid(Tetrahedron, (5, 5, 5)))
    # test_coloring(generate_grid(QuadraticTetrahedron, (5, 5, 5)))
    test_coloring(generate_grid(Hexahedron, (5, 5, 5)))
    # test_coloring(generate_grid(QuadraticHexahedron, (5, 5, 5)))

    # color only a subset
    test_coloring(generate_grid(Line, (5,)), 1:3)
    test_coloring(generate_grid(Triangle, (5, 5)), Set{Int}(1:(3^2)))
    test_coloring(generate_grid(Quadrilateral, (5, 5)), Set{Int}(1:(3^2)))
    test_coloring(generate_grid(Tetrahedron, (5, 5, 5)), Set{Int}(1:(3^3)))
    test_coloring(generate_grid(Hexahedron, (5, 5, 5)), Set{Int}(1:(3^3)))
    # unconnected subset
    test_coloring(generate_grid(Triangle, (10, 10)), union(Set(1:10), Set(70:80)))

    #Special case with one and zero elements in the sets
    test_coloring(generate_grid(Quadrilateral, (2, 2)), [1])
    test_coloring(generate_grid(Quadrilateral, (2, 2)), Int[])

    # duplicate entries in the cellset are ignored
    let grid = generate_grid(Quadrilateral, (3, 3))
        @test create_coloring(grid, [2, 2, 4]) == create_coloring(grid, [2, 4])
        @test create_coloring(grid, [5, 5]) == create_coloring(grid, [5])
    end
end

@testset "grid coloring only accesses cells in the loop" begin
    # Catch full-grid traversal for small (or empty) subsets without relying on
    # timing or allocation thresholds.
    struct ColoringSubsetGrid{G} <: Ferrite.AbstractGrid{3}
        grid::G
        allowed_cells::BitSet
    end
    Ferrite.getncells(grid::ColoringSubsetGrid) = getncells(grid.grid)
    Ferrite.getnnodes(grid::ColoringSubsetGrid) = getnnodes(grid.grid)
    function Ferrite.getcells(grid::ColoringSubsetGrid, cell::Int)
        cell in grid.allowed_cells || error("accessed cell outside the loop: $cell")
        return getcells(grid.grid, cell)
    end

    grid = generate_grid(Hexahedron, (4, 4, 4))
    conflicts = AssemblyConflicts(grid)
    skeleton = interfaceskeleton(ExclusiveTopology(grid), grid)
    facets = collect(getfacetset(grid, "left"))
    for set in (
            Int[], [1], [17, 1, 2, 17], 3:8, Set([1, 2, 64]),
            FacetIndex[], facets[1:3:end],
            NTuple{2, FacetIndex}[], skeleton[1:11:end],
        )
        cells = BitSet(Iterators.flatten(Ferrite._item_cells(item) for item in set))
        subset_grid = ColoringSubsetGrid(grid, cells)
        for alg in (ColoringAlgorithm.Greedy, ColoringAlgorithm.WorkStream)
            @test create_coloring(subset_grid, set; alg) == create_coloring(conflicts, set; alg)
        end
    end
end

@testset "facet and interface coloring" begin
    function node_adjacent(grid, a, b)
        na = Set(Ferrite.get_node_ids(getcells(grid, a)))
        return a == b || any(v -> v in na, Ferrite.get_node_ids(getcells(grid, b)))
    end
    item_cells(facet::FacetIndex) = (facet[1],)
    item_cells(interface::NTuple{2, FacetIndex}) = (interface[1][1], interface[2][1])
    # Interfaces of the grid (restricted to cellset), independently derived from the
    # facet skeleton and the facet neighborhood
    function reference_interfaces(grid, topology, cellvec)
        ref = NTuple{2, FacetIndex}[]
        neighborhood = Ferrite.get_facet_facet_neighborhood(topology, grid)
        for facet in Ferrite.facetskeleton(topology, grid)
            neighbors = neighborhood[facet[1], facet[2]]
            isempty(neighbors) && continue
            fb = neighbors[1]
            (insorted(facet[1], cellvec) && insorted(fb[1], cellvec)) || continue
            push!(ref, (FacetIndex(facet[1], facet[2]), FacetIndex(fb[1], fb[2])))
        end
        return ref
    end
    # The colors partition `set` and no two items of the same color conflict
    function check_item_coloring(grid, set)
        T = eltype(set)
        for alg in (ColoringAlgorithm.WorkStream, ColoringAlgorithm.Greedy)
            colors = create_coloring(grid, set; alg)
            @test colors isa Vector{Vector{T}}
            # Partition: every item exactly once
            @test sum(length, colors; init = 0) == length(set)
            @test Set{T}(Iterators.flatten(colors)) == Set{T}(set)
            # Validity: no two items of the same color have node-adjacent (or equal) cells
            for color in colors
                for x in 1:length(color), y in (x + 1):length(color)
                    for p in item_cells(color[x]), q in item_cells(color[y])
                        @test !node_adjacent(grid, p, q)
                    end
                end
            end
        end
        return
    end

    @testset "interfaces" begin
        function check_interface_coloring(grid; cellset = 1:getncells(grid))
            topology = ExclusiveTopology(grid)
            cellvec = sort!(unique!(collect(Int, cellset)))
            interfaces = filter(interfaceskeleton(topology, grid)) do (a, b)
                insorted(a[1], cellvec) && insorted(b[1], cellvec)
            end
            @test Set(interfaces) == Set(reference_interfaces(grid, topology, cellvec))
            check_item_coloring(grid, interfaces)
            return
        end
        check_interface_coloring(generate_grid(Quadrilateral, (5, 4)))
        check_interface_coloring(generate_grid(Triangle, (4, 4)))
        check_interface_coloring(generate_grid(Hexahedron, (3, 3, 3)))
        check_interface_coloring(generate_grid(Tetrahedron, (3, 3, 3)))
        check_interface_coloring(generate_grid(Line, (10,)))
        check_interface_coloring(generate_grid(Quadrilateral, (5, 4)); cellset = 1:10)
        check_interface_coloring(generate_grid(Quadrilateral, (5, 4)); cellset = [1, 2, 3, 7, 8, 20])
        check_interface_coloring(generate_grid(Quadrilateral, (5, 4)); cellset = Int[])

        for grid in (generate_grid(Quadrilateral, (5, 4)), generate_grid(Tetrahedron, (2, 2, 2)))
            topology = ExclusiveTopology(grid)
            skeleton = interfaceskeleton(topology, grid)
            # Arbitrary subset of the skeleton
            check_item_coloring(grid, skeleton[1:3:end])
            # Any iterable of interfaces works
            @test create_coloring(grid, Set(skeleton[1:3:end])) isa
                Vector{Vector{NTuple{2, FacetIndex}}}
            # The cached skeleton is not modified
            skeleton_copy = copy(skeleton)
            create_coloring(grid, skeleton)
            @test interfaceskeleton(topology, grid) == skeleton_copy
            # Input validation
            @test_throws ArgumentError create_coloring(grid, [skeleton[1], skeleton[1]])
            badinterface = (FacetIndex(getncells(grid) + 1, 1), FacetIndex(1, 1))
            @test_throws ArgumentError create_coloring(grid, [badinterface])
        end

        # Determinism
        let grid = generate_grid(Hexahedron, (4, 4, 4)), topology = ExclusiveTopology(grid)
            skeleton = interfaceskeleton(topology, grid)
            @test create_coloring(grid, skeleton) == create_coloring(grid, skeleton)
        end

        # InterfaceIterator over an explicit set of interfaces (e.g. one color)
        let grid = generate_grid(Quadrilateral, (5, 4)), topology = ExclusiveTopology(grid)
            dh = DofHandler(grid)
            add!(dh, :u, DiscontinuousLagrange{RefQuadrilateral, 1}())
            close!(dh)
            ninterfaces = count(Returns(true), InterfaceIterator(dh, topology))
            colors = create_coloring(dh, interfaceskeleton(topology, grid))
            total = 0
            for color in colors
                n = 0
                for ic in InterfaceIterator(dh, color)
                    n += 1
                    @test length(interfacedofs(ic)) == 8
                    @test (FacetIndex(cellid(ic.a), ic.a.current_facet_id[]), FacetIndex(cellid(ic.b), ic.b.current_facet_id[])) in color
                end
                @test n == length(color)
                total += n
            end
            @test total == ninterfaces
            # Grid based iterator with a set works too
            @test count(Returns(true), InterfaceIterator(grid, first(colors))) == length(first(colors))
        end

        # Colored threaded interface assembly matches a serial sweep in the same color
        # order bitwise (within a color the write sets are disjoint for cell-interior-dof
        # interpolations), and skeleton-order serial assembly up to summation order.
        let grid = generate_grid(Hexahedron, (4, 4, 4)), topology = ExclusiveTopology(grid)
            dh = DofHandler(grid)
            add!(dh, :u, DiscontinuousLagrange{RefHexahedron, 1}())
            close!(dh)
            colors = create_coloring(dh, interfaceskeleton(topology, grid))
            ndpc = ndofs_per_cell(dh)
            Ki = [i * 1.0e-3 + j * 1.0e-6 for i in 1:(2 * ndpc), j in 1:(2 * ndpc)]
            function assemble_interfaces(dh, iterators, K)
                for it in iterators
                    assembler = start_assemble(K; fillzero = false)
                    for ic in it
                        assemble!(assembler, interfacedofs(ic), Ki)
                    end
                end
                return K
            end
            K0 = allocate_matrix(dh; topology, interface_coupling = trues(1, 1))
            Kserial = assemble_interfaces(dh, (InterfaceIterator(dh, topology),), copy(K0))
            Kcolorserial = assemble_interfaces(dh, (InterfaceIterator(dh, color) for color in colors), copy(K0))
            Kthreaded = copy(K0)
            for color in colors
                chunks = collect(Iterators.partition(color, max(1, cld(length(color), Threads.nthreads()))))
                Threads.@threads for chunk in chunks
                    assemble_interfaces(dh, (InterfaceIterator(dh, collect(chunk)),), Kthreaded)
                end
            end
            @test Kcolorserial.nzval == Kthreaded.nzval
            @test Kserial.nzval ≈ Kthreaded.nzval
        end
    end

    @testset "boundary facets" begin
        for grid in (
                generate_grid(Line, (10,)), generate_grid(Quadrilateral, (5, 4)),
                generate_grid(Triangle, (4, 4)), generate_grid(Hexahedron, (3, 3, 3)),
                generate_grid(Tetrahedron, (3, 3, 3)),
            )
            # Single facetset, union of all facetsets (corner cells have several facets in
            # the set), and an empty set
            check_item_coloring(grid, getfacetset(grid, "left"))
            boundary = union(values(grid.facetsets)...)
            check_item_coloring(grid, boundary)
            check_item_coloring(grid, collect(boundary))
            check_item_coloring(grid, FacetIndex[])
        end

        let grid = generate_grid(Quadrilateral, (5, 4))
            boundary = collect(union(values(grid.facetsets)...))
            # Without shared dofs only the corner cells (two facets in the set) conflict
            dh = DofHandler(grid)
            add!(dh, :u, DiscontinuousLagrange{RefQuadrilateral, 1}())
            close!(dh)
            greedy = ColoringAlgorithm.Greedy
            @test length(create_coloring(dh, boundary; alg = greedy)) == 2
            @test length(create_coloring(dh, getfacetset(grid, "left"); alg = greedy)) == 1
            # Determinism
            @test create_coloring(grid, boundary) == create_coloring(grid, boundary)
            # Input validation
            @test_throws ArgumentError create_coloring(grid, [boundary[1], boundary[1]])
            @test_throws ArgumentError create_coloring(grid, [FacetIndex(getncells(grid) + 1, 1)])
        end

        # Colored threaded facet assembly matches serial assembly
        let grid = generate_grid(Quadrilateral, (6, 6))
            dh = DofHandler(grid)
            add!(dh, :u, Lagrange{RefQuadrilateral, 1}())
            close!(dh)
            facetset = union(values(grid.facetsets)...)
            colors = create_coloring(grid, facetset)
            ndpc = ndofs_per_cell(dh)
            Ke = [i * 1.0e-3 + j * 1.0e-6 for i in 1:ndpc, j in 1:ndpc]
            function assemble_facets(dh, iterators, K)
                for it in iterators
                    assembler = start_assemble(K; fillzero = false)
                    for fc in it
                        assemble!(assembler, celldofs(fc), Ke)
                    end
                end
                return K
            end
            K0 = allocate_matrix(dh)
            Kserial = assemble_facets(dh, (FacetIterator(dh, facetset),), copy(K0))
            Kcolorserial = assemble_facets(dh, (FacetIterator(dh, color) for color in colors), copy(K0))
            Kthreaded = copy(K0)
            for color in colors
                chunks = collect(Iterators.partition(color, max(1, cld(length(color), Threads.nthreads()))))
                Threads.@threads for chunk in chunks
                    assemble_facets(dh, (FacetIterator(dh, collect(chunk)),), Kthreaded)
                end
            end
            @test Kcolorserial.nzval == Kthreaded.nzval
            @test Kserial.nzval ≈ Kthreaded.nzval
        end
    end
end

@testset "dof based coloring" begin
    item_cells(cell::Int) = (cell,)
    item_cells(facet::FacetIndex) = (facet[1],)
    item_cells(interface::NTuple{2, FacetIndex}) = (interface[1][1], interface[2][1])
    # The dofs an item writes to: the dofs of its cells, and with constraint
    # condensation the masters of the affinely constrained dofs
    function write_set(dh, ch, item)
        W = Set{Int}()
        for c in item_cells(item), d in celldofs(dh, c)
            push!(W, d)
            ch === nothing && continue
            i = get(ch.dofmapping, d, nothing)
            i === nothing && continue
            coeffs = ch.dofcoefficients[i]
            coeffs === nothing || foreach(((m, _),) -> push!(W, m), coeffs)
        end
        return W
    end
    # The conflict graph is exact and the colors are valid partitions
    function check_dof_coloring(dh, set; ch = nothing)
        ws = AssemblyConflicts(dh, ch)
        items = Ferrite._collect_items(eltype(set), getncells(Ferrite.get_grid(dh)), set)
        Ws = [write_set(dh, ch, item) for item in items]
        graph = Ferrite._item_conflict_graph(ws, items)
        for x in eachindex(items)
            neighbors = view(graph.adj, Ferrite._neighbor_range(graph, x))
            @test allunique(neighbors)
            for y in eachindex(items)
                x == y && continue
                @test (y in neighbors) == !isdisjoint(Ws[x], Ws[y])
            end
        end
        for alg in (ColoringAlgorithm.WorkStream, ColoringAlgorithm.Greedy)
            colors = ch === nothing ? create_coloring(dh, set; alg) : create_coloring(dh, ch, set; alg)
            @test colors == create_coloring(ws, set; alg)
            @test colors isa Vector{Vector{eltype(items)}}
            @test sum(length, colors; init = 0) == length(items)
            @test Set(Iterators.flatten(colors)) == Set(items)
            for color in colors
                Wc = [write_set(dh, ch, item) for item in color]
                for x in eachindex(Wc), y in (x + 1):lastindex(Wc)
                    @test isdisjoint(Wc[x], Wc[y])
                end
            end
        end
        return
    end
    function check_all_items(dh; ch = nothing)
        grid = Ferrite.get_grid(dh)
        topology = ExclusiveTopology(grid)
        check_dof_coloring(dh, 1:getncells(grid); ch)
        check_dof_coloring(dh, collect(1:2:getncells(grid)); ch)
        check_dof_coloring(dh, Int[]; ch)
        check_dof_coloring(dh, collect(union(values(grid.facetsets)...)); ch)
        check_dof_coloring(dh, interfaceskeleton(topology, grid); ch)
        return
    end

    for (grid, ips) in (
            (
                generate_grid(Triangle, (4, 4)),
                (
                    Lagrange{RefTriangle, 1}(), Lagrange{RefTriangle, 2}()^2,
                    DiscontinuousLagrange{RefTriangle, 1}(), Nedelec{RefTriangle, 1}(),
                    RaviartThomas{RefTriangle, 1}(), CrouzeixRaviart{RefTriangle, 1}(),
                ),
            ),
            (
                generate_grid(Hexahedron, (2, 2, 2)),
                (Lagrange{RefHexahedron, 1}(), Nedelec{RefHexahedron, 1}()),
            ),
        )
        for ip in ips
            dh = DofHandler(grid)
            add!(dh, :u, ip)
            close!(dh)
            check_all_items(dh)
        end
    end

    # Mixed fields, and a DofHandler on a subdomain with two SubDofHandlers
    let grid = generate_grid(Quadrilateral, (5, 4))
        dh = DofHandler(grid)
        add!(dh, :u, Lagrange{RefQuadrilateral, 1}()^2)
        add!(dh, :p, DiscontinuousLagrange{RefQuadrilateral, 0}())
        close!(dh)
        check_all_items(dh)
        addcellset!(grid, "a", x -> x[1] < -0.2)
        addcellset!(grid, "b", x -> x[1] > 0.4)
        dh = DofHandler(grid)
        sdh = SubDofHandler(dh, getcellset(grid, "a"))
        add!(sdh, :u, Lagrange{RefQuadrilateral, 1}())
        sdh = SubDofHandler(dh, getcellset(grid, "b"))
        add!(sdh, :u, Lagrange{RefQuadrilateral, 1}())
        add!(sdh, :v, Lagrange{RefQuadrilateral, 1}())
        close!(dh)
        check_all_items(dh)
    end

    # Discontinuous fields: cells never conflict, interfaces only through a common cell
    let grid = generate_grid(Hexahedron, (3, 3, 3))
        dh = DofHandler(grid)
        add!(dh, :u, DiscontinuousLagrange{RefHexahedron, 1}())
        close!(dh)
        @test create_coloring(dh; alg = ColoringAlgorithm.Greedy) == [collect(1:getncells(grid))]
        # Greedy coloring of the interface ("line") graph needs few colors (about Δ + 1
        # where Δ is the max number of facet neighbors)
        skeleton = interfaceskeleton(ExclusiveTopology(grid), grid)
        @test length(create_coloring(dh, skeleton; alg = ColoringAlgorithm.Greedy)) <= 7 # Δ = 6
    end

    # Constraint condensation: periodic and arbitrary affine constraints
    let grid = generate_grid(Quadrilateral, (6, 4))
        dh = DofHandler(grid)
        add!(dh, :u, Lagrange{RefQuadrilateral, 1}())
        close!(dh)
        ch = ConstraintHandler(dh)
        add!(ch, PeriodicDirichlet(:u, collect_periodic_facets(grid, "left", "right")))
        add!(ch, Dirichlet(:u, getfacetset(grid, "bottom"), Returns(0.0)))
        close!(ch)
        check_all_items(dh; ch)
        dh = DofHandler(grid)
        add!(dh, :u, DiscontinuousLagrange{RefQuadrilateral, 1}())
        close!(dh)
        ch = ConstraintHandler(dh)
        n = ndofs(dh)
        add!(ch, AffineConstraint(1, [n => 0.5, n ÷ 2 => 0.5], 0.0))
        add!(ch, AffineConstraint(5, [n - 3 => 1.0], 0.0))
        close!(ch)
        check_all_items(dh; ch)
        # The cells now conflict only through the constraints
        @test length(create_coloring(dh, ch; alg = ColoringAlgorithm.Greedy)) == 2
    end

    # Grid based AssemblyConflicts give the same colorings as the grid based methods
    for grid in (
            generate_grid(Triangle, (5, 4)), generate_grid(Hexahedron, (3, 3, 3)),
            generate_grid(Tetrahedron, (2, 2, 2)),
        )
        cc = AssemblyConflicts(grid)
        skeleton = interfaceskeleton(ExclusiveTopology(grid), grid)
        sets = (
            1:getncells(grid), collect(1:3:getncells(grid)), Int[],
            collect(union(values(grid.facetsets)...)), skeleton, skeleton[1:2:end],
        )
        for alg in (ColoringAlgorithm.WorkStream, ColoringAlgorithm.Greedy), set in sets
            @test create_coloring(cc, set; alg) == create_coloring(grid, set; alg)
        end
    end

    # Input validation
    let grid = generate_grid(Quadrilateral, (2, 2))
        dh = DofHandler(grid)
        add!(dh, :u, Lagrange{RefQuadrilateral, 1}())
        @test_throws ArgumentError AssemblyConflicts(dh)
        close!(dh)
        @test_throws ArgumentError AssemblyConflicts(dh, ConstraintHandler(dh))
        dh2 = DofHandler(grid)
        add!(dh2, :u, Lagrange{RefQuadrilateral, 1}())
        close!(dh2)
        @test_throws ArgumentError AssemblyConflicts(dh, close!(ConstraintHandler(dh2)))
        @test_throws ArgumentError create_coloring(dh, [FacetIndex(1, 1), FacetIndex(1, 1)])
    end
end
