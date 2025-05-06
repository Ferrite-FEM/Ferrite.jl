@testset "Field continuity" begin
    # Integration testing to ensure that correct continuity across facets are obtained
    # after dofdistribution with use of cell and facet values.

    function find_matching_facet(grid, facet::FacetIndex)
        cell, facetnr = facet
        facet_vertices = Set(Ferrite.facets(getcells(grid, cell))[facetnr])
        for cnr in 1:getncells(grid)
            cnr == cell && continue
            for (i, f_vert) in enumerate(Ferrite.facets(getcells(grid, cnr)))
                facet_vertices == Set(f_vert) && return FacetIndex(cnr, i)
            end
        end
        return nothing
    end

    function test_continuity(
            dh::DofHandler, facet::FacetIndex;
            transformation_function::Function,
            value_function::Function = function_value,
            fieldname = :u
        )
        # transformation_function: (v,n) -> z
        # Examples
        # * Tangential continuity: fun(v, n) = v - (v ⋅ n)*n
        # * Normal continuity: fun(v, n) = v ⋅ n
        # value_function: (fe_v, q_point, ue) -> z

        # Find the matching FaceIndex
        cellnr, facetnr = facet
        facet2 = find_matching_facet(dh.grid, facet)
        facet2 === nothing && return false

        # Pick "random" points on the facet
        cell = getcells(dh.grid, cellnr)
        sdh1 = dh.subdofhandlers[dh.cell_to_subdofhandler[cellnr]]
        ipg1 = geometric_interpolation(typeof(cell))
        ipf1 = Ferrite.getfieldinterpolation(sdh1, fieldname)
        fqr = FacetQuadratureRule{Ferrite.getrefshape(ipg1)}(8)
        fv = FacetValues(fqr, ipf1, ipg1)
        cell_coords = getcoordinates(dh.grid, cellnr)
        inds = randperm(getnquadpoints(fv))[1:min(4, getnquadpoints(fv))]

        # Random dof vector to test continuity
        u = rand(ndofs(dh))

        # Calculate coordinates and function values for these
        point_coords = zeros(eltype(cell_coords), length(inds))
        point_normal = similar(point_coords)
        fun_vals = zeros(typeof(shape_value(fv, 1, 1)), length(inds))
        reinit!(fv, cell, cell_coords, facetnr)
        ue = u[celldofs(dh, cellnr)]
        for (i, q_point) in enumerate(inds)
            point_coords[i] = spatial_coordinate(fv, q_point, cell_coords)
            point_normal[i] = getnormal(fv, q_point)
            fun_vals[i] = value_function(fv, q_point, ue)
        end

        # Calculate function values on the other cell
        cell2 = getcells(dh.grid, facet2[1])
        cell_coords2 = getcoordinates(dh.grid, facet2[1])
        ipg2 = geometric_interpolation(typeof(cell2))
        sdh2 = dh.subdofhandlers[dh.cell_to_subdofhandler[facet2[1]]]
        ipf2 = Ferrite.getfieldinterpolation(sdh2, fieldname)
        local_coords = map(x -> Ferrite.find_local_coordinate(ipg2, cell_coords2, x, Ferrite.NewtonLineSearchPointFinder()), point_coords)
        @assert all(first.(local_coords)) # check that find_local_coordinate converged
        ξs = collect(last.(local_coords)) # Extract the local coordinates
        qr = QuadratureRule{Ferrite.getrefshape(ipg2)}(zeros(length(ξs)), ξs)
        cv = CellValues(qr, ipf2, ipg2)
        reinit!(cv, cell2, cell_coords2)
        ue2 = u[celldofs(dh, facet2[1])]
        for q_point in 1:getnquadpoints(cv)
            @assert spatial_coordinate(cv, q_point, cell_coords2) ≈ point_coords[q_point]
            # Approximate points can contribute to the inaccuracies
            n = point_normal[q_point]
            v1 = fun_vals[q_point]
            v2 = value_function(cv, q_point, ue2)
            @test isapprox(transformation_function(v1, n), transformation_function(v2, n); atol = 1.0e-6)
        end
        return true
    end

    tupleshift(t::NTuple{N}, shift::Int) where {N} = ntuple(i -> t[mod(i - 1 - shift, N) + 1], N)
    #tupleshift(t::NTuple, shift::Int) = tuple(circshift(SVector(t), shift)...)
    cell_permutations(cell::Quadrilateral) = (Quadrilateral(tupleshift(cell.nodes, shift)) for shift in 0:3)
    cell_permutations(cell::Triangle) = (Triangle(tupleshift(cell.nodes, shift)) for shift in 0:2)
    cell_permutations(cell::QuadraticTriangle) = (QuadraticTriangle((tupleshift(cell.nodes[1:3], shift)..., tupleshift(cell.nodes[4:6], shift)...)) for shift in 0:3)
    cell_permutations(cell::QuadraticQuadrilateral) = (QuadraticQuadrilateral((tupleshift(cell.nodes[1:4], shift)..., tupleshift(cell.nodes[5:8], shift)..., cell.nodes[9])) for shift in 0:4)

    function cell_permutations(cell::Hexahedron)
        idx = ( #Logic on refshape: Select 1st and 2nd vertex (must be neighbours)
            # The next follows to create inward vector with RHR, and then 4th is in same plane.
            # The last four must be the neighbours on the other plane to the first four (same order)
            (1, 2, 3, 4, 5, 6, 7, 8), (1, 4, 8, 5, 2, 3, 7, 6), (1, 5, 6, 2, 4, 8, 7, 3),
            (2, 1, 5, 6, 3, 4, 8, 7), (2, 3, 4, 1, 6, 7, 8, 5), (2, 6, 7, 3, 1, 5, 8, 4),
            (3, 2, 6, 7, 4, 1, 5, 8), (3, 4, 1, 2, 7, 8, 5, 6), (3, 7, 8, 4, 2, 6, 5, 1),
            (4, 1, 2, 3, 8, 5, 6, 7), (4, 3, 7, 8, 1, 2, 6, 5), (4, 8, 5, 1, 3, 7, 6, 1),
            (5, 1, 4, 8, 6, 2, 3, 7), (5, 6, 2, 1, 8, 7, 3, 4), (5, 8, 7, 6, 1, 4, 3, 2),
            (6, 2, 1, 5, 7, 3, 4, 8), (6, 5, 8, 7, 2, 1, 4, 3), (6, 7, 3, 2, 5, 8, 4, 1),
            (7, 3, 2, 6, 8, 4, 1, 5), (7, 6, 5, 8, 3, 2, 1, 4), (7, 8, 4, 3, 6, 5, 1, 2),
            (8, 4, 3, 7, 5, 1, 2, 6), (8, 5, 1, 4, 7, 6, 2, 3), (8, 7, 6, 5, 4, 3, 2, 1),
        )
        return (Hexahedron(ntuple(i -> cell.nodes[perm[i]], 8)) for perm in idx)
    end

    function cell_permutations(cell::Tetrahedron)
        idx = (
            (1, 2, 3, 4), (1, 3, 4, 2), (1, 4, 2, 3),
            (2, 1, 4, 3), (2, 3, 1, 4), (2, 4, 3, 1),
            (3, 1, 2, 4), (3, 2, 4, 1), (3, 4, 1, 2),
            (4, 1, 3, 2), (4, 3, 2, 1), (4, 2, 1, 3),
        )
        return (Tetrahedron(ntuple(i -> cell.nodes[perm[i]], 4)) for perm in idx)
    end

    continuity_function(ip::Interpolation) = continuity_function(Ferrite.conformity(ip))
    continuity_function(::Ferrite.H1Conformity) = ((v, _) -> v)
    continuity_function(::Ferrite.HcurlConformity) = ((v, n) -> v - n * (v ⋅ n)) # Tangent continuity
    continuity_function(::Ferrite.HdivConformity) = ((v, n) -> v ⋅ n) # Normal continuity

    nel = 3

    cell_types = Dict(
        RefTriangle => [Triangle, QuadraticTriangle],
        RefQuadrilateral => [Quadrilateral, QuadraticQuadrilateral],
        RefTetrahedron => [Tetrahedron],
        RefHexahedron => [Hexahedron]
    )

    test_ips = [
        Lagrange{RefTriangle, 2}(), Lagrange{RefQuadrilateral, 2}(), Lagrange{RefHexahedron, 2}()^3, # Test should also work for identity mapping
        Nedelec{RefTriangle, 1}(), Nedelec{RefTriangle, 2}(), Nedelec{RefQuadrilateral, 1}(), Nedelec{RefTetrahedron, 1}(), Nedelec{RefHexahedron, 1}(),
        RaviartThomas{RefTriangle, 1}(), RaviartThomas{RefTriangle, 2}(), RaviartThomas{RefQuadrilateral, 1}(), RaviartThomas{RefTetrahedron, 1}(), RaviartThomas{RefHexahedron, 1}(),
        BrezziDouglasMarini{RefTriangle, 1}(),
    ]
    @testset "Non-mixed grid" begin
        for ip in test_ips
            RefShape = getrefshape(ip)
            dim = Ferrite.getrefdim(ip) # dim = sdim = rdim
            p1, p2 = (rand(Vec{dim}), ones(Vec{dim}) + rand(Vec{dim}))
            transfun(x) = typeof(x)(i -> sinpi(x[mod(i, length(x)) + 1] + i / 3)) / 10

            for CT in cell_types[RefShape]
                grid = generate_grid(CT, ntuple(_ -> nel, dim), p1, p2)
                # Smoothly distort grid (to avoid spuriously badly deformed elements).
                # A distorted grid is important to properly test the geometry mapping
                # for 2nd order elements.
                transform_coordinates!(grid, x -> (x + transfun(x)))
                cellnr = getncells(grid) ÷ 2 + 1 # Should be a cell in the center
                basecell = getcells(grid, cellnr)
                @testset "$CT, $ip" begin
                    for testcell in cell_permutations(basecell)
                        grid.cells[cellnr] = testcell
                        dh = DofHandler(grid)
                        add!(dh, :u, ip)
                        close!(dh)
                        cnt = 0
                        for facetnr in 1:nfacets(RefShape)
                            fi = FacetIndex(cellnr, facetnr)
                            # Check continuity of function value according to continuity_function
                            found_matching = test_continuity(dh, fi; transformation_function = continuity_function(ip))
                            cnt += found_matching
                        end
                        @assert cnt > 0
                    end
                end
            end
        end
    end

    # Test continuity for 2D mixed grid, Quadrilaterals and Triangles
    test_ips = [
        (Nedelec, 1), #(Nedelec, 2), # 2nd order Nedelec on Quadrilaterals not yet implemented
        (RaviartThomas, 1), #(RaviartThomas, 2) # 2nd order RT on Quadrilaterals not yet implemented
    ]

    @testset "Quad in Tri-grid" begin
        trigrid = generate_grid(Triangle, (3, 3))
        mixgrid, cellnr = grid_with_inserted_quad(trigrid, (9, 10); update_sets = false)
        dim = 2
        p1, p2 = (rand(Vec{dim}), ones(Vec{dim}) + rand(Vec{dim}))
        transfun(x) = typeof(x)(i -> sinpi(x[mod(i, length(x)) + 1] + i / 3)) / 10
        basecell = getcells(mixgrid, cellnr)

        for (ip, order) in test_ips
            ip1 = ip{RefTriangle, order}(); set1 = setdiff(1:getncells(mixgrid), cellnr)
            ip2 = ip{RefQuadrilateral, order}(); set2 = Set(cellnr)

            @testset "$ip, order = $order" begin
                for testcell in cell_permutations(basecell)
                    mixgrid.cells[cellnr] = testcell
                    dh = DofHandler(mixgrid)
                    sdh1 = SubDofHandler(dh, set1)
                    add!(sdh1, :u, ip1)
                    sdh2 = SubDofHandler(dh, set2)
                    add!(sdh2, :u, ip2)
                    close!(dh)
                    cnt = 0
                    for facetnr in 1:nfacets(testcell)
                        fi = FacetIndex(cellnr, facetnr)
                        # Check continuity of function value according to continuity_function
                        found_matching = test_continuity(dh, fi; transformation_function = continuity_function(ip1))
                        cnt += found_matching
                    end
                    @assert cnt > 0
                end
            end
        end
    end
end
