# Imports for parallel (isolated) test execution:
include(joinpath(@__DIR__, "test_utils.jl"))

@testset "InterfaceValues" begin
    function test_interfacevalues(grid::Ferrite.AbstractGrid, iv::InterfaceValues; tol = 0)
        ip_here = Ferrite.function_interpolation(iv.here)
        ip_there = Ferrite.function_interpolation(iv.there)
        rdim = Ferrite.getrefdim(ip_here)
        n_basefuncs = getnbasefunctions(ip_here) + getnbasefunctions(ip_there)

        @test getnbasefunctions(iv) == n_basefuncs

        for ic in InterfaceIterator(grid)
            reinit!(iv, ic)
            coords_here, coords_there = getcoordinates(ic)
            nqp = getnquadpoints(iv)
            # Should have same quadrature points
            @test nqp == getnquadpoints(iv.here) == getnquadpoints(iv.there)
            for qp in 1:nqp
                # If correctly synced quadrature points coordinates should match
                @test isapprox(
                    spatial_coordinate(iv, qp, coords_here, coords_there; here = true),
                    spatial_coordinate(iv, qp, coords_here, coords_there; here = false); atol = tol
                )
                for i in 1:getnbasefunctions(iv)
                    here = i <= getnbasefunctions(iv.here)
                    shapevalue = shape_value(iv, qp, i; here = here)
                    shape_avg = shape_value_average(iv, qp, i)
                    shape_jump = shape_value_jump(iv, qp, i)

                    shapegrad = shape_gradient(iv, qp, i; here = here)
                    shapegrad_avg = shape_gradient_average(iv, qp, i)
                    shapegrad_jump = shape_gradient_jump(iv, qp, i)

                    normal = getnormal(iv, qp; here = false)
                    # Test values (May be removed as it mirrors implementation)
                    if i > getnbasefunctions(iv.here)
                        @test shapevalue ≈ shape_value(iv.there, qp, i - getnbasefunctions(iv.here))
                        @test shapegrad ≈ shape_gradient(iv.there, qp, i - getnbasefunctions(iv.here))

                        @test shape_jump ≈ shapevalue
                        @test shapegrad_jump ≈ shapegrad
                    else
                        normal = getnormal(iv, qp)
                        @test shapevalue ≈ shape_value(iv.here, qp, i)
                        @test shapegrad ≈ shape_gradient(iv.here, qp, i)

                        @test shape_jump ≈ -shapevalue
                        @test shapegrad_jump ≈ -shapegrad
                    end

                    @test shape_avg ≈ 0.5 * shapevalue
                    @test shapegrad_avg ≈ 0.5 * shapegrad

                end
            end
            @test_throws ErrorException("Invalid base function $(n_basefuncs + 1). Interface has only $(n_basefuncs) base functions") shape_value_jump(iv, 1, n_basefuncs + 1)
            @test_throws ErrorException("Invalid base function $(n_basefuncs + 1). Interface has only $(n_basefuncs) base functions") shape_gradient_average(iv, 1, n_basefuncs + 1)

            # Test function* copied from facetvalues tests
            nbf_a = Ferrite.getngeobasefunctions(iv.here)
            nbf_b = Ferrite.getngeobasefunctions(iv.there)
            for here in (true, false)
                u_a = zeros(Vec{rdim, Float64}, nbf_a)
                u_b = zeros(Vec{rdim, Float64}, nbf_b)
                u_scal_a = zeros(nbf_a)
                u_scal_b = zeros(nbf_b)
                H = rand(Tensor{2, rdim})
                V = rand(Tensor{1, rdim})
                for i in 1:nbf_a
                    xs = coords_here
                    u_a[i] = H ⋅ xs[i]
                    u_scal_a[i] = V ⋅ xs[i]
                end
                for i in 1:nbf_b
                    xs = coords_there
                    u_b[i] = H ⋅ xs[i]
                    u_scal_b[i] = V ⋅ xs[i]
                end
                u = vcat(u_a, u_b)
                u_scal = vcat(u_scal_a, u_scal_b)
                u_vector = reinterpret(Float64, u)
                for i in 1:getnquadpoints(iv)
                    if ip_here isa Ferrite.ScalarInterpolation
                        @test function_gradient(iv, i, u, here = here) ≈ H
                        @test function_gradient(iv, i, u_scal, here = here) ≈ V

                        @test isapprox(function_value_average(iv, i, u_scal), function_value(iv, i, u_scal, here = here); atol = tol)
                        @test all(function_value_jump(iv, i, u_scal) .<= 30 * eps(Float64))
                        @test isapprox(function_gradient_average(iv, i, u_scal), function_gradient(iv, i, u_scal, here = here); atol = tol)
                        @test all(function_gradient_jump(iv, i, u_scal) .<= 30 * eps(Float64))

                        @test isapprox(function_value_average(iv, i, u), function_value(iv, i, u, here = here); atol = tol)
                        @test all(function_value_jump(iv, i, u) .<= 30 * eps(Float64))
                        @test isapprox(function_gradient_average(iv, i, u), function_gradient(iv, i, u, here = here); atol = tol)
                        @test all(function_gradient_jump(iv, i, u) .<= 30 * eps(Float64))
                    else # func_interpol isa Ferrite.VectorInterpolation
                        @test function_gradient(iv, i, u_vector; here = here) ≈ H
                        @test isapprox(function_value_average(iv, i, u_vector), function_value(iv, i, u_vector, here = here); atol = tol)
                        @test all(function_value_jump(iv, i, u_vector) .<= 30 * eps(Float64))
                        @test isapprox(function_gradient_average(iv, i, u_vector), function_gradient(iv, i, u_vector, here = here); atol = tol)
                        @test all(function_gradient_jump(iv, i, u_vector) .<= 30 * eps(Float64))
                    end
                end
                # Test of volume
                vol = 0.0
                for i in 1:getnquadpoints(iv)
                    vol += getdetJdV(iv, i)
                end
                xs = here ? coords_here : coords_there
                face = here ? Ferrite.getcurrentfacet(iv.here) : Ferrite.getcurrentfacet(iv.there)
                func_interpol = here ? ip_here : ip_there
                let ip_base = func_interpol isa VectorizedInterpolation ? func_interpol.ip : func_interpol
                    x_face = xs[[Ferrite.dirichlet_facetdof_indices(ip_base)[face]...]]
                    @test vol ≈ calculate_facet_area(ip_base, x_face, face)
                end
            end
        end
    end
    getcelltypedim(::Type{<:Ferrite.AbstractCell{shape}}) where {dim, shape <: Ferrite.AbstractRefShape{dim}} = dim
    for (cell_shape, scalar_interpol, quad_rule) in (
            #TODO: update interfaces for lines
            (Line, DiscontinuousLagrange{RefLine, 1}(), FacetQuadratureRule{RefLine}(2)),
            (QuadraticLine, DiscontinuousLagrange{RefLine, 2}(), FacetQuadratureRule{RefLine}(2)),
            (Quadrilateral, DiscontinuousLagrange{RefQuadrilateral, 1}(), FacetQuadratureRule{RefQuadrilateral}(2)),
            (QuadraticQuadrilateral, DiscontinuousLagrange{RefQuadrilateral, 2}(), FacetQuadratureRule{RefQuadrilateral}(2)),
            (Triangle, DiscontinuousLagrange{RefTriangle, 1}(), FacetQuadratureRule{RefTriangle}(2)),
            (QuadraticTriangle, DiscontinuousLagrange{RefTriangle, 2}(), FacetQuadratureRule{RefTriangle}(2)),
            (Hexahedron, DiscontinuousLagrange{RefHexahedron, 1}(), FacetQuadratureRule{RefHexahedron}(2)),
            # (QuadraticQuadrilateral, Serendipity{RefQuadrilateral, 2}(), FacetQuadratureRule{RefQuadrilateral}(2)),
            (Tetrahedron, DiscontinuousLagrange{RefTetrahedron, 1}(), FacetQuadratureRule{RefTetrahedron}(2)),
            # (QuadraticTetrahedron, Lagrange{RefTetrahedron, 2}(), FacetQuadratureRule{RefTetrahedron}(2)),
            (Wedge, DiscontinuousLagrange{RefPrism, 1}(), FacetQuadratureRule{RefPrism}(2)),
            (Pyramid, DiscontinuousLagrange{RefPyramid, 1}(), FacetQuadratureRule{RefPyramid}(2)),
        )
        dim = getcelltypedim(cell_shape)
        grid = generate_grid(cell_shape, ntuple(i -> 2, dim))
        ip = scalar_interpol isa DiscontinuousLagrange ? Lagrange{Ferrite.getrefshape(scalar_interpol), Ferrite.getorder(scalar_interpol)}() : scalar_interpol
        @testset "faces nodes indices" begin
            cell = getcells(grid, 1)
            geom_ip_facets_indices = Ferrite.facetdof_indices(ip)
            Ferrite.getrefdim(ip) > 1 && (geom_ip_facets_indices = Tuple([facet[collect(facet .∉ Ref(interior))] for (facet, interior) in [(geom_ip_facets_indices[i], Ferrite.facetdof_interior_indices(ip)[i]) for i in 1:Ferrite.nfacets(ip)]]))
            facets_indices = Ferrite.reference_facets(Ferrite.getrefshape(Ferrite.geometric_interpolation(typeof(cell))))
            node_ids = Ferrite.get_node_ids(cell)
            cellfacets = Ferrite.facets(cell)
            @test getindex.(Ref(node_ids), collect.(facets_indices)) == cellfacets == getindex.(Ref(node_ids), collect.(geom_ip_facets_indices))
        end
        @testset "error paths" begin
            cell = getcells(grid, 1)
            dim == 1 && @test_throws ErrorException("1D elements don't use transformations for interfaces.") Ferrite.InterfaceOrientationInfo(cell, cell, 1, 1)
            @test_throws ArgumentError("unknown facet number") Ferrite.element_to_facet_transformation(Vec{dim, Float64}(ntuple(_ -> 0.0, dim)), Ferrite.getrefshape(cell), 100)
            @test_throws ArgumentError("unknown facet number") Ferrite.facet_to_element_transformation(Vec{dim - 1, Float64}(ntuple(_ -> 0.0, dim - 1)), Ferrite.getrefshape(cell), 100)
        end
        func_interpol = scalar_interpol
        for func_interpol in (scalar_interpol, VectorizedInterpolation(scalar_interpol))
            iv = cell_shape ∈ (QuadraticLine, QuadraticQuadrilateral, QuadraticTriangle, QuadraticTetrahedron) ?
                InterfaceValues(quad_rule, func_interpol, ip) : InterfaceValues(quad_rule, func_interpol)
            test_interfacevalues(grid, iv)
        end
    end
    @testset "construction errors" begin
        @test_throws ArgumentError InterfaceValues(FacetQuadratureRule{RefTriangle}(1), Lagrange{RefQuadrilateral, 1}())
        @test_throws ArgumentError InterfaceValues(FacetQuadratureRule{RefTriangle}(1), Lagrange{RefTriangle, 1}(), Lagrange{RefQuadrilateral, 1}())
        @test_throws ArgumentError InterfaceValues(FacetQuadratureRule{RefTriangle}(1), Lagrange{RefQuadrilateral, 1}(), Lagrange{RefQuadrilateral, 1}())
        @test_throws ArgumentError InterfaceValues(FacetQuadratureRule{RefTriangle}(1), Lagrange{RefQuadrilateral, 1}(), Lagrange{RefTriangle, 1}())
        @test_throws ArgumentError InterfaceValues(FacetQuadratureRule{RefTriangle}(1), Lagrange{RefTriangle, 1}(), FacetQuadratureRule{RefTriangle}(1), Lagrange{RefQuadrilateral, 1}())
    end
    # Custom quadrature
    @testset "Custom quadrature interface values" begin
        cell_shape = Tetrahedron
        scalar_interpol = DiscontinuousLagrange{RefTetrahedron, 1}()
        # From https://www.researchgate.net/publication/258241862_Application_of_Composite_Numerical_Integrations_Using_Gauss-Radau_and_Gauss-Lobatto_Quadrature_Rules?enrichId=rgreq-a5675bf95a198061d6e153e39f856f53-XXX&enrichSource=Y292ZXJQYWdlOzI1ODI0MTg2MjtBUzo5ODgzMzU0MzQ2NzAxOUAxNDAwNTc1MTYxNjA2&el=1_x_2&_esc=publicationCoverPdf
        points = Vec{2, Float64}.([[0.0, 0.844948974278318], [0.205051025721682, 0.694948974278318], [0.487979589711327, 0.487979589711327], [0.0, 0.355051025721682], [0.29202041028867254, 0.29202041028867254], [0.694948974278318, 0.205051025721682], [0.0, 0.0], [0.355051025721682, 0.0], [0.844948974278318, 0.0]])
        # Weights resulted in 4 times the volume [-1, 1] -> so /4 to get [0, 1]
        weights = [0.096614387479324, 0.308641975308642, 0.087870061825481, 0.187336229804627, 0.677562036939952, 0.308641975308642, 0.049382716049383, 0.187336229804627, 0.096614387479324] / 4
        quad_rule = Ferrite.create_facet_quad_rule(RefTetrahedron, weights, points)
        dim = getcelltypedim(cell_shape)
        grid = generate_grid(cell_shape, ntuple(i -> 2, dim))
        @testset "faces nodes indices" begin
            ip = scalar_interpol isa DiscontinuousLagrange ? Lagrange{Ferrite.getrefshape(scalar_interpol), Ferrite.getorder(scalar_interpol)}() : scalar_interpol
            cell = getcells(grid, 1)
            geom_ip_facets_indices = Ferrite.facetdof_indices(ip)
            Ferrite.getrefdim(ip) > 1 && (geom_ip_facets_indices = Tuple([facet[collect(facet .∉ Ref(interior))] for (facet, interior) in [(geom_ip_facets_indices[i], Ferrite.facedof_interior_indices(ip)[i]) for i in 1:Ferrite.nfaces(ip)]]))
            facets_indices = Ferrite.reference_facets(Ferrite.getrefshape(Ferrite.geometric_interpolation(typeof(cell))))
            node_ids = Ferrite.get_node_ids(cell)
            @test getindex.(Ref(node_ids), collect.(facets_indices)) == Ferrite.faces(cell) == getindex.(Ref(node_ids), collect.(geom_ip_facets_indices))
        end
        @testset "error paths" begin
            cell = getcells(grid, 1)
            @test_throws ArgumentError("unknown facet number") Ferrite.element_to_facet_transformation(Vec{dim, Float64}(ntuple(_ -> 0.0, dim)), Ferrite.getrefshape(cell), 100)
            @test_throws ArgumentError("unknown facet number") Ferrite.facet_to_element_transformation(Vec{dim - 1, Float64}(ntuple(_ -> 0.0, dim - 1)), Ferrite.getrefshape(cell), 100)
        end
        for func_interpol in (scalar_interpol, VectorizedInterpolation(scalar_interpol))
            iv = InterfaceValues(quad_rule, func_interpol)
            test_interfacevalues(grid, iv; tol = 5 * eps(Float64))
        end
    end
    # @testset "Mixed elements 2D grids" begin # TODO: this shouldn't work because it should change the FacetValues object
    #     dim = 2
    #     nodes = [Node((-1.0, 0.0)), Node((0.0, 0.0)), Node((1.0, 0.0)), Node((-1.0, 1.0)), Node((0.0, 1.0))]
    #     cells = [
    #                 Quadrilateral((1,2,5,4)),
    #                 Triangle((3,5,2)),
    #             ]

    #     grid = Grid(cells, nodes)
    #     topology = ExclusiveTopology(grid)
    #     test_interfacevalues(grid,
    #     DiscontinuousLagrange{RefQuadrilateral, 1}(), FacetQuadratureRule{RefQuadrilateral}(2),
    #     DiscontinuousLagrange{RefTriangle, 1}(), FacetQuadratureRule{RefTriangle}(2))
    # end
    @testset "Unordered nodes 3D" begin
        @testset "Hexahedron" begin
            nodes = [
                Node((-1.0, 0.0, 0.0)), Node((0.0, 0.0, 0.0)), Node((1.0, 0.0, 0.0)),
                Node((-1.0, 1.0, 0.0)), Node((0.0, 1.0, 0.0)), Node((1.0, 1.0, 0.0)),
                Node((-1.0, 0.0, 1.0)), Node((0.0, 0.0, 1.0)), Node((1.0, 0.0, 1.0)),
                Node((-1.0, 1.0, 1.0)), Node((0.0, 1.0, 1.0)), Node((1.0, 1.0, 1.0)),
            ]
            cells = [
                Hexahedron((1, 2, 5, 4, 7, 8, 11, 10)),
                Hexahedron((5, 6, 12, 11, 2, 3, 9, 8)),
            ]
            grid = Grid(cells, nodes)
            test_interfacevalues(
                grid,
                InterfaceValues(FacetQuadratureRule{RefHexahedron}(2), DiscontinuousLagrange{RefHexahedron, 1}())
            )
            orientation_info = Ferrite.InterfaceOrientationInfo(getcells(grid, 1), getcells(grid, 2), 3, 5)
            @testset "Interface Orientation" begin
                @test orientation_info.flipped == true
                @test Ferrite.get_transformation_matrix(orientation_info) isa Tensor{2, 3}
            end
            @testset "Flipped normal Interface Orientation" begin
                nodes = [
                    Node((-1.0, 0.0, 0.0)), Node((0.0, 0.0, 0.0)), Node((1.0, 0.0, 0.0)),
                    Node((-1.0, 1.0, 0.0)), Node((0.0, 1.0, 0.0)), Node((1.0, 1.0, 0.0)),
                    Node((-1.0, 0.0, 1.0)), Node((0.0, 0.0, 1.0)), Node((1.0, 0.0, 1.0)),
                    Node((-1.0, 1.0, 1.0)), Node((0.0, 1.0, 1.0)), Node((1.0, 1.0, 1.0)),
                ]
                cells = [
                    Hexahedron((1, 4, 5, 2, 7, 10, 11, 8)),
                    Hexahedron((5, 6, 12, 11, 2, 3, 9, 8)),
                ]
                grid = Grid(cells, nodes)
                orientation_info = Ferrite.InterfaceOrientationInfo(getcells(grid, 1), getcells(grid, 2), 4, 5)
                @test orientation_info.flipped == false
                @test Ferrite.get_transformation_matrix(orientation_info) isa Tensor{2, 3}
            end
        end
        @testset "Tetrahedron" begin
            nodes = [
                Node((0.0, 0.0, 0.0)), Node((1.0, 0.0, 0.0)), Node((0.0, 1.0, 0.0)),
                Node((0.0, 0.0, 1.0)), Node((-1.0, 0.0, 0.0)),
            ]
            cells = [
                Tetrahedron((1, 2, 3, 4)),
                Tetrahedron((1, 3, 5, 4)),
            ]
            grid = Grid(cells, nodes)
            test_interfacevalues(
                grid,
                InterfaceValues(FacetQuadratureRule{RefTetrahedron}(2), DiscontinuousLagrange{RefTetrahedron, 1}())
            )
            orientation_info = Ferrite.InterfaceOrientationInfo(getcells(grid, 1), getcells(grid, 2), 4, 2)
            @testset "Interface Orientation" begin
                @test orientation_info.flipped == true
                @test Ferrite.get_transformation_matrix(orientation_info) isa Tensor{2, 3}
            end
            @testset "Flipped normal Interface Orientation" begin
                nodes = [
                    Node((0.0, 0.0, 0.0)), Node((1.0, 0.0, 0.0)), Node((0.0, 1.0, 0.0)),
                    Node((0.0, 0.0, 1.0)), Node((-1.0, 0.0, 0.0)),
                ]
                cells = [
                    Tetrahedron((1, 2, 4, 3)),
                    Tetrahedron((1, 3, 5, 4)),
                ]
                grid = Grid(cells, nodes)
                orientation_info = Ferrite.InterfaceOrientationInfo(getcells(grid, 1), getcells(grid, 2), 4, 2)
                @test orientation_info.flipped == false
                @test Ferrite.get_transformation_matrix(orientation_info) isa Tensor{2, 3}
            end
        end
    end
    @testset "Interface dof_range" begin
        grid = generate_grid(Quadrilateral, (3, 3))
        ip_u = DiscontinuousLagrange{RefQuadrilateral, 1}()^2
        ip_p = DiscontinuousLagrange{RefQuadrilateral, 1}()
        qr_facet = FacetQuadratureRule{RefQuadrilateral}(2)
        iv = InterfaceValues(qr_facet, ip_p)
        @test iv == InterfaceValues(iv.here, iv.there)
        dh = DofHandler(grid)
        add!(dh, :u, ip_u)
        add!(dh, :p, ip_p)
        add!(dh, :_p, ip_p)
        close!(dh)
        ic = first(InterfaceIterator(dh))
        @test dof_range(ic, :p) == (9:12, 25:28)
    end
    # Test copy
    iv = InterfaceValues(FacetQuadratureRule{RefQuadrilateral}(2), DiscontinuousLagrange{RefQuadrilateral, 1}())
    ivc = copy(iv)
    @test typeof(iv) == typeof(ivc)
    for fname in fieldnames(typeof(iv))
        v = getfield(iv, fname)
        vc = getfield(ivc, fname)
        if hasmethod(pointer, Tuple{typeof(v)})
            @test pointer(v) != pointer(vc)
        end
        v isa FacetValues && continue
        for fname in fieldnames(typeof(vc))
            v2 = getfield(v, fname)
            vc2 = getfield(vc, fname)
            if hasmethod(pointer, Tuple{typeof(v2)})
                @test pointer(v2) != pointer(vc2)
            end
            @test v2 == vc2
        end
    end
    @testset "undefined transformation matrix error path" begin
        it = Ferrite.InterfaceOrientationInfo{DummyRefShapes.RefDodecahedron, DummyRefShapes.RefDodecahedron}(false, 0, 0, 1, 1)
        @test_throws ArgumentError("transformation is not implemented") Ferrite.get_transformation_matrix(it)
    end
    @testset "show" begin
        iv = InterfaceValues(FacetQuadratureRule{RefQuadrilateral}(2), Lagrange{RefQuadrilateral, 2}())
        showstring = sprint(show, MIME"text/plain"(), iv)
        @test contains(showstring, "InterfaceValues with")
    end
    @testset "AffineInterfaceTransformation" begin
        # Conforming interfaces: the explicit affine transformation must reproduce the
        # standard (vertex-derived InterfaceOrientationInfo) reinit! exactly.
        function test_conforming_equivalence(grid, iv, iv_affine)
            cells = Ferrite.getcells(grid)
            for ic in InterfaceIterator(grid)
                fiA, fiB = ic.a.current_facet_id[], ic.b.current_facet_id[]
                cA, cB = cellid(ic.a), cellid(ic.b)
                coords_here, coords_there = getcoordinates(ic)
                reinit!(iv, ic)
                trans = Ferrite.AffineInterfaceTransformation(cells[cA], coords_here, fiA, cells[cB], coords_there, fiB)
                reinit!(iv_affine, cells[cA], coords_here, fiA, cells[cB], coords_there, fiB, trans)
                for qp in 1:getnquadpoints(iv)
                    @test spatial_coordinate(iv_affine, qp, coords_here, coords_there; here = false) ≈
                        spatial_coordinate(iv, qp, coords_here, coords_there; here = true)
                    for i in 1:getnbasefunctions(iv.there)
                        @test shape_value(iv_affine.there, qp, i) ≈ shape_value(iv.there, qp, i)
                    end
                end
            end
        end
        grid = generate_grid(Quadrilateral, (2, 2))
        ip = Lagrange{RefQuadrilateral, 1}()
        iv = InterfaceValues(FacetQuadratureRule{RefQuadrilateral}(2), ip)
        test_conforming_equivalence(grid, iv, copy(iv))
        # ... including a rotated/flipped hexahedron pair (cf. the "Unordered nodes 3D" testset)
        hexnodes = [
            Node((-1.0, 0.0, 0.0)), Node((0.0, 0.0, 0.0)), Node((1.0, 0.0, 0.0)),
            Node((-1.0, 1.0, 0.0)), Node((0.0, 1.0, 0.0)), Node((1.0, 1.0, 0.0)),
            Node((-1.0, 0.0, 1.0)), Node((0.0, 0.0, 1.0)), Node((1.0, 0.0, 1.0)),
            Node((-1.0, 1.0, 1.0)), Node((0.0, 1.0, 1.0)), Node((1.0, 1.0, 1.0)),
        ]
        hexgrid = Grid([Hexahedron((1, 2, 5, 4, 7, 8, 11, 10)), Hexahedron((5, 6, 12, 11, 2, 3, 9, 8))], hexnodes)
        ip3 = Lagrange{RefHexahedron, 1}()
        iv3 = InterfaceValues(FacetQuadratureRule{RefHexahedron}(2), ip3)
        test_conforming_equivalence(hexgrid, iv3, copy(iv3))
        hexgrid2 = generate_grid(Hexahedron, (2, 2, 2))
        test_conforming_equivalence(hexgrid2, iv3, copy(iv3))

        # Hand-built 2D hanging-node interface: coarse cell 1 next to two fine cells.
        #   4-------3
        #   |       |6--5
        #   |   1   |2 3|
        #   |       |    (nodes 6/7 hang at the midpoints of facet (2,3) of cell 1)
        #   +       7--+
        nodes2d = Node.([Vec(0.0, 0.0), Vec(2.0, 0.0), Vec(2.0, 2.0), Vec(0.0, 2.0), Vec(3.0, 2.0), Vec(2.0, 1.0), Vec(3.0, 1.0), Vec(3.0, 0.0)])
        cells2d = [
            Quadrilateral((1, 2, 3, 4)),      # coarse
            Quadrilateral((2, 8, 7, 6)),      # fine lower, its facet 4 = (6, 2) ⊂ coarse facet 2 = (2, 3)
            Quadrilateral((6, 7, 5, 3)),      # fine upper, its facet 4 = (3, 6) ⊂ coarse facet 2
        ]
        hgrid = Grid(cells2d, nodes2d)
        ivh = InterfaceValues(FacetQuadratureRule{RefQuadrilateral}(2), Lagrange{RefQuadrilateral, 1}())
        u_lin(x) = 1.0 + 2.0 * x[1] - 3.0 * x[2]  # linear field: exactly represented on both sides
        for (fine, ffine) in ((2, 4), (3, 4))
            ca = Ferrite.getcells(hgrid, fine); cb = Ferrite.getcells(hgrid, 1)
            coords_a = getcoordinates(hgrid, fine); coords_b = getcoordinates(hgrid, 1)
            trans = Ferrite.AffineInterfaceTransformation(ca, coords_a, ffine, cb, coords_b, 2)
            reinit!(ivh, ca, coords_a, ffine, cb, coords_b, 2, trans)
            ue = [u_lin(coords_a[i]) for i in 1:4]
            ut = [u_lin(coords_b[i]) for i in 1:4]
            for qp in 1:getnquadpoints(ivh)
                xa = spatial_coordinate(ivh, qp, coords_a, coords_b; here = true)
                xb = spatial_coordinate(ivh, qp, coords_a, coords_b; here = false)
                @test xa ≈ xb
                @test function_value(ivh, qp, vcat(ue, ut); here = true) ≈
                    function_value(ivh, qp, vcat(ue, ut); here = false)
                @test function_value(ivh, qp, vcat(ue, ut); here = true) ≈ u_lin(xa)
            end
        end

        # Error paths: non-nested facets and unsupported (3D triangular) facets
        ca = Ferrite.getcells(hgrid, 2); cb = Ferrite.getcells(hgrid, 1)
        @test_throws ArgumentError Ferrite.AffineInterfaceTransformation(
            ca, getcoordinates(hgrid, 2), 2, cb, getcoordinates(hgrid, 1), 2
        )
        tetgrid = generate_grid(Tetrahedron, (1, 1, 1))
        teta = Ferrite.getcells(tetgrid, 1); tetb = Ferrite.getcells(tetgrid, 2)
        @test_throws ArgumentError Ferrite.AffineInterfaceTransformation(
            teta, getcoordinates(tetgrid, 1), 1, tetb, getcoordinates(tetgrid, 2), 1
        )
    end
end # of testset


# --- InterfaceValues on the facet skeleton of a non-conforming (AMR) grid ---
@testset "InterfaceValues on facet skeleton" begin
    # Every skeleton pair — conforming, hanging, across trees, rotated macro elements —
    # must reinit! an InterfaceValues via AffineInterfaceTransformation such that the two
    # sides' quadrature points coincide physically, and a globally linear field (whose
    # nodal values automatically satisfy the hanging midpoint constraints) is continuous
    # with continuous gradient across the interface.
    function check_interfacevalues(forest, refshape)
        dim = refshape === RefQuadrilateral ? 2 : 3
        grid = Ferrite.AMR.creategrid(forest)
        skel = Ferrite.facetskeleton(forest)
        cells = Ferrite.getcells(grid)
        iv = InterfaceValues(FacetQuadratureRule{refshape}(2), Lagrange{refshape, 1}())
        u_lin(x) = 1.0 + 2.0 * x[1] - 3.0 * x[dim] + (dim == 3 ? 0.5 * x[2] : 0.0)
        ∇u_lin = Tensors.gradient(u_lin, zero(Vec{dim}))
        for (fiA, fiB) in skel
            cA, fA = fiA[1], fiA[2]
            cB, fB = fiB[1], fiB[2]
            coordsA = getcoordinates(grid, cA)
            coordsB = getcoordinates(grid, cB)
            trans = Ferrite.AffineInterfaceTransformation(cells[cA], coordsA, fA, cells[cB], coordsB, fB)
            reinit!(iv, cells[cA], coordsA, fA, cells[cB], coordsB, fB, trans)
            ue = [u_lin.(coordsA); u_lin.(coordsB)]
            for qp in 1:getnquadpoints(iv)
                xh = spatial_coordinate(iv, qp, coordsA, coordsB; here = true)
                xt = spatial_coordinate(iv, qp, coordsA, coordsB; here = false)
                @test xh ≈ xt
                @test function_value(iv, qp, ue; here = true) ≈ u_lin(xh)
                @test function_value(iv, qp, ue; here = false) ≈ u_lin(xh)
                @test function_gradient(iv, qp, ue; here = true) ≈ ∇u_lin
                @test function_gradient(iv, qp, ue; here = false) ≈ ∇u_lin
                # opposing outward normals (straight facets for these grids)
                @test getnormal(iv, qp; here = true) ≈ -getnormal(iv, qp; here = false)
            end
        end
        return
    end

    # 2D multi-level, intra- + inter-tree, conforming + hanging
    forest = ForestBWG(generate_grid(Quadrilateral, (2, 2)), 5)
    Ferrite.refine!(forest, [1])
    Ferrite.balanceforest!(forest)
    Ferrite.refine!(forest, [1, 2, 7])
    Ferrite.balanceforest!(forest)
    check_interfacevalues(forest, RefQuadrilateral)

    # 2D rotated macro element
    grid = generate_grid(Quadrilateral, (2, 2))
    grid.cells[2] = Quadrilateral((grid.cells[2].nodes[2], grid.cells[2].nodes[3], grid.cells[2].nodes[4], grid.cells[2].nodes[1]))
    forest = ForestBWG(grid, 3)
    Ferrite.AMR.refine_octant!(forest.cells[2], forest.cells[2].leaves[1])
    check_interfacevalues(forest, RefQuadrilateral)

    # 3D multi-level, intra- + inter-tree
    forest = ForestBWG(generate_grid(Hexahedron, (2, 2, 2)), 4)
    Ferrite.refine!(forest, [1])
    Ferrite.balanceforest!(forest)
    Ferrite.refine!(forest, [1])
    Ferrite.balanceforest!(forest)
    check_interfacevalues(forest, RefHexahedron)

    # 3D rotated macro element (cf. "hanging nodes" testset)
    grid = generate_grid(Hexahedron, (2, 2, 2))
    grid.cells[1] = Hexahedron((grid.cells[1].nodes[2], grid.cells[1].nodes[3], grid.cells[1].nodes[4], grid.cells[1].nodes[1], grid.cells[1].nodes[6], grid.cells[1].nodes[7], grid.cells[1].nodes[8], grid.cells[1].nodes[5]))
    grid.cells[1] = Hexahedron((grid.cells[1].nodes[2], grid.cells[1].nodes[3], grid.cells[1].nodes[4], grid.cells[1].nodes[1], grid.cells[1].nodes[6], grid.cells[1].nodes[7], grid.cells[1].nodes[8], grid.cells[1].nodes[5]))
    forest = ForestBWG(grid, 3)
    Ferrite.AMR.refine_octant!(forest.cells[1], forest.cells[1].leaves[1])
    check_interfacevalues(forest, RefHexahedron)
end
