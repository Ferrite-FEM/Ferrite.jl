@testset "InterfaceValues" begin
    function test_interfacevalues(grid::Ferrite.AbstractGrid, iv::InterfaceValues; tol = 0)
        ip_here = Ferrite.function_interpolation(iv.here)
        ip_there = Ferrite.function_interpolation(iv.there)
        ndim = Ferrite.getdim(ip_here)
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
                @test isapprox(spatial_coordinate(iv, qp, coords_here, coords_there; here = true),
                    spatial_coordinate(iv, qp, coords_here, coords_there; here = false); atol = tol)
                for i in 1:getnbasefunctions(iv)
                    here = i <= getnbasefunctions(iv.here)
                    shapevalue = shape_value(iv, qp, i; here = here)
                    shape_avg = shape_value_average(iv, qp, i)
                    shape_jump = shape_value_jump(iv, qp, i)

                    shapegrad = shape_gradient(iv, qp, i; here = here)
                    shapegrad_avg = shape_gradient_average(iv, qp, i)
                    shapegrad_jump = shape_gradient_jump(iv, qp, i)

                    normal = getnormal(iv, qp; here=false)
                    # Test values (May be removed as it mirrors implementation)
                    if i > getnbasefunctions(iv.here)
                        @test shapevalue ≈ shape_value(iv.there, qp, i - getnbasefunctions(iv.here))
                        @test shapegrad ≈ shape_gradient(iv.there, qp, i - getnbasefunctions(iv.here))

                        @test shape_jump ≈ -shapevalue
                        @test shapegrad_jump ≈ -shapegrad
                    else
                        normal = getnormal(iv, qp)
                        @test shapevalue ≈ shape_value(iv.here, qp, i)
                        @test shapegrad ≈ shape_gradient(iv.here, qp, i)

                        @test shape_jump ≈ shapevalue
                        @test shapegrad_jump ≈ shapegrad
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
                u_a = Vec{ndim, Float64}[zero(Tensor{1,ndim}) for i in 1: nbf_a]
                u_b = Vec{ndim, Float64}[zero(Tensor{1,ndim}) for i in 1: nbf_b]
                u_scal_a = zeros(nbf_a)
                u_scal_b = zeros(nbf_b)
                H = rand(Tensor{2, ndim})
                V = rand(Tensor{1, ndim})
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
                face = here ? Ferrite.getcurrentface(iv.here) : Ferrite.getcurrentface(iv.there)
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
            geom_ip_faces_indices = Ferrite.facetdof_indices(ip)
            Ferrite.getdim(ip) > 1 && (geom_ip_faces_indices = Tuple([face[collect(face .∉ Ref(interior))] for (face, interior) in [(geom_ip_faces_indices[i], Ferrite.facetdof_interior_indices(ip)[i]) for i in 1:Ferrite.nfacets(ip)]]))
            faces_indices = Ferrite.reference_facets(Ferrite.getrefshape(Ferrite.default_interpolation(typeof(cell))))
            node_ids = Ferrite.get_node_ids(cell)
            cellfacets = Ferrite.facets(cell)
            @test getindex.(Ref(node_ids), collect.(faces_indices)) == cellfacets == getindex.(Ref(node_ids), collect.(geom_ip_faces_indices))
        end
        @testset "error paths" begin
            cell = getcells(grid, 1)
            dim == 1 && @test_throws ErrorException("1D elements don't use transformations for interfaces.") Ferrite.InterfaceOrientationInfo(cell,cell,1,1)
            @test_throws ArgumentError("unknown facet number") Ferrite.element_to_facet_transformation(Vec{dim,Float64}(ntuple(_->0.0, dim)), Ferrite.getrefshape(cell), 100)
            @test_throws ArgumentError("unknown facet number") Ferrite.facet_to_element_transformation(Vec{dim-1,Float64}(ntuple(_->0.0, dim-1)), Ferrite.getrefshape(cell), 100)
        end
        func_interpol = scalar_interpol
        for func_interpol in (scalar_interpol, VectorizedInterpolation(scalar_interpol))
            iv = cell_shape ∈ (QuadraticLine, QuadraticQuadrilateral, QuadraticTriangle, QuadraticTetrahedron) ? 
                InterfaceValues(quad_rule, func_interpol, ip) : InterfaceValues(quad_rule, func_interpol)
            test_interfacevalues(grid, iv)
        end
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
            geom_ip_faces_indices = Ferrite.facetdof_indices(ip)
            Ferrite.getdim(ip) > 1 && (geom_ip_faces_indices = Tuple([face[collect(face .∉ Ref(interior))] for (face, interior) in [(geom_ip_faces_indices[i], Ferrite.facedof_interior_indices(ip)[i]) for i in 1:nfaces(ip)]]))
            faces_indices = Ferrite.reference_facets(Ferrite.getrefshape(Ferrite.default_interpolation(typeof(cell))))
            node_ids = Ferrite.get_node_ids(cell)
            @test getindex.(Ref(node_ids), collect.(faces_indices)) == Ferrite.faces(cell) == getindex.(Ref(node_ids), collect.(geom_ip_faces_indices))
        end
        @testset "error paths" begin
            cell = getcells(grid, 1)
            @test_throws ArgumentError("unknown facet number") Ferrite.element_to_facet_transformation(Vec{dim,Float64}(ntuple(_->0.0, dim)), Ferrite.getrefshape(cell), 100)
            @test_throws ArgumentError("unknown facet number") Ferrite.facet_to_element_transformation(Vec{dim-1,Float64}(ntuple(_->0.0, dim-1)), Ferrite.getrefshape(cell), 100)
        end
        for func_interpol in (scalar_interpol, VectorizedInterpolation(scalar_interpol))
            iv = InterfaceValues(quad_rule, func_interpol)
            test_interfacevalues(grid, iv; tol = 5*eps(Float64))
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
        dim = 2
        nodes = [Node((-1.0, 0.0, 0.0)), Node((0.0, 0.0, 0.0)), Node((1.0, 0.0, 0.0)), 
                Node((-1.0, 1.0, 0.0)), Node((0.0, 1.0, 0.0)), Node((1.0, 1.0, 0.0)), 
                Node((-1.0, 0.0, 1.0)), Node((0.0, 0.0, 1.0)), Node((1.0, 0.0, 1.0)), 
                Node((-1.0, 1.0, 1.0)), Node((0.0, 1.0, 1.0)), Node((1.0, 1.0, 1.0)), 
                ]
        cells = [
                    Hexahedron((1,2,5,4,7,8,11,10)),
                    Hexahedron((5,6,12,11,2,3,9,8)),
                ]

        grid = Grid(cells, nodes)
        test_interfacevalues(grid,
            InterfaceValues(FacetQuadratureRule{RefHexahedron}(2), DiscontinuousLagrange{RefHexahedron, 1}()))
    end
    @testset "Interface dof_range" begin
        grid = generate_grid(Quadrilateral,(3,3))
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
        v isa Ferrite.ScalarWrapper && continue
        vc = getfield(ivc, fname)
        if hasmethod(pointer, Tuple{typeof(v)})
            @test pointer(v) != pointer(vc)
        end
        v isa FacetValues && continue
        for fname in fieldnames(typeof(vc))
            v2 = getfield(v, fname)
            v2 isa Ferrite.ScalarWrapper && continue
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
        iv = InterfaceValues(FacetQuadratureRule{RefQuadrilateral}(2), Lagrange{RefQuadrilateral,2}())
        showstring = sprint(show, MIME"text/plain"(), iv)
        @test contains(showstring, "InterfaceValues with")
    end
end # of testset
