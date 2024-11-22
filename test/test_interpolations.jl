using Ferrite: reference_shape_value, reference_shape_gradient

@testset "interpolations" begin #=
    @testset "Value Type $value_type" for value_type in (Float32, Float64)
        @testset "Correctness of $interpolation" for interpolation in (
                Lagrange{RefLine, 1}(),
                Lagrange{RefLine, 2}(),
                Lagrange{RefQuadrilateral, 1}(),
                Lagrange{RefQuadrilateral, 2}(),
                Lagrange{RefQuadrilateral, 3}(),
                Lagrange{RefTriangle, 1}(),
                Lagrange{RefTriangle, 2}(),
                Lagrange{RefTriangle, 3}(),
                Lagrange{RefTriangle, 4}(),
                Lagrange{RefTriangle, 5}(),
                Lagrange{RefHexahedron, 1}(),
                Lagrange{RefHexahedron, 2}(),
                Serendipity{RefQuadrilateral, 2}(),
                Serendipity{RefHexahedron, 2}(),
                Lagrange{RefTetrahedron, 1}(),
                Lagrange{RefTetrahedron, 2}(),
                Lagrange{RefPrism, 1}(),
                Lagrange{RefPrism, 2}(),
                Lagrange{RefPyramid, 1}(),
                Lagrange{RefPyramid, 2}(),
                #
                DiscontinuousLagrange{RefLine, 0}(),
                DiscontinuousLagrange{RefQuadrilateral, 0}(),
                DiscontinuousLagrange{RefHexahedron, 0}(),
                DiscontinuousLagrange{RefTriangle, 0}(),
                DiscontinuousLagrange{RefTetrahedron, 0}(),
                DiscontinuousLagrange{RefLine, 1}(),
                DiscontinuousLagrange{RefQuadrilateral, 1}(),
                DiscontinuousLagrange{RefHexahedron, 1}(),
                DiscontinuousLagrange{RefTriangle, 1}(),
                DiscontinuousLagrange{RefTetrahedron, 1}(),
                DiscontinuousLagrange{RefPrism, 1}(),
                DiscontinuousLagrange{RefPyramid, 1}(),
                #
                BubbleEnrichedLagrange{RefTriangle, 1}(),
                #
                CrouzeixRaviart{RefTriangle, 1}(),
                CrouzeixRaviart{RefTetrahedron, 1}(),
                RannacherTurek{RefQuadrilateral, 1}(),
                RannacherTurek{RefHexahedron, 1}(),
            )
            # Test of utility functions
            ref_dim = Ferrite.getrefdim(interpolation)
            ref_shape = Ferrite.getrefshape(interpolation)
            func_order = Ferrite.getorder(interpolation)
            @test typeof(interpolation) <: Interpolation{ref_shape, func_order}

            # Note that not every element formulation exists for every order and dimension.
            applicable(Ferrite.getlowerorder, interpolation) && @test typeof(Ferrite.getlowerorder(interpolation)) <: Interpolation{ref_shape, func_order - 1}
            @testset "transform face points" begin
                # Test both center point and random points on the face
                ref_coord = Ferrite.reference_coordinates(Lagrange{ref_shape, 1}())
                for face in 1:nfacets(interpolation)
                    face_nodes = Ferrite.reference_facets(ref_shape)[face]
                    center_coord = [0.0 for _ in 1:ref_dim]
                    rand_coord = [0.0 for _ in 1:ref_dim]
                    rand_weights = rand(length(face_nodes))
                    rand_weights /= sum(rand_weights)
                    for (i, node) in pairs(face_nodes)
                        center_coord += ref_coord[node] / length(face_nodes)
                        rand_coord += rand_weights[i] .* ref_coord[node]
                    end
                    for point in (center_coord, rand_coord)
                        vec_point = Vec{ref_dim}(point)
                        cell_to_face = Ferrite.element_to_facet_transformation(vec_point, ref_shape, face)
                        face_to_cell = Ferrite.facet_to_element_transformation(cell_to_face, ref_shape, face)
                        @test vec_point ≈ face_to_cell
                    end
                end
            end
            n_basefuncs = getnbasefunctions(interpolation)
            coords = Ferrite.reference_coordinates(interpolation)
            @test length(coords) == n_basefuncs
            f(x) = [reference_shape_value(interpolation, Tensor{1, ref_dim}(x), i) for i in 1:n_basefuncs]

            #TODO prefer this test style after 1.6 is removed from CI
            # @testset let x = sample_random_point(ref_shape) # not compatible with Julia 1.6
            x = Vec{ref_dim, value_type}(sample_random_point(ref_shape))
            random_point_testset = @testset "Random point test" begin
                # Check gradient evaluation
                @test vec(ForwardDiff.jacobian(f, Array(x))') ≈
                    reinterpret(value_type, [reference_shape_gradient(interpolation, x, i) for i in 1:n_basefuncs])
                # Check partition of unity at random point.
                @test sum([reference_shape_value(interpolation, x, i) for i in 1:n_basefuncs]) ≈ 1.0
                # Check if the important functions are consistent
                @test_throws ArgumentError reference_shape_value(interpolation, x, n_basefuncs + 1)
                # Idempotency test
                @test reference_shape_value(interpolation, x, n_basefuncs) == reference_shape_value(interpolation, x, n_basefuncs)
            end
            # Remove after 1.6 is removed from CI (see above)
            # Show coordinate in case failure (see issue #811)
            !isempty(random_point_testset.results) && println("^^^^^Random point test failed at $x for $interpolation !^^^^^")

            # Test whether we have for each entity corresponding dof indices (possibly empty)
            @test length(Ferrite.vertexdof_indices(interpolation)) == Ferrite.nvertices(interpolation)
            @test length(Ferrite.facedof_indices(interpolation)) == Ferrite.nfaces(interpolation)
            @test length(Ferrite.facedof_interior_indices(interpolation)) == Ferrite.nfaces(interpolation)
            @test length(Ferrite.edgedof_indices(interpolation)) == Ferrite.nedges(interpolation)
            @test length(Ferrite.edgedof_interior_indices(interpolation)) == Ferrite.nedges(interpolation)
            # We have at least as many edge/face dofs as we have edge/face interior dofs
            @test all(length.(Ferrite.facedof_interior_indices(interpolation)) .<= length.(Ferrite.facedof_indices(interpolation)))
            @test all(length.(Ferrite.edgedof_interior_indices(interpolation)) .<= length.(Ferrite.edgedof_indices(interpolation)))
            # The total number of dofs must match the number of base functions
            totaldofs = sum(length.(Ferrite.vertexdof_indices(interpolation)); init = 0)
            totaldofs += sum(length.(Ferrite.facedof_interior_indices(interpolation)); init = 0)
            totaldofs += sum(length.(Ferrite.edgedof_interior_indices(interpolation)); init = 0)
            totaldofs += length(Ferrite.volumedof_interior_indices(interpolation))
            @test totaldofs == n_basefuncs

            # The dof indices are valid.
            @test all([all(0 .< i .<= n_basefuncs) for i in Ferrite.vertexdof_indices(interpolation)])
            @test all([all(0 .< i .<= n_basefuncs) for i in Ferrite.facedof_indices(interpolation)])
            @test all([all(0 .< i .<= n_basefuncs) for i in Ferrite.facedof_interior_indices(interpolation)])
            @test all([all(0 .< i .<= n_basefuncs) for i in Ferrite.edgedof_indices(interpolation)])
            @test all([all(0 .< i .<= n_basefuncs) for i in Ferrite.edgedof_interior_indices(interpolation)])
            @test all([all(0 .< i .<= n_basefuncs) for i in Ferrite.volumedof_interior_indices(interpolation)])

            # Check for evaluation type correctness of interpolation
            @testset "return type correctness dof $dof" for dof in 1:n_basefuncs
                @test (@inferred reference_shape_value(interpolation, x, dof)) isa value_type
                @test (@inferred reference_shape_gradient(interpolation, x, dof)) isa Vec{ref_dim, value_type}
            end

            # Check for dirac delta property of interpolation
            @testset "dirac delta property of dof $dof" for dof in 1:n_basefuncs
                for k in 1:n_basefuncs
                    N_dof = reference_shape_value(interpolation, coords[dof], k)
                    if k == dof
                        @test N_dof ≈ 1.0
                    else
                        factor = interpolation isa Lagrange{RefQuadrilateral, 3} ? 200 : 4
                        @test N_dof ≈ 0.0 atol = factor * eps(value_type)
                    end
                end
            end

            # Test that facedof_indices(...) return in counter clockwise order (viewing from the outside)
            if interpolation isa Lagrange
                function __outward_normal(coords::Vector{<:Vec{1}}, nodes)
                    n = coords[nodes[1]]
                    return n / norm(n)
                end
                function __outward_normal(coords::Vector{<:Vec{2}}, nodes)
                    p1 = coords[nodes[1]]
                    p2 = coords[nodes[2]]
                    n = Vec{2}((p2[2] - p1[2], - p2[1] + p1[1]))
                    return n / norm(n)
                end
                function __outward_normal(coords::Vector{<:Vec{3}}, nodes)
                    p1 = coords[nodes[1]]
                    p2 = coords[nodes[2]]
                    p3 = coords[nodes[3]]
                    n = (p3 - p2) × (p1 - p2)
                    return n / norm(n)
                end
                _bfunc = if ref_dim == 3
                    Ferrite.facedof_indices(interpolation)
                elseif ref_dim == 2
                    Ferrite.edgedof_indices(interpolation)
                elseif ref_dim == 1
                    Ferrite.vertexdof_indices(interpolation)
                end
                for (facenodes, normal) in zip(_bfunc, reference_normals(interpolation))
                    @test __outward_normal(coords, facenodes) ≈ normal
                end
            end

            # regression for https://github.com/Ferrite-FEM/Ferrite.jl/issues/520
            #=interpolation_type = typeof(interpolation).name.wrapper
    if func_order > 1 && interpolation_type != Ferrite.Serendipity
        first_order = interpolation_type{ref_shape,1}()
        for (highorderface, firstorderface) in zip(Ferrite.facedof_indices(interpolation), Ferrite.facedof_indices(first_order))
            for (h_node, f_node) in zip(highorderface, firstorderface)
                @test h_node == f_node
            end
        end
        if ref_dim > 2
            for (highorderedge, firstorderedge) in zip(Ferrite.edgedof_indices(interpolation), Ferrite.edgedof_indices(first_order))
                for (h_node, f_node) in zip(highorderedge, firstorderedge)
                    @test h_node == f_node
                end
            end
        end
    end=#

            # VectorizedInterpolation
            v_interpolation_1 = interpolation^2
            v_interpolation_2 = (d = 2; interpolation^d)
            @test getnbasefunctions(v_interpolation_1) ==
                getnbasefunctions(v_interpolation_2) ==
                getnbasefunctions(interpolation) * 2
            # pretty printing
            @test repr("text/plain", v_interpolation_1) == repr(v_interpolation_1.ip) * "^2"

            # Check for evaluation type correctness of vectorized interpolation
            v_interpolation_3 = interpolation^ref_dim
            @testset "vectorized case of return type correctness of dof $dof" for dof in 1:n_basefuncs
                @test @inferred(reference_shape_value(v_interpolation_1, x, dof)) isa Vec{2, value_type}
                @test @inferred(reference_shape_gradient(v_interpolation_3, x, dof)) isa Tensor{2, ref_dim, value_type}
            end
        end # correctness testset

        @test Ferrite.reference_coordinates(DiscontinuousLagrange{RefTriangle, 0}()) ≈ [Vec{2, Float64}((1 / 3, 1 / 3))]
        @test Ferrite.reference_coordinates(DiscontinuousLagrange{RefQuadrilateral, 0}()) ≈ [Vec{2, Float64}((0, 0))]
        @test Ferrite.reference_coordinates(DiscontinuousLagrange{RefTetrahedron, 0}()) ≈ [Vec{3, Float64}((1 / 4, 1 / 4, 1 / 4))]
        @test Ferrite.reference_coordinates(DiscontinuousLagrange{RefHexahedron, 0}()) ≈ [Vec{3, Float64}((0, 0, 0))]

        # Test discontinuous interpolations related functions
        d_ip = DiscontinuousLagrange{RefQuadrilateral, 1}()
        d_ip_t = DiscontinuousLagrange{RefQuadrilateral, 1}

        ip = Lagrange{RefQuadrilateral, 1}()
        ip_t = Lagrange{RefQuadrilateral, 1}

        @test Ferrite.is_discontinuous(ip) == false
        @test Ferrite.is_discontinuous(ip_t) == false
        @test Ferrite.is_discontinuous(d_ip) == true
        @test Ferrite.is_discontinuous(d_ip_t) == true
    end =#
    @testset "Correctness of AD of embedded interpolations" begin
        ip = Lagrange{RefHexahedron, 2}()^3
        ξ = rand(Vec{3, Float64})
        for I in 1:getnbasefunctions(ip)
            #Call StaticArray-version
            H_sa, G_sa, V_sa = Ferrite._reference_shape_hessian_gradient_and_value_static_array(ip, ξ, I)
            #Call tensor AD version
            H, G, V = Ferrite.reference_shape_hessian_gradient_and_value(ip, ξ, I)

            @test V ≈ V_sa
            @test G ≈ G_sa
            @test H ≈ H_sa
        end
        ips = Lagrange{RefQuadrilateral, 2}()
        vdim = 3
        ipv = ips^vdim
        ξ = rand(Vec{2, Float64})
        for ipv_ind in 1:getnbasefunctions(ipv)
            ips_ind, v_ind = fldmod1(ipv_ind, vdim)
            H, G, V = Ferrite.reference_shape_hessian_gradient_and_value(ipv, ξ, ipv_ind)
            h, g, v = Ferrite.reference_shape_hessian_gradient_and_value(ips, ξ, ips_ind)
            @test h ≈ H[v_ind, :, :]
            @test g ≈ G[v_ind, :]
            @test v ≈ V[v_ind]
        end
    end

    reference_cell(::Type{RefTriangle}) = Triangle((1, 2, 3))
    reference_cell(::Type{RefQuadrilateral}) = Quadrilateral((1, 2, 3, 4))
    reference_cell(::Type{RefTetrahedron}) = Tetrahedron((1, 2, 3, 4))
    reference_cell(::Type{RefHexahedron}) = Hexahedron((ntuple(identity, 8)))

    function line_integral(qr::QuadratureRule{RefLine}, ip, shape_nr, x0, Δx, L, v, f)
        val = 0.0
        for (ξ1d, w) in zip(Ferrite.getpoints(qr), Ferrite.getweights(qr))
            ξ = x0 + (ξ1d[1] / 2) * Δx  # ::Vec
            s = (ξ1d[1] + 1) / 2        # ∈ [0, 1]
            Nξ = reference_shape_value(ip, ξ, shape_nr)
            dΩ = (w * L / 2)
            val += (Nξ ⋅ v) * f(s) * dΩ
        end
        return val
    end

    # Required properties of shape value Nⱼ of an edge-elements (Hcurl) on an edge with direction v, length L, and dofs ∈ 𝔇
    # 1) Unit property: ∫(Nⱼ ⋅ v f(s) dS) = 1/length(𝔇) ∀ ∈ 𝔇
    #    Must hold for
    #    length(𝔇) ≥ 1: f(s) = 1
    #    length(𝔇) = 2: f(s) = 1 - s or f(s) = s for 1st and 2nd dof, respectively.
    #    Additionally, should be zero for
    #    length(𝔇) = 2: f(s) = s or f(s) = 1 - s for 1st and 2nd dof, respectively.
    #    s is the path parameter ∈[0,1] along the positive direction of the path.
    # 2) Zero along other edges: Nⱼ ⋅ v = 0 if j∉𝔇
    @testset "H(curl) on RefCell" begin
        lineqr = QuadratureRule{RefLine}(20)
        for ip in (Nedelec{2, RefTriangle, 1}(), Nedelec{2, RefTriangle, 2}(), Nedelec{3, RefTetrahedron, 1}(), Nedelec{3, RefHexahedron, 1}())
            cell = reference_cell(getrefshape(ip))
            edges = Ferrite.edges(cell)
            dofs = Ferrite.edgedof_interior_indices(ip)
            x = Ferrite.reference_coordinates(geometric_interpolation(typeof(cell)))
            @testset "$(getrefshape(ip)), order=$(Ferrite.getorder(ip))" begin
                for (edge_nr, (i1, i2)) in enumerate(edges)
                    Δx = x[i2] - x[i1]
                    x0 = (x[i1] + x[i2]) / 2
                    L = norm(Δx)
                    v = Δx / L
                    for (idof, shape_nr) in enumerate(dofs[edge_nr])
                        nedgedofs = length(dofs[edge_nr])
                        f(x) = nedgedofs == 1 ? 1.0 : (idof == 1 ? 1 - x : x)
                        s = line_integral(lineqr, ip, shape_nr, x0, Δx, L, v, f)
                        @test s ≈ one(s) / nedgedofs
                        if nedgedofs == 2
                            g(x) = idof == 1 ? x : 1 - x
                            @test 1 ≈ 1 + line_integral(lineqr, ip, shape_nr, x0, Δx, L, v, g)
                        end
                    end
                    for (j_edge, shape_nrs) in enumerate(dofs)
                        j_edge == edge_nr && continue
                        for shape_nr in shape_nrs
                            for ξ in (x[i1] + r * Δx for r in [0.0, rand(3)..., 1.0])
                                @test abs(reference_shape_value(ip, ξ, shape_nr) ⋅ v) < eps() * 100
                            end
                        end
                    end
                end
            end
        end
    end

    # Required properties of shape value Nⱼ of an edge-elements (Hdiv) on an edge with normal n, length L, and dofs ∈ 𝔇
    # 1) Unit property: ∫(Nⱼ ⋅ n f(s) dS) = 1/length(𝔇) ∀ j ∈ 𝔇
    #    Must hold for
    #    length(𝔇) ≥ 1: f(s) = 1
    #    length(𝔇) = 2: f(s) = 1 - s or f(s) = s for 1st and 2nd dof, respectively.
    #    Additionally, should be zero for
    #    length(𝔇) = 2: f(s) = s or f(s) = 1 - s for 1st and 2nd dof, respectively.
    #    s is the path parameter ∈[0,1] along the positive direction of the path.
    # 2) Zero normal component on other edges: Nⱼ ⋅ n = 0 if j∉𝔇
    @testset "H(div) on RefCell" begin
        lineqr = QuadratureRule{RefLine}(20)
        for ip in (
                RaviartThomas{2, RefTriangle, 1}(),
                RaviartThomas{2, RefTriangle, 2}(),
                Ferrite.BrezziDouglasMarini{2, RefTriangle, 1}(),
            )
            cell = reference_cell(getrefshape(ip))
            cell_facets = Ferrite.facets(cell)
            dofs = Ferrite.facetdof_interior_indices(ip)
            x = Ferrite.reference_coordinates(geometric_interpolation(typeof(cell)))
            normals = reference_normals(geometric_interpolation(typeof(cell)))
            @testset "$ip" begin
                for (facet_nr, (i1, i2)) in enumerate(cell_facets)
                    @testset "Facet $facet_nr" begin
                        Δx = x[i2] - x[i1]
                        x0 = (x[i1] + x[i2]) / 2
                        L = norm(Δx)
                        n = normals[facet_nr]
                        for (idof, shape_nr) in enumerate(dofs[facet_nr])
                            nfacetdofs = length(dofs[facet_nr])
                            f(x) = nfacetdofs == 1 ? 1.0 : (idof == 1 ? 1 - x : x)
                            s = line_integral(lineqr, ip, shape_nr, x0, Δx, L, n, f)
                            @test s ≈ one(s) / nfacetdofs
                            if nfacetdofs == 2
                                g(x) = idof == 1 ? x : 1 - x
                                @test 1 ≈ 1 + line_integral(lineqr, ip, shape_nr, x0, Δx, L, n, g)
                            end
                        end
                        for (j_facet, shape_nrs) in enumerate(dofs)
                            j_facet == facet_nr && continue
                            for shape_nr in shape_nrs
                                for ξ in (x[i1] + r * Δx for r in [0.0, rand(3)..., 1.0])
                                    @test abs(reference_shape_value(ip, ξ, shape_nr) ⋅ n) < eps() * 100
                                end
                            end
                        end
                    end
                end
            end
        end
    end

    tupleshift(t::NTuple{N}, shift::Int) where {N} = ntuple(i -> t[mod(i - 1 - shift, N) + 1], N)
    #tupleshift(t::NTuple, shift::Int) = tuple(circshift(SVector(t), shift)...)
    cell_permutations(cell::Quadrilateral) = (Quadrilateral(tupleshift(cell.nodes, shift)) for shift in 0:3)
    cell_permutations(cell::Triangle) = (Triangle(tupleshift(cell.nodes, shift)) for shift in 0:2)
    cell_permutations(cell::QuadraticTriangle) = (QuadraticTriangle((tupleshift(cell.nodes[1:3], shift)..., tupleshift(cell.nodes[4:6], shift)...)) for shift in 0:3)

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

    @testset "Hcurl and Hdiv" begin
        include(joinpath(@__DIR__, "InterpolationTestUtils.jl"))
        import .InterpolationTestUtils as ITU
        nel = 3
        hdiv_check(v, n) = v ⋅ n        # Hdiv (normal continuity)
        hcurl_check(v, n) = v - n * (v ⋅ n) # Hcurl (tangent continuity)
        transformation_functions = ((Nedelec, hcurl_check), (RaviartThomas, hdiv_check), (Ferrite.BrezziDouglasMarini, hdiv_check))

        for CT in (Triangle, QuadraticTriangle, Tetrahedron, Hexahedron)
            dim = Ferrite.getrefdim(CT) # dim = sdim = rdim
            p1, p2 = (rand(Vec{dim}), ones(Vec{dim}) + rand(Vec{dim}))
            grid = generate_grid(CT, ntuple(_ -> nel, dim), p1, p2)
            # Smoothly distort grid (to avoid spuriously badly deformed elements).
            # A distorted grid is important to properly test the geometry mapping
            # for 2nd order elements.
            transfun(x) = typeof(x)(i -> sinpi(x[mod(i, length(x)) + 1] + i / 3)) / 10
            transform_coordinates!(grid, x -> (x + transfun(x)))
            cellnr = getncells(grid) ÷ 2 + 1 # Should be a cell in the center
            basecell = getcells(grid, cellnr)
            RefShape = Ferrite.getrefshape(basecell)
            for order in (1, 2)
                for (IPT, transformation_function) in transformation_functions
                    dim == 3 && order > 1 && continue
                    IPT == RaviartThomas && (dim == 3 || order > 1) && continue
                    IPT == RaviartThomas && (RefShape == RefHexahedron) && continue
                    IPT == Ferrite.BrezziDouglasMarini && !(RefShape == RefTriangle && order == 1) && continue
                    ip = IPT{dim, RefShape, order}()
                    @testset "$CT, $ip" begin
                        for testcell in cell_permutations(basecell)
                            grid.cells[cellnr] = testcell
                            dh = DofHandler(grid)
                            add!(dh, :u, ip)
                            close!(dh)
                            for facetnr in 1:nfacets(RefShape)
                                fi = FacetIndex(cellnr, facetnr)
                                # Check continuity of tangential function value
                                ITU.test_continuity(dh, fi; transformation_function)
                            end
                            # Check gradient calculation
                            ITU.test_gradient(dh, cellnr)
                        end
                    end
                end
            end
        end
    end

    @testset "Hcurl and Hdiv BC" begin
        hdiv_ips = (
            RaviartThomas{2, RefTriangle, 1}(),
            RaviartThomas{2, RefTriangle, 2}(),
            Ferrite.BrezziDouglasMarini{2, RefTriangle, 1}(),
        )
        hdiv_check(v, n) = v ⋅ n

        hcurl_ips = (
            Nedelec{2, RefTriangle, 1}(),
            Nedelec{2, RefTriangle, 2}(),
        )
        function hcurl_check(v, n::Vec{2}) # 3d not supported yet
            t = rotate(n, π / 2)
            return v ⋅ t
        end
        for (f, interpolations) in ((hdiv_check, hdiv_ips), (hcurl_check, hcurl_ips))
            for ip in interpolations
                @testset "$ip" begin
                    RefShape = Ferrite.getrefshape(ip)
                    CT = typeof(reference_cell(RefShape))
                    dim = Ferrite.getrefdim(CT) # dim=sdim=vdim
                    grid = generate_grid(CT, ntuple(Returns(2), dim))
                    qr = FacetQuadratureRule{RefShape}(4)
                    fv = FacetValues(qr, ip, geometric_interpolation(CT))
                    dh = close!(add!(DofHandler(grid), :u, ip))
                    for bval in (0.0, 1.0)
                        for side in ("left", "right", "top", "bottom")
                            a = zeros(ndofs(dh))
                            ch = ConstraintHandler(dh)
                            add!(ch, Dirichlet(:u, getfacetset(grid, side), Returns(bval)))
                            close!(ch)
                            apply!(a, ch)
                            test_val = 0.0
                            for (cellidx, facetidx) in getfacetset(grid, side)
                                reinit!(fv, getcells(grid, cellidx), getcoordinates(grid, cellidx), facetidx)
                                ae = a[celldofs(dh, cellidx)]
                                val = 0.0
                                for q_point in 1:getnquadpoints(fv)
                                    dΓ = getdetJdV(fv, q_point)
                                    val += f(function_value(fv, q_point, ae), getnormal(fv, q_point)) * dΓ
                                end
                                test_val += f === hdiv_check ? val : abs(val)
                            end
                            @test abs(test_val - 2 * bval) < 1.0e-6
                        end
                    end
                end
            end
        end
    end

    @testset "Errors for entitydof_indices on VectorizedInterpolations" begin
        ip = Lagrange{RefQuadrilateral, 2}()^2
        @test_throws ArgumentError Ferrite.vertexdof_indices(ip)
        @test_throws ArgumentError Ferrite.edgedof_indices(ip)
        @test_throws ArgumentError Ferrite.facedof_indices(ip)
        @test_throws ArgumentError Ferrite.facetdof_indices(ip)

        @test_throws ArgumentError Ferrite.edgedof_interior_indices(ip)
        @test_throws ArgumentError Ferrite.facedof_interior_indices(ip)
        @test_throws ArgumentError Ferrite.volumedof_interior_indices(ip)
        @test_throws ArgumentError Ferrite.facetdof_interior_indices(ip)
    end


end # testset
