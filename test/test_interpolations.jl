using Ferrite: reference_shape_value, reference_shape_gradient

"""
    test_interpolation_properties(ip::Interpolation)

This function tests the following
interpolation properties. All base interpolations should pass this test, but
`VectorizedInterpolation`s do not which is ok as these are
special-cased in the code base.

A) Length of `<entity>dof_indices` and `<entity>dof_interior_indices`
   matches number of reference shape entities (e.g. `edge`)
B) Numbering convention
   - vertices -> edges -> faces -> volume
   - Numbered in entity order (e.g. edge 1 has lower indices than edge 2)
   - Continuous and increasing numbering within entity (e.g. `edgedof_interior_indices(ip)[edgenr]`)
     can be `(4, 5)`, but not `(4, 6)` or `(5, 4)`.
C) Lower-dimensional entities' dof indices + current interior dof indices
   matches current dof indices (e.g. vertexdof + edge_interior => edgedofs) for each entity.
D) The dof indices values matches `1:N` without duplication (follows from B, but also checked separately)
E) All `N` base functions are implemented + `ArgumentError` if `i=0` or `i=N+1`
F) Interpolation accessor functions versus type parameters (e.g. same refshape)
G) `conformity` and `mapping_type` is defined
"""
function test_interpolation_properties(ip::Interpolation{RefShape, FunOrder}) where {RefShape, FunOrder}
    return @testset "Interpolation properties: $ip" begin
        # test accessor functions and type parameters
        @test RefShape == getrefshape(ip)
        @test Ferrite.getrefdim(RefShape) == Ferrite.getrefdim(ip)
        @test Ferrite.getorder(ip) == FunOrder

        rdim = Ferrite.getrefdim(ip)

        as_vector(t::Tuple) = collect(as_vector.(t))
        as_vector(i::Int) = i
        dof_data = (
            vert = as_vector(Ferrite.vertexdof_indices(ip)),
            edge = as_vector(Ferrite.edgedof_indices(ip)),
            face = as_vector(Ferrite.facedof_indices(ip)),
            edge_i = as_vector(Ferrite.edgedof_interior_indices(ip)),
            face_i = as_vector(Ferrite.facedof_interior_indices(ip)),
            vol_i = as_vector(Ferrite.volumedof_interior_indices(ip)),
            n = getnbasefunctions(ip),
        )

        refshape_data = (
            nvertices = as_vector(Ferrite.nvertices(RefShape)),
            edges = as_vector(Ferrite.reference_edges(RefShape)),
            faces = as_vector(Ferrite.reference_faces(RefShape)),
            rdim = rdim,
        )

        # Test A-D
        _test_interpolation_properties(dof_data, refshape_data)

        # Test E: All base functions implemented and infers correct types.
        # Argument errors for 0th and n+1 indices.
        for T in (Float64, Float32)
            ξ = zero(Vec{Ferrite.getrefdim(ip), T})
            @test_throws ArgumentError Ferrite.reference_shape_value(ip, ξ, 0)
            for i in 1:getnbasefunctions(ip)
                @test (@inferred Ferrite.reference_shape_value(ip, ξ, i)) isa Ferrite.shape_value_type(ip, T)
            end
            @test_throws ArgumentError Ferrite.reference_shape_value(ip, ξ, getnbasefunctions(ip) + 1)
        end

        # Test that property functions are defined, runs, and, if possible, give expected type
        Ferrite.mapping_type(ip) # Dry-run just to catch if it isn't defined
        @test Ferrite.conformity(ip) isa Union{Ferrite.L2Conformity, Ferrite.HdivConformity, Ferrite.HcurlConformity, Ferrite.H1Conformity}
    end
end

# Brake out to avoid compiling new function for each interpolation
function _test_interpolation_properties(dofs::NamedTuple, rs::NamedTuple)
    collect_all_dofs(t::Union{Tuple, Vector}) = vcat(Int[], collect.(t)...)
    # Check match to reference shape (A)
    @test length(dofs.vert) == rs.nvertices
    @test length(dofs.edge) == length(rs.edges)
    @test length(dofs.edge_i) == length(rs.edges)
    @test length(dofs.face) == length(rs.faces)
    @test length(dofs.face_i) == length(rs.faces)

    # Check numbering convention (B) and entity matching
    all_dofs = Int[]
    # Vertices numbered first
    append!(all_dofs, collect_all_dofs(dofs.vert))
    @test all(all_dofs .== 1:length(all_dofs))
    if rs.rdim ≥ 1 # Test edges
        # Edges numbered next, no gaps or missing numbers. Sorted by edge number.
        all_edofs_i = collect_all_dofs(dofs.edge_i)
        @test all(all_edofs_i .== length(all_dofs) .+ (1:length(all_edofs_i)))
        # - all edge dofs include both vertexdofs and interior edegdofs, and nothing more.
        append!(all_dofs, all_edofs_i)
        @test all(all_dofs .== 1:length(all_dofs))
        @test length(all_dofs) == length(collect_all_dofs(dofs.vert)) + length(all_edofs_i)
        # Coarse check for C
        @test Set(collect_all_dofs(dofs.edge)) == Set(1:length(all_dofs))
        # - test each edge individually (Detailed check for C)
        for (edge_nr, edge_vertices) in enumerate(rs.edges)
            vdofs_e = Int[] # dofs.vert for vertices belonging to the current edge
            for j in edge_vertices # vertices in edge i
                isempty(dofs.vert[j]) || append!(vdofs_e, collect(dofs.vert[j]))
            end
            @test Set(dofs.edge[edge_nr]) == Set(vcat(vdofs_e, collect(dofs.edge_i[edge_nr])))
        end
    end
    if rs.rdim ≥ 2 # Test faces
        # Face numbered next, no gaps or missing numbers. Sorted by face number.
        all_fdofs_i = collect_all_dofs(dofs.face_i)
        @test all(all_fdofs_i .== length(all_dofs) .+ (1:length(all_fdofs_i)))
        # - all dofs now include vertex dofs, edge dofs and face dofs, but not volume dofs.
        append!(all_dofs, all_fdofs_i)
        @test all(all_dofs .== 1:length(all_dofs))
        # Coarse check for C
        @test Set(collect_all_dofs(dofs.face)) == Set(1:length(all_dofs))
        # - test each face individually (Detailed check for C)
        for (facenr, face_verts) in enumerate(rs.faces)
            vdofs_f = Int[]
            for j in face_verts # vertices in face i
                vdof_indices = dofs.vert[j]
                isempty(vdof_indices) || append!(vdofs_f, collect(vdof_indices))
            end
            edofs_f = Int[] # Interior edgedofs for edges belong to current face
            for (edgenr, edge_verts) in enumerate(rs.edges)
                # Both edge vertices belong to face => edge belongs to face
                (edge_verts[1] ∈ face_verts && edge_verts[2] ∈ face_verts) || continue
                append!(edofs_f, collect(dofs.edge_i[edgenr]))
            end
            @test Set(dofs.face[facenr]) == Set(vcat(vdofs_f, edofs_f, collect(dofs.face_i[facenr])))
        end
    end
    # Test volume
    # We always test this, since volumedofs are also used by lower-dimensional
    # discontinuous inteprolations to make them internal to the cell, e.g. DiscontinuousLagrange
    # Volumedofs numbered last
    append!(all_dofs, collect(dofs.vol_i))
    @test all(all_dofs .== 1:length(all_dofs))        # Numbering convention

    # Test D: getnbasefunctions matching number of dof indices
    return @test length(all_dofs) == dofs.n
end
@testset "interpolations" begin
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
        # Standard test all base interpolations must fullfill
        test_interpolation_properties(interpolation)

        ref_dim = Ferrite.getrefdim(interpolation)
        ref_shape = Ferrite.getrefshape(interpolation)
        func_order = Ferrite.getorder(interpolation)

        # Note that not every element formulation exists for every order and dimension.
        if applicable(Ferrite.getlowerorder, interpolation)
            @test isa(Ferrite.getlowerorder(interpolation), Interpolation{ref_shape, func_order - 1})
        end

        n_basefuncs = getnbasefunctions(interpolation)
        coords = Ferrite.reference_coordinates(interpolation)
        @test length(coords) == n_basefuncs

        @testset "Value Type $value_type" for value_type in (Float32, Float64)
            @testset let x = Vec{ref_dim, value_type}(sample_random_point(ref_shape))
                # Check gradient evaluation
                f(ξ) = [reference_shape_value(interpolation, Vec{ref_dim}(ξ), i) for i in 1:n_basefuncs]
                @test vec(ForwardDiff.jacobian(f, Array(x))') ≈
                    reinterpret(value_type, [reference_shape_gradient(interpolation, x, i) for i in 1:n_basefuncs])
                # Check partition of unity at random point.
                @test sum([reference_shape_value(interpolation, x, i) for i in 1:n_basefuncs]) ≈ 1.0
                # Check if the important functions are consistent
                @test_throws ArgumentError reference_shape_value(interpolation, x, n_basefuncs + 1)
                # Idempotency test
                @test reference_shape_value(interpolation, x, n_basefuncs) == reference_shape_value(interpolation, x, n_basefuncs)

                # Check for evaluation type correctness of interpolation
                for dof in 1:n_basefuncs
                    @test (@inferred reference_shape_value(interpolation, x, dof)) isa value_type
                    @test (@inferred reference_shape_gradient(interpolation, x, dof)) isa Vec{ref_dim, value_type}
                end
            end
        end

        # Check for dirac delta property of interpolation
        @testset "dirac delta property of dof $dof" for dof in 1:n_basefuncs
            for k in 1:n_basefuncs
                N_dof = reference_shape_value(interpolation, coords[dof], k)
                if k == dof
                    @test N_dof ≈ 1.0
                else
                    factor = interpolation isa Lagrange{RefQuadrilateral, 3} ? 200 : 4
                    @test N_dof ≈ 0.0 atol = factor * eps(typeof(N_dof))
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
            normals = reference_normals(getrefshape(interpolation))
            for (facetnodes, normal) in zip(Ferrite.facetdof_indices(interpolation), normals)
                @test __outward_normal(coords, facetnodes) ≈ normal
            end
        end

        # regression for https://github.com/Ferrite-FEM/Ferrite.jl/issues/520
        interpolation_type = typeof(interpolation).name.wrapper
        if func_order > 1 && interpolation_type != Ferrite.Serendipity
            first_order = interpolation_type{ref_shape, 1}()
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
        end

        @testset "VectorizedInterpolation" begin
            v_interpolation_1 = interpolation^2
            v_interpolation_2 = (d = 2; interpolation^d)
            @test getnbasefunctions(v_interpolation_1) ==
                getnbasefunctions(v_interpolation_2) ==
                getnbasefunctions(interpolation) * 2
            # pretty printing
            @test repr("text/plain", v_interpolation_1) == repr(v_interpolation_1.ip) * "^2"

            # Check for evaluation type correctness of vectorized interpolation
            v_interpolation_3 = interpolation^ref_dim

            @testset "Value Type $value_type" for value_type in (Float32, Float64)
                x = Vec{ref_dim, value_type}(sample_random_point(getrefshape(v_interpolation_1)))
                @testset "vectorized case of return type correctness of dof $dof" for dof in 1:n_basefuncs
                    @test @inferred(reference_shape_value(v_interpolation_1, x, dof)) isa Vec{2, value_type}
                    @test @inferred(reference_shape_gradient(v_interpolation_3, x, dof)) isa Tensor{2, ref_dim, value_type}
                end
            end
        end
    end

    @testset "Discontinuous interpolations" begin
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
    end

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

    @testset "H(curl) and H(div)" begin
        Hcurl_interpolations = [
            Nedelec{RefTriangle, 1}(), Nedelec{RefTriangle, 2}(), Nedelec{RefQuadrilateral, 1}(),
            Nedelec{RefTetrahedron, 1}(), Nedelec{RefHexahedron, 1}(),
        ]
        Hdiv_interpolations = [
            RaviartThomas{RefTriangle, 1}(), RaviartThomas{RefTriangle, 2}(), RaviartThomas{RefQuadrilateral, 1}(),
            RaviartThomas{RefTetrahedron, 1}(), RaviartThomas{RefHexahedron, 1}(),
            BrezziDouglasMarini{RefTriangle, 1}(),
        ]

        # These reference moments define the functionals that an interpolation should fulfill
        ## Raviart-Thomas on RefTriangle
        reference_edge_moment(::RaviartThomas{RefTriangle, 1}, edge_shape_nr, s) = 1
        reference_edge_moment(::RaviartThomas{RefTriangle, 2}, edge_shape_nr, s) = edge_shape_nr == 1 ? (1 - s) : s
        reference_face_moment(::RaviartThomas{RefTriangle, 2}, face_shape_nr, s1, s2) = face_shape_nr == 1 ? Vec(1, 0) : Vec(0, 1)

        ## Raviart-Thomas on RefQuadrilateral
        reference_edge_moment(::RaviartThomas{RefQuadrilateral, 1}, edge_shape_nr, s) = 1

        ## Raviart-Thomas on RefTetrahedron
        reference_face_moment(::RaviartThomas{RefTetrahedron, 1}, face_shape_nr, s1, s2) = 1

        ## Raviart-Thomas on RefHexahedron
        reference_face_moment(::RaviartThomas{RefHexahedron, 1}, face_shape_nr, s1, s2) = 1

        ## Brezzi-Douglas-Marini on RefTriangle
        reference_edge_moment(::BrezziDouglasMarini{RefTriangle, 1}, edge_shape_nr, s) = edge_shape_nr == 1 ? (1 - s) : s

        ## Nedelec on RefTriangle
        reference_edge_moment(::Nedelec{RefTriangle, 1}, edge_shape_nr, s) = 1
        reference_edge_moment(::Nedelec{RefTriangle, 2}, edge_shape_nr, s) = edge_shape_nr == 1 ? (1 - s) : s
        reference_face_moment(::Nedelec{RefTriangle, 2}, face_shape_nr, s1, s2) = face_shape_nr == 1 ? Vec(1, 0) : Vec(0, 1)

        ## Nedelec on RefQuadrilateral
        reference_edge_moment(::Nedelec{RefQuadrilateral, 1}, edge_shape_nr, s) = 1

        ## Nedelec on RefTetrahedron
        reference_edge_moment(::Nedelec{RefTetrahedron, 1}, edge_shape_nr, s) = 1

        ## Nedelec on RefHexahedron
        reference_edge_moment(::Nedelec{RefHexahedron, 1}, edge_shape_nr, s) = 1

        """
            integrate_edge(f)

        Integrate f(s) on the unit line, s ∈ [0, 1]
        """
        function integrate_edge(f::Function)
            val, _ = hquadrature(f, 0, 1; atol = 1.0e-8)
            return val
        end

        """
            integrate_face(f)

        Integrate f(s1, s2) on the unit square, (s1,s2) ∈ [0, 1] × [0, 1]
        """
        function integrate_face(f::Function)
            val, _ = hcubature(z -> f(z[1], z[2]), (0, 0), (1, 1); atol = 1.0e-8)
            return val
        end

        parameterize_edge(edge_coords, s) = edge_coords[1] + (edge_coords[2] - edge_coords[1]) * s

        function parameterize_face(face_coords, s1, s2)
            ξ0 = face_coords[1]
            v1 = face_coords[2] - ξ0
            v2 = face_coords[end] - ξ0
            if length(face_coords) == 3 # Triangle
                return ξ0 + s1 * (1 - s2 / 2) * v1 + s2 * (1 - s1 / 2) * v2
            elseif length(face_coords) == 4 # Quadrilateral
                return ξ0 + s1 * v1 + s2 * v2
            else
                throw(ArgumentError("length(face_coords) must be 3 or 4"))
            end
        end

        # parameterize_volume(volume_coords, s1, s2, s3)
        # TODO parameterization of volume (Tetrahedron, Hexahedron, Prism, Pyramid)

        edge_weight(ξ::F, s) where {F <: Function} = norm(gradient(ξ, s))
        face_weight(ξ::F, s1, s2) where {F <: Function} = norm(gradient(s -> ξ(s, s2), s1) × gradient(s -> ξ(s1, s), s2))

        function test_interpolation_functionals(ip::Interpolation)
            @testset "functionals $ip" begin
                test_interpolation_functionals(Ferrite.conformity(ip), Val(Ferrite.getrefdim(ip)), ip)
            end
        end

        function test_interpolation_functionals(::Ferrite.HdivConformity, ::Val{2}, ip::Interpolation)
            RefShape = getrefshape(ip)
            ipg = Lagrange{RefShape, 1}()

            # Test functionals associated with the edges
            for edge_nr in 1:Ferrite.nedges(RefShape)
                edge_coords = getindex.((Ferrite.reference_coordinates(ipg),), Ferrite.reference_edges(RefShape)[edge_nr])
                dof_inds = Ferrite.edgedof_interior_indices(ip)[edge_nr]

                ξ(s) = parameterize_edge(edge_coords, s)
                normal = reference_normals(RefShape)[edge_nr]
                for (edge_shape_nr, shape_nr) in pairs(dof_inds)
                    f(s) = reference_edge_moment(ip, edge_shape_nr, s) * (reference_shape_value(ip, ξ(s), shape_nr) ⋅ normal) * edge_weight(ξ, s)
                    @test integrate_edge(f) ≈ 1
                    # Test that the functional is zero for the other shape functions
                    for other_shape_nr in 1:getnbasefunctions(ip)
                        other_shape_nr == shape_nr && continue
                        g(s) = reference_edge_moment(ip, edge_shape_nr, s) * (reference_shape_value(ip, ξ(s), other_shape_nr) ⋅ normal) * edge_weight(ξ, s)
                        @test integrate_edge(g) + 1 ≈ 1 # integrate_edge(g) ≈ 0
                    end
                end

                # Test that normal components of other shape functions are zero on this edge
                # Stronger requirement than functionals being zero.
                for shape_nr in 1:getnbasefunctions(ip)
                    shape_nr in dof_inds && continue
                    for s in range(0, 1, 5)
                        @test abs(reference_shape_value(ip, ξ(s), shape_nr) ⋅ normal) < 1.0e-10
                    end
                end
            end

            # Test functionals associated with the faces
            for face_nr in 1:Ferrite.nfaces(RefShape)
                face_coords = getindex.((Ferrite.reference_coordinates(ipg),), Ferrite.reference_faces(RefShape)[face_nr])
                dof_inds = Ferrite.facedof_interior_indices(ip)[face_nr]

                ξ(s1, s2) = parameterize_face(face_coords, s1, s2)
                for (face_shape_nr, shape_nr) in pairs(dof_inds)
                    f(s1, s2) = reference_face_moment(ip, face_shape_nr, s1, s2) ⋅ reference_shape_value(ip, ξ(s1, s2), shape_nr) * face_weight(ξ, s1, s2)
                    @test integrate_face(f) ≈ 1

                    # Test that the functional is zero for the other shape functions
                    for other_shape_nr in 1:getnbasefunctions(ip)
                        other_shape_nr == shape_nr && continue
                        g(s1, s2) = reference_face_moment(ip, face_shape_nr, s1, s2) ⋅ reference_shape_value(ip, ξ(s1, s2), other_shape_nr) * face_weight(ξ, s1, s2)
                        @test integrate_face(g) + 1 ≈ 1 # integrate_edge(g) ≈ 0
                    end
                end
            end
        end

        # 3D, H(div)
        function test_interpolation_functionals(::Ferrite.HdivConformity, ::Val{3}, ip::Interpolation)
            RefShape = getrefshape(ip)
            ipg = Lagrange{RefShape, 1}()

            # Test functionals associated with the faces
            for face_nr in 1:Ferrite.nfaces(RefShape)
                face_coords = getindex.((Ferrite.reference_coordinates(ipg),), Ferrite.reference_faces(RefShape)[face_nr])
                normal = reference_normals(RefShape)[face_nr]
                dof_inds = Ferrite.facedof_interior_indices(ip)[face_nr]

                ξ(s1, s2) = parameterize_face(face_coords, s1, s2)
                for (face_shape_nr, shape_nr) in pairs(dof_inds)
                    f(s1, s2) = reference_face_moment(ip, face_shape_nr, s1, s2) * (reference_shape_value(ip, ξ(s1, s2), shape_nr) ⋅ normal) * face_weight(ξ, s1, s2)
                    @test integrate_face(f) ≈ 1

                    # Test that the functional is zero for the other shape functions
                    for other_shape_nr in 1:getnbasefunctions(ip)
                        other_shape_nr == shape_nr && continue
                        g(s1, s2) = reference_face_moment(ip, face_shape_nr, s1, s2) * (reference_shape_value(ip, ξ(s1, s2), other_shape_nr) ⋅ normal) * face_weight(ξ, s1, s2)
                        @test integrate_face(g) + 1 ≈ 1 # integrate_edge(g) ≈ 0
                    end
                end

                # Test that normal components of other shape functions are zero on this edge
                # Stronger requirement than functionals being zero.
                for shape_nr in 1:getnbasefunctions(ip)
                    shape_nr in dof_inds && continue
                    for s1 in range(0, 1, 5), s2 in range(0, 1, 5)
                        @test abs(reference_shape_value(ip, ξ(s1, s2), shape_nr) ⋅ normal) < 1.0e-10
                    end
                end
            end

            # Test functionals associated with the volume
            @assert length(Ferrite.volumedof_interior_indices(ip)) == 0 # Should be supported when testing `ip`s with those.
        end

        # H(curl)
        function test_interpolation_functionals(::Ferrite.HcurlConformity, ::Union{Val{2}, Val{3}}, ip::Interpolation)
            RefShape = getrefshape(ip)
            ipg = Lagrange{RefShape, 1}()

            # Test functionals associated with the edges
            for edge_nr in 1:Ferrite.nedges(RefShape)
                edge_coords = getindex.((Ferrite.reference_coordinates(ipg),), Ferrite.reference_edges(RefShape)[edge_nr])
                tangent = normalize(edge_coords[2] - edge_coords[1])
                dof_inds = Ferrite.edgedof_interior_indices(ip)[edge_nr]

                ξ(s) = parameterize_edge(edge_coords, s)
                for (edge_shape_nr, shape_nr) in pairs(dof_inds)
                    f(s) = reference_edge_moment(ip, edge_shape_nr, s) * (reference_shape_value(ip, ξ(s), shape_nr) ⋅ tangent) * edge_weight(ξ, s)
                    @test integrate_edge(f) ≈ 1
                end

                for shape_nr in 1:getnbasefunctions(ip)
                    shape_nr in dof_inds && continue # Already tested
                    for s in range(0, 1, 5)
                        @test abs(reference_shape_value(ip, ξ(s), shape_nr) ⋅ tangent) < 1.0e-10
                    end
                end
            end

            # Test functionals associated with the faces
            for face_nr in 1:Ferrite.nfaces(RefShape)
                face_coords = getindex.((Ferrite.reference_coordinates(ipg),), Ferrite.reference_faces(RefShape)[face_nr])
                dof_inds = Ferrite.facedof_interior_indices(ip)[face_nr]

                ξ(s1, s2) = parameterize_face(face_coords, s1, s2)
                for (face_shape_nr, shape_nr) in pairs(dof_inds)
                    f(s1, s2) = reference_face_moment(ip, face_shape_nr, s1, s2) ⋅ reference_shape_value(ip, ξ(s1, s2), shape_nr) * face_weight(ξ, s1, s2)
                    @test integrate_face(f) ≈ 1
                end
            end

            # Test functionals associated with volume
            @assert length(Ferrite.volumedof_interior_indices(ip)) == 0 # Test not supported yet, but needs to be if introduced ip with volumedofs.
        end

        @testset "Interpolation properties" begin
            test_interpolation_properties.(Hcurl_interpolations)
            test_interpolation_properties.(Hdiv_interpolations)
        end
        @testset "Interpolation functionals" begin
            test_interpolation_functionals.(Hcurl_interpolations)
            test_interpolation_functionals.(Hdiv_interpolations)
        end

        # Test boundary conditions (ProjectedDirichlet)
        # Depending on the interpolation, we have different polynomials orders on the facets
        _facet_poly_order(ip::Nedelec) = Ferrite.getorder(ip) - 1
        _facet_poly_order(ip::RaviartThomas) = Ferrite.getorder(ip) - 1
        _facet_poly_order(ip::BrezziDouglasMarini) = Ferrite.getorder(ip)
        # Based on this order, p_facet, we expect that we should fullfill different criteria.
        # * If we prescribe a polynomial function to ProjectedDirichlet with a lower or equal order
        #   than p_facet, we expect that the interpolation should match the provided function pointwise.
        # * If we prescribe a polynomial function with higher order, but lower order than what the quadrature
        #   rule in ProjectedDirichlet can integrate exactly, we expect that the integral over the boundary are equal.
        # * If we prescribe a polynomial one order lower than we can integrate exactly, and p_facet ≥ 1,
        #   we expect that the integrated linear moment equation, ∫ x f(x) dx, is integrated exactly
        # The following tests check those properties for the H(div) and H(curl) interpolations.

        cell_type(::Type{RefLine}) = Line
        cell_type(::Type{RefTriangle}) = Triangle
        cell_type(::Type{RefQuadrilateral}) = Quadrilateral
        cell_type(::Type{RefTetrahedron}) = Tetrahedron
        cell_type(::Type{RefHexahedron}) = Hexahedron
        function _setup_dh_fv_for_bc_test(ip::Interpolation; nel = 3, qr_order = 6)
            RefShape = Ferrite.getrefshape(ip)
            CT = cell_type(RefShape)
            dim = Ferrite.getrefdim(CT) # dim=sdim=vdim
            grid = generate_grid(CT, ntuple(Returns(nel), dim), -0.25 * ones(Vec{dim}), 0.2 * ones(Vec{dim}))
            transform_coordinates!(grid, x -> x + 0.25 * Vec(ntuple(i -> abs(x[dim - i + 1])^(1 + i / dim), dim)))
            qr = FacetQuadratureRule{RefShape}(qr_order)
            fv = FacetValues(qr, ip, geometric_interpolation(CT))
            dh = close!(add!(DofHandler(grid), :u, ip))
            return dh, (fv,)
        end
        function _setup_dh_fv_for_bc_test(ips::Tuple{Interpolation{<:RefTriangle}, Interpolation{<:RefQuadrilateral}}; nel = 3, qr_order = 6)
            dim = 2
            trigrid = generate_grid(Triangle, (3, 3), -0.25 * ones(Vec{dim}), 0.2 * ones(Vec{dim}))
            #trigrid = generate_grid(Triangle, (3, 3))
            mixgrid, nr_quad = grid_with_inserted_quad(trigrid, (3, 4); update_sets = false) # Quad on bottom facet.
            empty!(mixgrid.facetsets)
            addfacetset!(mixgrid, "bottom", x -> x[2] ≈ -0.25)
            transform_coordinates!(mixgrid, x -> x + 0.25 * Vec(ntuple(i -> abs(x[dim - i + 1])^(1 + i / dim), dim)))
            qr_tri = FacetQuadratureRule{RefTriangle}(qr_order)
            fv_tri = FacetValues(qr_tri, ips[1], geometric_interpolation(Triangle))
            qr_quad = FacetQuadratureRule{RefQuadrilateral}(qr_order)
            fv_quad = FacetValues(qr_quad, ips[2], geometric_interpolation(Quadrilateral))
            dh = DofHandler(mixgrid)
            sdh_tri = SubDofHandler(dh, setdiff(1:getncells(mixgrid), Set(nr_quad)))
            add!(sdh_tri, :u, ips[1])
            sdh_quad = SubDofHandler(dh, Set(nr_quad))
            add!(sdh_quad, :u, ips[2])
            close!(dh)
            return dh, (fv_tri, fv_quad)
        end

        function test_bc_integral(
                f_bc::Function, check_fun::Function, moment_fun::Function, dh::DofHandler, facetset, fvs::Tuple;
                tol = 1.0e-6, custom_qr_order = nothing, pointwise_check, kwargs...
            )
            grid = Ferrite.get_grid(dh)
            dbc = if custom_qr_order === nothing
                ProjectedDirichlet(:u, facetset, f_bc)
            else
                ProjectedDirichlet(:u, facetset, f_bc; qr_order = custom_qr_order)
            end
            ch = close!(add!(ConstraintHandler(dh), dbc))
            a = zeros(ndofs(dh))
            apply!(a, ch)
            fv1 = first(fvs)
            test_val = zero(f_bc(get_node_coordinate(grid, 1), 0.0, getnormal(fv1, 1)))
            check_val = zero(check_fun(zero(Ferrite.shape_value_type(fv1)), getnormal(fv1, 1)))
            @assert typeof(test_val) === typeof(check_val)
            for (sdh, fv) in zip(dh.subdofhandlers, fvs)
                tv, cv = test_bc_integral(f_bc, check_fun, moment_fun, sdh, a, facetset, fv; pointwise_check, kwargs...)
                test_val += tv
                check_val += cv
            end
            @test norm(test_val - check_val) < tol
        end

        function test_bc_integral(
                f_bc::Function, check_fun::Function, moment_fun::Function, sdh::SubDofHandler, a, facetset, fv::FacetValues;
                pointwise_check, patchwise_check = true
            )
            grid = sdh.dh.grid
            test_val = zero(f_bc(get_node_coordinate(grid, 1), 0.0, getnormal(fv, 1)))
            check_val = zero(check_fun(zero(Ferrite.shape_value_type(fv)), getnormal(fv, 1)))
            @assert typeof(test_val) === typeof(check_val)
            for (cellidx, facetidx) in facetset
                cellidx ∈ sdh.cellset || continue
                cell_coords = getcoordinates(grid, cellidx)
                reinit!(fv, getcells(grid, cellidx), cell_coords, facetidx)
                ae = a[celldofs(sdh, cellidx)]
                for q_point in 1:getnquadpoints(fv)
                    dΓ = getdetJdV(fv, q_point)
                    u = function_value(fv, q_point, ae)
                    n = getnormal(fv, q_point)
                    x = spatial_coordinate(fv, q_point, cell_coords)
                    check_fun_val = check_fun(u, n)
                    bc_fun_val = f_bc(x, 0.0, n)
                    check_val += moment_fun(x) * check_fun_val * dΓ
                    test_val += moment_fun(x) * bc_fun_val * dΓ
                    if pointwise_check
                        @test check_fun_val ≈ bc_fun_val
                    end
                end
                if patchwise_check
                    @test check_val ≈ test_val
                end
            end
            return test_val, check_val
        end

        @testset "H(div) BC" begin
            for ip in Hdiv_interpolations
                @testset "$ip" begin
                    dh, fv = _setup_dh_fv_for_bc_test(ip)
                    linear_x1(x, _, _) = x[1]
                    quadratic_x1(x, _, _) = (x[1] - 0.3)^2
                    nonlinear(x, _, _) = 100 * (x[1] - 0.3)^4
                    funs = [
                        (Returns(0.0), true),
                        (Returns(rand()), true),
                        (linear_x1, _facet_poly_order(ip) ≥ 1),
                        (quadratic_x1, _facet_poly_order(ip) ≥ 2),
                        (nonlinear, false), # Only test integral quantities using high-order integration.
                    ]
                    for (f_bc, pointwise_check) in funs
                        @testset "f_bc = $f_bc" begin
                            for facetset in values(dh.grid.facetsets)
                                custom_qr_order = f_bc === nonlinear ? 5 : nothing
                                test_bc_integral(f_bc, ⋅, Returns(1), dh, facetset, fv; pointwise_check, custom_qr_order)
                                if _facet_poly_order(ip) ≥ 1 && f_bc !== nonlinear # Linear facet polynomial should pass moment test.
                                    test_bc_integral(f_bc, ⋅, x -> x[1], dh, facetset, fv; pointwise_check = false)
                                end
                            end
                        end
                    end
                end
            end
        end

        @testset "H(curl) BC" begin
            ips = Any[Hcurl_interpolations...]
            push!(ips, (Nedelec{RefTriangle, 1}(), Nedelec{RefQuadrilateral, 1}())) # Mixed case
            for interp in ips
                ip = isa(interp, Tuple) ? interp[1] : interp
                dim = Ferrite.getrefdim(getrefshape(ip))
                dim == 3 && continue # TODO: Not yet implemented
                @testset "$interp" begin
                    dh, fvs = _setup_dh_fv_for_bc_test(interp)
                    v3 = rand()
                    linear_x1(x, _, _) = x[1] * Vec((0.0, 0.0, v3))
                    quadratic_x1(x, _, _) = (x[1] - 0.3)^2 * Vec((0.0, 0.0, v3))
                    nonlinear(x, _, _) = Vec((0.0, 0.0, v3)) * (100 * (x[1] + 0.3)^4)
                    funs = [
                        (Returns(zero(Vec{3})), true),
                        (Returns(Vec((0.0, 0.0, rand()))), true),
                        (linear_x1, _facet_poly_order(ip) ≥ 1),
                        (quadratic_x1, _facet_poly_order(ip) ≥ 2),
                        (nonlinear, false),
                    ]
                    for (f_bc, pointwise_check) in funs
                        @testset "f_bc = $f_bc" begin
                            for facetset in values(dh.grid.facetsets)
                                custom_qr_order = f_bc === nonlinear ? 5 : nothing
                                test_bc_integral(f_bc, ×, Returns(1.0), dh, facetset, fvs; pointwise_check, custom_qr_order)
                                if _facet_poly_order(ip) ≥ 1  && f_bc !== nonlinear
                                    test_bc_integral(f_bc, ×, x -> x[1], dh, facetset, fvs; pointwise_check = false)
                                end
                            end
                        end
                    end
                end
            end
        end

        @testset "ProjectedDirichlet error path" begin
            dh_H1, _ = _setup_dh_fv_for_bc_test(Lagrange{RefTriangle, 1}()^2; nel = 1, qr_order = 1)
            dh_L2, _ = _setup_dh_fv_for_bc_test(DiscontinuousLagrange{RefTriangle, 1}()^2; nel = 1, qr_order = 1)
            for dh in (dh_H1, dh_L2)
                dbc = ProjectedDirichlet(:u, Set([FacetIndex(1, 1)]), Returns(zero(Vec{2})))
                ch = add!(ConstraintHandler(dh), dbc)
                @test_throws ArgumentError close!(ch)
            end
        end

    end

end # testset
