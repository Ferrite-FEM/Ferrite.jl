using Ferrite: reference_shape_value, reference_shape_gradient

"""
    test_interpolation_properties(ip::Interpolation)

This function tests the following implementation details for an
interpolation. All base interpolations should pass this test, but
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

        # Test E: All base functions implemented.
        # Argument errors for 0th and n+1 indices.
        ξ = zero(Vec{Ferrite.getrefdim(ip)})
        @test_throws ArgumentError Ferrite.reference_shape_value(ip, ξ, 0)
        for i in 1:getnbasefunctions(ip)
            @test Ferrite.reference_shape_value(ip, ξ, i) isa Ferrite.shape_value_type(ip, Float64)
        end
        @test_throws ArgumentError Ferrite.reference_shape_value(ip, ξ, getnbasefunctions(ip) + 1)
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

    reference_cell(::Type{RefTriangle}) = Triangle((1, 2, 3))
    reference_cell(::Type{RefQuadrilateral}) = Quadrilateral((1, 2, 3, 4))
    reference_cell(::Type{RefTetrahedron}) = Tetrahedron((1, 2, 3, 4))
    reference_cell(::Type{RefHexahedron}) = Hexahedron((ntuple(identity, 8)))
    function facet_parameterization(::Type{RefShape}, ξ, facet_id) where {RefShape <: Ferrite.AbstractRefShape{2}}
        # facet = edge
        return edge_parameterization(RefShape, ξ, facet_id)
    end

    """
        edge_parameterization(::Type{<:AbstractRefShape}, ξ, edge_id)

    An edge is parameterized by the normalized curve coordinate `s [0, 1]`,
    increasing in the positive edge direction.
    """
    function edge_parameterization(::Type{RefShape}, ξ, edge_id) where {RefShape <: Ferrite.AbstractRefShape}
        ipg = Lagrange{RefShape, 1}() # Reference shape always described by 1st order Lagrange ip.
        refcoords = Ferrite.reference_coordinates(ipg)
        i1, i2 = Ferrite.edgedof_indices(ipg)[edge_id]
        ξ1, ξ2 = (refcoords[i1], refcoords[i2])
        Δξ = ξ2 - ξ1
        L = norm(Δξ)
        s = (ξ - ξ1) ⋅ normalize(Δξ) / L
        @assert norm(ξ - ξ1) ≈ L * s # Ensure ξ is on the line ξ1 - ξ2
        @assert -eps(L) ≤ s ≤ (1 + eps(L)) # Ensure ξ is between ξ1 and ξ2
        return s
    end

    function facet_parameterization(::Type{<:Ferrite.AbstractRefShape{3}}, ξ, facet_id)
        # Not implemented (not yet defined in Ferrite what this should be),
        # but to support testing interpolations with a single facedof interior index,
        # we return `nothing` just to allow running the code as long as the output isn't used.
        return nothing
    end

    function integrate_facet(fv::FacetValues, f::Function, shapenr::Int, cell::Ferrite.AbstractCell{RefShape}) where {RefShape}
        facet_id = Ferrite.getcurrentfacet(fv)
        function qpoint_contribution(q_point)
            ξ = Ferrite.getpoints(fv.fqr, facet_id)[q_point]
            # facet parameterization: 1D [0, 1], 2D ([0, 1], [0, 1])
            s = facet_parameterization(RefShape, ξ, facet_id)
            n = getnormal(fv, q_point)
            facet_sign = Ferrite.get_direction(Ferrite.function_interpolation(fv), shapenr, cell)
            N = shape_value(fv, q_point, shapenr) * facet_sign # Ensure no reorientation.
            return f(s, N, n) * getdetJdV(fv, q_point)
        end
        val = qpoint_contribution(1)
        for q_point in 2:getnquadpoints(fv)
            val += qpoint_contribution(q_point)
        end
        return val
    end

    function integrate_edge(ev::EdgeValues, f::Function, shapenr::Int, cell::Ferrite.AbstractCell{RefShape}) where {RefShape}
        edge_id = Ferrite.getcurrentedge(ev)
        function qpoint_contribution(q_point)
            ξ = Ferrite.getpoints(ev.eqr, edge_id)[q_point]
            s = edge_parameterization(RefShape, ξ, edge_id)
            t = Ferrite.gettangent(ev, q_point)
            edge_sign = Ferrite.get_direction(Ferrite.function_interpolation(ev), shapenr, cell)
            N = shape_value(ev, q_point, shapenr) * edge_sign # Ensure no reorientation
            return f(s, N, t) * getdetJdV(ev, q_point)
        end
        val = qpoint_contribution(1)
        for q_point in 2:getnquadpoints(ev)
            val += qpoint_contribution(q_point)
        end
        return val
    end

    Hcurl_interpolations = [Nedelec{2, RefTriangle, 1}(), Nedelec{2, RefTriangle, 2}()] # Nedelec{3, RefTetrahedron, 1}(), Nedelec{3, RefHexahedron, 1}()]
    Hdiv_interpolations = [RaviartThomas{2, RefTriangle, 1}(), RaviartThomas{2, RefTriangle, 2}(), BrezziDouglasMarini{2, RefTriangle, 1}()]

    test_interpolation_properties.(Hcurl_interpolations)
    test_interpolation_properties.(Hdiv_interpolations)

    # Required properties of shape value Nⱼ of an edge-elements (Hcurl) on an edge with direction v, length L, and dofs ∈ 𝔇
    # 1) Unit property: ∫(Nⱼ ⋅ v f(s) dS) = 1 ∀ ∈ 𝔇
    #    Must hold for
    #    length(𝔇) ≥ 1: f(s) = 1
    #    length(𝔇) = 2: f(s) = 1 - s or f(s) = s for 1st and 2nd dof, respectively.
    #    Additionally, should be zero for
    #    length(𝔇) = 2: f(s) = s or f(s) = 1 - s for 1st and 2nd dof, respectively.
    #    s is the path parameter ∈[0,1] along the positive direction of the path.
    # 2) Zero along other edges: Nⱼ ⋅ v = 0 if j∉𝔇
    @testset "H(curl) on RefCell" begin
        for ip in Hcurl_interpolations
            cell = reference_cell(getrefshape(ip))
            geo_ip = geometric_interpolation(cell)
            ev = EdgeValues(EdgeQuadratureRule{getrefshape(ip)}(20), ip, geo_ip)
            edges = Ferrite.edges(cell)
            dofs = Ferrite.edgedof_interior_indices(ip)
            x = Ferrite.reference_coordinates(geo_ip)
            test_points_line = [Vec((ξ,)) for ξ in [-1.0, rand(3)..., 1.0]]
            @testset "$ip" begin
                for (edge_nr, (i1, i2)) in enumerate(edges)
                    reinit!(ev, cell, x, edge_nr)
                    for (idof, shape_nr) in enumerate(dofs[edge_nr])
                        nedgedofs = length(dofs[edge_nr])
                        if nedgedofs == 1
                            @test 1 ≈ integrate_edge(ev, (_, N, t) -> N ⋅ t, shape_nr, cell)
                        elseif nedgedofs == 2
                            f(s, N, t) = (idof == 1 ? 1 - s : s) * N ⋅ t
                            @test 1 ≈ integrate_edge(ev, f, shape_nr, cell)
                            g(s, N, t) = (idof == 1 ? s : 1 - s) * N ⋅ t
                            @test 1 ≈ 1 + integrate_edge(ev, g, shape_nr, cell)
                        end
                    end
                    # Check that tangential component is zero for dofs not belong to edge
                    t = gettangent(ev, 1) # Constant tangent since we work on ref cell
                    for ξ_1d in test_points_line
                        ξ = Ferrite.edge_to_cell_transformation(ξ_1d, getrefshape(ip), edge_nr)
                        for (j_edge, shape_nrs) in enumerate(dofs)
                            j_edge == edge_nr && continue
                            for shape_nr in shape_nrs
                                @test reference_shape_value(ip, ξ, shape_nr) ⋅ t + 1 ≈ 1
                            end
                        end
                    end
                end
            end
        end
    end

    # Required properties of shape value Nⱼ of an edge-elements (Hdiv) on an edge with normal n, length L, and dofs ∈ 𝔇
    # 1) Unit property: ∫(Nⱼ ⋅ n f(s) dS) = 1 ∀ j ∈ 𝔇
    #    Must hold for
    #    length(𝔇) ≥ 1: f(s) = 1
    #    length(𝔇) = 2: f(s) = 1 - s or f(s) = s for 1st and 2nd dof, respectively.
    #    Additionally, should be zero for
    #    length(𝔇) = 2: f(s) = s or f(s) = 1 - s for 1st and 2nd dof, respectively.
    #    s is the path parameter ∈[0,1] along the positive direction of the path.
    # 2) Zero normal component on other edges: Nⱼ ⋅ n = 0 if j∉𝔇
    @testset "H(div) on RefCell" begin
        reference_moment_functions(::RaviartThomas{2, RefTriangle, 1}) = (Returns(1.0),)
        reference_moment_functions(::RaviartThomas{2, RefTriangle, 2}) = (s -> 1 - s, s -> s)
        reference_moment_functions(::BrezziDouglasMarini{2, RefTriangle, 1}) = (s -> 1 - s, s -> s)

        for ip in Hdiv_interpolations
            cell = reference_cell(getrefshape(ip))
            fqr = FacetQuadratureRule{getrefshape(ip)}(4)
            fv = FacetValues(fqr, ip, Lagrange{getrefshape(ip), 1}())
            cell_facets = Ferrite.facets(cell)
            dofs = Ferrite.facetdof_interior_indices(ip)
            x = Ferrite.reference_coordinates(geometric_interpolation(typeof(cell)))
            normals = reference_normals(geometric_interpolation(typeof(cell)))
            test_points_facet = [Vec((ξ,)) for ξ in [-1.0, rand(3)..., 1.0]] #TODO: generalize to work for faces and not only lines
            @testset "$ip" begin
                for (facet_nr, (i1, i2)) in enumerate(cell_facets)
                    reinit!(fv, reference_cell(getrefshape(ip)), x, facet_nr)
                    @testset "Facet $facet_nr" begin
                        n = normals[facet_nr]
                        for (rmf_idx, rm_fun) in enumerate(reference_moment_functions(ip))
                            f(s, N, nq) = rm_fun(s) * (N ⋅ nq)
                            for (idof, shape_nr) in enumerate(dofs[facet_nr])
                                if idof == rmf_idx
                                    @test 1 ≈ integrate_facet(fv, f, shape_nr, cell)
                                else
                                    @test 1 ≈ 1 + integrate_facet(fv, f, shape_nr, cell)
                                end
                            end
                        end
                        for (j_facet, shape_nrs) in enumerate(dofs)
                            j_facet == facet_nr && continue
                            for shape_nr in shape_nrs
                                for ξ_onfacet in test_points_facet
                                    ξ = Ferrite.facet_to_cell_transformation(ξ_onfacet, getrefshape(ip), facet_nr)
                                    @test reference_shape_value(ip, ξ, shape_nr) ⋅ n + 1 ≈ 1
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

        cell_types = Dict(
            RefTriangle => [Triangle, QuadraticTriangle],
            RefQuadrilateral => [Quadrilateral, QuadraticQuadrilateral],
            RefTetrahedron => [Tetrahedron],
            RefHexahedron => [Hexahedron]
        )

        for (ips, check_function) in ((Hcurl_interpolations, hcurl_check), (Hdiv_interpolations, hdiv_check))
            for ip in ips
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
                            for facetnr in 1:nfacets(RefShape)
                                fi = FacetIndex(cellnr, facetnr)
                                # Check continuity of function value according to check_function
                                ITU.test_continuity(dh, fi; transformation_function = check_function)
                            end
                            # Check gradient calculation
                            ITU.test_gradient(dh, cellnr)
                        end
                    end
                end
            end
        end
    end

    function _setup_dh_fv(ip; nel = 2, qr_order = 4)
        RefShape = Ferrite.getrefshape(ip)
        CT = typeof(reference_cell(RefShape))
        dim = Ferrite.getrefdim(CT) # dim=sdim=vdim
        #grid = generate_grid(CT, ntuple(Returns(nel), dim), -0.25 * ones(Vec{dim}), 0.2 * ones(Vec{dim}))
        grid = generate_grid(CT, ntuple(Returns(nel), dim))
        qr = FacetQuadratureRule{RefShape}(qr_order)
        fv = FacetValues(qr, ip, geometric_interpolation(CT))
        dh = close!(add!(DofHandler(grid), :u, ip))
        return dh, fv
    end

    function test_bc_integral(f_bc::Function, check_fun::Function, dh, facetset, fv; tol = 1.0e-6)
        grid = Ferrite.get_grid(dh)
        dbc = WeakDirichlet(:u, facetset, f_bc)
        ch = close!(add!(ConstraintHandler(dh), dbc))
        a = zeros(ndofs(dh))
        apply!(a, ch)
        test_val = zero(f_bc(get_node_coordinate(grid, 1), 0.0, getnormal(fv, 1)))
        check_val = zero(check_fun(zero(Ferrite.shape_value_type(fv)), getnormal(fv, 1)))
        @assert typeof(test_val) === typeof(check_val)
        for (cellidx, facetidx) in facetset
            cell_coords = getcoordinates(grid, cellidx)
            reinit!(fv, getcells(grid, cellidx), cell_coords, facetidx)
            ae = a[celldofs(dh, cellidx)]
            for q_point in 1:getnquadpoints(fv)
                dΓ = getdetJdV(fv, q_point)
                u = function_value(fv, q_point, ae)
                n = getnormal(fv, q_point)
                x = spatial_coordinate(fv, q_point, cell_coords)
                check_val += check_fun(u, n) * dΓ
                test_val += f_bc(x, 0.0, n) * dΓ
            end
        end
        @test norm(test_val - check_val) < tol
    end

    @testset "H(div) BC" begin
        for ip in Hdiv_interpolations
            @testset "$ip" begin
                dh, fv = _setup_dh_fv(ip)
                linear_x1(x, _, _) = x[1]
                for f_bc in (Returns(0.0), Returns(1.0), linear_x1)
                    @testset "f_bc = $f_bc" begin
                        for facetset in values(dh.grid.facetsets)
                            test_bc_integral(f_bc, ⋅, dh, facetset, fv)
                        end
                    end
                end
            end
        end
    end

    @testset "H(curl) BC" begin
        for ip in Hcurl_interpolations
            @testset "$ip" begin
                dh, fv = _setup_dh_fv(ip)
                dim = Ferrite.getrefdim(getrefshape(ip))
                @assert dim == 2 # 3d not supported yet
                v3 = rand()
                linear_x1(x, _, _) = x[1] * Vec((0.0, 0.0, v3))
                for f_bc in (Returns(zero(Vec{3})), Returns(Vec((0.0, 0.0, rand()))), linear_x1)
                    @testset "f_bc = $f_bc" begin
                        for facetset in values(dh.grid.facetsets)
                            test_bc_integral(f_bc, ×, dh, facetset, fv)
                        end
                    end
                end
            end
        end
    end
end # testset
