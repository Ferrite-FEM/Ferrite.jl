# Shared helpers for the test_interpolations* files.
using Ferrite

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
            vert = as_vector(Ferrite.vertexdof_indices(ip)::NTuple{<:Any, NTuple{<:Any, Int}}),
            edge = as_vector(Ferrite.edgedof_indices(ip)::NTuple{<:Any, NTuple{<:Any, Int}}),
            face = as_vector(Ferrite.facedof_indices(ip)::NTuple{<:Any, NTuple{<:Any, Int}}),
            edge_i = as_vector(Ferrite.edgedof_interior_indices(ip)::NTuple{<:Any, NTuple{<:Any, Int}}),
            face_i = as_vector(Ferrite.facedof_interior_indices(ip)::NTuple{<:Any, NTuple{<:Any, Int}}),
            vol_i = as_vector(Ferrite.volumedof_interior_indices(ip)::NTuple{<:Any, Int}),
            n = getnbasefunctions(ip)::Int,
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

        # Test F: dof functionals
        fs = Ferrite.dof_functionals(ip)
        @test fs isa NTuple{getnbasefunctions(ip), DofFunctional}
        if all(f -> f isa PointValue, fs)
            # Point-value dofs must satisfy the Kronecker delta property at their
            # reference coordinates
            coords = Ferrite.reference_coordinates(ip)
            @test map(f -> f.x, fs) == Tuple(coords)
            for j in 1:getnbasefunctions(ip), i in 1:getnbasefunctions(ip)
                Nij = Ferrite.reference_shape_value(ip, coords[j], i)
                @test isapprox(Nij, i == j ? one(Nij) : zero(Nij); atol = 200 * eps(Float64))
            end
        else
            # Moment-type layouts must have uniform per-entity signatures so that dof
            # distribution can compare them entity-wise
            for entity_functionals in (Ferrite.vertexdof_functionals, Ferrite.edgedof_functionals, Ferrite.facedof_functionals, Ferrite.facetdof_functionals)
                sigs = unique(filter(!isempty, collect(entity_functionals(ip))))
                @test length(sigs) <= 1
            end
        end
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
