@testset "interpolations" begin

@testset "$interpolation" for interpolation in (
                      Lagrange{1, RefCube, 1}(),
                      Lagrange{1, RefCube, 2}(),
                      Lagrange{2, RefCube, 1}(),
                      Lagrange{2, RefCube, 2}(),
                      Lagrange{2, RefTetrahedron, 1}(),
                      Lagrange{2, RefTetrahedron, 2}(),
                      #Lagrange{2, RefTetrahedron, 3}(),
                      #Lagrange{2, RefTetrahedron, 4}(),
                      #Lagrange{2, RefTetrahedron, 5}(),
                      Lagrange{3, RefCube, 1}(),
                      Lagrange{3, RefCube, 2}(),
                      Serendipity{2, RefCube, 2}(),
                      Serendipity{3, RefCube, 2}(),
                      Lagrange{3, RefTetrahedron, 1}(),
                      Lagrange{3, RefTetrahedron, 2}(),
                      Lagrange{3, RefPrism, 1}(),
                      Lagrange{3, RefPrism, 2}(),
                      #
                      DiscontinuousLagrange{1, RefCube, 0}(),
                      DiscontinuousLagrange{2, RefCube, 0}(),
                      DiscontinuousLagrange{3, RefCube, 0}(),
                      DiscontinuousLagrange{2, RefTetrahedron, 0}(),
                      DiscontinuousLagrange{3, RefTetrahedron, 0}(),
                      DiscontinuousLagrange{1, RefCube, 1}(),
                      DiscontinuousLagrange{2, RefCube, 1}(),
                      DiscontinuousLagrange{3, RefCube, 1}(),
                      DiscontinuousLagrange{2, RefTetrahedron, 1}(),
                      DiscontinuousLagrange{3, RefTetrahedron, 1}(),
                      #
                      BubbleEnrichedLagrange{2,RefTetrahedron,1}(),
                      #
                      CrouzeixRaviart{2,1}(),
    )

    # Test of utility functions
    ndim = Ferrite.getdim(interpolation)
    r_shape = Ferrite.getrefshape(interpolation)
    func_order = Ferrite.getorder(interpolation)
    @test typeof(interpolation) <: Interpolation{ndim,r_shape,func_order}

    # Note that not every element formulation exists for every order and dimension.
    applicable(Ferrite.getlowerdim, interpolation) && @test typeof(Ferrite.getlowerdim(interpolation)) <: Interpolation{ndim-1}
    applicable(Ferrite.getlowerorder, interpolation) && @test typeof(Ferrite.getlowerorder(interpolation)) <: Interpolation{ndim,r_shape,func_order-1}

    # Check partition of unity at random point.
    n_basefuncs = getnbasefunctions(interpolation)
    x = rand(Tensor{1, ndim})
    f = (x) -> Ferrite.value(interpolation, Tensor{1, ndim}(x))
    @test vec(ForwardDiff.jacobian(f, Array(x))') ≈
           reinterpret(Float64, Ferrite.derivative(interpolation, x))
    @test sum(Ferrite.value(interpolation, x)) ≈ 1.0

    # Check if the important functions are consistens
    coords = Ferrite.reference_coordinates(interpolation)
    @test length(coords) == n_basefuncs
    @test Ferrite.value(interpolation, length(coords), x) == Ferrite.value(interpolation, length(coords), x)
    @test_throws BoundsError Ferrite.value(interpolation, length(coords)+1, x)

    # Test whether we have for each entity corresponding dof indices (possibly empty)
    @test length(Ferrite.vertexdof_indices(interpolation)) == Ferrite.nvertices(interpolation)
    @test length(Ferrite.edgedof_indices(interpolation)) == Ferrite.nedges(interpolation)
    @test length(Ferrite.edgedof_interior_indices(interpolation)) == Ferrite.nedges(interpolation)
    @test length(Ferrite.facedof_indices(interpolation)) == Ferrite.nfaces(interpolation)
    @test length(Ferrite.facedof_interior_indices(interpolation)) == Ferrite.nfaces(interpolation)

    # We have at least as many edge/face dofs as we have edge/face interior dofs
    @test all(length.(Ferrite.edgedof_interior_indices(interpolation)) .<= length.(Ferrite.edgedof_indices(interpolation)))
    @test all(length.(Ferrite.facedof_interior_indices(interpolation)) .<= length.(Ferrite.facedof_indices(interpolation)))

    # The total number of dofs must match the number of base functions
    @test sum(length.(Ferrite.vertexdof_indices(interpolation));init=0) + sum(length.(Ferrite.edgedof_interior_indices(interpolation));init=0) + sum(length.(Ferrite.facedof_interior_indices(interpolation));init=0) + sum(length.(Ferrite.celldof_interior_indices(interpolation));init=0) == n_basefuncs

    # The dof indices are valid.
    @test all([all(0 .< i .<= n_basefuncs) for i ∈ Ferrite.vertexdof_indices(interpolation)])
    @test all([all(0 .< i .<= n_basefuncs) for i ∈ Ferrite.facedof_indices(interpolation)])
    @test all([all(0 .< i .<= n_basefuncs) for i ∈ Ferrite.edgedof_indices(interpolation)])
    @test all([all(0 .< i .<= n_basefuncs) for i ∈ Ferrite.facedof_interior_indices(interpolation)])
    @test all([all(0 .< i .<= n_basefuncs) for i ∈ Ferrite.edgedof_interior_indices(interpolation)])
    @test all([all(0 .< i .<= n_basefuncs) for i ∈ Ferrite.celldof_interior_indices(interpolation)])

    # Check for dirac delta property of interpolation
    @testset "dirac delta property of dof $dof" for dof in 1:n_basefuncs
        N_dof = Ferrite.value(interpolation, coords[dof])
        for k in 1:n_basefuncs
            if k == dof
                @test N_dof[k] ≈ 1.0
            else
                @test N_dof[k] ≈ 0.0 atol=4eps(Float64) #broken=typeof(interpolation)==Lagrange{2, RefTetrahedron, 5}&&dof==4&&k==18
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
        for (facenodes, normal) in zip(Ferrite.facedof_indices(interpolation), reference_normals(interpolation))
            @test __outward_normal(coords, facenodes) ≈ normal
        end
    end

    # regression for https://github.com/Ferrite-FEM/Ferrite.jl/issues/520
    interpolation_type = typeof(interpolation).name.wrapper
    if func_order > 1 && interpolation_type != Ferrite.Serendipity
        first_order = interpolation_type{ndim,r_shape,1}() 
        for (highorderface, firstorderface) in zip(Ferrite.facedof_indices(interpolation), Ferrite.facedof_indices(first_order))
            for (h_node, f_node) in zip(highorderface, firstorderface)
                @test h_node == f_node
            end
        end
        if ndim > 2
            for (highorderedge, firstorderedge) in zip(Ferrite.edgedof_indices(interpolation), Ferrite.edgedof_indices(first_order))
                for (h_node, f_node) in zip(highorderedge, firstorderedge)
                    @test h_node == f_node
                end
            end
        end
    end
end

@test Ferrite.reference_coordinates(DiscontinuousLagrange{2,RefTetrahedron,0}()) ≈ [Vec{2,Float64}((1/3,1/3))]
@test Ferrite.reference_coordinates(DiscontinuousLagrange{2,RefCube,0}()) ≈ [Vec{2,Float64}((0,0))]
@test Ferrite.reference_coordinates(DiscontinuousLagrange{3,RefTetrahedron,0}()) ≈ [Vec{3,Float64}((1/4,1/4,1/4))]
@test Ferrite.reference_coordinates(DiscontinuousLagrange{3,RefCube,0}()) ≈ [Vec{3,Float64}((0,0,0))]

end
