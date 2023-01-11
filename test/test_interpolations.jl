@testset "interpolations" begin

@testset "$interpolation" for interpolation in (
                      Lagrange{1, RefCube, 1}(),
                      Lagrange{1, RefCube, 2}(),
                      Lagrange{2, RefCube, 1}(),
                      Lagrange{2, RefCube, 2}(),
                      Lagrange{2, RefTetrahedron, 1}(),
                      Lagrange{2, RefTetrahedron, 2}(),
                      Lagrange{2, RefTetrahedron, 3}(),
                      Lagrange{2, RefTetrahedron, 4}(),
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

    n_basefuncs = getnbasefunctions(interpolation)
    x = rand(Tensor{1, ndim})
    f = (x) -> Ferrite.value(interpolation, Tensor{1, ndim}(x))
    @test vec(ForwardDiff.jacobian(f, Array(x))') ≈
           reinterpret(Float64, Ferrite.derivative(interpolation, x))
    @test sum(Ferrite.value(interpolation, x)) ≈ 1.0

    # Check for dirac delta property of interpolation
    coords = Ferrite.reference_coordinates(interpolation)
    @testset "dirac delta property of dof $dof" for dof in 1:n_basefuncs
        N_dof = Ferrite.value(interpolation, coords[dof])
        for k in 1:n_basefuncs
            if k == dof
                @test N_dof[k] ≈ 1.0
            else
                @test N_dof[k] ≈ 0.0 atol=4eps(Float64)
            end
        end
    end

    # Test that faces(...) return in counter clockwise order (viewing from the outside)
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
        for (facenodes, normal) in zip(Ferrite.faces(interpolation), reference_normals(interpolation))
            @test __outward_normal(coords, facenodes) ≈ normal
        end
    end
    # regression for https://github.com/Ferrite-FEM/Ferrite.jl/issues/520
    interpolation_type = typeof(interpolation).name.wrapper
    if func_order > 1 && interpolation_type != Ferrite.Serendipity
        first_order = interpolation_type{ndim,r_shape,1}() 
        for (highorderface, firstorderface) in zip(Ferrite.faces(interpolation), Ferrite.faces(first_order))
            for (h_node, f_node) in zip(highorderface, firstorderface)
                @test h_node == f_node
            end
        end
        if ndim > 2
            for (highorderedge, firstorderedge) in zip(Ferrite.edges(interpolation), Ferrite.edges(first_order))
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
