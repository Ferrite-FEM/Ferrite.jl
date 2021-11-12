@testset "interpolations" begin

for interpolation in (Lagrange{1, RefCube, 1}(),
                      Lagrange{1, RefCube, 2}(),
                      Lagrange{2, RefCube, 1}(),
                      Lagrange{2, RefCube, 2}(),
                      Lagrange{2, RefTetrahedron, 1}(),
                      Lagrange{2, RefTetrahedron, 2}(),
                      Lagrange{3, RefCube, 1}(),
                      Lagrange{3, RefCube, 2}(),
                      Serendipity{2, RefCube, 2}(),
                      Serendipity{3, RefCube, 2}(),
                      Lagrange{3, RefTetrahedron, 1}(),
                      Lagrange{3, RefTetrahedron, 2}(),
                      #
                      DiscontinuousLagrange{1, RefCube, 0}(),
                      DiscontinuousLagrange{2, RefCube, 0}(),
                      DiscontinuousLagrange{3, RefCube, 0}(),
                      DiscontinuousLagrange{2, RefTetrahedron, 0}(),
                      DiscontinuousLagrange{3, RefTetrahedron, 0}(),
                      #
                      BubbleEnrichedLagrange{2,RefTetrahedron,1}(),
                      #
                      CrouzeixRaviart{2,1}(),)

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

    coords = Ferrite.reference_coordinates(interpolation)
    for node in 1:n_basefuncs
        N_node = Ferrite.value(interpolation, coords[node])
        for k in 1:node
            if k == node
                @test N_node[k] ≈ 1.0
            else
                @test N_node[k] ≈ 0.0
            end
        end
    end
end

end
