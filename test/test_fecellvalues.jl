@testset "FECellValues" begin
for (function_space, quad_rule) in  ((Lagrange{1, RefCube, 1}(), QuadratureRule{1, RefCube}(2)),
                                     (Lagrange{1, RefCube, 2}(), QuadratureRule{1, RefCube}(2)),
                                     (Lagrange{2, RefCube, 1}(), QuadratureRule{2, RefCube}(2)),
                                     (Lagrange{2, RefCube, 2}(), QuadratureRule{2, RefCube}(2)),
                                     (Lagrange{2, RefTetrahedron, 1}(), QuadratureRule{2, RefTetrahedron}(2)),
                                     (Lagrange{2, RefTetrahedron, 2}(), QuadratureRule{2, RefTetrahedron}(2)),
                                     (Lagrange{3, RefCube, 1}(), QuadratureRule{3, RefCube}(2)),
                                     (Serendipity{2, RefCube, 2}(), QuadratureRule{2, RefCube}(2)),
                                     (Lagrange{3, RefTetrahedron, 1}(), QuadratureRule{3, RefTetrahedron}(2)))

    fe_cv = FECellValues(quad_rule, function_space)
    ndim = functionspace_n_dim(function_space)
    n_basefuncs = n_basefunctions(function_space)

    x = valid_coordinates(function_space)
    reinit!(fe_cv, x)

    # We test this by applying a given deformation gradient on all the nodes.
    # Since this is a linear deformation we should get back the exact values
    # from the interpolation.
    u = Vec{ndim, Float64}[zero(Tensor{1,ndim}) for i in 1:n_basefuncs]
    u_scal = zeros(n_basefuncs)
    H = rand(Tensor{2, ndim})
    V = rand(Tensor{1, ndim})
    for i in 1:n_basefuncs
        u[i] = H ⋅ x[i]
        u_scal[i] = V ⋅ x[i]
    end

    for i in 1:length(points(quad_rule))
        @test function_vector_gradient(fe_cv, i, u) ≈ H
        @test function_vector_symmetric_gradient(fe_cv, i, u) ≈ 0.5(H + H')
        @test function_vector_divergence(fe_cv, i, u) ≈ trace(H)
        @test function_scalar_gradient(fe_cv, i, u_scal) ≈ V
        function_scalar_value(fe_cv, i, u_scal)
        function_vector_value(fe_cv, i, u)
    end

    # Test of volume
    vol = 0.0
    for i in 1:length(points(quad_rule))
        vol += detJdV(fe_cv,i)
    end
    @test vol ≈ calculate_volume(function_space, x)

    # Test of utility functions
    @test get_functionspace(fe_cv) == function_space
    @test get_geometricspace(fe_cv) == function_space
    @test get_quadrule(fe_cv) == quad_rule

    # Test quadrature rule after reinit! with ref. coords
    x = reference_coordinates(function_space)
    reinit!(fe_cv, x)
    vol = 0.0
    for i in 1:length(points(quad_rule))
        vol += detJdV(fe_cv,i)
    end
    @test vol ≈ reference_volume(function_space)

end

end # of testset
