@testset "FEFaceValues" begin
for (function_space, quad_rule) in  ((Lagrange{1, RefCube, 1}(), QuadratureRule(Dim{0}, RefCube(), 2)),
                                     (Lagrange{1, RefCube, 2}(), QuadratureRule(Dim{0}, RefCube(), 2)),
                                     (Lagrange{2, RefCube, 1}(), QuadratureRule(Dim{1}, RefCube(), 2)),
                                     (Lagrange{2, RefCube, 2}(), QuadratureRule(Dim{1}, RefCube(), 2)),
                                     (Lagrange{2, RefTetrahedron, 1}(), QuadratureRule(Dim{1}, RefTetrahedron(), 2)),
                                     (Lagrange{2, RefTetrahedron, 2}(), QuadratureRule(Dim{1}, RefTetrahedron(), 2)),
                                     (Lagrange{3, RefCube, 1}(), QuadratureRule(Dim{2}, RefCube(), 2)),
                                     (Serendipity{2, RefCube, 2}(), QuadratureRule(Dim{1}, RefCube(), 2)),
                                     (Lagrange{3, RefTetrahedron, 1}(), QuadratureRule(Dim{2}, RefTetrahedron(), 2)))

    fefv = FEFaceValues(quad_rule, function_space)
    ndim = n_dim(function_space)
    n_basefuncs = n_basefunctions(function_space)

    function valid_nodes(fs::JuAFEM.FunctionSpace)
        x = JuAFEM.reference_coordinates(fs)
        return [x[i] + 0.1 * rand(typeof(x[i])) for i in 1:length(x)]
    end

    x = valid_nodes(function_space)
    for boundary in 1:JuAFEM.n_boundaries(function_space)
        reinit!(fefv, x, boundary)
        boundary_quad_rule = get_quadrule(fefv)
        @test JuAFEM.current_boundary(fefv) == boundary

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

        for i in 1:length(points(boundary_quad_rule))
            @test function_vector_gradient(fefv, i, u) ≈ H
            @test function_vector_symmetric_gradient(fefv, i, u) ≈ 0.5(H + H')
            @test function_vector_divergence(fefv, i, u) ≈ trace(H)
            @test function_scalar_gradient(fefv, i, u_scal) ≈ V
            function_scalar_value(fefv, i, u_scal)
            function_vector_value(fefv, i, u)
        end

        # Test of volume
        vol = 0.0
        for i in 1:length(points(boundary_quad_rule))
            vol += detJdV(fefv,i)
        end
        # @test vol ≈ JuAFEM.reference_volume(function_space, boundary) # TODO: Add function that calculates the volume for an object

        # Test of utility functions
        @test get_functionspace(fefv) == function_space
        @test get_geometricspace(fefv) == function_space

        # Test quadrature rule after reinit! with ref. coords
        x = JuAFEM.reference_coordinates(function_space)
        reinit!(fefv, x, boundary)
        vol = 0.0
        for i in 1:length(points(boundary_quad_rule))
            vol += detJdV(fefv, i)
        end
        @test vol ≈ JuAFEM.reference_volume(function_space, boundary)
    end

end

end # of testset
