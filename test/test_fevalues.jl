@testset "FEValues" begin
# For testing equality between constructors
function Base.:(==)(fev1::FEValues, fev2::FEValues)
    fev1.N == fev2.N &&
    fev1.dNdx == fev2.dNdx &&
    fev1.dNdξ == fev2.dNdξ &&
    fev1.detJdV == fev2.detJdV &&
    fev1.quad_rule == fev2.quad_rule &&
    fev1.function_space == fev2.function_space &&
    fev1.dMdξ == fev2.dMdξ &&
    fev1.geometric_space == fev2.geometric_space
end

for (function_space, quad_rule) in  ((Lagrange{1, RefCube, 1}(), QuadratureRule(Dim{1}, RefCube(), 2)),
                                     (Lagrange{1, RefCube, 2}(), QuadratureRule(Dim{1}, RefCube(), 2)),
                                     (Lagrange{2, RefCube, 1}(), QuadratureRule(Dim{2}, RefCube(), 2)),
                                     (Lagrange{2, RefCube, 2}(), QuadratureRule(Dim{2}, RefCube(), 2)),
                                     (Lagrange{2, RefTetrahedron, 1}(), QuadratureRule(Dim{2}, RefTetrahedron(), 2)),
                                     (Lagrange{2, RefTetrahedron, 2}(), QuadratureRule(Dim{2}, RefTetrahedron(), 2)),
                                     (Lagrange{3, RefCube, 1}(), QuadratureRule(Dim{3}, RefCube(), 2)),
                                     (Serendipity{2, RefCube, 2}(), QuadratureRule(Dim{2}, RefCube(), 2)),
                                     (Lagrange{3, RefTetrahedron, 1}(), QuadratureRule(Dim{3}, RefTetrahedron(), 2)))

    fev = FEValues(quad_rule, function_space)
    @test fev == FEValues(Float64, quad_rule, function_space)
    @test fev == FEValues(quad_rule, function_space, function_space)
    ndim = n_dim(function_space)
    n_basefuncs = n_basefunctions(function_space)

    function valid_nodes(fs::JuAFEM.FunctionSpace)
        x = JuAFEM.reference_coordinates(fs)
        return [x[i] + 0.1 * rand(typeof(x[i])) for i in 1:length(x)]
    end

    x = valid_nodes(function_space)
    reinit!(fev, x)

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
        @test function_vector_gradient(fev, i, u) ≈ H
        @test function_vector_symmetric_gradient(fev, i, u) ≈ 0.5(H + H')
        @test function_vector_divergence(fev, i, u) ≈ trace(H)
        @test function_scalar_gradient(fev, i, u_scal) ≈ V
        function_scalar_value(fev, i, u_scal)
        function_vector_value(fev, i, u)
    end

    # Test of volume
    vol = 0.0
    for i in 1:length(points(quad_rule))
        vol += detJdV(fev,i)
    end
    # @test vol ≈ JuAFEM.reference_volume(function_space) # TODO: Add function that calculates the volume for an object

    # Test of utility functions
    @test get_functionspace(fev) == function_space
    @test get_geometricspace(fev) == function_space
    @test get_quadrule(fev) == quad_rule

    # Test quadrature rule after reinit! with ref. coords
    x = JuAFEM.reference_coordinates(function_space)
    reinit!(fev, x)
    vol = 0.0
    for i in 1:length(points(quad_rule))
        vol += detJdV(fev,i)
    end
    @test vol ≈ JuAFEM.reference_volume(function_space)

end

end
