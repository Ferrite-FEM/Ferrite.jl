using ForwardDiff

@testset "fevalues" begin

@testset "function space derivatives and sums" begin

for functionspace in (Lagrange{1, RefCube, 1}(),
                      Lagrange{1, RefCube, 2}(),
                      Lagrange{2, RefCube, 1}(),
                      Lagrange{2, RefTetrahedron, 1}(),
                      Lagrange{2, RefTetrahedron, 2}(),
                      Lagrange{3, RefCube, 1}(),
                      Serendipity{2, RefCube, 2}(),
                      Lagrange{3, RefTetrahedron, 1}())
    ndim = n_dim(functionspace)
    n_basefuncs = n_basefunctions(functionspace)
    x = rand(Tensor{1, ndim})
    f = (x) -> JuAFEM.value(functionspace, Tensor{1, ndim}(x))
    @test vec(ForwardDiff.jacobian(f, extract_components(x))') ≈
           reinterpret(Float64, JuAFEM.derivative(functionspace, x), (ndim * n_basefuncs,))
    @test sum(JuAFEM.value(functionspace, x)) ≈ 1.0

    coords = JuAFEM.reference_coordinates(functionspace)
    for node in 1:n_basefuncs
        N_node = JuAFEM.value(functionspace, coords[node])
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


@testset "function interpolations" begin


    for (function_space, quad_rule) in  ((Lagrange{1, RefCube, 1}(), QuadratureRule(Dim{1}, RefCube(), 2)),
                                         (Lagrange{1, RefCube, 2}(), QuadratureRule(Dim{1}, RefCube(), 2)),
                                         (Lagrange{2, RefCube, 1}(), QuadratureRule(Dim{2}, RefCube(), 2)),
                                         (Lagrange{2, RefTetrahedron, 1}(), QuadratureRule(Dim{2}, RefTetrahedron(), 2)),
                                         (Lagrange{2, RefTetrahedron, 2}(), QuadratureRule(Dim{2}, RefTetrahedron(), 2)),
                                         (Lagrange{3, RefCube, 1}(), QuadratureRule(Dim{3}, RefCube(), 2)),
                                         (Serendipity{2, RefCube, 2}(), QuadratureRule(Dim{2}, RefCube(), 2)),
                                         (Lagrange{3, RefTetrahedron, 1}(), QuadratureRule(Dim{3}, RefTetrahedron(), 2)))

        fev = FEValues(quad_rule, function_space)
        ndim = n_dim(function_space)
        n_basefuncs = n_basefunctions(function_space)
        x = Vec{ndim, Float64}[rand(Tensor{1,ndim}) for i in 1:n_basefuncs]
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

        for i in 1:length(JuAFEM.points(quad_rule))
            @test function_vector_gradient(fev, i, u) ≈ H
            @test function_vector_symmetric_gradient(fev, i, u) ≈ 0.5(H + H')
            @test function_vector_divergence(fev, i, u) ≈ trace(H)
            @test function_scalar_gradient(fev, i, u_scal) ≈ V
            function_scalar_value(fev, i, u_scal)
            function_vector_value(fev, i, u)
        end
    end

end
end
