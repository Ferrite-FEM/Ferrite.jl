@testset "interpolations" begin

for interpolation in (Lagrange{1, RefCube, 1}(),
                      Lagrange{1, RefCube, 2}(),
                      Lagrange{2, RefCube, 1}(),
                      Lagrange{2, RefCube, 2}(),
                      Lagrange{2, RefTetrahedron, 1}(),
                      Lagrange{2, RefTetrahedron, 2}(),
                      Lagrange{3, RefCube, 1}(),
                      Serendipity{2, RefCube, 2}(),
                      Lagrange{3, RefTetrahedron, 1}(),
                      Lagrange{3, RefTetrahedron, 2}())

    # Test of utility functions
    ndim = JuAFEM.getdim(interpolation)
    r_shape = JuAFEM.getrefshape(interpolation)
    func_order = JuAFEM.getorder(interpolation)
    @test typeof(interpolation) <: Interpolation{ndim,r_shape,func_order}
    @test typeof(JuAFEM.getlowerdim(interpolation)) <: Interpolation{ndim-1}
    @test typeof(JuAFEM.getlowerorder(interpolation)) <: Interpolation{ndim,r_shape,func_order-1}

    n_basefuncs = getnbasefunctions(interpolation)
    x = rand(Tensor{1, ndim})
    f = (x) -> JuAFEM.value(interpolation, Tensor{1, ndim}(x))
    @test vec(ForwardDiff.jacobian(f, Array(x))') ≈
           reinterpret(Float64, JuAFEM.derivative(interpolation, x), (ndim * n_basefuncs,))
    @test sum(JuAFEM.value(interpolation, x)) ≈ 1.0

    coords = reference_coordinates(interpolation)
    for node in 1:n_basefuncs
        N_node = JuAFEM.value(interpolation, coords[node])
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
