@testset "function spaces" begin

for functionspace in (Lagrange{1, RefCube, 1}(),
                      Lagrange{1, RefCube, 2}(),
                      Lagrange{2, RefCube, 1}(),
                      Lagrange{2, RefCube, 2}(),
                      Lagrange{2, RefTetrahedron, 1}(),
                      Lagrange{2, RefTetrahedron, 2}(),
                      Lagrange{3, RefCube, 1}(),
                      Serendipity{2, RefCube, 2}(),
                      Lagrange{3, RefTetrahedron, 1}())

    # Test of utility functions
    ndim = getdim(functionspace)
    r_shape = getrefshape(functionspace)
    func_order = getorder(functionspace)
    @test typeof(functionspace) <: FunctionSpace{ndim,r_shape,func_order}
    @test typeof(JuAFEM.getlowerdim(functionspace)) <: FunctionSpace{ndim-1}
    @test typeof(JuAFEM.getlowerorder(functionspace)) <: FunctionSpace{ndim,r_shape,func_order-1}

    n_basefuncs = getnbasefunctions(functionspace)
    x = rand(Tensor{1, ndim})
    f = (x) -> JuAFEM.value(functionspace, Tensor{1, ndim}(x))
    @test vec(ForwardDiff.jacobian(f, Array(x))') ≈
           reinterpret(Float64, JuAFEM.derivative(functionspace, x), (ndim * n_basefuncs,))
    @test sum(JuAFEM.value(functionspace, x)) ≈ 1.0

    coords = reference_coordinates(functionspace)
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