import JuAFEM: Square, Triangle

using ForwardDiff

@testset "Utility testing" begin

@testset "assemble" begin
    K = zeros(3,3)
    Ke = [1 2 3; 4 5 6; 7 8 9]
    edof = [3, 2, 1]
    @test assemble(edof, K, Ke) == [9.0 8.0 7.0; 6.0 5.0 4.0; 3.0 2.0 1.0]
    # Test non square Ke
    @test_throws DimensionMismatch assemble(edof, K, [1 2 3; 4 5 6])
    # Test wrong length of edof
    @test_throws DimensionMismatch assemble([1, 2, 3, 4], K, Ke)
end

@testset "solveq" begin
    K = [ 1 -1  0  0
         -1  3 -2  0
          0 -2  3 -1
          0  0 -1  1]

    f = [0, 0, 1, 0]

    bc = [1 0
          4 0]
    (a, fb) = solveq(K, f, bc)
    a_analytic = [0.0, 0.4, 0.6, 0.0]
    fb_analytic = [-0.4, 0.0, 0.0, -0.6]

    @test norm(a - a_analytic) / norm(a_analytic) < 1e-15
    @test norm(fb - fb_analytic) / norm(fb_analytic) < 1e-15
    # Test wrong size of K
    @test_throws DimensionMismatch (a, fb) = solveq(K[:,1:2], f, bc)
    # Test wrong length of f
    @test_throws DimensionMismatch (a, fb) = solveq(K, [1, 2], bc)
end

@testset "extract" begin
    a = [0.0 5.0 7.0 9.0 11.0]
    edof = [2 4
            3 5]'
    @test extract(edof, a) == [5.0 9.0; 7.0 11.0]'
end

@testset "gen_quad_mesh" begin
    p1 = [0.0, 0.0]
    p2 = [1.0, 1.]
    nelx = 2
    nely = 3
    ndofs = 2
    Edof, Ex, Ey, B1, B2, B3, B4, coord, dof = gen_quad_mesh(p1, p2, nelx, nely, ndofs)

    # Reference values
    Edof_r =
    [1 2 3 4 9 10 7 8
     3 4 5 6 11 12 9 10
     7 8 9 10 15 16 13 14
     9 10 11 12 17 18 15 16
     13 14 15 16 21 22 19 20
     15 16 17 18 23 24 21 22]'

    Ex_r =
    [0.0 0.5 0.5 0.0
     0.5 1.0 1.0 0.5
     0.0 0.5 0.5 0.0
     0.5 1.0 1.0 0.5
     0.0 0.5 0.5 0.0
     0.5 1.0 1.0 0.5]'

    Ey_r =
    [0.0 0.0 0.3333333333333333 0.3333333333333333
     0.0 0.0 0.3333333333333333 0.3333333333333333
     0.3333333333333333 0.3333333333333333 0.6666666666666666 0.6666666666666666
     0.3333333333333333 0.3333333333333333 0.6666666666666666 0.6666666666666666
     0.6666666666666666 0.6666666666666666 1.0 1.0
     0.6666666666666666 0.6666666666666666 1.0 1.0]'

    B1_r = [1 2; 3 4; 5 6]'
    B2_r = [5 6; 11 12; 17 18; 23 24]'
    B3_r = [23 24; 21 22; 19 20]'
    B4_r = [19 20; 13 14; 7 8; 1 2]'
    @test Edof == Edof_r
    @test norm(Ex - Ex_r) / norm(Ex_r) < 1e-15
    @test norm(Ey - Ey_r) / norm(Ey_r) < 1e-15
    @test B1 == B1_r
    @test B2 == B2_r
    @test B3 == B3_r
    @test B4 == B4_r
end

@testset "static condensation" begin
    K = [1 2 3 4;
         4 5 6 7;
         7 8 9 10;
         11 12 13 14]

    f = [1, 2, 3, 4]

    cd = [1, 2]

    K1, f1 = statcon(K, f, cd)

    K1_calfem = [0.000000000000004  -2.999999999999993;
                2.666666666666661  -0.333333333333346]

    f1_calfem = [
       0.000000000000001
      -0.333333333333334]

    @test K1 ≈ K1_calfem
    @test f1 ≈ f1_calfem
end

@testset "coordxtr + topologyxtr" begin

    Dof = [1 2;
           3 4;
           5 6;
           7 8]'

    Coord = [1.0 3.0;
             2.0 4.0;
             5.0 6.0
             7.0 8.0]'

    Edof = [1 2 3 4 5 6;
            1 2 3 4 7 8]'

    Ex, Ey, Ez = coordxtr(Edof,Coord,Dof, 3)

    @test Ex == [1.0 2.0 5.0;
                  1.0 2.0 7.0]'

    @test Ey == [3.0 4.0 6.0;
                  3.0 4.0 8.0]'

    topo = topologyxtr(Edof,Coord,Dof, 3)

    @test topo == [1 2 3;
                    1 2 4]'
end

@testset "function space derivatives and sums" begin

    for functionspace in (Lagrange{1, Square, 1}(),
                          Lagrange{1, Square, 2}(),
                          Lagrange{2, Square, 1}(),
                          Lagrange{2, Triangle, 1}(),
                          Lagrange{2, Triangle, 2}(),
                          Lagrange{3, Square, 1}(),
                          Serendipity{2, Square, 2}())
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

end
