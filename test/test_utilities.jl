import JuAFEM: det_spec, inv_spec

facts("Utility testing") do

context("assemble") do
    K = zeros(3,3)
    Ke = [1 2 3; 4 5 6; 7 8 9]
    edof = [1, 3, 2, 1]
    @fact assemble(edof, K, Ke) --> [9.0 8.0 7.0; 6.0 5.0 4.0; 3.0 2.0 1.0]
    # Test non square Ke
    @fact_throws DimensionMismatch assemble(edof, K, [1 2 3; 4 5 6])
    # Test wrong length of edof
    @fact_throws DimensionMismatch assemble([1, 2, 3], K, Ke)
end

context("solveq") do
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

    @fact norm(a - a_analytic) / norm(a_analytic) --> roughly(0.0, atol=1e-15)
    @fact norm(fb - fb_analytic) / norm(fb_analytic) --> roughly(0.0, atol=1e-15)
    # Test wrong size of K
    @fact_throws DimensionMismatch (a, fb) = solveq(K[:,1:2], f, bc)
    # Test wrong length of f
    @fact_throws DimensionMismatch (a, fb) = solveq(K, [1, 2], bc)
end

context("extract") do
    a = [0.0 5.0 7.0 9.0 11.0]
    edof = [1 2 4
            2 3 5]'
    @fact extract(edof, a) --> [5.0 9.0; 7.0 11.0]'
end

context("linalg") do
    J = [4. 2.; 3. 8.]
    J_inv = [0.3076923076923077 -0.07692307692307693; -0.11538461538461539 0.15384615384615385]
    @fact norm(inv_spec(J) - J_inv) / norm(J_inv) --> roughly(0.0, atol=1e-15)
    @fact det_spec(J) --> roughly(det(J))
    srand(1234)
    J = rand(3,3)
    @fact inv_spec(J) --> roughly(inv(J))
    @fact det_spec(J) --> roughly(det(J))
end


context("gen_quad_mesh") do
    p1 = [0.0, 0.0]
    p2 = [1.0, 1.]
    nelx = 2
    nely = 3
    ndofs = 2
    Edof, Ex, Ey, B1, B2, B3, B4, coord, dof = gen_quad_mesh(p1, p2, nelx, nely, ndofs)

    # Reference values
    Edof_r =
    [1 1 2 3 4 9 10 7 8
     2 3 4 5 6 11 12 9 10
     3 7 8 9 10 15 16 13 14
     4 9 10 11 12 17 18 15 16
     5 13 14 15 16 21 22 19 20
     6 15 16 17 18 23 24 21 22]'
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
    @fact Edof --> Edof_r
    @fact norm(Ex - Ex_r) / norm(Ex_r) --> roughly(0.0, atol=1e-15)
    @fact norm(Ey - Ey_r) / norm(Ey_r) --> roughly(0.0, atol=1e-15)
    @fact B1 --> B1_r
    @fact B2 --> B2_r
    @fact B3 --> B3_r
    @fact B4 --> B4_r
end

context("static condensation") do
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

    @fact K1 --> roughly(K1_calfem)
    @fact f1 --> roughly(f1_calfem)
end

context("coordxtr + topologyxtr") do

    Dof = [1 2;
           3 4;
           5 6;
           7 8]'

    Coord = [1.0 3.0;
             2.0 4.0;
             5.0 6.0
             7.0 8.0]'

    Edof = [1 1 2 3 4 5 6;
            2 1 2 3 4 7 8]'

    Ex, Ey, Ez = coordxtr(Edof,Coord,Dof, 3)

    @fact Ex --> [1.0 2.0 5.0;
                  1.0 2.0 7.0]'

    @fact Ey --> [3.0 4.0 6.0;
                  3.0 4.0 8.0]'

    topo = topologyxtr(Edof,Coord,Dof, 3)

    @fact topo --> [1 2 3;
                    1 2 4]'
end


end
