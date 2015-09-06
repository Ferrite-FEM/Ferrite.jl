facts("assemble test") do
    K = zeros(3,3)
    Ke = [1 2 3; 4 5 6; 7 8 9]
    edof = [1  3 2 1]
    @fact assemble(edof, K, Ke) --> [9.0 8.0 7.0; 6.0 5.0 4.0; 3.0 2.0 1.0]
    # Test non square Ke
    @fact_throws DimensionMismatch assemble(edof, K, [1 2 3; 4 5 6])
    # Test wrong length of edof
    @fact_throws DimensionMismatch assemble([1 2 3], K, Ke)
end

facts("solve_eq_sys test") do
    K = [ 1 -1  0  0
         -1  3 -2  0
          0 -2  3 -1
          0  0 -1  1]

    f = [0, 0, 1, 0]

    bc = [1 0
          4 0]
    (a, fb) = solve_eq_sys(K, f, bc)
    a_analytic = [0.0, 0.4, 0.6, 0.0]
    fb_analytic = [-0.4, 0.0, 0.0, -0.6]

    @fact norm(a - a_analytic) / norm(a_analytic) --> roughly(0.0, atol=1e-15)
    @fact norm(fb - fb_analytic) / norm(fb_analytic) --> roughly(0.0, atol=1e-15)
    # Test wrong size of K
    @fact_throws DimensionMismatch (a, fb) = solve_eq_sys(K[:,1:2], f, bc)
    # Test wrong length of f
    @fact_throws DimensionMismatch (a, fb) = solve_eq_sys(K, [1, 2], bc)
end

facts("extract_eldisp test") do
    a = [0.0 5.0 7.0 9.0 11.0]
    edof = [1  2 4
            2  3 5]
    @fact extract_eldisp(edof, a) --> [5.0 9.0; 7.0 11.0]
end


facts("inv2x2 test") do
    J = [4. 2.; 3. 8.]
    J_inv = [0.3076923076923077 -0.07692307692307693; -0.11538461538461539 0.15384615384615385]
    @fact norm(inv2x2(J) - J_inv) / norm(J_inv) --> roughly(0.0, atol=1e-15)
end
