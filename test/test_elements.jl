facts("spring testing") do
    @fact spring1e(3.0) --> [3.0 -3.0; -3.0 3.0]
    @fact spring1s(3.0, [3.0, 1.0]) --> 3.0 * (1.0 - 3.0)
end


facts("plani4e testing") do

    K, f = plani4e([0, 1, 1.5, 0.5], [0.0, 0.2, 0.8, 0.6], [2, 2, 2], hooke(2, 210e9, 0.3), [1.0, 2.5])

    K_calfem =
  1e11*[1.367692307692308   0.067307692307692  -1.012307692307693   0.740384615384616   0.059230769230770  -0.740384615384616  -0.414615384615385  -0.067307692307692;
   0.067307692307692   2.121538461538462   0.336538461538462   0.576153846153845  -0.740384615384616   0.449615384615385   0.336538461538462  -3.147307692307693;
  -1.012307692307693   0.336538461538462   4.340000000000000  -2.759615384615385  -0.414615384615385   0.336538461538462  -2.913076923076923   2.086538461538462;
   0.740384615384616   0.576153846153845  -2.759615384615385   8.163076923076922  -0.067307692307692  -3.147307692307692   2.086538461538461  -5.591923076923075;
   0.059230769230770  -0.740384615384616  -0.414615384615385  -0.067307692307692   1.367692307692308   0.067307692307692  -1.012307692307693   0.740384615384616;
  -0.740384615384616   0.449615384615385   0.336538461538462  -3.147307692307692   0.067307692307692   2.121538461538462   0.336538461538462   0.576153846153845;
  -0.414615384615385   0.336538461538462  -2.913076923076923   2.086538461538461  -1.012307692307693   0.336538461538462   4.340000000000000  -2.759615384615385;
  -0.067307692307692  -3.147307692307693   2.086538461538462  -5.591923076923075   0.740384615384616   0.576153846153845  -2.759615384615385   8.163076923076924]

   f_calfem =
  [0.250;
   0.625;
   0.250;
   0.625;
   0.250;
   0.625;
   0.250;
   0.625]

    @fact norm(K - K_calfem) / norm(K) --> roughly(0.0, atol=1e-15)
    @fact norm(f - f_calfem) / norm(f) --> roughly(0.0, atol=1e-15)


    # Patch test the element:
    # Set up a 4 element patch:
    # 17,18---15,16----13,14
    #   |       |        |
    #  7,8-----5,6-----11,12
    #   |       |        |
    #  1,2-----3,4------9,10
    # Set dirichlet boundary conditions such that u_x = u_y = 0.1x + 0.05y
    # Solve and see that middle node is at correct position
    function patch_test()
        Coord = [0 0
                 1 0
                 1 1
                 0 1
                 2 0
                 2 1
                 2 2
                 1 2
                 0 2]

        Dof = [1 2
               3 4
               5 6
               7 8
               9 10
               11 12
               13 14
               15 16
               17 18]

        Edof = [1 1 2 3 4 5 6 7 8;
                2 3 4 9 10 11 12 5 6;
                3 5 6 11 12 13 14 15 16;
                4 7 8 5 6 15 16 17 18]

        function get_coord(dof)
          node = div(dof+1, 2)
          if dof % 2 == 0
              return Coord[node, 2]
          else
              return Coord[node, 1]
          end
        end

        ux = 0.1
        uy = 0.05
        bc_dofs = [1, 2, 3, 4, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 7, 8]
        bc = zeros(length(bc_dofs), 2)
        for i in 1:size(bc, 1)
          dof = bc_dofs[i]
          node = div(dof+1, 2)
          coord = Coord[node, :]
          bc[i, 1] = dof
          bc[i, 2] = ux * coord[1] + uy * coord[2]
        end

          a = start_assemble()
          D = hooke(2, 250e9, 0.3)
          for e in 1:size(Edof, 1)
            ex = [get_coord(i) for i in Edof[e, 2:2:end]]
            ey = [get_coord(i) for i in Edof[e, 3:2:end]]
            Ke, _ = plani4e(ex, ey, [2, 1, 2], D)
            assemble(Edof[e, :], a, Ke)
          end
          K = end_assemble(a)
          a, _ = solve_eq_sys(K, zeros(18), bc)
          d_free = setdiff(collect(1:18), convert(Vector{Int}, bc[:,1]))
          @fact a[d_free] --> roughly([ux + uy, ux + uy])
      end
      patch_test()

end


facts("bar test") do
    # From example 3.2 in the book Strukturmekanik
    ex = [0.  1.6]; ey = [0. -1.2]
    elem_prop = [200.e9 1.0e-3]
    Ke = bar2e(ex, ey, elem_prop)
    ed = [0. 0. -0.3979 -1.1523]*1e-3
    N = bar2s(ex, ey, elem_prop, ed)
    Ke_ref = [ 64  -48. -64  48
              -48   36   48 -36
              -64   48   64 -48
               48  -36  -48  36]*1e6
    N_ref = 37.306e3
    @fact norm(Ke - Ke_ref) / norm(Ke_ref) --> roughly(0.0, atol=1e-15)
    @fact abs(N - N_ref) / N_ref --> roughly(0.0, atol=1e-15)
end
