using JuAFEM

# Element dofs
Edof=[1   1  2  5  6;
      2   5  6  7  8;
      3   3  4  5  6];

E = 2.0e11
A = 5.0e-4
ep = [E A]

Ex = [0.0 1.6
      1.6 1.6
      0.0 1.6]

Ey = [0.0 0.0
      0.0 1.2
      1.2 0.0]

eldraw2(Ex, Ey)

# Allocate space
K = zeros(8, 8)
f = zeros(8, 1)

# Element stiffness matrices
Ke1 = bar2e(Ex[1,:], Ey[1,:], ep)
Ke2 = bar2e(Ex[2,:], Ey[2,:], ep)
Ke3 = bar2e(Ex[3,:], Ey[3,:], ep)

K = assemble(Edof[1,:], K, Ke1)
K = assemble(Edof[2,:], K, Ke2)
K = assemble(Edof[3,:], K, Ke3)

# Boundary conditions
f[6] = -100e3

bc= [1  0
     2  0
     3  0
     4  0
     7  0
     8  0]

# Solve the equation system
(a, r) = solveq(K, f, bc)

# Compute sectional forces
el_disp = extract(Edof, a);

N1 = bar2s(Ex[1,:], Ey[1,:], ep, el_disp[1,:])
N2 = bar2s(Ex[2,:], Ey[2,:], ep, el_disp[2,:])
N3 = bar2s(Ex[3,:], Ey[3,:], ep, el_disp[3,:])

eldisp2(Ex, Ey, el_disp, [1, 2, 0], 50.0)


