using JuAFEM

grid = generate_grid(Triangle, (2, 2))
dh = DofHandler(grid)
quad = Lagrange{2,RefTetrahedron,2}()
lin = Lagrange{2,RefTetrahedron,1}()
push!(dh, :u, 2, quad)
push!(dh, :p, 1, lin)
close!(dh)

ch = ConstraintHandler(dh)
add!(ch, Dirichlet(:u, getfaceset(grid, "left"), (x,t) -> x, [1, 2]))
add!(ch, Dirichlet(:u, getfaceset(grid, "bottom"), (x,t) -> x[1], [1]))
add!(ch, Dirichlet(:u, getfaceset(grid, "right"), (x,t) -> x[2], 2))
add!(ch, Dirichlet(:p, getfaceset(grid, "top"), (x,t) -> x[1]))
close!(ch)



ch2 = ConstraintHandler2(dh)
add!(ch2, Dirichlet2(:u, getfaceset(grid, "left"), (x,t) -> x, [1, 2]))
add!(ch2, Dirichlet2(:u, getfaceset(grid, "bottom"), (x,t) -> x[1], [1]))
add!(ch2, Dirichlet2(:u, getfaceset(grid, "right"), (x,t) -> x[2], 2))
add!(ch2, Dirichlet2(:p, getfaceset(grid, "top"), (x,t) -> x[1]))
close!(ch2)
