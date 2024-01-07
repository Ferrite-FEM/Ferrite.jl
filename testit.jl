using Ferrite

grid = generate_grid(Quadrilateral, (20,20))
dh = DofHandler(grid)
add!(dh, :u, Lagrange{RefQuadrilateral,1}())
close!(dh)
u = zeros(ndofs(dh))
apply_analytical!(u, dh, :u, x -> sinpi(x[1])*cospi(x[2]))

VTKFile("continuous", grid) do vtk
    write_solution(vtk, dh, u)
end

VTKFile("discontinuous", grid; write_discontinuous=true) do vtk
    write_solution(vtk, dh, u)
end

