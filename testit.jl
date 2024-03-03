using Ferrite

grid = generate_grid(Quadrilateral, (20,20))
dh = DofHandler(grid)
add!(dh, :u, Lagrange{RefQuadrilateral,1}())
close!(dh)
u = zeros(ndofs(dh))
f(x) = sinpi(x[1])*cospi(x[2])
f(x) = x[1] + x[2]
apply_analytical!(u, dh, :u, f)

VTKFile("continuous", grid) do vtk
    write_solution(vtk, dh, u)
end

VTKFile("discontinuous", grid; write_discontinuous=true) do vtk
    write_solution(vtk, dh, u)
end

dh = DofHandler(grid)
ip = DiscontinuousLagrange{RefQuadrilateral,1}()
add!(dh, :u, ip)
close!(dh)

function calculate_u(dh, ip, fun)
    u = zeros(ndofs(dh))
    ref_points = Ferrite.reference_coordinates(ip)
    qr = QuadratureRule{RefQuadrilateral}(zeros(length(ref_points)), ref_points)
    cv = CellValues(qr, ip)
    for cell in CellIterator(dh)
        reinit!(cv, cell)
        for q_point in 1:getnquadpoints(qr)
            x = spatial_coordinate(cv, q_point, getcoordinates(cell))
            v = fun(x)
            u[celldofs(cell)[q_point]] = v + cellid(cell)/getncells(dh.grid)
        end
    end
    return u
end

u = calculate_u(dh, ip, f)

VTKFile("really_discontinuous", grid; write_discontinuous=true) do vtk
    write_solution(vtk, dh, u)
end

VTKFile("not_discontinuous", grid) do vtk
    write_solution(vtk, dh, u)
end

VTKFile("really_discontinuous_auto", dh) do vtk
    write_solution(vtk, dh, u)
end
