using Ferrite

function get_2d_grid()
    # GIVEN: two cells, a quad and a triangle sharing one face
    cells = [
        Quadrilateral((1, 2, 3, 4)),
        Triangle((3, 2, 5))
        ]
    coords = zeros(Vec{2,Float64}, 5)
    nodes = [Node(coord) for coord in zeros(Vec{2,Float64}, 5)]
    return Grid(cells,nodes)
end

grid = get_2d_grid()
dh = DofHandler(grid);
sdh1 = SubDofHandler(dh, Set([1,2]))
add!(sdh1, :u, Lagrange{Ferrite.RefAnyshape{2},1}())
close!(dh)