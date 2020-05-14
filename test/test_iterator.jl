@testset "iterators" begin

grid = generate_grid(Triangle, (3, 1))
dh = DofHandler(grid)
push!(dh, :u, 3)
close!(dh)

mdh = MixedDofHandler(grid)
push!(mdh, :u, 3)
close!(mdh)

dim = JuAFEM.getdim(grid)

for _iterator in [CellIterator(dh), 
                  CellIterator(dh, 2:3), 
                  CellIterator(grid), 
                  CellIterator(grid, 2:3),
                  CellIterator(mdh), 
                  CellIterator(mdh, 2:3)]

    for celldata in _iterator
        coords = zeros(Vec{dim}, JuAFEM.nnodes_per_cell(grid, cellid(celldata)))
        getcoordinates!(coords, grid, cellid(celldata))
        @test all(getcoordinates(celldata) .== coords)
        @test all(getnodes(celldata) .== grid.cells[cellid(celldata)].nodes)
    end

end


#Cant loop over cellset with different celltypes
grid = get_2d_grid()
# WHEN: adding a scalar field for each cell and generating dofs
field1 = create_field(name=:u, spatial_dim=2, field_dim=1, order=1, cellshape=RefCube)
field2 = create_field(name=:u, spatial_dim=2, field_dim=1, order=1, cellshape=RefTetrahedron)
mdh = MixedDofHandler(grid);
push!(mdh, FieldHandler([field1], Set(1)));
push!(mdh, FieldHandler([field2], Set(2)));
close!(mdh)

@test_throws ErrorException CellIterator(mdh)
@test_throws ErrorException CellIterator(grid)
CellIterator(mdh, [1]) #does not throw error
CellIterator(mdh, [2]) #does not throw error
CellIterator(grid, [1]) #does not throw error
CellIterator(grid, [2]) #does not throw error


end