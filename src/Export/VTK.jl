cell_to_vtkcell(::Type{Line}) = VTKCellTypes.VTK_LINE
cell_to_vtkcell(::Type{QuadraticLine}) = VTKCellTypes.VTK_QUADRATIC_EDGE

cell_to_vtkcell(::Type{Quadrilateral}) = VTKCellTypes.VTK_QUAD
cell_to_vtkcell(::Type{QuadraticQuadrilateral}) = VTKCellTypes.VTK_BIQUADRATIC_QUAD
cell_to_vtkcell(::Type{Triangle}) = VTKCellTypes.VTK_TRIANGLE
cell_to_vtkcell(::Type{QuadraticTriangle}) = VTKCellTypes.VTK_QUADRATIC_TRIANGLE
cell_to_vtkcell(::Type{Cell{2,8,4}}) = VTKCellTypes.VTK_QUADRATIC_QUAD

cell_to_vtkcell(::Type{Hexahedron}) = VTKCellTypes.VTK_HEXAHEDRON
cell_to_vtkcell(::Type{Tetrahedron}) = VTKCellTypes.VTK_TETRA
cell_to_vtkcell(::Type{QuadraticTetrahedron}) = VTKCellTypes.VTK_QUADRATIC_TETRA

"""
```julia
vtk_grid(filename::AbstractString, grid::Grid)
```

Create a unstructured VTK grid from a `Grid`. Return a `DatasetFile`
which data can be appended to, see `vtk_point_data`, `vtk_cell_data`.
"""
function WriteVTK.vtk_grid{dim, N, T}(filename::AbstractString, grid::Grid{dim, N, T})
    celltype = cell_to_vtkcell(getcelltype(grid))
    cls = MeshCell[]
    for cell in CellIterator(grid)
        push!(cls, MeshCell(celltype, getnodes(cell)))
    end
    coords = reinterpret(T, getnodes(grid), (dim, getnnodes(grid)))
    return vtk_grid(filename, coords, cls)
end

function WriteVTK.vtk_point_data{dim, T}(vtk::WriteVTK.DatasetFile, data::Vector{Vec{dim, T}}, name::AbstractString)
    npoints = length(data)
    data = reinterpret(T, data, (dim, npoints))
    return vtk_point_data(vtk, data, name)
end

function vtk_nodeset{dim}(vtk::WriteVTK.DatasetFile, grid::Grid{dim}, nodeset::String)
    z = zeros(getnnodes(grid))
    z[getnodeset(grid, nodeset)] = 1.0
    vtk_point_data(vtk, z, nodeset)
end

function vtk_cellset{dim}(vtk::WriteVTK.DatasetFile, grid::Grid{dim}, cellset::String)
    z = zeros(getncells(grid))
    z[getcellset(grid, cellset)] = 1.0
    vtk_cell_data(vtk, z, cellset)
end

