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
function WriteVTK.vtk_grid(filename::AbstractString, grid::Grid{dim, T}) where {dim, T}
    
    cls = MeshCell[]
    for (cellgroupid, cellgroup) in enumerate(grid.cellgroups)#CellIterator(grid)
        _celltype = getcelltype(grid, cellgroupid)
        celltype = cell_to_vtkcell(_celltype)
        for (cellid, cells) in enumerate(cellgroup)
            nodes_tuple = getcellnodes(grid, cellgroupid, cellid)
            nodes = Int[]
            for node in nodes_tuple 
                push!(nodes, node)
            end
            push!(cls, MeshCell(celltype, nodes))
        end
    end
    coords = reinterpret(T, getnodes(grid), (dim, getnnodes(grid)))
    return vtk_grid(filename, coords, cls)
end

function WriteVTK.vtk_point_data(vtk::WriteVTK.DatasetFile, data::Vector{Vec{dim, T}}, name::AbstractString) where {dim, T}
    npoints = length(data)
    data = reinterpret(T, data, (dim, npoints))
    return vtk_point_data(vtk, data, name)
end

function vtk_nodeset(vtk::WriteVTK.DatasetFile, grid::Grid{dim}, nodeset::String) where {dim}
    z = zeros(getnnodes(grid))
    z[collect(getnodeset(grid, nodeset))] = 1.0
    vtk_point_data(vtk, z, nodeset)
end

function vtk_cellset(vtk::WriteVTK.DatasetFile, grid::Grid{dim}, cellset::String) where {dim}
    z = zeros(getncells(grid))
    z[collect(getcellset(grid, cellset))] = 1.0
    vtk_cell_data(vtk, z, cellset)
end

