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
    vtk_grid(filename::AbstractString, grid::Grid)

Create a unstructured VTK grid from a `Grid`. Return a `DatasetFile`
which data can be appended to, see `vtk_point_data` and `vtk_cell_data`.
"""
function WriteVTK.vtk_grid(filename::AbstractString, grid::Grid{dim,N,T}) where {dim,N,T}
    celltype = cell_to_vtkcell(getcelltype(grid))
    cls = MeshCell[]
    for cell in CellIterator(grid)
        push!(cls, MeshCell(celltype, copy(getnodes(cell))))
    end
    coords = reshape(reinterpret(T, getnodes(grid)), (dim, getnnodes(grid)))
    return vtk_grid(filename, coords, cls)
end

"""
    vtk_point_data(vtk, data::Vector{<:Vec}, name)

Write the vector field data to the vtk file.
"""
function WriteVTK.vtk_point_data(vtk::WriteVTK.DatasetFile, data::Vector{Vec{dim,T}}, name::AbstractString) where {dim,T}
    npoints = length(data)
    data = reshape(reinterpret(T, data), (dim, npoints))
    return vtk_point_data(vtk, data, name)
end

function vtk_nodeset(vtk::WriteVTK.DatasetFile, grid::Grid{dim}, nodeset::String) where {dim}
    z = zeros(getnnodes(grid))
    z[collect(getnodeset(grid, nodeset))] .= 1.0
    vtk_point_data(vtk, z, nodeset)
end

"""
    vtk_cellset(vtk, grid::Grid)

Export all cell sets in the grid. Each cell set is exported with
`vtk_cell_data` with value 1 if the cell is in the set, and 0 otherwise.
"""
function vtk_cellset(vtk::WriteVTK.DatasetFile, grid::Grid, cellsets=keys(grid.cellsets))
    z = zeros(getncells(grid))
    for cellset in cellsets
        z .= 0.0
        z[collect(getcellset(grid, cellset))] .= 1.0
        vtk_cell_data(vtk, z, cellset)
    end
    return vtk
end

"""
    vtk_cellset(vtk, grid::Grid, cellset::String)

Export the cell set specified by `cellset` as cell data with value 1 if
the cell is in the set and 0 otherwise.
"""
vtk_cellset(vtk::WriteVTK.DatasetFile, grid::Grid, cellset::String) =
    vtk_cellset(vtk, grid, [cellset])

function WriteVTK.vtk_grid(filename::AbstractString, grid::MixedGrid{dim,C,T}) where {dim,C,T}
    #celltype = cell_to_vtkcell(getcelltype(grid))
    cls = MeshCell[]
    #for cell in CellIterator(grid)
    for cell in grid.cells
        celltype = JuAFEM.cell_to_vtkcell(typeof(cell))
        push!(cls, MeshCell(celltype, collect(cell.nodes)))
    end
    coords = reshape(reinterpret(T, getnodes(grid)), (dim, getnnodes(grid)))
    return vtk_grid(filename, coords, cls)
end

import JuAFEM.field_offset
function WriteVTK.vtk_point_data(vtkfile, dh::MixedDofHandler, u::Vector, suffix="")
    for f in 1:length(dh.field_names)  # TODO nfields(dh) funkar inte i Main-scope - kolla upp
        @debug println("exporting field $(dh.field_names[f])")
        field_dim = dh.field_dims[f]
        space_dim = field_dim == 2 ? 3 : field_dim
        data = fill(0.0, space_dim, getnnodes(dh.grid))
        offset = field_offset(dh, dh.field_names[f])
        #for cell in CellIterator(dh)
        for (cellnum, cell) in enumerate(dh.grid.cells)
            n = ndofs_per_cell(dh, cellnum)
            eldofs = zeros(Int, n)
            _celldofs = celldofs!(eldofs, dh, cellnum)
            counter = 1
            for node in cell.nodes
                for d in 1:dh.field_dims[f]
                    data[d, node] = u[_celldofs[counter + offset]]
                    @debug println("  exporting $(u[_celldofs[counter + offset]]) for dof#$(_celldofs[counter + offset])")
                    counter += 1
                end
            end
        end
        vtk_point_data(vtkfile, data, string(dh.field_names[f], suffix))
    end
    return vtkfile
end
