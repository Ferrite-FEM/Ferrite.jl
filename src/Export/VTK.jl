module VTK

import WriteVTK

using ..JuAFEM
import ..JuAFEM: @debug

"""
    VTK.save(vtk)

Save the vtk-file `vtk` to disk.
"""
save(vtk) = WriteVTK.vtk_save(vtk)

# Grid utility
cell_to_vtkcell(::Type{Line}) = WriteVTK.VTKCellTypes.VTK_LINE
cell_to_vtkcell(::Type{QuadraticLine}) = WriteVTK.VTKCellTypes.VTK_QUADRATIC_EDGE

cell_to_vtkcell(::Type{Quadrilateral}) = WriteVTK.VTKCellTypes.VTK_QUAD
cell_to_vtkcell(::Type{QuadraticQuadrilateral}) = WriteVTK.VTKCellTypes.VTK_BIQUADRATIC_QUAD
cell_to_vtkcell(::Type{Triangle}) = WriteVTK.VTKCellTypes.VTK_TRIANGLE
cell_to_vtkcell(::Type{QuadraticTriangle}) = WriteVTK.VTKCellTypes.VTK_QUADRATIC_TRIANGLE
cell_to_vtkcell(::Type{Cell{2,8,4}}) = WriteVTK.VTKCellTypes.VTK_QUADRATIC_QUAD

cell_to_vtkcell(::Type{Hexahedron}) = WriteVTK.VTKCellTypes.VTK_HEXAHEDRON
cell_to_vtkcell(::Type{Tetrahedron}) = WriteVTK.VTKCellTypes.VTK_TETRA
cell_to_vtkcell(::Type{QuadraticTetrahedron}) = WriteVTK.VTKCellTypes.VTK_QUADRATIC_TETRA

"""
    VTK.grid(filename::AbstractString, grid::Grid)

Create a unstructured VTK grid from a `Grid`.
See [`VTK.point_data`](@ref), [`VTK.cell_data`](@ref),
[`VTK.nodeset`](@ref), [`VTK.cellset`](@ref).
"""
function grid(filename::AbstractString, grid::Grid{dim,N,T}) where {dim,N,T}
    celltype = cell_to_vtkcell(getcelltype(grid))
    cls = WriteVTK.MeshCell[]
    for cell in CellIterator(grid)
        push!(cls, WriteVTK.MeshCell(celltype, copy(getnodes(cell))))
    end
    coords = reinterpret(T, getnodes(grid), (dim, getnnodes(grid)))
    return WriteVTK.vtk_grid(filename, coords, cls)
end

"""
    VTK.grid(filename::AbstractString, dh::DofHandler)

Create a unstructured VTK grid from the grid in the DofHandler.
See [`VTK.point_data`](@ref), [`VTK.cell_data`](@ref),
[`VTK.nodeset`](@ref), [`VTK.cellset`](@ref).
"""
grid(filename::AbstractString, dh::DofHandler) = WriteVTK.vtk_grid(filename, dh.grid)

# for do-blocks
function grid(f::Function, args...; kwargs...)
    vtk = grid(args...; kwargs...)
    try
        f(vtk)
    finally
        return save(vtk)
    end
end

"""
    VTK.nodeset(vtk, grid::Grid, setname::String)

Write a point data field with the value of the nodes in `setname` set to 1,
and the rest of the nodes to 0.
"""
function nodeset(vtk::WriteVTK.DatasetFile, grid::Grid{dim}, setname::String) where {dim}
    z = zeros(getnnodes(grid))
    z[collect(getnodeset(grid, setname))] = 1.0
    WriteVTK.vtk_point_data(vtk, z, setname)
end

"""
    VTK.cellset(vtk, grid::Grid, setname::String)

Write a cell data field with the value of the cells in `setname` set to 1,
and the rest of the cells to 0.
"""
function vtk_cellset(vtk::WriteVTK.DatasetFile, grid::Grid{dim}, setname::String) where {dim}
    z = zeros(getncells(grid))
    z[collect(getcellset(grid, setname))] = 1.0
    vtk_cell_data(vtk, z, setname)
end

# Solution
"""
    VTK.point_data(vtk, data::Vector, name)

Write the data field `data` to the `vtk` file.
"""
point_data(vtk, data, name) = WriteVTK.vtk_point_data(vtk, data, name)
function point_data(vtk::WriteVTK.DatasetFile, data::Vector{Vec{dim,T}}, name::AbstractString) where {dim,T}
    npoints = length(data)
    data = reinterpret(T, data, (dim, npoints))
    return WriteVTK.vtk_point_data(vtk, data, name)
end

"""
    VTK.point_data(vtk, dh:DofHandler, u::Vector)

Write the solution `u` to the vtk-file as point data.
"""
function point_data(vtkfile, dh::DofHandler, u::Vector)
    for f in 1:nfields(dh)
        @debug println("exporting field $(dh.field_names[f])")
        field_dim = dh.field_dims[f]
        space_dim = field_dim == 2 ? 3 : field_dim
        data = fill(0.0, space_dim, getnnodes(dh.grid))
        offset = field_offset(dh, dh.field_names[f])
        for cell in CellIterator(dh)
            _celldofs = celldofs(cell)
            counter = 1
            for node in getnodes(cell)
                for d in 1:dh.field_dims[f]
                    data[d, node] = u[_celldofs[counter + offset]]
                    @debug println("  exporting $(u[_celldofs[counter + offset]]) for dof#$(_celldofs[counter + offset])")
                    counter += 1
                end
            end
        end
        WriteVTK.vtk_point_data(vtkfile, data, string(dh.field_names[f]))
    end
    return vtkfile
end

# BC
"""
    VTK.point_data(vtk, ch::ConstraintHandler)

Write the constraints in `ch` to the vtk-file, with constrained
faces set to 1, and unconstrained faces set to 0.
"""
function point_data(vtkfile, ch::ConstraintHandler)
    unique_fields = []
    for dbc in ch.dbcs
        push!(unique_fields, dbc.field_name)
    end
    unique_fields = unique(unique_fields) # TODO v0.7: unique!(unique_fields)

    for field in unique_fields
        nd = ndim(ch.dh, field)
        data = zeros(Float64, nd, getnnodes(ch.dh.grid))
        for dbc in ch.dbcs
            dbc.field_name != field && continue
            for (cellidx, faceidx) in dbc.faces
                for facenode in faces(ch.dh.grid.cells[cellidx])[faceidx]
                    for component in dbc.components
                        data[component, facenode] = 1
                    end
                end
            end
        end
        WriteVTK.vtk_point_data(vtkfile, data, string(field, "_bc"))
    end
    return vtkfile
end



end # module VTK
