cell_to_vtkcell(::Type{Line}) = VTKCellTypes.VTK_LINE
cell_to_vtkcell(::Type{Line2D}) = VTKCellTypes.VTK_LINE
cell_to_vtkcell(::Type{Line3D}) = VTKCellTypes.VTK_LINE
cell_to_vtkcell(::Type{QuadraticLine}) = VTKCellTypes.VTK_QUADRATIC_EDGE

cell_to_vtkcell(::Type{Quadrilateral}) = VTKCellTypes.VTK_QUAD
cell_to_vtkcell(::Type{Quadrilateral3D}) = VTKCellTypes.VTK_QUAD
cell_to_vtkcell(::Type{QuadraticQuadrilateral}) = VTKCellTypes.VTK_BIQUADRATIC_QUAD
cell_to_vtkcell(::Type{Triangle}) = VTKCellTypes.VTK_TRIANGLE
cell_to_vtkcell(::Type{QuadraticTriangle}) = VTKCellTypes.VTK_QUADRATIC_TRIANGLE
cell_to_vtkcell(::Type{Cell{2,8,4}}) = VTKCellTypes.VTK_QUADRATIC_QUAD

cell_to_vtkcell(::Type{Hexahedron}) = VTKCellTypes.VTK_HEXAHEDRON
cell_to_vtkcell(::Type{Cell{3,20,6}}) = VTKCellTypes.VTK_QUADRATIC_HEXAHEDRON
cell_to_vtkcell(::Type{Tetrahedron}) = VTKCellTypes.VTK_TETRA
cell_to_vtkcell(::Type{QuadraticTetrahedron}) = VTKCellTypes.VTK_QUADRATIC_TETRA

"""
    vtk_grid(filename::AbstractString, grid::Grid)

Create a unstructured VTK grid from a `Grid`. Return a `DatasetFile`
which data can be appended to, see `vtk_point_data` and `vtk_cell_data`.
"""
function WriteVTK.vtk_grid(filename::AbstractString, grid::Grid{dim,C,T}; compress::Bool=true) where {dim,C,T}
    cls = MeshCell[]
    for cell in grid.cells
        celltype = Ferrite.cell_to_vtkcell(typeof(cell))
        push!(cls, MeshCell(celltype, collect(cell.nodes)))
    end
    coords = reshape(reinterpret(T, getnodes(grid)), (dim, getnnodes(grid)))
    return vtk_grid(filename, coords, cls; compress=compress)
end

# ParaView uses this ordering for SymmetricTensors. Regular Tensors
# aren't recognized as tensorfields, but let's use the same ordering anyway
const PARAVIEW_VOIGT_ORDER = ([1], [1 3; 4 2], [1 4 6; 7 2 5; 9 8 3])
function toparaview!(v, x::Vec{D}) where D
    v[1:D] .= x
end
function toparaview!(v, x::SecondOrderTensor{D}) where D
    tovoigt!(v, x; order=PARAVIEW_VOIGT_ORDER[D])
end
function toparaview!(v, x::SymmetricTensor{2,2})
    # Since 2D tensors are padded to 3D we need to place the output in
    # the correct positions with this view
    tovoigt!(@view(v[[1, 2, 4]]), x; order=PARAVIEW_VOIGT_ORDER[2])
end

"""
    vtk_point_data(vtk, data::Vector{<:AbstractTensor}, name)

Write the tensor field `data` to the vtk file. Two-dimensional tensors are padded with zeros.

For second order tensors the following indexing ordering is used:
`[11, 22, 33, 12, 23, 13, 21, 32, 31]`. This is the order that ParaView uses.
"""
function WriteVTK.vtk_point_data(
    vtk::WriteVTK.DatasetFile,
    data::Vector{S},
    name::AbstractString
    ) where {O, D, T, M, S <: Union{Tensor{O, D, T, M}, SymmetricTensor{O, D, T, M}}}
    noutputs = S <: SymmetricTensor{2,2} ? 6 : # Pad 2D SymmetricTensors to 3D
               S <: Vec{2} ? 3 : # Pad 2D Vec to 3D
               M
    npoints = length(data)
    out = zeros(T, noutputs, npoints)
    for i in 1:npoints
        toparaview!(@view(out[:, i]), data[i])
    end
    return vtk_point_data(vtk, out, name)
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
function vtk_cellset(vtk::WriteVTK.DatasetFile, grid::AbstractGrid, cellsets=keys(grid.cellsets))
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
vtk_cellset(vtk::WriteVTK.DatasetFile, grid::AbstractGrid, cellset::String) =
    vtk_cellset(vtk, grid, [cellset])

function WriteVTK.vtk_point_data(vtkfile, dh::AbstractDofHandler, u::Vector, suffix="")

    fieldnames = Ferrite.getfieldnames(dh)  # all primary fields

    for name in fieldnames
        data = reshape_to_nodes(dh, u, name)
        vtk_point_data(vtkfile, data, string(name, suffix))
    end

    return vtkfile
end
