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

nodes_to_vtkorder(cell::AbstractCell) = collect(cell.nodes)

"""
    vtk_grid(filename::AbstractString, grid::Grid)

Create a unstructured VTK grid from a `Grid`. Return a `DatasetFile`
which data can be appended to, see `vtk_point_data` and `vtk_cell_data`.
"""
function WriteVTK.vtk_grid(filename::AbstractString, grid::Grid{dim,C,T}; compress::Bool=true) where {dim,C,T}
    cls = MeshCell[]
    for cell in grid.cells
        celltype = Ferrite.cell_to_vtkcell(typeof(cell))
        push!(cls, MeshCell(celltype, nodes_to_vtkorder(cell)))
    end
    coords = reshape(reinterpret(T, getnodes(grid)), (dim, getnnodes(grid)))
    return vtk_grid(filename, coords, cls; compress=compress)
end

function toparaview!(v, x::Vec{D}) where D
    v[1:D] .= x
end
function toparaview!(v, x::SecondOrderTensor{D}) where D
    tovoigt!(v, x)
end

"""
    vtk_point_data(vtk, data::Vector{<:AbstractTensor}, name)

Write the tensor field `data` to the vtk file. Two-dimensional tensors are padded with zeros.

For second order tensors the following indexing ordering is used:
`[11, 22, 33, 23, 13, 12, 32, 31, 21]`. This is the default Voigt order in Tensors.jl.
"""
function WriteVTK.vtk_point_data(
    vtk::WriteVTK.DatasetFile,
    data::Vector{S},
    name::AbstractString
    ) where {O, D, T, M, S <: Union{Tensor{O, D, T, M}, SymmetricTensor{O, D, T, M}}}
    noutputs = S <: Vec{2} ? 3 : M # Pad 2D Vec to 3D
    npoints = length(data)
    out = zeros(T, noutputs, npoints)
    for i in 1:npoints
        toparaview!(@view(out[:, i]), data[i])
    end
    return vtk_point_data(vtk, out, name; component_names=component_names(S))
end

function component_names(::Type{S}) where S
    names =
        S <:             Vec{1}   ? ["x"] :
        S <:             Vec      ? ["x", "y", "z"] : # Pad 2D Vec to 3D
        S <:          Tensor{2,1} ? ["xx"] :
        S <: SymmetricTensor{2,1} ? ["xx"] :
        S <:          Tensor{2,2} ? ["xx", "yy", "xy", "yx"] :
        S <: SymmetricTensor{2,2} ? ["xx", "yy", "xy"] :
        S <:          Tensor{2,3} ? ["xx", "yy", "zz", "yz", "xz", "xy", "zy", "zx", "yx"] :
        S <: SymmetricTensor{2,3} ? ["xx", "yy", "zz", "yz", "xz", "xy"] :
                                    nothing
    return names
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
