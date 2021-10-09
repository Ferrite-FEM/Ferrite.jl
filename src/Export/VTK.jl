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

"""
    vtk_point_data(vtk, data::Vector{<:SymmetricTensor{2}}, name)

Write the second order tensor field `data` to the vtk file. Twodimensional tensors are padded with zeros.
The tensor field is written such that Paraview recognizes the tensor components, where `XX` corresponds to the `[1,1]` component and so on.
"""
function WriteVTK.vtk_point_data(
    vtk::WriteVTK.DatasetFile,
    data::Union{
        Vector{SymmetricTensor{2,dim,T,M}}
        },
    name::AbstractString
    ) where {dim,T,M}

    if dim == 1
        noutputs = 1
        sort_paraview = [1]
    elseif dim == 2
        noutputs = 6
        sort_paraview = [1,4,2]
    elseif dim == 3
        noutputs = 6
        sort_paraview = [1,4,6,2,5,3]
    end

    npoints = length(data)
    out = zeros(T, noutputs, npoints)
    out[sort_paraview, :] .= reshape(reinterpret(T, data), (M, npoints))

    return vtk_point_data(vtk, out, name)
end

"""
    vtk_point_data(vtk, data::Vector{<:Tensor}, name)

Write the tensor field data to the vtk file. Only writes the tensor values available in `data`.
In the vtu-file, ordering of the tensor components is column-wise (just like Julia).
[1 2
 3 4] => 1, 3, 2, 4
"""
function WriteVTK.vtk_point_data(
    vtk::WriteVTK.DatasetFile,
    data::Union{
        Vector{Tensor{order,dim,T,M}},
        Vector{SymmetricTensor{order,dim,T,M}}
        },
    name::AbstractString
    ) where {order,dim,T,M}

    npoints = length(data)
    out = zeros(T, M, npoints)
    out[1:M, :] .= reshape(reinterpret(T, data), (M, npoints))
    return vtk_point_data(vtk, out, name)
end

"""
    vtk_point_data(vtk, data::Vector{<:Vec}, name)

Write the vector field data to the vtk file.
"""
function WriteVTK.vtk_point_data(vtk::WriteVTK.DatasetFile, data::Vector{Vec{dim,T}}, name::AbstractString) where {dim,T}
    npoints = length(data)
    out = zeros(T, (dim == 2 ? 3 : dim), npoints)
    out[1:dim, :] .= reshape(reinterpret(T, data), (dim, npoints))
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