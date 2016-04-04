function get_cell_type(nen, ndim)
    # TODO: This is a bad way of figuring out the eltype
    if nen == 3 && ndim == 2
        cell =  VTKCellType.VTK_TRIANGLE
    elseif nen == 4 && ndim == 2
        cell = VTKCellType.VTK_QUAD
    elseif nen == 4 && ndim == 2
        cell = VTKCellType.VTK_HEXAHEDRON
    elseif nen == 4 && ndim == 3
        cell = VTKCellType.VTK_TETRA
    end
    return cell
end

function pad_zeros(points, ndim, nnodes)
    if ndim == 3
        points = points
    elseif ndim == 2
        points = [points; zeros(nnodes)']
    elseif ndim == 1
        points = [points; zeros(nnodes)'; zeros(nnodes)']
    end
    return points
end

"""
Creates an unstructured VTK grid from the element topology and coordinates.


    vtk_grid(topology::Matrix{Int}, coord::Matrix, filename::AbstractString)


**Arguments**

* `topology` A matrix where each column contains the vertices of the element
* `coord` A matrix of the coordinates, one column per coordinate
* `filename`: Name of the file when it is saved to disk

**Results:**

* `::DatasetFile`

**Example:**

```julia
julia> coords = [0.0 0.0; 1.0 0.0; 0.5 1.0; 1.5 1.0]'
2x4 Array{Float64,2}:
 0.0  1.0  0.5  1.5
 0.0  0.0  1.0  1.0

julia> topology = [1 2 3; 2 4 3]'
3x2 Array{Int64,2}:
 1  2
 2  4
 3  3

julia> vtkobj = vtk_grid(topology, coords, "example");

julia> vtk_save(vtkobj)
1-element Array{UTF8String,1}:
 "example.vtu"
```

**Details**

This is a thin wrapper around the `vtk_grid` function from the [`WriteVTK`](https://github.com/jipolanco/WriteVTK.jl) package.

For infromation how to add cell data and point data to the resulting VTK object as well as how to write it to a file see
[https://github.com/jipolanco/WriteVTK.jl#generating-an-unstructured-vtk-file](https://github.com/jipolanco/WriteVTK.jl#generating-an-unstructured-vtk-file)
"""
function vtk_grid(topology::Matrix{Int}, coord::Matrix, filename::AbstractString)

    nele = size(topology, 2)
    nen = size(topology,1)
    nnodes = size(coord, 2)
    ndim = size(coord, 1)
    if ndim > 3
        throw(ArgumentError("dimension > 3, maybe you transposed the input coord matrix"))
    end

    cell = get_cell_type(nen, ndim)

    points = coord
    points = pad_zeros(points, ndim, nnodes)

    cells = MeshCell[MeshCell(cell, topology[:,i]) for i = 1:nele]

    vtk = vtk_grid(filename, points, cells)
    return vtk
end
