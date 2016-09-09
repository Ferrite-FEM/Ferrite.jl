function get_cell_type(nen, ndim)
    # TODO: This is a bad way of figuring out the eltype
    if nen == 3 && ndim == 2
        cell =  VTKCellTypes.VTK_TRIANGLE
    elseif nen == 4 && ndim == 2
        cell = VTKCellTypes.VTK_QUAD
    elseif nen == 4 && ndim == 2
        cell = VTKCellTypes.VTK_HEXAHEDRON
    elseif nen == 4 && ndim == 3
        cell = VTKCellTypes.VTK_TETRA
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

function vtk_grid(topology::Matrix{Int}, coord::Matrix, filename::AbstractString)
    Base.depwarn("vtk_grid(topology::Matrix{Int}, coord::Matrix, filename::AbstractString) is deprecated, use vtk_grid{dim,T}(filename::AbstractString, coords::Vector{Vec{dim,T}}, topology::Matrix{Int}, celltype::VTKCellTypes.VTKCellType) instead", :vtk_grid)
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

"""
Creates an unstructured VTK grid from the element topology and coordinates.


    vtk_grid{dim,T}(filename::AbstractString, coords::Vector{Vec{dim,T}}, topology::Matrix{Int}, celltype::VTKCellTypes.VTKCellType)


**Arguments**

* `filename` Name (or path) of the file when it is saved to disk, eg `filename = "myfile"`, or `filename = "/results/myfile"` to store it in the folder results
* `coords` A vector of the node coordinates
* `topology` A matrix where each column contains the nodes which connects the element
* `celltype` The definition of the celltype in the grid, see [https://github.com/jipolanco/WriteVTK.jl#defining-cells](https://github.com/jipolanco/WriteVTK.jl#defining-cells)

**Results:**

* `::DatasetFile`

**Example:**

```julia
julia> coords = [Vec{2}((0.0,0.0)), Vec{2}((1.0,0.0)), Vec{2}((1.5,1.5)), Vec{2}((0.0,1.0))]
4-element Array{ContMechTensors.Tensor{1,2,Float64,2},1}:
 [0.0,0.0]
 [1.0,0.0]
 [1.5,1.5]
 [0.0,1.0]

julia> topology = [1 2 4; 2 3 4]'
3×2 Array{Int64,2}:
 1  2
 2  3
 4  4

julia> celltype = VTKCellTypes.VTK_TRIANGLE;

julia> vtkobj = vtk_grid("example", coords, topology, celltype);

julia> vtk_save(vtkobj)
1-element Array{String,1}:
 "example.vtu"
```

**Details**

This is a thin wrapper around the `vtk_grid` function from the [`WriteVTK`](https://github.com/jipolanco/WriteVTK.jl) package.

For infromation how to add cell data and point data to the resulting VTK object as well as how to write it to a file see
[https://github.com/jipolanco/WriteVTK.jl#generating-an-unstructured-vtk-file](https://github.com/jipolanco/WriteVTK.jl#generating-an-unstructured-vtk-file)
"""
function vtk_grid{dim,T}(filename::AbstractString, coords::Vector{Vec{dim,T}}, topology::Matrix{Int}, celltype::VTKCellTypes.VTKCellType)

    Nel = size(topology,2)
    Npts = length(coords)
    coords = reinterpret(T,coords,(dim,Npts))
    cells = MeshCell[]
    for el in 1:Nel
        push!(cells, MeshCell(celltype, topology[:,el]))
    end

    return vtk_grid(filename,coords,cells)
end
