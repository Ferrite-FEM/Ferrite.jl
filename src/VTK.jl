export vtk_nodeset, vtk_cellset

"""
Creates an unstructured VTK grid from the element topology and coordinates.

```julia
vtk_grid{dim,T}(filename::AbstractString, coords::Vector{Vec{dim,T}}, topology::Matrix{Int}, celltype::VTKCellTypes.VTKCellType)
```

**Arguments**

* `filename`: name (or path) of the file when it is saved to disk, eg `filename = "myfile"`, or `filename = "/results/myfile"` to store it in the folder results
* `coords`: a vector of the node coordinates
* `topology`: a matrix where each column contains the nodes which connects the element
* `celltype`: the definition of the celltype in the grid, see [https://github.com/jipolanco/WriteVTK.jl#defining-cells](https://github.com/jipolanco/WriteVTK.jl#defining-cells)

**Results:**

* `::DatasetFile`

**Example:**

```jldoctest
julia> coords = [Vec{2}((0.0,0.0)), Vec{2}((1.0,0.0)), Vec{2}((1.5,1.5)), Vec{2}((0.0,1.0))];

julia> topology = [1 2 4; 2 3 4]';

julia> celltype = VTKCellTypes.VTK_TRIANGLE;

julia> vtkobj = vtk_grid("example", coords, topology, celltype);

julia> vtk_save(vtkobj)
1-element Array{String,1}:
 "example.vtu"
```

**Details**

This is a thin wrapper around the function `vtk_grid` from the [`WriteVTK`](https://github.com/jipolanco/WriteVTK.jl) package.

For information how to add cell data and point data to the resulting VTK object as well as how to write it to a file see
[https://github.com/jipolanco/WriteVTK.jl#generating-an-unstructured-vtk-file](https://github.com/jipolanco/WriteVTK.jl#generating-an-unstructured-vtk-file)
"""
function vtk_grid{dim,T}(filename::AbstractString, coords::Vector{Vec{dim,T}}, topology::Matrix{Int}, celltype::VTKCellTypes.VTKCellType)

    nel = size(topology,2)
    npts = length(coords)
    coords = reinterpret(T,coords,(dim,npts))
    cells = MeshCell[]
    for el in 1:nel
        push!(cells, MeshCell(celltype, topology[:,el]))
    end

    return vtk_grid(filename,coords,cells)
end

"""
Returns the VTKCellType corresponding to the input `Interpolation`

```julia
getVTKtype(ip::Interpolation)
```

**Arguments**

* `ip`: The interpolation

**Results:**

* `::VTKCellType`: The cell type, see [https://github.com/jipolanco/WriteVTK.jl#generating-an-unstructured-vtk-file](https://github.com/jipolanco/WriteVTK.jl#generating-an-unstructured-vtk-file)

**Example:**

```jldoctest
julia> ip = Lagrange{2, RefCube, 1}()
JuAFEM.Lagrange{2,JuAFEM.RefCube,1}()

julia> getVTKtype(ip)
WriteVTK.VTKCellTypes.VTKCellType("VTK_QUAD", 0x09, 4)
```
"""
getVTKtype(::Lagrange{1, RefCube, 1}) = VTKCellTypes.VTK_LINE
getVTKtype(::Lagrange{1, RefCube, 2}) = VTKCellTypes.VTK_QUADRATIC_EDGE

getVTKtype(::Lagrange{2, RefCube, 1}) = VTKCellTypes.VTK_QUAD
getVTKtype(::Lagrange{2, RefCube, 2}) = VTKCellTypes.VTK_BIQUADRATIC_QUAD
getVTKtype(::Lagrange{2, RefTetrahedron, 1}) = VTKCellTypes.VTK_TRIANGLE
getVTKtype(::Lagrange{2, RefTetrahedron, 2}) = VTKCellTypes.VTK_QUADRATIC_TRIANGLE
getVTKtype(::Serendipity{2, RefCube, 2}) = VTKCellTypes.VTK_QUADRATIC_QUAD

getVTKtype(::Lagrange{3, RefCube, 1}) = VTKCellTypes.VTK_HEXAHEDRON
getVTKtype(::Lagrange{3, RefTetrahedron, 1}) = VTKCellTypes.VTK_TETRA

getVTKtype(::Type{Cell{1,2}}) = VTKCellTypes.VTK_LINE
getVTKtype(::Type{Cell{1,3}}) = VTKCellTypes.VTK_QUADRATIC_EDGE

getVTKtype(::Type{Cell{2,4}}) = VTKCellTypes.VTK_QUAD
getVTKtype(::Type{Cell{2,9}}) = VTKCellTypes.VTK_BIQUADRATIC_QUAD
getVTKtype(::Type{Cell{2,3}}) = VTKCellTypes.VTK_TRIANGLE
getVTKtype(::Type{Cell{2,6}}) = VTKCellTypes.VTK_QUADRATIC_TRIANGLE
getVTKtype(::Type{Cell{2,8}}) = VTKCellTypes.VTK_QUADRATIC_QUAD

getVTKtype(::Type{Cell{3,8}}) = VTKCellTypes.VTK_HEXAHEDRON
getVTKtype(::Type{Cell{3,4}}) = VTKCellTypes.VTK_TETRA


function vtk_grid{dim, N, T}(filename::AbstractString, grid::Grid{dim, N, T})
    coords = reinterpret(T, getnodes(grid), (dim, getnnodes(grid)))

    celltype = getVTKtype(getcelltype(grid))
    cls = MeshCell[]
    for cell in 1:getncells(grid)
        push!(cls, MeshCell(celltype, collect(grid.cells[cell].nodes)))
    end

    return vtk_grid(filename, coords, cls)
end

function vtk_point_data{dim, T}(vtk::DatasetFile, data::Vector{Vec{dim, T}}, name::AbstractString)
    npoints = length(data)
    data = reinterpret(T, data, (dim, npoints))
    return vtk_point_data(vtk, data, name)
end

function vtk_nodeset{dim}(vtk::DatasetFile, grid::Grid{dim}, nodeset::String)
    z = zeros(getnnodes(grid))
    z[getnodeset(grid, nodeset)] = 1.0
    vtk_point_data(vtk, z, nodeset)
end

function vtk_cellset{dim}(vtk::DatasetFile, grid::Grid{dim}, cellset::String)
    z = zeros(getncells(grid))
    z[getcellset(grid, cellset)] = 1.0
    vtk_cell_data(vtk, z, cellset)
end

