
"""
    VTKFile(filename::AbstractString, grid::AbstractGrid; kwargs...)
    VTKFile(filename::AbstractString, dh::DofHandler; kwargs...)

Create a `VTKFile` that contains an unstructured VTK grid.
The keyword arguments are forwarded to `WriteVTK.vtk_grid`, see
[Data Formatting Options](https://juliavtk.github.io/WriteVTK.jl/stable/grids/syntax/#Data-formatting-options)

This file handler can be used to to write data with

* [`write_solution`](@ref)
* [`write_cell_data`](@ref)
* [`write_projection`](@ref)
* [`write_node_data`](@ref).
* [`Ferrite.write_cellset`](@ref)
* [`Ferrite.write_nodeset`](@ref)
* [`Ferrite.write_constraints`](@ref)

It is necessary to call `close(::VTKFile)` to save the data after writing
to the file handler. Using the supported `do`-block does this automatically:
```julia
VTKFile(filename, grid) do vtk
    write_solution(vtk, dh, u)
    write_cell_data(vtk, celldata)
end
```
"""
struct VTKFile{VTK<:WriteVTK.VTKFile}
    vtk::VTK
end
function VTKFile(filename::String, dh::DofHandler; kwargs...)
    return VTKFile(filename, get_grid(dh); kwargs...)
end
function VTKFile(filename::String, grid::AbstractGrid; kwargs...)
    vtk = create_vtk_grid(filename, grid; kwargs...)
    return VTKFile(vtk)
end
# Makes it possible to use the `do`-block syntax
function VTKFile(f::Function, args...; kwargs...)
    vtk = VTKFile(args...; kwargs...)
    try
        f(vtk)
    finally
        close(vtk)
    end
end

Base.close(vtk::VTKFile) = WriteVTK.vtk_save(vtk.vtk)

function Base.show(io::IO, ::MIME"text/plain", vtk::VTKFile)
    open_str = isopen(vtk.vtk) ? "open" : "closed"
    filename = vtk.vtk.path
    print(io, "VTKFile for the $open_str file \"$(filename)\".")
end

"""
    VTKFileCollection(name::String, grid::AbstractGrid; kwargs...)
    VTKFileCollection(name::String, dh::DofHandler; kwargs...)

Create a paraview data file (.pvd) that can be used to
save multiple vtk file along with a time stamp. The keyword arguments
are forwarded to `WriteVTK.paraview_collection`.

See [`addstep!`](@ref) for examples for how to use `VTKFileCollection`.
```
"""
mutable struct VTKFileCollection{P<:WriteVTK.CollectionFile,G_DH}
    pvd::P
    grid_or_dh::G_DH
    name::String
    step::Int
end
function VTKFileCollection(name::String, grid_or_dh::Union{AbstractGrid,AbstractDofHandler}; kwargs...)
    pvd = WriteVTK.paraview_collection(name; kwargs...)
    basename = string(first(split(pvd.path, ".pvd")))
    return VTKFileCollection(pvd, grid_or_dh, basename, 0)
end
Base.close(pvd::VTKFileCollection) = WriteVTK.vtk_save(pvd.pvd)

"""
    addstep!(f::Function, pvd::VTKFileCollection, t::Real, [grid_or_dh]; kwargs...)

Add a step at time `t` by writing a `VTKFile` to `pvd`.
The keyword arguments are forwarded to `WriteVTK.vtk_grid`.
If required, a new grid can be used by supplying the grid or dofhandler as the last argument.
Should be used in a do-block, e.g.
```julia
filename = "myoutput"
pvd = VTKFileCollection(filename, grid)
for (n, t) in pairs(timevector)
    # Calculate, e.g., the solution `u` and the stress `σeff`
    addstep!(pvd, t) do io
        write_cell_data(io, σeff, "Effective stress")
        write_solution(io, dh, u)
    end
end
close(pvd)
```
"""
function addstep!(f::Function, pvd::VTKFileCollection, t, grid=pvd.grid_or_dh; kwargs...)
    pvd.step += 1
    VTKFile(string(pvd.name, "_", pvd.step), grid; kwargs...) do vtk
        f(vtk)
        pvd.pvd[t] = vtk.vtk # Add to collection
    end
end

"""
    addstep!(pvd::VTKFileCollection, vtk::VTKFile, t)

As an alternative to using the `addstep!(pvd, t) do` block, it is
also possible to add a specific `vtk` at time `t` to `pvd`.
Note that this will close the `vtk`. Example
```julia
filename = "myoutput"
pvd = VTKFileCollection(filename, grid)
for (n, t) in pairs(timevector)
    # Calculate, e.g., the solution `u` and the stress `σeff`
    vtk = VTKFile(string(filename, "_", n), dh)
    write_cell_data(vtk, σeff, "Effective stress")
    write_solution(vtk, dh, u)
    addstep!(pvd, vtk, t)
end
close(pvd)
```
"""
function addstep!(pvd::VTKFileCollection, vtk::VTKFile, t)
    pvd.step += 1
    pvd.pvd[t] = vtk.vtk
end

cell_to_vtkcell(::Type{Line}) = VTKCellTypes.VTK_LINE
cell_to_vtkcell(::Type{QuadraticLine}) = VTKCellTypes.VTK_QUADRATIC_EDGE

cell_to_vtkcell(::Type{Quadrilateral}) = VTKCellTypes.VTK_QUAD
cell_to_vtkcell(::Type{QuadraticQuadrilateral}) = VTKCellTypes.VTK_BIQUADRATIC_QUAD
cell_to_vtkcell(::Type{Triangle}) = VTKCellTypes.VTK_TRIANGLE
cell_to_vtkcell(::Type{QuadraticTriangle}) = VTKCellTypes.VTK_QUADRATIC_TRIANGLE
cell_to_vtkcell(::Type{SerendipityQuadraticQuadrilateral}) = VTKCellTypes.VTK_QUADRATIC_QUAD

cell_to_vtkcell(::Type{Hexahedron}) = VTKCellTypes.VTK_HEXAHEDRON
cell_to_vtkcell(::Type{SerendipityQuadraticHexahedron}) = VTKCellTypes.VTK_QUADRATIC_HEXAHEDRON
cell_to_vtkcell(::Type{QuadraticHexahedron}) = VTKCellTypes.VTK_TRIQUADRATIC_HEXAHEDRON
cell_to_vtkcell(::Type{Tetrahedron}) = VTKCellTypes.VTK_TETRA
cell_to_vtkcell(::Type{QuadraticTetrahedron}) = VTKCellTypes.VTK_QUADRATIC_TETRA
cell_to_vtkcell(::Type{Wedge}) = VTKCellTypes.VTK_WEDGE
cell_to_vtkcell(::Type{Pyramid}) = VTKCellTypes.VTK_PYRAMID

nodes_to_vtkorder(cell::AbstractCell) = collect(cell.nodes)
nodes_to_vtkorder(cell::Pyramid) = cell.nodes[[1,2,4,3,5]]
nodes_to_vtkorder(cell::QuadraticHexahedron) = [
    cell.nodes[1], # faces
    cell.nodes[2],
    cell.nodes[3],
    cell.nodes[4],
    cell.nodes[5],
    cell.nodes[6],
    cell.nodes[7],
    cell.nodes[8],
    cell.nodes[9], # edges
    cell.nodes[10],
    cell.nodes[11],
    cell.nodes[12],
    cell.nodes[13],
    cell.nodes[14],
    cell.nodes[15],
    cell.nodes[16],
    cell.nodes[17],
    cell.nodes[18],
    cell.nodes[19],
    cell.nodes[20],
    cell.nodes[25], # faces
    cell.nodes[23],
    cell.nodes[22],
    cell.nodes[24],
    cell.nodes[21],
    cell.nodes[26],
    cell.nodes[27], # interior
]

function create_vtk_griddata(grid::Grid{dim,C,T}) where {dim,C,T}
    cls = WriteVTK.MeshCell[]
    for cell in getcells(grid)
        celltype = Ferrite.cell_to_vtkcell(typeof(cell))
        push!(cls, WriteVTK.MeshCell(celltype, nodes_to_vtkorder(cell)))
    end
    coords = reshape(reinterpret(T, getnodes(grid)), (dim, getnnodes(grid)))
    return coords, cls
end

function create_vtk_grid(filename::AbstractString, grid::Grid{dim,C,T}; kwargs...) where {dim,C,T}
    coords, cls = create_vtk_griddata(grid)
    return WriteVTK.vtk_grid(filename, coords, cls; kwargs...)
end

function toparaview!(v, x::Vec{D}) where D
    v[1:D] .= x
end
function toparaview!(v, x::SecondOrderTensor)
    tovoigt!(v, x)
end

function _vtk_write_node_data(
    vtk::WriteVTK.DatasetFile,
    nodedata::Vector{S},
    name::AbstractString
    ) where {O, D, T, M, S <: Union{Tensor{O, D, T, M}, SymmetricTensor{O, D, T, M}}}
    noutputs = S <: Vec{2} ? 3 : M # Pad 2D Vec to 3D
    npoints = length(nodedata)
    out = zeros(T, noutputs, npoints)
    for i in 1:npoints
        toparaview!(@view(out[:, i]), nodedata[i])
    end
    return WriteVTK.vtk_point_data(vtk, out, name; component_names=component_names(S))
end
function _vtk_write_node_data(vtk::WriteVTK.DatasetFile, nodedata::Vector{<:Real}, name::AbstractString)
    return WriteVTK.vtk_point_data(vtk, nodedata, name)
end
function _vtk_write_node_data(vtk::WriteVTK.DatasetFile, nodedata::Matrix{<:Real}, name::AbstractString; component_names=nothing)
    return WriteVTK.vtk_point_data(vtk, nodedata, name; component_names=component_names)
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

"""
    write_solution(vtk::VTKFile, dh::AbstractDofHandler, u::Vector, suffix="")

Save the values at the nodes in the degree of freedom vector `u` to `vtk`.
Each field in `dh` will be saved separately, and `suffix` can be used to append
to the fieldname.

`u` can also contain tensorial values, but each entry in `u` must correspond to a
degree of freedom in `dh`, see [`write_node_data`](@ref write_node_data) for details.
Use `write_node_data` directly when exporting values that are already
sorted by the nodes in the grid.
"""
function write_solution(vtk::VTKFile, dh::AbstractDofHandler, u::Vector, suffix="")
    fieldnames = Ferrite.getfieldnames(dh)  # all primary fields
    for name in fieldnames
        data = _evaluate_at_grid_nodes(dh, u, name, #=vtk=# Val(true))
        _vtk_write_node_data(vtk.vtk, data, string(name, suffix))
    end
    return vtk
end

"""
    write_projection(vtk::VTKFile, proj::L2Projector, vals::Vector, name::AbstractString)

Project `vals` to the grid nodes with `proj` and save to `vtk`.
"""
function write_projection(vtk::VTKFile, proj::L2Projector, vals, name)
    data = _evaluate_at_grid_nodes(proj, vals, #=vtk=# Val(true))::Matrix
    @assert size(data, 2) == getnnodes(get_grid(proj.dh))
    _vtk_write_node_data(vtk.vtk, data, name; component_names=component_names(eltype(vals)))
    return vtk
end

"""
    write_cell_data(vtk::VTKFile, celldata::AbstractVector, name::String)

Write the `celldata` that is ordered by the cells in the grid to the vtk file.
"""
function write_cell_data(vtk::VTKFile, celldata, name)
    WriteVTK.vtk_cell_data(vtk.vtk, celldata, name)
end

"""
    write_node_data(vtk::VTKFile, nodedata::Vector{Real}, name)
    write_node_data(vtk::VTKFile, nodedata::Vector{<:AbstractTensor}, name)

Write the `nodedata` that is ordered by the nodes in the grid to `vtk`.

When `nodedata` contains `Tensors.Vec`s, each component is exported.
Two-dimensional vectors are padded with zeros.

When `nodedata` contains second order tensors, the index order,
`[11, 22, 33, 23, 13, 12, 32, 31, 21]`, follows the default Voigt order in Tensors.jl.
"""
function write_node_data(vtk::VTKFile, nodedata, name)
    _vtk_write_node_data(vtk.vtk, nodedata, name)
end


"""
    write_nodeset(vtk::VTKFile, grid::AbstractGrid, nodeset::String)

Write nodal values of 1 for nodes in `nodeset`, and 0 otherwise
"""
function write_nodeset(vtk, grid::AbstractGrid, nodeset::String)
    z = zeros(getnnodes(grid))
    z[collect(getnodeset(grid, nodeset))] .= 1.0
    write_node_data(vtk, z, nodeset)
    return vtk
end

"""
    write_cellset(vtk, grid::AbstractGrid)
    write_cellset(vtk, grid::AbstractGrid, cellset::String)
    write_cellset(vtk, grid::AbstractGrid, cellsets::Union{AbstractVector{String},AbstractSet{String})

Write all cell sets in the grid with name according to their keys and
celldata 1 if the cell is in the set, and 0 otherwise. It is also possible to
only export a single `cellset`, or multiple `cellsets`.
"""
function write_cellset(vtk, grid::AbstractGrid, cellsets=keys(getcellsets(grid)))
    z = zeros(getncells(grid))
    for cellset in cellsets
        fill!(z, 0)
        z[collect(getcellset(grid, cellset))] .= 1.0
        write_cell_data(vtk, z, cellset)
    end
    return vtk
end
write_cellset(vtk, grid::AbstractGrid, cellset::String) = write_cellset(vtk, grid, [cellset])

"""
    write_constraints(vtk::VTKFile, ch::ConstraintHandler)

Saves the dirichlet boundary conditions to a vtkfile.
Values will have a 1 where bcs are active and 0 otherwise
"""
function write_constraints(vtk, ch::ConstraintHandler)
    unique_fields = []
    for dbc in ch.dbcs
        push!(unique_fields, dbc.field_name)
    end
    unique!(unique_fields)

    for field in unique_fields
        nd = n_components(ch.dh, field)
        data = zeros(Float64, nd, getnnodes(get_grid(ch.dh)))
        for dbc in ch.dbcs
            dbc.field_name != field && continue
            if eltype(dbc.facets) <: BoundaryIndex
                functype = boundaryfunction(eltype(dbc.facets))
                for (cellidx, facetidx) in dbc.facets
                    for facetnode in functype(getcells(get_grid(ch.dh), cellidx))[facetidx]
                        for component in dbc.components
                            data[component, facetnode] = 1
                        end
                    end
                end
            else
                for nodeidx in dbc.facets
                    for component in dbc.components
                        data[component, nodeidx] = 1
                    end
                end
            end
        end
        write_node_data(vtk, data, string(field, "_bc"))
    end
    return vtk
end

"""
    write_cell_colors(vtk::VTKFile, grid::AbstractGrid, cell_colors, name="coloring")

Write cell colors (see [`create_coloring`](@ref)) to a VTK file for visualization.

In case of coloring a subset, the cells which are not part of the subset are represented as color 0.
"""
function write_cell_colors(vtk, grid::AbstractGrid, cell_colors::AbstractVector{<:AbstractVector{<:Integer}}, name="coloring")
    color_vector = zeros(Int, getncells(grid))
    for (i, cells_color) in enumerate(cell_colors)
        for cell in cells_color
            color_vector[cell] = i
        end
    end
    write_cell_data(vtk, color_vector, name)
end
