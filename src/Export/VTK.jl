struct VTKStream{VTK<:WriteVTK.DatasetFile}
    vtk::VTK
end

"""
    VTKStream(filename::AbstractString, grid::AbstractGrid; kwargs...)

Create a `Ferrite.VTKStream` that contains an unstructured VTK grid from
a `DofHandler` (limited functionality if only a `Grid` is given). 
This stream can be used to to write data with 
[`write_solution`](@ref), [`write_celldata`](@ref), [`write_nodedata`](@ref),
[`write_projected`](@ref), 
[`write_cellset`](@ref), [`write_nodeset`](@ref), and [`write_constraints`](@ref).

The keyword arguments are forwarded to `WriteVTK.vtk_grid`, see 
[Data Formatting Options](https://juliavtk.github.io/WriteVTK.jl/stable/grids/syntax/#Data-formatting-options)

It is necessary to call [`close`](@ref) to save the data after writing to the stream, 
or alternatively use the `do`-block syntax which does this implicitly, e.g.,
```julia
VTKStream(filename, grid) do vtks
    write_solution(vtks, dh, u)
    write_celldata(vtks, grid, celldata)
end
"""
function VTKStream(filename::String, grid::AbstractGrid; kwargs...)
    vtk = create_vtk_grid(filename, grid; kwargs...)
    return VTKStream(vtk)
end
# Makes it possible to use the `do`-block syntax
function VTKStream(f::Function, args...; kwargs...)
    vtks = VTKStream(args...; kwargs...)
    try
        f(vtks)
    finally
        close(vtks)
    end
end

"""
    Base.close(vtks::VTKStream)

Close the vtk stream and save the data to disk. 
"""
Base.close(vtks::VTKStream) = WriteVTK.vtk_save(vtks.vtk)

function Base.show(io::IO, ::MIME"text/plain", vtks::VTKStream)
    open_str = WriteVTK.isopen(vtks.vtk) ? "open" : "closed"
    filename = vtks.vtk.path
    print(io, "VTKStream for the $open_str file \"$(filename)\".")
end

# Support ParaviewCollection
function Base.setindex!(pvd::WriteVTK.CollectionFile, vtks::VTKStream, time::Real)
    WriteVTK.collection_add_timestep(pvd, vtks, time)
end
function WriteVTK.collection_add_timestep(pvd::WriteVTK.CollectionFile, vtks::VTKStream, time::Real)
    WriteVTK.collection_add_timestep(pvd, vtks.vtk, time)
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

nodes_to_vtkorder(cell::AbstractCell) = collect(cell.nodes)
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

function create_vtk_grid(filename::AbstractString, grid::Grid{dim,C,T}; kwargs...) where {dim,C,T}
    cls = MeshCell[]
    for cell in getcells(grid)
        celltype = Ferrite.cell_to_vtkcell(typeof(cell))
        push!(cls, MeshCell(celltype, nodes_to_vtkorder(cell)))
    end
    coords = reshape(reinterpret(T, getnodes(grid)), (dim, getnnodes(grid)))
    return vtk_grid(filename, coords, cls; kwargs...)
end

function toparaview!(v, x::Vec{D}) where D
    v[1:D] .= x
end
function toparaview!(v, x::SecondOrderTensor)
    tovoigt!(v, x)
end

function _vtk_write_nodedata(
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
function _vtk_write_nodedata(vtk::WriteVTK.DatasetFile, nodedata::Vector{<:Real}, name::AbstractString)
    return WriteVTK.vtk_point_data(vtk, nodedata, name)
end
function _vtk_write_nodedata(vtk::WriteVTK.DatasetFile, nodedata::Matrix{<:Real}, name::AbstractString; component_names=nothing)
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
    write_solution(vtks::VTKStream, dh::AbstractDofHandler, u::Vector, suffix="")

Save the values at the nodes in the degree of freedom vector `u` to a 
the a vtk file for each field in `DofHandler` in `vtks`. 
`suffix` can be used to append to the fieldname.

`u` can also contain tensorial values, but each entry in `u` must correspond to a degree of freedom in `dh`,
see [`write_nodedata`](@ref) for details. This function should be used directly when exporting values already 
sorted by the nodes in the grid. 
"""
function write_solution(vtks::VTKStream, dh::AbstractDofHandler, u::Vector, suffix="")
    fieldnames = Ferrite.getfieldnames(dh)  # all primary fields
    for name in fieldnames
        data = _evaluate_at_grid_nodes(dh, u, name, #=vtk=# Val(true))
        _vtk_write_nodedata(vtks.vtk, data, string(name, suffix))
    end
    return vtks
end

"""
    write_projected(vtks::VTKStream, proj::L2Projector, vals::Vector, name::AbstractString)

Write `vals` that have been projected with `proj` to the vtk file in `vtks`
"""
function write_projected(vtks::VTKStream, proj::L2Projector, vals, name)
    data = _evaluate_at_grid_nodes(proj, vals, #=vtk=# Val(true))::Matrix
    @assert size(data, 2) == getnnodes(get_grid(proj.dh))
    _vtk_write_nodedata(vtks.vtk, data, name; component_names=component_names(eltype(vals)))
    return vtks
end

"""
    write_celldata(vtks::VTKStream, celldata, name)

Write the `celldata` that is ordered by the cells in the grid to the vtk file.
"""
function write_celldata(vtks::VTKStream, ::AbstractGrid, celldata, name)
    WriteVTK.vtk_cell_data(vtks.vtk, celldata, name)
end

"""
    write_nodedata(vtks::VTKStream, nodedata::Vector{Real}, name)
    write_nodedata(vtks::VTKStream, nodedata::Vector{<:AbstractTensor}, name)
    
Write the `nodedata` that is ordered by the nodes in the grid to the vtk file.

When `nodedata` contains `Tensors.Vec`s, each component is exported. 
Two-dimensional vectors are padded with zeros.

When `nodedata` contains second order tensors, the index order, 
`[11, 22, 33, 23, 13, 12, 32, 31, 21]`, follows the default Voigt order in Tensors.jl.
"""
function write_nodedata(vtks::VTKStream, ::AbstractGrid, nodedata, name)
    _vtk_write_nodedata(vtks.vtk, nodedata, name)
end


"""
    write_nodeset(vtks::VTKStream, grid::AbstractGrid, nodeset::String)

Export nodal values of 1 for nodes in `nodeset`, and 0 otherwise
"""
function write_nodeset(vtks, grid::AbstractGrid, nodeset::String)
    z = zeros(getnnodes(grid))
    z[collect(getnodeset(grid, nodeset))] .= 1.0
    write_nodedata(vtks, grid, z, nodeset)
    return vtks
end

"""
    write_cellset(vtks, grid::AbstractGrid)
    write_cellset(vtks, grid::AbstractGrid, cellset::String)
    write_cellset(vtks, grid::AbstractGrid, cellsets::Union{AbstractVector{String},AbstractSet{String})

Export all cell sets in the grid with name according to their keys and 
celldata 1 if the cell is in the set, and 0 otherwise. It is also possible to 
only export a single `cellset`, or multiple `cellsets`. 
"""
function write_cellset(vtks, grid::AbstractGrid, cellsets=keys(getcellsets(getgrid(vtks))))
    z = zeros(getncells(grid))
    for cellset in cellsets
        fill!(z, 0)
        z[collect(getcellset(grid, cellset))] .= 1.0
        write_celldata(vtks, grid, z, cellset)
    end
    return vtks
end
write_cellset(vtks, grid::AbstractGrid, cellset::String) = write_cellset(vtks, grid, [cellset])

"""
    write_dirichlet(vtks::VTKStream, ch::ConstraintHandler)

Saves the dirichlet boundary conditions to a vtkfile.
Values will have a 1 where bcs are active and 0 otherwise
"""
function write_dirichlet(vtks, ch::ConstraintHandler)    
    unique_fields = []
    for dbc in ch.dbcs
        push!(unique_fields, dbc.field_name)
    end
    unique!(unique_fields)

    for field in unique_fields
        nd = getfielddim(ch.dh, field)
        data = zeros(Float64, nd, getnnodes(get_grid(ch.dh)))
        for dbc in ch.dbcs
            dbc.field_name != field && continue
            if eltype(dbc.faces) <: BoundaryIndex
                functype = boundaryfunction(eltype(dbc.faces))
                for (cellidx, faceidx) in dbc.faces
                    for facenode in functype(getcells(get_grid(ch.dh), cellidx))[faceidx]
                        for component in dbc.components
                            data[component, facenode] = 1
                        end
                    end
                end
            else
                for nodeidx in dbc.faces
                    for component in dbc.components
                        data[component, nodeidx] = 1
                    end
                end
            end
        end
        write_nodedata(vtks, get_grid(ch.dh), data, string(field, "_bc"))
    end
    return vtks
end

"""
    write_cell_colors(vtks::VTKStream, grid::AbstractGrid, cell_colors, name="coloring")

Write cell colors (see [`create_coloring`](@ref)) to a VTK file for visualization.

In case of coloring a subset, the cells which are not part of the subset are represented as color 0.
"""
function write_cell_colors(vtks, grid::AbstractGrid, cell_colors::AbstractVector{<:AbstractVector{<:Integer}}, name="coloring")
    color_vector = zeros(Int, getncells(grid))
    for (i, cells_color) in enumerate(cell_colors)
        for cell in cells_color
            color_vector[cell] = i
        end
    end
    write_celldata(vtks, grid, color_vector, name)
end
