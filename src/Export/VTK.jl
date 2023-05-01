struct VTKStream{VTK<:WriteVTK.DatasetFile, DH<:Union{DofHandler,Grid}}
    filename::String # Just to allow easy printing
    vtk::VTK
    grid_or_dh::DH
end
"""
    Base.close(vtks::VTKStream)

Close the vtk stream and save the data to disk. 
"""
Base.close(vtks::VTKStream) = WriteVTK.vtk_save(vtks.vtk)

"""
    open_vtk(filename::AbstractString, dh::Union{DofHandler,Grid}; kwargs...)

Create a `Ferrite.VTKStream` that contains an unstructured VTK grid from
a `DofHandler` (limited functionality if only a `Grid` is given). 
This stream can be used to to write data with 
[`write_solution`](@ref), [`write_celldata`](@ref), [`write_nodedata`](@ref),
[`write_cellset`](@ref), [`write_nodeset`](@ref), and [`write_constraints`](@ref).

The keyword arguments are forwarded to `WriteVTK.vtk_grid`, see 
[Data Formatting Options](https://juliavtk.github.io/WriteVTK.jl/stable/grids/syntax/#Data-formatting-options)

It is necessary to call [`close`](@ref) to save the data after writing to the stream, 
or alternatively use the `do`-block syntax which does this implicitly, e.g.,
```julia
open_vtk(filename, dh) do vtk
    write_solution(vtk, u)
    write_celldata(vtk, celldata)
end
"""
function open_vtk(filename::String, grid_or_dh; kwargs...)
    vtk = create_vtk_grid(filename, grid_or_dh; kwargs...)
    return VTKStream(filename, vtk, grid_or_dh)
end
# Makes it possible to use the `do`-block syntax
function open_vtk(f::Function, args...; kwargs...)
    vtks = open_vtk(args...; kwargs...)
    try
        f(vtks)
    finally
        close(vtks)
    end
end

isopen(vtks::VTKStream) = WriteVTK.isopen(vtks.vtk)

function show(io::IO, ::MIME"text/plain", vtks::VTKStream)
    open_str = isopen(vtk) ? "open" : "closed"
    if isa(vtks.grid_or_dh, DofHandler)
        dh_string = "DofHandler"
    elseif isa(vtks.grid_or_dh, Grid) 
        dh_string = "Grid" 
    else
        dh_string = string(typeof(vtks.grid_or_dh))
    end
    print(io, "VTKStream for the $open_str file \"$(vtk.path)\" based on a $dh_string")
end

getgrid(vtks::VTKStream{<:Any,<:DofHandler}) = vtks.grid_or_dh.grid
getgrid(vtks::VTKStream{<:Any,<:Grid}) = vtks.grid_or_dh

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
cell_to_vtkcell(::Type{Tetrahedron}) = VTKCellTypes.VTK_TETRA
cell_to_vtkcell(::Type{QuadraticTetrahedron}) = VTKCellTypes.VTK_QUADRATIC_TETRA
cell_to_vtkcell(::Type{Wedge}) = VTKCellTypes.VTK_WEDGE

nodes_to_vtkorder(cell::AbstractCell) = collect(cell.nodes)

function create_vtk_grid(filename::AbstractString, grid::Grid{dim,C,T}; kwargs...) where {dim,C,T}
    cls = MeshCell[]
    for cell in grid.cells
        celltype = Ferrite.cell_to_vtkcell(typeof(cell))
        push!(cls, MeshCell(celltype, nodes_to_vtkorder(cell)))
    end
    coords = reshape(reinterpret(T, getnodes(grid)), (dim, getnnodes(grid)))
    return vtk_grid(filename, coords, cls; kwargs...)
end
function create_vtk_grid(filename::AbstractString, dh::AbstractDofHandler; kwargs...)
    create_vtk_grid(filename, dh.grid; kwargs...)
end

function toparaview!(v, x::Vec{D}) where D
    v[1:D] .= x
end
function toparaview!(v, x::SecondOrderTensor{D}) where D
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
    return WriteVTK.vtk_point_data(vtk, out, name; component_names=get_component_names(S))
end
function _vtk_write_nodedata(vtk::WriteVTK.DatasetFile, nodedata::Vector{<:Real}, name::AbstractString)
    return WriteVTK.vtk_point_data(vtk, nodedata, name)
end
function _vtk_write_nodedata(vtk::WriteVTK.DatasetFile, nodedata::Matrix{<:Real}, name::AbstractString; component_names=nothing)
    return WriteVTK.vtk_point_data(vtk, nodedata, name; component_names=component_names)
end

function get_component_names(::Type{S}) where S
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


function _vtk_write_solution(vtkfile, dh, u::Vector, suffix)
    fieldnames = Ferrite.getfieldnames(dh)  # all primary fields

    for name in fieldnames
        data = reshape_to_nodes(dh, u, name)
        _vtk_write_nodedata(vtkfile, data, string(name, suffix))
    end
    return nothing
end

"""
    write_solution(vtks::VTKStream, u::Vector, suffix="")

Save the values at the nodes in the degree of freedom vector `u` to a 
the a vtk file for each field in `DofHandler` in `vtks`. 
`suffix` can be used to append to the fieldname.
`vtks` is typically created by the [`open_vtk`](@ref) function. 

`u` can also contain tensorial values, but each entry in `u` must correspond to a degree of freedom in `dh`,
see [`write_nodedata`](@ref) for details. This function should be used directly when exporting values already 
sorted by the nodes in the grid. 
"""
function write_solution(vtks::VTKStream, u, suffix="")
    _vtk_write_solution(vtks.vtk, vtks.grid_or_dh, u, suffix)
    return vtks
end

"""
    write_projected(vtks::VTKStream, proj::L2Projector, vals::Vector, name::AbstractString)

Write `vals` that have been projected with `proj` to the vtk file in `vtks`
"""
function write_projected(vtks::VTKStream, proj::L2Projector, vals, name)
    if getgrid(vtks) !== proj.dh.grid
        @warn("The grid saved in VTKStream and L2Projector are not aliased, no checks are performed to ensure that they are equal")
    end
    nodedata = reshape_to_nodes(proj, vals)
    @assert size(nodedata, 2) == getnnodes(proj.dh.grid)
    _vtk_write_nodedata(vtks.vtk, nodedata, name; component_names=get_component_names(eltype(vals)))
    return vtks
end

"""
    write_celldata(vtks::VTKStream, celldata, name)

Write the `celldata` that is ordered by the cells in the grid to the vtk file.
"""
function write_celldata(vtks::VTKStream, celldata, name)
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
function write_nodedata(vtks::VTKStream, nodedata, name)
    _vtk_write_nodedata(vtks.vtk, nodedata, name)
end


"""
    write_nodeset(vtks::VTKStream, nodeset::String)

Export nodal values of 1 for nodes in `nodeset`, and 0 otherwise
"""
function write_nodeset(vtks::VTKStream, nodeset::String)
    grid = getgrid(vtks)
    z = zeros(getnnodes(grid))
    z[collect(getnodeset(grid, nodeset))] .= 1.0
    WriteVTK.vtk_point_data(vtks.vtk, z, nodeset)
    return vtks
end

"""
    write_cellset(vtk)
    write_cellset(vtk, cellset::String)
    write_cellset(vtk, cellsets::Union{AbstractVector{String},AbstractSet{String})

Export all cell sets in the grid with name according to their keys and 
celldata 1 if the cell is in the set, and 0 otherwise. It is also possible to 
only export a single `cellset`, or multiple `cellsets`. 
"""
function write_cellset(vtks::VTKStream, cellsets=keys(grid.cellsets))
    grid = getgrid(vtks)
    z = zeros(getncells(grid))
    for cellset in cellsets
        z .= 0.0
        z[collect(getcellset(grid, cellset))] .= 1.0
        vtk_cell_data(vtks.vtk, z, cellset)
    end
    return vtks
end
write_cellset(vtks::VTKStream, cellset::String) = write_cellset(vtks, [cellset])

"""
    write_dirichlet(vtks::VTKStream, ch::ConstraintHandler)

Saves the dirichlet boundary conditions to a vtkfile.
Values will have a 1 where bcs are active and 0 otherwise
"""
function write_dirichlet(vtks::VTKStream, ch::ConstraintHandler)
    
    if getgrid(vtks) !== ch.dh.grid 
        @warn("The grid saved in VTKStream and ConstraintHandler are not aliased, no checks are performed to ensure that they are equal")
    end
    vtkfile = vtks.vtk

    unique_fields = []
    for dbc in ch.dbcs
        push!(unique_fields, dbc.field_name)
    end
    unique!(unique_fields)

    for field in unique_fields
        nd = getfielddim(ch.dh, field)
        data = zeros(Float64, nd, getnnodes(ch.dh.grid))
        for dbc in ch.dbcs
            dbc.field_name != field && continue
            if eltype(dbc.faces) <: BoundaryIndex
                functype = boundaryfunction(eltype(dbc.faces))
                for (cellidx, faceidx) in dbc.faces
                    for facenode in functype(ch.dh.grid.cells[cellidx])[faceidx]
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
        WriteVTK.vtk_point_data(vtkfile, data, string(field, "_bc"))
    end
    return vtkfile
end