"""
    DofHandler

A `DofHandler` takes care of the degrees of freedom in the system.

```jldoctest dh
julia> g = generate_grid(Tetrahedron, (2,2,2));

julia> dh = DofHandler(g)
```

We can now add fields of a certain name and dimension to the dofhandler:

```jldoctest dh
julia> push!(dh, :temperature, 1);

julia> push!(dh, :displacements, 3);
```

When we are done with adding the fields, we need to close it:

```jldoctest dh
julia> close!(dh)
DofHandler
  Fields:
    temperature dim: 1
    displacements dim: 3
  Total dofs: 108
  Dofs per cell: 16
```

Can now do queries:

```jldoctest dh
julia> ndofs(dh)
108

julia> ndofs_per_cell(dh)
16

julia> global_dofs = zeros(Int, ndofs_per_cell(dh))

julia> dofs_cell!(global_dofs, dh, 3);

julia> global_dofs
16-element Array{Int64,1}:
 13
 37
 53
  6
  7
  8
 14
 15
 16
 38
 39
 40
 54
 55
 56
```

Can use it to export

```jldoctest dh
julia> a = rand(ndofs(dh))

julia> vtkfile = vtk_grid(grid)

julia> vtk_point_data(vtkfile)
```

"""

mutable struct DofContainer
    name::Symbol
    ndims::Int
end

# TODO: Make this immutable?
mutable struct DofHandler{dim,T}
    dofs_nodes::Vector{Int}
    #dofs_cells::Vector{Int} # Alt1
    dofs_cells::Vector{Vector{Int}} # Alt2 

    nodes_offset::Vector{Int}
    #cells_offset::Vector{Int} # Alt1
    cells_offset::Vector{Vector{Int}} #Alt2

    field_names::Vector{Vector{DofContainer}} #cellgroups dofs
    
    closed::ScalarWrapper{Bool}
    grid::Grid{dim, T}
end

function DofHandler(m::Grid, alt::Int)
    field_names = Vector{Vector{DofContainer}}()
    for cellgroup in 1:length(m.cellgroups)
        push!(field_names, Vector{DofHandler}())
    end
    DofHandler(Vector{Int}(0), Vector{Vector{Int}}(0), Vector{Int}(0), Vector{Vector{Int}}(0), field_names, ScalarWrapper(false), m)
end

function Base.show(io::IO, dh::DofHandler)
    println(io, "DofHandler")
    for (cellgroup_id, cellgroup_dofs) in enumerate(dh.field_names)
        println(io, "    Cellgroup ", cellgroup_id, ", nel: ", length(dh.grid.cellgroups[cellgroup_id]))
        if length(cellgroup_dofs) == 0
            println(io, "      This cellgroup is empty")
        end
        for dof in cellgroup_dofs
            println(io, "      Name:", dof.name, ", dim: ", dof.ndims)
        end
    end
    if !isclosed(dh)
        println(io, "  Not closed!")
    else
        println(io, "  Total dofs: ", ndofs(dh))
        println(io, "  Total nels: ", getncells(dh.grid))
    end
end

ndofs(dh::DofHandler) = length(dh.dofs_nodes)
isclosed(dh::DofHandler) = dh.closed[]
dofs_node(dh::DofHandler, i::Int) = dh.dof_nodes[:, i]
ncellgroups(dh) = length(dh.grid.cellgroups)

# Stores the dofs for the cell with number `i` into the vector `global_dofs`
function celldofs!(global_dofs::Vector{Int}, dh::DofHandler, cellgroup::Int, i::Int)
    @assert isclosed(dh)
    @assert length(global_dofs) == ndofs_per_cell(dh, cellgroup)
    celldofs = getcelldofs(dh, cellgroup, i)

    @inbounds for j in 1:ndofs_per_cell(dh, cellgroup)
        global_dofs[j] = celldofs[j]
    end
    return global_dofs
end

#getcelldofs(dh::DofHandler, cellid::Int) = dh.dofs_cells[(dh.cells_offset[cellid]+1):dh.cells_offset[cellid+1]]
export getcelldofs
getcelldofs(dh::DofHandler, cellgroup::Int, cellid::Int) = dh.dofs_cells[cellgroup][(dh.cells_offset[cellgroup][cellid]+1):dh.cells_offset[cellgroup][cellid+1]]
getnodedofs(dh::DofHandler, nodeid::Int) = dh.dofs_nodes[(dh.nodes_offset[nodeid]+1):dh.nodes_offset[nodeid+1]]

# Add a field to all cellgroups, ex `push!(dh, :u, 3)`
function Base.push!(dh::DofHandler, name::Symbol, dim::Int)
    @assert !isclosed(dh)
    for i in 1:ncellgroups(dh)
        push!(dh.field_names[i], DofContainer(name, dim))
    end
    return dh
end

# Add a field to a specific cellgroup, ex `push!(dh, SOLIDS, :u, 3)`
function Base.push!(dh::DofHandler, cellgroup_id::Int, name::Symbol, dim::Int)
    @assert !isclosed(dh)
    #if name in dh.field_names[cellgroup_id]
    #    error("duplicate field name")
    #end

    push!(dh.field_names[cellgroup_id], DofContainer(name, dim))
    return dh
end

# Computes the number of dofs from which the field starts data
# For example [ux, uy, uz, T] --> dof_offset(dh, :temperature) = 4
function dof_offset(dh::DofHandler, cellgroupid::Int, field_name::Symbol)
    offset = 0
    i = 0
    for dof in dh.field_names[cellgroupid]
        i += 1
        if dof.name == field_name
            return offset
        else
            offset += dof.ndims
        end
    end
    error("unexisting field name $field_name among $(dh.field_names)")
end

function ndim(dh::DofHandler, cellgroup::Int, field_name::Symbol)

    for dof in dh.field_names[cellgroup]
        if dof.name == field_name
            return dof.ndims
        end
    end
    error("unexisting field name $field_name among $(dh.field_names)")

end

#All cell in a cellgroup has the same number of dofs
function ndofs_per_cell(dh::DofHandler, cellgroup::Int)
    nnodes_per_cell = nnodes(dh.grid.cellgroups[cellgroup][1])
    sum = 0
    for dof in dh.field_names[cellgroup]
        sum += dof.ndims
    end
    return sum*nnodes_per_cell
    
end

function ndofs_per_node(dh::DofHandler, nodeid::Int)
    @assert isclosed(dh) 
    return length(getnodedofs(dh, nodeid))
end

function ndims_cellgroup(dh::DofHandler, cellgroup::Int)
    sum = 0
    for dof in dh.field_names[cellgroup]
        sum += dof.ndims
    end
    return sum
end

function close!(dh::DofHandler)
    @assert !isclosed(dh)  

    #Get number of dofs per node and cell
    node_ndims = Vector{Int}(getnnodes(dh.grid))
    cells_ndims = Vector{Vector{Int}}()
    for (cellgroup_index, cellgroup) in enumerate(dh.grid.cellgroups)
        
        ndofs_per_cell_in_cellgroup = ndofs_per_cell(dh, cellgroup_index)
        ndofs_per_node = ndims_cellgroup(dh, cellgroup_index)
        cellgroup_ndims = Vector{Int}()
        for (cellid, cell) in enumerate(cellgroup)
            for node_id in cell.nodes
                node_ndims[node_id] = ndofs_per_node
            end
            push!(cellgroup_ndims, ndofs_per_cell_in_cellgroup)
        end
        push!(cells_ndims, cellgroup_ndims)

    end 

    #Create the vector nodes_offset and cells_offset
    for cellgroup_ndim in cells_ndims

        tmp = Vector{Int}()
        push!(tmp,0)
        counter = 0
        for cell_ndim in cellgroup_ndim
            counter += cell_ndim
            push!(tmp, counter)
        end
        push!(dh.cells_offset, tmp)
    end

    ndofs_counter = 0;
    append!(dh.nodes_offset,0)
    for ndims in node_ndims
        ndofs_counter += ndims
        append!(dh.nodes_offset, ndofs_counter)
    end

    #Create dofs_nodes 
    dh.dofs_nodes = 1:ndofs_counter

    #Create dofs_cells
    for cellgroup in dh.grid.cellgroups

        tmp = Vector{Int}()
        for (cellid, cell) in enumerate(cellgroup)
            for node_id in cell.nodes
                nodedofs = getnodedofs(dh, node_id)
                append!(tmp, nodedofs)
            end
        end
        push!(dh.dofs_cells, tmp)
    end

    dh.closed[] = true
    return dh
end


getnvertices(::Type{JuAFEM.Cell{dim, N, M}}) where {dim, N, M} = N

# Creates a sparsity pattern from the dofs in a dofhandler.
# Returns a sparse matrix with the correct pattern.
@inline create_sparsity_pattern(dh::DofHandler) = _create_sparsity_pattern(dh, false)
@inline create_symmetric_sparsity_pattern(dh::DofHandler) = Symmetric(_create_sparsity_pattern(dh, true), :U)

function _create_sparsity_pattern(dh::DofHandler, sym::Bool)
    #sym not used...
    I = Int[];
    J = Int[];
    for (cellgroup_id, cellgroup) in enumerate(dh.grid.cellgroups)     
        n = ndofs_per_cell(dh, cellgroup_id)
        global_dofs = zeros(Int, n)
        for element_id in 1:length(cellgroup)
            celldofs!(global_dofs, dh, cellgroup_id, element_id)
            @inbounds for j in 1:n, i in 1:n
                dofi = global_dofs[i]
                dofj = global_dofs[j]
                push!(I, dofi)
                push!(J, dofj)
            end
        end
    end

    for d in 1:ndofs(dh)
        push!(I, d)
        push!(J, d)
    end
    V = zeros(length(I))
    K = sparse(I, J, V)
    return K
end

WriteVTK.vtk_grid(filename::AbstractString, dh::DofHandler) = vtk_grid(filename, dh.grid)

# Exports the FE field `u` to `vtkfile`
function WriteVTK.vtk_point_data(vtkfile, dh::DofHandler, u::Vector)
    offset = 0
    for i in 1:length(dh.field_names)
        ndim_field = dh.dof_dims[i]
        space_dim = ndim_field == 2 ? 3 : ndim_field
        data = zeros(space_dim, getnnodes(dh.grid))
        for j in 1:size(dh.dofs_nodes, 2)
            for k in 1:ndim_field
                data[k, j] = u[dh.dofs_nodes[k + offset, j]]
            end
        end
        vtk_point_data(vtkfile, data, string(dh.field_names[i]))
        offset += ndim_field
    end
    return vtkfile
end

# Exports the FE field `u` to `vtkfile`
function WriteVTK.vtk_point_data(vtkfile, dh::DofHandler, u::Vector)
    offset = 0
    for (cellgroupid, cellgroup) in enumerate(dh.grid.cellgroups)
        offset = 0
        for dof in dh.field_names[cellgroupid]
            ndim_field = dof.ndims
            space_dim = ndim_field == 2 ? 3 : ndim_field #???
            data = zeros(space_dim, getnnodes(dh.grid))
            for j in 1:getnnodes(dh.grid)
                nodedofs = getnodedofs(dh, j)
                for k in 1:ndim_field
                    data[k, j] = u[nodedofs[k + offset]]
                end
            end
            vtk_point_data(vtkfile, data, string(dof.name))
            offset += ndim_field
        end
    end
    return vtkfile
end
