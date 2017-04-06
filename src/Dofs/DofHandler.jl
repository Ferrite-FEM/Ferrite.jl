export DofHandler, close!, ndofs, ndofs_per_cell, celldofs!, celldofs,
       create_sparsity_pattern, create_symmetric_sparsity_pattern

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

julia> vtk_point_data(vtkfile
```

"""
# TODO: Make this immutable?
type DofHandler{dim, N, T, M}
    dofs_cells::Matrix{Int}
    field_names::Vector{Symbol}
    field_interpolations,::Vector{Interpolation}
    field_dims::Vector{Int}
    closed::ScalarWrapper{Bool}
    grid::Grid{dim, N, T, M}
end

function create_dofs!{dim}(dh::DofHandler{dim})
    dofs_per_element = 0
    for (space, dim) in zip(dh.field_interpolations, dh.field_dims)
        dofs_per_element += get_nbasefunctions(space) * dim
    end

    element_dofs = zeros(Int, dofs_per_element, getncells(dh.grid))

    free_dof_number = 1
    for space in dh.dof_spaces
        if dim >= 1
            free_dof_number = create_vertex_dofs!(dh.grid, space, element_dofs, free_dof_number)
            free_dof_number = create_edge_dofs!(dh, space, free_dof_number)
        end
        if dim >= 2
            free_dof_number = create_face_dofs!(dh, space, free_dof_number)
        end

        if dim >= 3
            free_dof_number = create_cell_dofs!(dh, space, free_dof_number)
        end
    end
end

function create_vertex_dofs!(grid, space, free_dof_number)
    # vertex -> dof number
    vertex_dofs = Dict{Int, Int}()
    for element in 1:n_elements
        vertices = getcells(dh.grid, element).nodes

        for vertex in
            if !haskey(vertex_dofs, vertex)
                vertex_dofs[vertex] = free_dof_number
                element_dofs[element, ]
                free_dof_number += 1
            end
        end
    end
end

function DofHandler(grid::Grid)
    DofHandler(Matrix{Int}(0, 0), Matrix{Int}(0, 0), Symbol[], Interpolation[], ScalarWrapper(false), grid)
end

function show(io::IO, dh::DofHandler)
    println(io, "DofHandler")
    println(io, "  Fields:")
    for i in 1:length(dh.field_names)
        println(io, "    ", dh.field_names[i], " interpolation: ", dh.dof_spaces[i])
    end
    if !isclosed(dh)
        println(io, "  Not closed!")
    else
        println(io, "  Total dofs: ", ndofs(dh))
        print(io, "  Dofs per cell: ", ndofs_per_cell(dh))
    end
end

ndofs_per_cell(dh::DofHandler) = size(dh.dofs_cells, 1)
isclosed(dh::DofHandler) = dh.closed[]
dofs_node(dh::DofHandler, i::Int) = dh.dof_nodes[:, i]

# Stores the dofs for the cell with number `i` into the vector `global_dofs`
function celldofs!(global_dofs::Vector{Int}, dh::DofHandler, i::Int)
    @assert isclosed(dh)
    @assert length(global_dofs) == ndofs_per_cell(dh)
    @inbounds for j in 1:ndofs_per_cell(dh)
        global_dofs[j] = dh.dofs_cells[j, i]
    end
    return global_dofs
end

# Add a collection of fields
function push!(dh::DofHandler, names::Vector{Interpolation}, dims)
    @assert length(names) == length(dims)
    for i in 1:length(names)
        push!(dh, names[i], dims[i])
    end
end

# Add a field to the dofhandler ex `push!(dh, :u, 3)`
function push!(dh::DofHandler, name::Symbol, interpolation::Interpolation, dim)
    @assert !isclosed(dh)
    if name in dh.field_names
        error("duplicate field name")
    end
    push!(dh.field_names, name)
    push!(dh.dof_spaces, interpolation)
    return dh
end

# Computes the number of dofs from which the field starts data
# For example [ux, uy, uz, T] --> dof_offset(dh, :temperature) = 4
function dof_offset(dh::DofHandler, field_name::Symbol)
    offset = 0
    i = 0
    for name in dh.field_names
        i += 1
        if name == field_name
            return offset
        else
            offset += dh.dof_dims[i]
        end
    end
    error("unexisting field name $field_name among $(dh.field_names)")
end

function ndim(dh::DofHandler, field_name::Symbol)
    i = 0
    for name in dh.field_names
        i += 1
        if name == field_name
            return dh.dof_dims[i]
        end
    end
    error("unexisting field name $field_name among $(dh.field_names)")
end

function close!(dh::DofHandler)
    @assert !isclosed(dh)
    create_element_dofs!(dh)
    dh.closed[] = true
    return dh
end


# Creates a sparsity pattern from the dofs in a dofhandler.
# Returns a sparse matrix with the correct pattern.
@inline create_sparsity_pattern(dh::DofHandler) = _create_sparsity_pattern(dh, false)
@inline create_symmetric_sparsity_pattern(dh::DofHandler) = Symmetric(_create_sparsity_pattern(dh, true), :U)

function _create_sparsity_pattern(dh::DofHandler, sym::Bool)
    ncells = getncells(dh.grid)
    n = ndofs_per_cell(dh)
    N = sym ? div(n*(n+1), 2) * ncells : n^2 * ncells
    I = Int[]; sizehint!(I, N)
    J = Int[]; sizehint!(J, N)
    global_dofs = zeros(Int, n)
    for element_id in 1:ncells
        celldofs!(global_dofs, dh, element_id)
        @inbounds for j in 1:n, i in 1:n
            dofi = global_dofs[i]
            dofj = global_dofs[j]
            sym && (dofi > dofj && continue)
            push!(I, dofi)
            push!(J, dofj)
        end
    end
    V = zeros(length(I))
    K = sparse(I, J, V)
    return K
end


vtk_grid(filename::AbstractString, dh::DofHandler) = vtk_grid(filename, dh.grid)

# Exports the FE field `u` to `vtkfile`
function vtk_grid(filename::AbstractString, dh::DofHandler, u::Vector)
    vtkfile = vtk_grid(filename, dh)
    vtk_point_data(vtkfile, dh, u)
    return vtkfile
end

function vtk_point_data(vtkfile, dh::DofHandler, u::Vector)
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
