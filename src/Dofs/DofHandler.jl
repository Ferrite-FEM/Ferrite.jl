export DofHandler, close!, ndofs, ndofs_per_cell, celldofs!, celldofs,
       create_sparsity_pattern, create_symmetric_sparsity_pattern

import Base.@pure

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

const DEBUG = true

@pure get_n_edges(ip::Interpolation) = get_n_edges(typeof(ip))
@pure get_n_edgedofs(ip::Interpolation) = get_n_edgedofs(typeof(ip))
@pure get_n_vertices(ip::Interpolation) = get_n_vertices(typeof(ip))
@pure getedgelist(ip::Interpolation) = getedge(typeof(ip))
@pure get_n_faces{order, refshape}(::Type{Interpolation{1, refshape, order}}) = 0
@pure get_n_faces{order, refshape}(::Type{Interpolation{2, refshape, order}}) = 0

@pure get_n_edges{order}(::Type{Lagrange{1, RefCube, order}}) = 1
@pure get_n_edges{order}(::Type{Lagrange{2, RefCube, order}}) = 4
@pure get_n_edges{order}(::Type{Lagrange{3, RefCube, order}}) = 12

@pure get_n_vertices{order}(::Type{Lagrange{1, RefCube, order}}) = 2
@pure get_n_vertices{order}(::Type{Lagrange{2, RefCube, order}}) = 4
@pure get_n_vertices{order}(::Type{Lagrange{3, RefCube, order}}) = 8

@pure get_n_faces(ip::Interpolation) = get_n_faces(typeof(ip))

@pure get_n_faces{order}(::Type{Lagrange{3, RefCube, order}}) = 6

@pure get_n_cells(ip::Interpolation) = get_n_cells(typeof(ip))
@pure get_n_cells{dim, order}(::Type{Lagrange{dim, RefCube, order}}) = 1

@pure getedgelist{order}(::Type{Lagrange{1, RefCube, order}}) = ((1,), (2,))
@pure getedgelist{order}(::Type{Lagrange{2, RefCube, order}}) = ((1,2), (2, 3), (3,4), (4, 1))

# Computes the number of degrees of freedom that sits in every edge for a SINGLE edge
@pure get_n_edgedofs{dim, order}(T::Type{Lagrange{dim, RefCube, order}}) = (order - 1)

@pure get_n_celldofs(ip::Interpolation) = get_n_celldofs(typeof(ip))
@pure get_n_celldofs{dim, order}(T::Type{Lagrange{dim, RefCube, order}}) = (order - 1)^dim

# DofHandler
type DofHandler{dim, N, T, M}
    dofs_cells::Matrix{Int}
    field_names::Vector{Symbol}
    interpolations::Vector{Interpolation}
    dof_dims::Vector{Int}
    closed::JuAFEM.ScalarWrapper{Bool}
    grid::Grid{dim, N, T, M}
end

function DofHandler(grid::Grid)
    DofHandler(Matrix{Int}(0, 0), Symbol[], Interpolation[], Int[], JuAFEM.ScalarWrapper(false), grid)
end

isclosed(dh::DofHandler) = dh.closed[]


# Add a field to the dofhandler ex `push!(dh, :u, 3)`
function Base.push!(dh::DofHandler, name::Symbol, interpolation::Interpolation, dim::Int)
    @assert !isclosed(dh)
    if name in dh.field_names
        error("duplicate field name")
    end
    push!(dh.field_names, name)
    push!(dh.interpolations, interpolation)
    push!(dh.dof_dims, dim)
    return dh
end

function Base.show(io::IO, dh::DofHandler)
    println(io, "DofHandler")
    println(io, "  Fields:")
    for i in 1:length(dh.field_names)
        println(io, "    ", dh.field_names[i], " with interpolation: ", typeof(dh.interpolations[i]))
    end
    if !isclosed(dh)
        println(io, "  Not closed!")
    else
        println(io, "  Total dofs: ", ndofs(dh))
        print(io, "  Dofs per cell: ", ndofs_per_cell(dh))
    end
end

immutable SortedNTuple{N, T}
  n::NTuple{N, T}
  SortedNTuple(n::NTuple{N, T}) = new{N,T}(swapsort(n))
end
SortedNTuple{N,T}(n::NTuple{N, T}) = SortedNTuple{N,T}(n)

immutable GlobalNode{N}
    dof_numbers::NTuple{N, Int} # Number of the dofs
end

immutable GlobalEdge{N}
    sign::Bool                  # "sign" of the edge
    dof_numbers::NTuple{N, Int} # Number of the dofs
end

# Returns the global vertices and the orientation of the edge. The orientation is -1 (false) if the global vertices goes from a higher to a lower vertex
function get_global_edgevertices_and_orientation{dim}(ip::Interpolation{dim, RefCube}, global_vertices, edge_number::Int)
  local_edge_numbers = getedgelist(typeof(ip))[edge_number]
  global_vertices_edge = global_vertices[local_edge_numbers[1]], global_vertices[local_edge_numbers[2]]
  return SortedNTuple(global_vertices_edge), global_vertices_edge[1] < global_vertices_edge[2]
end

immutable DofStorage{N, M}
  global_vertex_dofs::Dict{Int, GlobalNode{N}}
  global_edge_dofs::Dict{SortedNTuple{2}, GlobalEdge{M}}
  # Cell dofs are never shared and hence no need for a global lookup
  #
  cell_vertex_dofs::Array{Int, 3}
  cell_edge_dofs::Array{Int, 3}
  cell_cell_dofs::Array{Int, 2}
end

function DofStorage(grid, ip::Interpolation, dim)
  N = dim
  M = get_n_edgedofs(ip) * dim
  cell_vertex_dofs = zeros(Int, dim,                  get_n_vertices(ip), JuAFEM.getncells(grid))
  cell_edge_dofs =   zeros(Int, get_n_edgedofs(ip)*dim, get_n_edges(ip)   , JuAFEM.getncells(grid))
  cell_cell_dofs =   zeros(Int, get_n_celldofs(ip)*dim, JuAFEM.getncells(grid))
  DofStorage(Dict{Int, GlobalNode{N}}(),
             Dict{SortedNTuple{2}, GlobalEdge{M}}(),
             cell_vertex_dofs,
             cell_edge_dofs,
             cell_cell_dofs)
end

function add_cell_dofs{N, M}(grid, ip::Interpolation, dofstorage::DofStorage{N, M}, free_dof::Int)
  for i in 1:JuAFEM.getncells(grid)
    cell = grid.cells[i]
    new_dofs = free_dof:free_dof + size(dofstorage.cell_cell_dofs, 1) -1
    free_dof += length(new_dofs)
    DEBUG && println("Adding cell dofs: $(collect(new_dofs)), for global cell $i")
    dofstorage.cell_cell_dofs[:, i] = new_dofs
  end # cells
  return free_dof
end

function add_edge_dofs{N, M}(grid, ip::Interpolation, dofstorage::DofStorage{N, M}, free_dof::Int)
  for i in 1:JuAFEM.getncells(grid)
    cell = grid.cells[i]
    for local_edge in 1:get_n_edges(typeof(ip))
      global_edgevertices, orientation = get_global_edgevertices_and_orientation(ip, cell.nodes, local_edge)
      if !haskey(dofstorage.global_edge_dofs, global_edgevertices)
        new_dofs = ntuple(i -> free_dof + (i-1), Val{M})
        dofstorage.global_edge_dofs[global_edgevertices] = GlobalEdge{M}(orientation, new_dofs)
        current_dofs, same_orientation = new_dofs, true
        DEBUG && println("Adding edge dofs: $current_dofs, for global edge $(global_edgevertices.n)  local edge: $local_edge, orientation: $orientation")
        free_dof += M
      else
        global_edge_dofs = dofstorage.global_edge_dofs[global_edgevertices]
        current_dofs = global_edge_dofs.dof_numbers
        same_orientation = global_edge_dofs.sign == orientation
        DEBUG && println("Reusing edge dofs: $current_dofs, for global edge $(global_edgevertices.n) local edge: $local_edge, same orientation: $same_orientation")
      end
      if !same_orientation
        current_dofs = reverse(current_dofs)
      end
      dofstorage.cell_edge_dofs[:, local_edge, i] = collect(current_dofs) # TODO: collect = blä
    end
  end # cells
  return free_dof
end

function add_node_dofs{N, M}(grid, ip::Interpolation, dofstorage::DofStorage{N, M}, free_dof::Int)
  for i in 1:JuAFEM.getncells(grid)
    cell = grid.cells[i]
    for (local_vertex, global_vertex) in enumerate(cell.nodes)
      if !haskey(dofstorage.global_vertex_dofs, global_vertex)
        current_dofs = free_dof
        new_dofs = ntuple(i -> free_dof + (i-1), Val{N})
        DEBUG && println("Adding node dofs: $new_dofs for global vertex: $global_vertex local vertex: $local_vertex")
        dofstorage.global_vertex_dofs[global_vertex] = GlobalNode{N}(new_dofs)
        current_dofs = new_dofs
        free_dof += N
      else
        current_dofs = dofstorage.global_vertex_dofs[global_vertex].dof_numbers
        DEBUG && println("Reusing node dofs $current_dofs for global vertex: $global_vertex local vertex: $local_vertex")
      end
      dofstorage.cell_vertex_dofs[:, local_vertex, i] = collect(current_dofs) # TODO: collect = blä
    end
  end
  return free_dof
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
