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

DEBUG = true

# We use sorted tuples as dictionary keys for edges and surfaces.
# The tuple contain all the global vertices for that edge / surface
immutable SortedSVector{N, T}
  n::SVector{N, T}
  SortedSVector(n::SVector{N, T}) = new{N,T}(SVector(swapsort(n.data)))
end
SortedSVector{N,T}(n::SVector{N, T}) = SortedSVector{N,T}(n)


immutable GlobalNode{N}
    dof_numbers::SVector{N, Int} # Dof numbers in the node
end

immutable GlobalEdge{N}
    sign::Bool                  # "sign" of the local edge that created dofs for this global edge
    dof_numbers::SVector{N, Int} # Dof numbers in the edge
end

immutable GlobalSurface{N}
    sign::Bool                  # "sign" of the surface
    offset::Int
    dof_numbers::SVector{N, Int} # Number of the dofs
end

immutable DofStorage{N, M, O}
  # The dicts below are for looking up the object which stores the dofs for
  # a global vertex / edge / surface
  global_vertex_dofs::Dict{Int, GlobalNode{N}}
  global_edge_dofs::Dict{SortedSVector{2}, GlobalEdge{M}}
  global_surface_dofs::Dict{SortedSVector{3}, GlobalSurface{O}}
  # Cell dofs are never shared and hence no need for a global lookup
  # The arrays below are cell, (vertex / edge / surface / cell) -> dofs
  cell_vertex_dofs::Array{Int, 3}
  cell_edge_dofs::Array{Int, 3}
  cell_surface_dofs::Array{Int, 3}
  # There is only one set of cell dofs for each cell so one rank lower
  cell_cell_dofs::Array{Int, 2}
end


function DofStorage(grid, ip::Interpolation, n_fieldcomponents)
  n_vert = get_n_vertexdofs(ip) * n_fieldcomponents
  n_edge = get_n_edgedofs(ip)   * n_fieldcomponents
  n_surface = get_n_surfacedofs(ip)   * n_fieldcomponents
  O = get_n_celldofs(ip)   * n_fieldcomponents
  cell_vertex_dofs = zeros(Int, n_vert, get_n_vertices(ip), getncells(grid))
  cell_edge_dofs =   zeros(Int, n_edge, get_n_edges(ip),    getncells(grid))
  cell_surface_dofs =zeros(Int, n_surface, get_n_edges(ip), getncells(grid))
  cell_cell_dofs =   zeros(Int, O,                          getncells(grid))
  DofStorage(Dict{Int, GlobalNode{n_vert}}(),
             Dict{SortedSVector{2}, GlobalEdge{n_edge}}(),
             Dict{SortedSVector{3}, GlobalSurface{n_surface}}(),
             cell_vertex_dofs,
             cell_edge_dofs,
             cell_surface_dofs,
             cell_cell_dofs)
end

get_global_face_dofs(ip::Interpolation{1}, d::DofStorage) = d.global_vertex_dofs
get_global_face_dofs(ip::Interpolation{2}, d::DofStorage) = d.global_edge_dofs
get_global_face_dofs(ip::Interpolation{3}, d::DofStorage) = d.global_surface_dofs

# DofHandler
type DofHandler{dim, N, T, M}
    dofs_cells::Matrix{Int}
    field_names::Vector{Symbol}
    interpolations::Vector{Interpolation}
    dof_dims::Vector{Int}
    closed::JuAFEM.ScalarWrapper{Bool}
    dof_storage::Vector{DofStorage}
    grid::Grid{dim, N, T, M}
end

function DofHandler(grid::Grid)
    DofHandler(Matrix{Int}(0, 0), Symbol[], Interpolation[], Int[], JuAFEM.ScalarWrapper(false), DofStorage[], grid)
end

isclosed(dh::DofHandler) = dh.closed[]
function find_field(dh::DofHandler, field::Symbol)
  j = findfirst(i -> i == field, dh.field_names)
  j == 0 && error("did not find field $field")
  return j
end

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

# Returns the global vertices and the orientation of the edge. The orientation is -1 (false) if the global vertices goes from a higher to a lower vertex
function get_global_edgevertices_and_orientation(ip::Interpolation, global_vertices, edge_number::Int)
  local_edge_numbers = getedgelist(ip)[edge_number]
  global_vertices_edge = SVector(global_vertices[local_edge_numbers[1]], global_vertices[local_edge_numbers[2]])
  return SortedSVector(global_vertices_edge), global_vertices_edge[1] < global_vertices_edge[2]
end

function add_cell_dofs{N, M}(grid, ip::Interpolation, dofstorage::DofStorage{N, M}, free_dof::Int)
  get_n_celldofs(ip) <= 0 && return
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
  get_n_edgedofs(ip) <= 0 && return
  for i in 1:JuAFEM.getncells(grid)
    cell = grid.cells[i]
    for local_edge in 1:get_n_edges(ip)
      global_edgevertices, orientation = get_global_edgevertices_and_orientation(ip, cell.nodes, local_edge)
      if !haskey(dofstorage.global_edge_dofs, global_edgevertices)
        new_dofs = SVector(ntuple(i -> free_dof + (i-1), Val{M}))
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
  get_n_vertexdofs(ip) <= 0 && return
  for i in 1:JuAFEM.getncells(grid)
    cell = grid.cells[i]
    for (local_vertex, global_vertex) in enumerate(cell.nodes)
      if !haskey(dofstorage.global_vertex_dofs, global_vertex)
        current_dofs = free_dof
        new_dofs = SVector(ntuple(i -> free_dof + (i-1), Val{N}))
        DEBUG && println("Adding node dofs: $new_dofs for global vertex: $global_vertex local vertex: $local_vertex")
        @show new_dofs
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


function insert_into_celldof_matrix(cell_dofs::Matrix, storage, k, g::Grid, spacedim)
   l = 0
    for i in 1:getncells(g)
        l = k
        # Vertices
        for local_vertex in 1:size(storage.cell_vertex_dofs, 2)
          for dof in 1:size(storage.cell_vertex_dofs, 1)
            dof = storage.cell_vertex_dofs[dof, local_vertex, i]
            cell_dofs[l, i] = dof
            DEBUG && println("For cell $i, adding vertex dof $dof to number $l")
            l += 1
          end
        end
        if spacedim >= 2
          # Edges
          for local_edge in 1:size(storage.cell_edge_dofs, 2)
            for dof in 1:size(storage.cell_edge_dofs, 1)
              dof = storage.cell_edge_dofs[dof, local_edge, i]
              cell_dofs[l, i] = dof
              DEBUG && println("For cell $i, adding edge dof $dof to number $l")
              l += 1
            end
          end
        end

        #=
        if spacedim >= 3
          # Surfaces
          for local_vertex in 1:size(storage.cell_vertex_dofs, 2)
            for dof in 1:size(storage.cell_vertex_dofs, 1)
              dof = storage.cell_vertex_dofs[dof, local_vertex, i]
              cell_dofs[l, i] = dof
              DEBUG && println("For cell $i, adding dof $dof to number $l")
              l += 1
            end
          end
          =#

      # Cells
        for dof in 1:size(storage.cell_cell_dofs, 1)
          dof = storage.cell_cell_dofs[dof, i]
          cell_dofs[l, i] = dof
          DEBUG && println("For cell $i, adding cell dof $dof to number $l")
          l += 1
        end
    end
    return k + l - 1
end


function close!{dim}(dh::DofHandler{dim})
  free_dof = 1
  storages = []
  @show free_dof
  for i in 1:length(dh.interpolations)
    ip = dh.interpolations[i]
    storage = DofStorage(dh.grid, ip, dh.dof_dims[i])
    push!(storages, storage)
    free_dof = add_node_dofs(dh.grid, ip, storage, free_dof)
    if dim >= 2
      free_dof = add_edge_dofs(dh.grid, ip, storage, free_dof)
    end

    if dim >= 3
        # free_dof = add_surface_dofs(dh.grid, ip, storage, free_dof)
    end
    free_dof = add_cell_dofs(dh.grid, ip, storage, free_dof)
  end

  dofs_per_element = 0
  for i in 1:length(dh.interpolations)
      dofs_per_element += getnbasefunctions(dh.interpolations[i]) * dh.dof_dims[i]
  end

  cell_dofs = zeros(Int, dofs_per_element, getncells(dh.grid))

  k = 1
  for i in 1:length(dh.interpolations)
      k = insert_into_celldof_matrix(cell_dofs, storages[i], k, dh.grid, dim)
  end

  # Renumber dofs
  #=
  free_dof = 1
  dof_renumber = Dict{Int, Int}()
  for i in eachindex(cell_dofs)
    dof = cell_dofs[i]
    if !haskey(dof_renumber, dof)
      dof_renumber[dof] = free_dof
      cell_dofs[i] = free_dof
      free_dof += 1
    else
      cell_dofs[i] = dof_renumber[dof]
    end
  end
  =#

  dh.dofs_cells = cell_dofs
  dh.dof_storage = storages
  dh.closed[] = true
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
