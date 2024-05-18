# Sets

_check_setname(dict, name) = haskey(dict, name) && throw(ArgumentError("there already exists a set with the name: $name"))
_warn_emptyset(set, name) = length(set) == 0 && @warn("no entities added to the set with name: $name")

"""
    addcellset!(grid::AbstractGrid, name::String, cellid::Union{Set{Int}, Vector{Int}})
    addcellset!(grid::AbstractGrid, name::String, f::function; all::Bool=true)

Adds a cellset to the grid with key `name`.
Cellsets are typically used to define subdomains of the problem, e.g. two materials in the computational domain.
The `DofHandler` can construct different fields which live not on the whole domain, but rather on a cellset.
`all=true` implies that `f(x)` must return `true` for all nodal coordinates `x` in the cell if the cell
should be added to the set, otherwise it suffices that `f(x)` returns `true` for one node. 

```julia
addcellset!(grid, "left", Set((1,3))) #add cells with id 1 and 3 to cellset left
addcellset!(grid, "right", x -> norm(x[1]) < 2.0 ) #add cell to cellset right, if x[1] of each cell's node is smaller than 2.0
```
"""
function addcellset!(grid::AbstractGrid, name::String, cellid::Union{Set{Int},Vector{Int}})
    _addset!(grid, name, cellid, getcellsets(grid))
end

function addcellset!(grid::AbstractGrid, name::String, f::Function; all::Bool=true)
    _addset!(grid, name, create_cellset(grid, f; all), getcellsets(grid))
end

"""
    addnodeset!(grid::AbstractGrid, name::String, nodeid::Union{Vector{Int}, Set{Int}})
    addnodeset!(grid::AbstractGrid, name::String, f::Function)    

Adds a `nodeset::Set{Int}` to the `grid`'s `nodesets` with key `name`. Has the same interface as `addcellset`. 
However, instead of mapping a cell id to the `String` key, a set of node ids is returned.
"""
addnodeset!(grid::AbstractGrid, name::String, nodeid::Union{Vector{Int}, Set{Int}}) = 
    _addset!(grid, name, nodeid, getnodesets(grid))

addnodeset!(grid::AbstractGrid, name::String, f::Function) = 
    _addset!(grid, name, create_nodeset(grid, f), getnodesets(grid))

"""
    addfacetset!(grid::AbstractGrid, name::String, faceid::Union{Set{FacetIndex},Vector{FacetIndex}})
    addfacetset!(grid::AbstractGrid, name::String, f::Function; all::Bool=true) 

Adds a facetset to the grid with key `name`.
A facetset maps a `String` key to a `Set` of tuples corresponding to `(global_cell_id, local_facet_id)`.
Facetsets can be used to initialize `Dirichlet` boundary conditions for the `ConstraintHandler`.
`all=true` implies that `f(x)` must return `true` for all nodal coordinates `x` on the facet if the facet
should be added to the set, otherwise it suffices that `f(x)` returns `true` for one node. 

```julia
addfacetset!(grid, "right", Set((FacetIndex(2,2), FacetIndex(4,2)))) #see grid manual example for reference
addfacetset!(grid, "clamped", x -> norm(x[1]) ≈ 0.0) #see incompressible elasticity example for reference
```
"""
addfacetset!(grid::AbstractGrid, name::String, set::Union{Set{FacetIndex},Vector{FacetIndex}}) = 
    _addset!(grid, name, set, getfacetsets(grid))

addfacetset!(grid::AbstractGrid, name::String, f::Function; all::Bool=true) = 
    _addset!(grid, name, create_facetset(grid, f; all=all), getfacetsets(grid))

"""
    addvertexset!(grid::AbstractGrid, name::String, faceid::Union{Set{FaceIndex},Vector{FaceIndex}})
    addvertexset!(grid::AbstractGrid, name::String, f::Function) 

Adds a vertexset to the grid with key `name`.
A vertexset maps a `String` key to a `Set` of tuples corresponding to `(global_cell_id, local_vertex_id)`.
Vertexsets can be used to initialize `Dirichlet` boundary conditions for the `ConstraintHandler`.

```julia
addvertexset!(grid, "right", Set((VertexIndex(2,2), VertexIndex(4,2))))
addvertexset!(grid, "clamped", x -> norm(x[1]) ≈ 0.0)
```
"""
addvertexset!(grid::AbstractGrid, name::String, set::Union{Set{VertexIndex},Vector{VertexIndex}}) = 
    _addset!(grid, name, set, getvertexsets(grid))

addvertexset!(grid::AbstractGrid, name::String, f::Function) = 
    _addset!(grid, name, create_vertexset(grid, f; all=true), getvertexsets(grid))

function _addset!(grid::AbstractGrid, name::String, _set, dict::Dict)
    _check_setname(dict, name)
    set = Set(_set)
    _warn_emptyset(set, name)
    dict[name] = set
    grid
end

"""
addboundaryvertexset!(grid::AbstractGrid, topology::ExclusiveTopology, name::String, f::Function; all::Bool=true)

Adds a boundary vertexset to the grid with key `name`.
A vertexset maps a `String` key to a `Set` of tuples corresponding to `(global_cell_id,
local_vertex_id)`. `all=true` implies that `f(x)` must return `true` for all nodal
coordinates `x` on the face if the face should be added to the set, otherwise it suffices
that `f(x)` returns `true` for one node.
"""
function addboundaryvertexset!(grid::AbstractGrid, top::ExclusiveTopology, name::String, f::Function; kwargs...)
    set = create_boundaryvertexset(grid, top, f; kwargs...)
    return _addset!(grid, name, set, getvertexsets(grid))
end

"""
addboundaryfacetset!(grid::AbstractGrid, topology::ExclusiveTopology, name::String, f::Function; all::Bool=true)

Adds a boundary facetset to the grid with key `name`.
A facetset maps a `String` key to a `Set` of tuples corresponding to `(global_cell_id,
local_facet_id)`. Facetsets are used to initialize `Dirichlet` structs, that are needed to
specify the boundary for the `ConstraintHandler`. `all=true` implies that `f(x)` must return
`true` for all nodal coordinates `x` on the facet if the facet should be added to the set,
otherwise it suffices that `f(x)` returns `true` for one node.
"""
function addboundaryfacetset!(grid::AbstractGrid, top::ExclusiveTopology, name::String, f::Function; kwargs...)
    set = create_boundaryfacetset(grid, top, f; kwargs...)
    return _addset!(grid, name, set, getfacetsets(grid))
end

function _create_set(f::Function, grid::AbstractGrid, ::Type{BI}; all=true) where {BI <: BoundaryIndex}
    set = Set{BI}()
    for (cell_idx, cell) in enumerate(getcells(grid))
        for (entity_idx, entity) in enumerate(boundaryfunction(BI)(cell))
            pass = all
            for node_idx in entity
                v = f(get_node_coordinate(grid, node_idx))
                all ? (!v && (pass = false; break)) : (v && (pass = true; break))
            end
            pass && push!(set, BI(cell_idx, entity_idx))
        end
    end
    return set
end

# Given a boundary index, for example EdgeIndex(2, 1), add this to `set`, as well as any other `EdgeIndex` in the grid 
# pointing to the same edge (i.e. indices belong to neighboring cells)
function push_entity_instances!(set::Set{BI}, grid::AbstractGrid, top::ExclusiveTopology, entity::BI) where {BI <: BoundaryIndex}
    push!(set, entity) # Add the given entity
    cell = getcells(grid, entity[1])
    verts = boundaryfunction(BI)(cell)[entity[2]]
    for cell_idx in top.vertex_to_cell[verts[1]]# Since all vertices should be shared, the first one can be used here
        cell_entities = boundaryfunction(BI)(getcells(grid, cell_idx))
        for (entity_idx, cell_entity) in pairs(cell_entities)
            if all(x -> x in verts, cell_entity)
                push!(set, BI(cell_idx, entity_idx))
            end
        end
    end
    return set
end

# Create a `Set{BI}` whose entities are a subset of facets which do not have neighbors, i.e. that are on the boundary. 
# Note that this may include entities from cells incident to the facet, e.g. 
# ____  consider the case of a vertex boundary set, with f(x) only true on the right side. Then also the VertexIndex
# |\ |  belong to the lower left cell, in its lower right corner, is on the boundary, so this should be added too.  
# |_\|  That is done by the `push_entity_instances!` function. 
function _create_boundaryset(f::Function, grid::AbstractGrid, top::ExclusiveTopology, ::Type{BI}; all = true) where {BI <: BoundaryIndex}
    # Function barrier as get_facet_facet_neighborhood is not always type stable
    function _makeset(ff_nh)
        set = Set{BI}()
        for (ff_nh_idx, neighborhood) in pairs(ff_nh)
            # ff_nh_idx::CartesianIndex into Matrix{<:EntityNeighborhood}
            isempty(neighborhood) || continue # Skip any facets with neighbors (not on boundary)
            cell_idx  = ff_nh_idx[1]
            facet_nr = ff_nh_idx[2]
            cell = getcells(grid, cell_idx)
            facet_nodes = facets(cell)[facet_nr]
            for (subentity_idx, subentity_nodes) in pairs(boundaryfunction(BI)(cell))
                if Base.all(n -> n in facet_nodes, subentity_nodes)
                    pass = all
                    for node_idx in subentity_nodes
                        v = f(get_node_coordinate(grid, node_idx))
                        all ? (!v && (pass = false; break)) : (v && (pass = true; break))
                    end
                    pass && push_entity_instances!(set, grid, top, BI(cell_idx, subentity_idx))
                end
            end
        end
        return set
    end
    return _makeset(get_facet_facet_neighborhood(top, grid))::Set{BI}
end

function create_cellset(grid::AbstractGrid, f::Function; all::Bool=true)
    cells = Set{Int}()
    for (i, cell) in enumerate(getcells(grid))
        pass = all
        for node_idx in get_node_ids(cell)
            v = f(get_node_coordinate(grid, node_idx))
            all ? (!v && (pass = false; break)) : (v && (pass = true; break))
        end
        pass && push!(cells, i)
    end
    return cells 
end
function create_nodeset(grid::AbstractGrid, f::Function)
    nodes = Set{Int}()
    for (i, n) in pairs(getnodes(grid))
        f(get_node_coordinate(n)) && push!(nodes, i)
    end
    return nodes 
end
create_vertexset(grid::AbstractGrid, f::Function; kwargs...) = _create_set(f, grid, VertexIndex; kwargs...)
create_edgeset(  grid::AbstractGrid, f::Function; kwargs...) = _create_set(f, grid, EdgeIndex;   kwargs...)
create_faceset(  grid::AbstractGrid, f::Function; kwargs...) = _create_set(f, grid, FaceIndex;   kwargs...)
create_facetset( grid::AbstractGrid, f::Function; kwargs...) = _create_set(f, grid, FacetIndex;  kwargs...)

create_boundaryvertexset(grid::AbstractGrid, top::ExclusiveTopology, f::Function; kwargs...) = _create_boundaryset(f, grid, top, VertexIndex; kwargs...)
create_boundaryedgeset(  grid::AbstractGrid, top::ExclusiveTopology, f::Function; kwargs...) = _create_boundaryset(f, grid, top, EdgeIndex;   kwargs...)
create_boundaryfaceset(  grid::AbstractGrid, top::ExclusiveTopology, f::Function; kwargs...) = _create_boundaryset(f, grid, top, FaceIndex;   kwargs...)
create_boundaryfacetset( grid::AbstractGrid, top::ExclusiveTopology, f::Function; kwargs...) = _create_boundaryset(f, grid, top, FacetIndex;  kwargs...)

"""
    bounding_box(grid::AbstractGrid)

Computes the axis-aligned bounding box for a given grid, based on its node coordinates. 
Returns the minimum and maximum vertex coordinates of the bounding box.
"""
function bounding_box(grid::AbstractGrid{dim}) where {dim}
    T = get_coordinate_eltype(grid)
    min_vertex = Vec{dim}(i->typemax(T))
    max_vertex = Vec{dim}(i->typemin(T))
    for node in getnodes(grid)
        x = get_node_coordinate(node)
        _max_tmp = max_vertex # avoid type instability
        _min_tmp = min_vertex
        max_vertex = Vec{dim}(i -> max(x[i], _max_tmp[i]))
        min_vertex = Vec{dim}(i -> min(x[i], _min_tmp[i]))
    end
    return min_vertex, max_vertex
end
