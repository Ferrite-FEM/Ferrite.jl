"""
getfaceinstances(grid::AbstractGrid, topology::ExclusiveTopology, face::FaceIndex)

Returns all the faces as `Set{FaceIndex}` that share all their vertices with a given face
represented as `FaceIndex`. The returned set includes the input face.

```julia-repl
julia> using Ferrite; using Ferrite: getfaceinstances

julia> grid = generate_grid(Tetrahedron, (2,1,1));

julia> topology = ExclusiveTopology(grid);

julia> getfaceinstances(grid, topology, FaceIndex(4,2))
Set{FaceIndex} with 2 elements:
FaceIndex((6, 4))
FaceIndex((4, 2))
```
"""
function getfaceinstances end

"""
getedgeinstances(grid::AbstractGrid, topology::ExclusiveTopology, edge::EdgeIndex)

Returns all the edges as `Set{EdgeIndex}` that share all their vertices with a given edge
represented as `EdgeIndex`.
The returned set includes the input edge.

```julia-repl
julia> using Ferrite; using Ferrite: getedgeinstances

julia> grid = generate_grid(Tetrahedron, (2,1,1));

julia> topology = ExclusiveTopology(grid);

julia> getedgeinstances(grid, topology, EdgeIndex(4,2))
Set{EdgeIndex} with 3 elements:
EdgeIndex((4, 2))
EdgeIndex((9, 6))
EdgeIndex((7, 6))
```
"""
function getedgeinstances end

"""
getvertexinstances(grid::AbstractGrid, topology::ExclusiveTopology, vertex::EdgeIndex)

Returns all the vertices as `Set{::VertexIndex}` that use a given vertex represented as
`VertexIndex` in all cells.
The returned set includes the input vertex.

```julia-repl
julia> using Ferrite; using Ferrite: getvertexinstances

julia> grid = generate_grid(Tetrahedron,(2,1,1));

julia> topology = ExclusiveTopology(grid);

julia> getvertexinstances(grid, topology, VertexIndex(4,2))
Set{VertexIndex} with 8 elements:
VertexIndex((7, 4))
VertexIndex((10, 4))
VertexIndex((12, 4))
VertexIndex((6, 3))
VertexIndex((4, 2))
VertexIndex((9, 4))
VertexIndex((11, 4))
VertexIndex((8, 4))
```
"""
function getvertexinstances end

for (func,                entity_f,  entity_t) in (
(:getvertexinstances, :vertices, :VertexIndex),
(:getedgeinstances,   :edges,    :EdgeIndex),
(:getfaceinstances,   :faces,    :FaceIndex),
)
@eval begin
  function $(func)(grid::AbstractGrid, topology::ExclusiveTopology, entity::$(entity_t))
    _set = Set{$(entity_t)}()
    cells = getcells(grid)
    cell = cells[entity[1]]
    verts = $(entity_f)(cell)[entity[2]]
    # Since we are looking for an entity that share *all* vertices, the first one can be
    # used here to query potiential neighbor cells
    for cell_idx in topology.vertex_to_cell[verts[1]] # Since all vertices should be shared, the first one can be used here
        cell_entities = $(entity_f)(cells[cell_idx])
        for (entity_idx, cell_entity) in pairs(cell_entities)
            if all(x -> x in verts, cell_entity)
                push!(_set, $(entity_t)((cell_idx, entity_idx)))
            end
        end
    end
    return _set
  end
end
end

"""
filterfaces(grid::AbstractGrid, faces::Set{FaceIndex}, f::Function; all::Bool=true)

Returns the faces in `faces` that satisfy `f` as a `Set{FaceIndex}`.
`all=true` implies that `f(x)` must return `true` for all nodal coordinates `x` on the face
if the face should be added to the set, otherwise it suffices that `f(x)` returns `true` for
one node.

```julia-repl
julia> using Ferrite; using Ferrite: filterfaces

julia> grid = generate_grid(Tetrahedron, (2,2,2));

julia> topology = ExclusiveTopology(grid);

julia> addboundaryfaceset!(grid, topology, "b", x -> true);

julia> filterfaces(grid, grid.facesets["b"], x -> x[3] ≈ -1)
Set{FaceIndex} with 8 elements:
FaceIndex((7, 1))
FaceIndex((3, 1))
FaceIndex((21, 1))
FaceIndex((13, 1))
FaceIndex((19, 1))
FaceIndex((15, 1))
FaceIndex((1, 1))
FaceIndex((9, 1))
```
"""
function filterfaces end

"""
filteredges(grid::AbstractGrid, edges::Set{EdgeIndex}, f::Function; all::Bool=true)

Returns the edges in `edges` that satisfy `f` as a `Set{EdgeIndex}`.
`all=true` implies that `f(x)` must return `true` for all nodal coordinates `x` on the face
if the face should be added to the set, otherwise it suffices that `f(x)` returns `true` for
one node.

```julia-repl
julia> using Ferrite; using Ferrite: filteredges

julia> grid = generate_grid(Tetrahedron, (1,1,1));

julia> topology = ExclusiveTopology(grid);

julia> addboundaryedgeset!(grid, topology, "b", x -> true);

julia> filteredges(grid, grid.edgesets["b"], x -> x[3] ≈ -1)
Set{EdgeIndex} with 8 elements:
EdgeIndex((1, 2))
EdgeIndex((3, 2))
EdgeIndex((4, 3))
EdgeIndex((1, 3))
EdgeIndex((3, 3))
EdgeIndex((1, 1))
EdgeIndex((3, 1))
EdgeIndex((2, 3))
```
"""
function filteredges end

"""
filtervertices(grid::AbstractGrid, vertices::Set{VertexIndex}, f::Function; all::Bool=true)

Returns the vertices in `vertices` that satisfy `f` as a `Set{VertexIndex}`.
`all=true` implies that `f(x)` must return `true` for all nodal coordinates `x` on the face
if the face should be added to the set, otherwise it suffices that `f(x)` returns `true` for
one node.

```julia-repl
julia> using Ferrite; using Ferrite: filtervertices

julia> grid = generate_grid(Tetrahedron,(1,1,1));

julia> topology = ExclusiveTopology(grid);

julia> addboundaryvertexset!(grid, topology, "b", x -> true);

julia> filtervertices(grid, grid.vertexsets["b"], x -> x[3] ≈ -1)
Set{VertexIndex} with 12 elements:
VertexIndex((2, 3))
VertexIndex((4, 3))
VertexIndex((4, 1))
VertexIndex((3, 3))
VertexIndex((3, 2))
VertexIndex((1, 1))
VertexIndex((2, 1))
VertexIndex((3, 1))
VertexIndex((1, 3))
VertexIndex((5, 1))
VertexIndex((1, 2))
VertexIndex((6, 1))
```
"""
function filtervertices end

for (func,            entity_f,  entity_t) in (
(:filtervertices, :vertices, :VertexIndex),
(:filteredges,    :edges,    :EdgeIndex),
(:filterfaces,    :faces,    :FaceIndex),
)
@eval begin
    function $(func)(grid::AbstractGrid, set::Set{$(entity_t)}, f::Function; all::Bool=true)
        _set = Set{$(entity_t)}()
        cells = getcells(grid)
        for entity in set # entities can be edges/vertices in the face/edge
            cell = cells[entity[1]]
            cell_entities = $(entity_f)(cell)
            pass = all
            for node_idx in cell_entities[entity[2]] # using cell entities common with boundary face
                v = f(grid.nodes[node_idx].x)
                all ? (!v && (pass = false; break)) : (v && (pass = true; break))
            end
            pass && push!(_set, entity)
        end
        return _set
    end
end
end

"""
addboundaryfaceset!(grid::AbstractGrid, topology::ExclusiveTopology, name::String, f::Function; all::Bool=true)

Adds a boundary faceset to the grid with key `name`.
A faceset maps a `String` key to a `Set` of tuples corresponding to `(global_cell_id,
local_face_id)`. Facesets are used to initialize `Dirichlet` structs, that are needed to
specify the boundary for the `ConstraintHandler`. `all=true` implies that `f(x)` must return
`true` for all nodal coordinates `x` on the face if the face should be added to the set,
otherwise it suffices that `f(x)` returns `true` for one node.

```julia-repl
julia> using Ferrite

julia> grid = generate_grid(Tetrahedron, (1,1,1));

julia> topology = ExclusiveTopology(grid);

julia> addboundaryfaceset!(grid, topology, "b", x -> true);

julia> grid.facesets["b"]
Set{FaceIndex} with 12 elements:
FaceIndex((3, 1))
FaceIndex((4, 3))
FaceIndex((3, 3))
FaceIndex((4, 1))
FaceIndex((5, 1))
FaceIndex((2, 2))
FaceIndex((1, 4))
FaceIndex((2, 1))
FaceIndex((6, 1))
FaceIndex((6, 3))
FaceIndex((5, 3))
FaceIndex((1, 1))
```
"""
function addboundaryfaceset! end

"""
addboundaryedgeset!(grid::AbstractGrid, topology::ExclusiveTopology, name::String, f::Function; all::Bool=true)

Adds a boundary edgeset to the grid with key `name`.
An edgeset maps a `String` key to a `Set` of tuples corresponding to `(global_cell_id,
local_edge_id)`. `all=true` implies that `f(x)` must return `true` for all nodal coordinates
`x` on the face if the face should be added to the set, otherwise it suffices that `f(x)`
returns `true` for one node.

```julia-repl
julia> using Ferrite

julia> grid = generate_grid(Tetrahedron, (1,1,1));

julia> topology = ExclusiveTopology(grid);

julia> addboundaryedgeset!(grid, topology, "b", x -> true);

julia> grid.edgesets["b"]
Set{EdgeIndex} with 30 elements:
EdgeIndex((6, 6))
EdgeIndex((2, 1))
EdgeIndex((5, 3))
.
.
.
EdgeIndex((2, 5))
EdgeIndex((1, 4))
```
"""
function addboundaryedgeset! end

"""
addboundaryvertexset!(grid::AbstractGrid, topology::ExclusiveTopology, name::String, f::Function; all::Bool=true)

Adds a boundary vertexset to the grid with key `name`.
A vertexset maps a `String` key to a `Set` of tuples corresponding to `(global_cell_id,
local_vertex_id)`. `all=true` implies that `f(x)` must return `true` for all nodal
coordinates `x` on the face if the face should be added to the set, otherwise it suffices
that `f(x)` returns `true` for one node.

```julia-repl
julia> using Ferrite

julia> grid = generate_grid(Tetrahedron, (1,1,1));

julia> topology = ExclusiveTopology(grid);

julia> addboundaryvertexset!(grid, topology, "b", x -> true);

julia> grid.vertexsets["b"]
Set{VertexIndex} with 24 elements:
VertexIndex((2, 3))
VertexIndex((5, 2))
VertexIndex((4, 1))
.
.
.
VertexIndex((1, 4))
VertexIndex((4, 4))
```
"""
function addboundaryvertexset! end

for (func,                   entity_f,            entity_t,     filter_f,        instance_f,          destination) in (
(:addboundaryfaceset!,   :((_, x)->Set([x])), :FaceIndex,   :filterfaces,    :getfaceinstances,   :(grid.facesets)),
(:addboundaryedgeset!,   :getfaceedges,       :EdgeIndex,   :filteredges,    :getedgeinstances,   :(grid.edgesets)),
(:addboundaryvertexset!, :getfacevertices,    :VertexIndex, :filtervertices, :getvertexinstances, :(grid.vertexsets)),
)
@eval begin
    function $(func)(grid::AbstractGrid, topology::ExclusiveTopology, name::String, f::Function; all::Bool=true)
        _check_setname($(destination), name)
        _set = Set{$(entity_t)}()
        for (face_idx, neighborhood) in pairs(topology.face_face_neighbor)
            isempty(neighborhood) || continue # Skip any faces with neighbors (not on boundary)
            entities =  $(entity_f)(grid, FaceIndex((face_idx[1], face_idx[2])))
            for entity in $(filter_f)(grid, entities, f; all=all)
                union!(_set, $(instance_f)(grid, topology, entity))
            end
        end
        _warn_emptyset(_set, name)
        $(destination)[name] = _set
        return grid
    end
end
end
