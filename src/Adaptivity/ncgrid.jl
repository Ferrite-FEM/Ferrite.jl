"""
    HangingFacet

Topological record of one non-conforming facet interface of a [`NonConformingGrid`](@ref):
the coarse facet `coarse` is covered by the finer sub-facets `fine` (2 in 2D, 4 in 3D) of
the refined neighbour cells. Both sides are `(cell, local facet)` references in Ferrite
numbering. `fine` is unordered — the child position and relative orientation of each
sub-facet are derived from the shared corner node ids by the consumer (each fine facet
shares exactly one corner node with the coarse facet), so no octree conventions leak into
the record.
"""
struct HangingFacet
    coarse::Ferrite.FacetIndex
    fine::Vector{Ferrite.FacetIndex}
end

"""
    HangingEdge

Topological record of one non-conforming (hanging) edge of a 3D [`NonConformingGrid`](@ref):
the coarse edge `coarse` is covered by the two half-edges `fine`, each given by one
representative `(cell, local edge)` reference (several fine cells may share a half-edge;
any representative identifies the same mesh entity). Hanging edges that lie on a hanging
face are also reachable through the corresponding [`HangingFacet`](@ref) — consumers
deduplicate by constrained dof.
"""
struct HangingEdge
    coarse::Ferrite.EdgeIndex
    fine::NTuple{2, Ferrite.EdgeIndex}
end

"""
    ConformityInfo

Conformity information of a [`NonConformingGrid`](@ref): the purely topological
hanging-interface records ([`HangingFacet`](@ref)/[`HangingEdge`](@ref)) from which the
constraint builder derives the hanging-dof constraints for any interpolation, plus —
transitional, until all consumers are ported — the legacy node-level map
`hanging_nodes::Dict` from hanging vertex node id to its master node ids.

`complete` is `false` when the forest had level jumps > 1 across a tree interface
(unbalanced forest): such interfaces are not representable as records (the legacy
node-level constraints tolerate them by constraining only the top-level midpoints), and
the generic constraint builder refuses incomplete records instead of silently missing
constraints.

The container forwards `length`/`getindex`/`iterate`/`haskey`/`keys` to `hanging_nodes`
so legacy consumers of the plain dict keep working during the migration.
"""
struct ConformityInfo
    hanging_facets::Vector{HangingFacet}
    hanging_edges::Vector{HangingEdge}
    hanging_nodes::Dict{Int, Vector{Int}}
    complete::Bool
end

Base.length(ci::ConformityInfo) = length(ci.hanging_nodes)
Base.getindex(ci::ConformityInfo, i::Int) = ci.hanging_nodes[i]
Base.iterate(ci::ConformityInfo, state...) = iterate(ci.hanging_nodes, state...)
Base.haskey(ci::ConformityInfo, i) = haskey(ci.hanging_nodes, i)
Base.keys(ci::ConformityInfo) = keys(ci.hanging_nodes)

"""
    NonConformingGrid{dim, C<:AbstractCell, T<:Real, CIT} <: AbstractGrid}

A `NonConformingGrid` is a collection of `Cells` and `Node`s which covers the computational domain, together with Sets of cells, nodes, faces
and associated information about the conformity.
There are multiple helper structures to apply boundary conditions or define subdomains. They are gathered in the `cellsets`, `nodesets`,
`facesets`, `edgesets` and `vertexsets`.

This grid serves as an entry point for non-intrusive adaptive grid libraries.

# Fields
- `cells::Vector{C}`: stores all cells of the grid
- `nodes::Vector{Node{dim,T}}`: stores the `dim` dimensional nodes of the grid
- `cellsets::Dict{String,Set{Int}}`: maps a `String` key to a `Set` of cell ids
- `nodesets::Dict{String,Set{Int}}`: maps a `String` key to a `Set` of global node ids
- `facesets::Dict{String,Set{FaceIndex}}`: maps a `String` to a `Set` of `Set{FaceIndex} (global_cell_id, local_face_id)`
- `edgesets::Dict{String,Set{EdgeIndex}}`: maps a `String` to a `Set` of `Set{EdgeIndex} (global_cell_id, local_edge_id`
- `vertexsets::Dict{String,Set{VertexIndex}}`: maps a `String` key to a `Set` of local vertex ids
- `conformity_info::CIT`: a container for conformity information
- `boundary_matrix::SparseMatrixCSC{Bool,Int}`: optional, only needed by `onboundary` to check if a cell is on the boundary, see, e.g. Helmholtz example
"""
mutable struct NonConformingGrid{dim, C <: Ferrite.AbstractCell, T <: Real, CIT} <: Ferrite.AbstractGrid{dim}
    cells::Vector{C}
    nodes::Vector{Node{dim, T}}
    # Sets
    cellsets::Dict{String, OrderedSet{Int}}
    nodesets::Dict{String, OrderedSet{Int}}
    facetsets::Dict{String, OrderedSet{Ferrite.FacetIndex}}
    vertexsets::Dict{String, OrderedSet{Ferrite.VertexIndex}}
    conformity_info::CIT # TODO refine
    # Boundary matrix (faces per cell × cell)
    boundary_matrix::SparseMatrixCSC{Bool, Int}
end

function NonConformingGrid(
        cells::Vector{C},
        nodes::Vector{Node{dim, T}};
        cellsets::Dict{String, OrderedSet{Int}} = Dict{String, OrderedSet{Int}}(),
        nodesets::Dict{String, OrderedSet{Int}} = Dict{String, OrderedSet{Int}}(),
        facetsets::Dict{String, OrderedSet{Ferrite.FacetIndex}} = Dict{String, OrderedSet{Ferrite.FacetIndex}}(),
        vertexsets::Dict{String, OrderedSet{Ferrite.VertexIndex}} = Dict{String, OrderedSet{Ferrite.VertexIndex}}(),
        conformity_info,
        boundary_matrix::SparseMatrixCSC{Bool, Int} = spzeros(Bool, 0, 0)
    ) where {dim, C, T}
    return NonConformingGrid(cells, nodes, cellsets, nodesets, facetsets, vertexsets, conformity_info, boundary_matrix)
end

get_coordinate_type(::NonConformingGrid{dim, C, T}) where {dim, C, T} = Vec{dim, T}
