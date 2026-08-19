"""
    NonConformingGrid{dim, C <: AbstractCell, T <: Real, CIT} <: AbstractGrid{dim}

A `NonConformingGrid` is a collection of `Cells` and `Node`s which covers the computational domain, together with sets of cells, nodes,
facets and vertices, and associated information about the conformity.
There are multiple helper structures to apply boundary conditions or define subdomains. They are gathered in the `cellsets`, `nodesets`,
`facetsets` and `vertexsets`.

This grid serves as an entry point for non-intrusive adaptive grid libraries.

# Fields
- `cells::Vector{C}`: stores all cells of the grid
- `nodes::Vector{Node{dim,T}}`: stores the `dim` dimensional nodes of the grid
- `cellsets::Dict{String,OrderedSet{Int}}`: maps a `String` key to an `OrderedSet` of cell ids
- `nodesets::Dict{String,OrderedSet{Int}}`: maps a `String` key to an `OrderedSet` of global node ids
- `facetsets::Dict{String,OrderedSet{FacetIndex}}`: maps a `String` to an `OrderedSet` of `FacetIndex (global_cell_id, local_facet_id)`
- `vertexsets::Dict{String,OrderedSet{VertexIndex}}`: maps a `String` key to an `OrderedSet` of `VertexIndex (global_cell_id, local_vertex_id)`
- `conformity_info::CIT`: a container for conformity information

!!! note
    A grid produced by [`creategrid`](@ref Ferrite.AMR.creategrid) only carries the `cellsets`
    and `facetsets` of the base grid; its `nodesets` and `vertexsets` are empty.

!!! warning "Deprecated"
    `NonConformingGrid` is deprecated in favour of using the [`ForestBWG`](@ref) directly as
    the grid: the forest answers all `AbstractGrid` queries for its current refinement state,
    a `DofHandler` built on it detects staleness after refinement, and [`reclose!`](@ref)
    re-distributes the dofs. `NonConformingGrid` (and [`creategrid`](@ref
    Ferrite.AMR.creategrid)) keep working during the transition but will be removed.
"""
mutable struct NonConformingGrid{dim, C <: Ferrite.AbstractCell, T <: Real, CIT} <: Ferrite.AbstractGrid{dim}
    cells::Vector{C}
    nodes::Vector{Node{dim, T}}
    # Sets
    cellsets::Dict{String, OrderedSet{Int}}
    nodesets::Dict{String, OrderedSet{Int}}
    facetsets::Dict{String, OrderedSet{Ferrite.FacetIndex}}
    vertexsets::Dict{String, OrderedSet{Ferrite.VertexIndex}}
    # Hanging nodes. Currently a `Dict{Int, Vector{Int}}` mapping each hanging (slave) vertex
    # to its 2 or 4 master vertices. The layout will change once hanging edges/faces are
    # exposed as entities for higher-order fields — see the warning in the AMR devdocs.
    conformity_info::CIT
end

function NonConformingGrid(
        cells::Vector{C},
        nodes::Vector{Node{dim, T}};
        cellsets::Dict{String, OrderedSet{Int}} = Dict{String, OrderedSet{Int}}(),
        nodesets::Dict{String, OrderedSet{Int}} = Dict{String, OrderedSet{Int}}(),
        facetsets::Dict{String, OrderedSet{Ferrite.FacetIndex}} = Dict{String, OrderedSet{Ferrite.FacetIndex}}(),
        vertexsets::Dict{String, OrderedSet{Ferrite.VertexIndex}} = Dict{String, OrderedSet{Ferrite.VertexIndex}}(),
        conformity_info
    ) where {dim, C, T}
    return NonConformingGrid(cells, nodes, cellsets, nodesets, facetsets, vertexsets, conformity_info)
end

# Must be qualified: `get_coordinate_type` is not exported from Ferrite, so an unqualified
# definition here would create a separate `AMR.get_coordinate_type` instead of extending it.
Ferrite.get_coordinate_type(::NonConformingGrid{dim, C, T}) where {dim, C, T} = Vec{dim, T}

Ferrite.has_hanging_nodes(::NonConformingGrid) = true

# See the docstring at `conformity_info(::ForestBWG)` in forest.jl.
conformity_info(g::NonConformingGrid) = g.conformity_info

function Base.show(io::IO, ::MIME"text/plain", grid::NonConformingGrid)
    print(io, "$(typeof(grid)) with $(getncells(grid)) ")
    if isconcretetype(eltype(grid.cells))
        typestrs = [repr(eltype(grid.cells))]
    else
        typestrs = sort!(repr.(OrderedSet(typeof(x) for x in grid.cells)))
    end
    join(io, typestrs, '/')
    print(io, " cells, $(getnnodes(grid)) nodes and $(length(grid.conformity_info)) hanging nodes")
    return
end
