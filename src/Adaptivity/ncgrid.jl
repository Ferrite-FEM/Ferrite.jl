"""
    NonConformingGrid{dim, C<:AbstractCell, T<:Real, CIT} <: AbstractGrid}

A `NonConformingGrid` is a collection of `Cells` and `Node`s which covers the computational domain, together with Sets of cells, nodes, faces
and assocaited information about the conformity.
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
mutable struct NonConformingGrid{dim,C<:Ferrite.AbstractCell,T<:Real,CIT} <: Ferrite.AbstractGrid{dim}
    cells::Vector{C}
    nodes::Vector{Node{dim,T}}
    # Sets
    cellsets::Dict{String,OrderedSet{Int}}
    nodesets::Dict{String,OrderedSet{Int}}
    facetsets::Dict{String,OrderedSet{Ferrite.FacetIndex}}
    vertexsets::Dict{String,OrderedSet{Ferrite.VertexIndex}}
    conformity_info::CIT # TODO refine
    # Boundary matrix (faces per cell × cell)
    boundary_matrix::SparseMatrixCSC{Bool,Int}
end

function NonConformingGrid(
    cells::Vector{C},
    nodes::Vector{Node{dim,T}};
    cellsets::Dict{String,OrderedSet{Int}}=Dict{String,OrderedSet{Int}}(),
    nodesets::Dict{String,OrderedSet{Int}}=Dict{String,OrderedSet{Int}}(),
    facetsets::Dict{String,OrderedSet{Ferrite.FacetIndex}}=Dict{String,OrderedSet{Ferrite.FacetIndex}}(),
    vertexsets::Dict{String,OrderedSet{Ferrite.VertexIndex}}=Dict{String,OrderedSet{Ferrite.VertexIndex}}(),
    conformity_info,
    boundary_matrix::SparseMatrixCSC{Bool,Int}=spzeros(Bool, 0, 0)
    ) where {dim,C,T}
    return NonConformingGrid(cells, nodes, cellsets, nodesets, facetsets, vertexsets, conformity_info, boundary_matrix)
end

get_coordinate_type(::NonConformingGrid{dim,C,T}) where {dim,C,T} = Vec{dim,T}
