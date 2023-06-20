# This file defines iterators used for looping over a grid

struct UpdateFlags
    nodes::Bool
    coords::Bool
    dofs::Bool
end

UpdateFlags(; nodes::Bool=true, coords::Bool=true, dofs::Bool=true) =
    UpdateFlags(nodes, coords, dofs)


###############
## CellCache ##
###############

"""
    CellCache(grid::Grid)
    CellCache(dh::AbstractDofHandler)

Create a cache object with pre-allocated memory for the nodes, coordinates, and dofs of a
cell. The cache is updated for a new cell by calling `reinit!(cache, cellid)` where
`cellid::Int` is the cell id.

**Struct fields of `CellCache`**
 - `cc.nodes :: Vector{Int}`: global node ids
 - `cc.coords :: Vector{<:Vec}`: node coordinates
 - `cc.dofs :: Vector{Int}`: global dof ids (empty when constructing the cache from a grid)

**Methods with `CellCache`**
 - `reinit!(cc, i)`: reinitialize the cache for cell `i`
 - `cellid(cc)`: get the cell id of the currently cached cell
 - `getnodes(cc)`: get the global node ids of the cell
 - `get_cell_coordinates(cc)`: get the coordinates of the cell
 - `celldofs(cc)`: get the global dof ids of the cell
 - `reinit!(fev, cc)`: reinitialize [`CellValues`](@ref) or [`FaceValues`](@ref)

See also [`CellIterator`](@ref).
"""
struct CellCache{X,G<:AbstractGrid,DH<:Union{AbstractDofHandler,Nothing}}
    flags::UpdateFlags
    grid::G
    # Pretty useless to store this since you have it already for the reinit! call, but
    # needed for the CellIterator(...) workflow since the user doesn't necessarily control
    # the loop order in the cell subset.
    cellid::ScalarWrapper{Int}
    nodes::Vector{Int}
    coords::Vector{X}
    dh::DH
    dofs::Vector{Int}
end

function CellCache(grid::Grid{dim,C,T}, flags::UpdateFlags=UpdateFlags()) where {dim,C,T}
    N = nnodes_per_cell(grid)
    nodes = zeros(Int, N)
    coords = zeros(Vec{dim,T}, N)
    return CellCache(flags, grid, ScalarWrapper(-1), nodes, coords, nothing, Int[])
end

function CellCache(dh::DofHandler{dim}, flags::UpdateFlags=UpdateFlags()) where {dim}
    N = nnodes_per_cell(get_grid(dh))
    nodes = zeros(Int, N)
    coords = zeros(Vec{dim, get_coordinate_eltype(get_grid(dh))}, N)
    n = ndofs_per_cell(dh)
    celldofs = zeros(Int, n)
    return CellCache(flags, get_grid(dh), ScalarWrapper(-1), nodes, coords, dh, celldofs)
end

function reinit!(cc::CellCache, i::Int)
    cc.cellid[] = i
    if cc.flags.nodes
        resize!(cc.nodes, nnodes_per_cell(cc.grid, i))
        cellnodes!(cc.nodes, cc.grid, i)
    end
    if cc.flags.coords
        resize!(cc.coords, nnodes_per_cell(cc.grid, i))
        get_cell_coordinates!(cc.coords, cc.grid, i)
    end
    if cc.dh !== nothing && cc.flags.dofs
        resize!(cc.dofs, ndofs_per_cell(cc.dh, i))
        celldofs!(cc.dofs, cc.dh, i)
    end
    return cc
end

###############
## FaceCache ##
###############

"""
    FaceCache(grid::Grid, topology::ExclusiveTopology)
    FaceCache(dh::AbstractDofHandler, topology::ExclusiveTopology)

Create a cache object with pre-allocated memory for the coordinates and facets of a face in the face skeleton.
The cache is updated for a new cell by calling `reinit!(cache, this_face, neighbor_face = nothing)` where
`this_face::FaceIndex` and `neighbor_face::FaceIndex` are the interface facets.

**Struct fields of `FaceCache`**
 - `fc.this_coords :: Vector{<:Vec}`: current cell node coordinates
 - `fc.neighbor_coords :: Vector{<:Vec}`: neighbor cell node coordinates
 - `fc.this_face :: Vector{<:Vec}`: local face index for current cell
 - `fc.neighbor_face :: Vector{<:Vec}`: local face index for neighbor cell
 - `fc.orientation_info :: InterfaceOrientationInfo`: whether the neighbor orientation info relative to current face
 - `fc.grid :: AbstractGrid`: grid information used in iteration
 - `fc.topology :: ExclusiveTopology`: topology information used in iteration

**Methods with `FaceCache`**
 - `reinit!(cache::FaceCache, this_face::FaceIndex, neighbor_face::FaceIndex)`: reinitialize [`FaceCache`](@ref)

See also [`FaceIterator`](@ref).
"""
struct FaceCache{X,G<:AbstractGrid}
    this_coords::Vector{X}
    neighbor_coords::Vector{X}
    this_face::ScalarWrapper{Int}
    neighbor_face::ScalarWrapper{Int}
    orientation_info::InterfaceOrientationInfo
    # grid and topology information needed for iteration
    grid::G
    topology::ExclusiveTopology
end

function FaceCache(grid::Grid{dim,C,T}, topology::ExclusiveTopology) where {dim,C,T}
    N = nnodes_per_cell(grid)
    this_coords = zeros(Vec{dim,T}, N)
    neighbor_coords = zeros(Vec{dim,T}, N)
    return FaceCache(this_coords, neighbor_coords, ScalarWrapper(-1), ScalarWrapper(-1), InterfaceOrientationInfo(false, 0), grid, topology)
end

function FaceCache(dh::DofHandler{dim}, topology::ExclusiveTopology) where {dim}
    N = nnodes_per_cell(get_grid(dh))
    this_coords = zeros(Vec{dim, get_coordinate_eltype(get_grid(dh))}, N)
    neighbor_coords = zeros(Vec{dim, get_coordinate_eltype(get_grid(dh))}, N)
    return FaceCache(this_coords, neighbor_coords, ScalarWrapper(-1), ScalarWrapper(-1), InterfaceOrientationInfo(false, 0), get_grid(dh), topology)
end

function reinit!(cache::FaceCache, this_face::FaceIndex, neighbor_face::Union{Nothing, FaceIndex} = nothing)
    resize!(cache.this_coords, nnodes_per_cell(cache.grid, this_face[1]))
    get_cell_coordinates!(cache.this_coords, cache.grid, this_face[1])
    cache.this_face[] = this_face[2]
    if !isnothing(neighbor_face)
        resize!(cache.neighbor_coords, nnodes_per_cell(cache.grid, neighbor_face[1]))
        get_cell_coordinates!(cache.neighbor_coords, cache.grid, neighbor_face[1])
        cache.neighbor_face[] = neighbor_face[2]
        return cache
    end
    cache.neighbor_face[] = -1 
    return cache
end

# reinit! FEValues with CellCache
reinit!(cv::CellValues, cc::CellCache) = reinit!(cv, cc.coords)
reinit!(fv::FaceValues, cc::CellCache, f::Int) = reinit!(fv, cc.coords, f)
reinit!(fv::FaceValues, fc::FaceCache, here::Union{Nothing, Bool} = nothing) = begin
    isnothing(here) && (fc.neighbor_face != -1) && @warn "Current facet has a neighbor, consider using `here::Bool` to specify which facet to reninit"    
    (isnothing(here) || here) && reinit!(fv, fc.this_coords, fc.this_face)
    (!isnothing(here) && here) || reinit!(fv, fc.neighbor_coords, fc.neighbor_face)
end
reinit!(iv::InterfaceValues, fc::FaceCache) = begin
    @assert fc.neighbor_face == -1 "Current face does not belong to an interface or is on boundary"    
    reinit!(iv, fc.this_coords, fc.this_face, fc.neighbor_coords, fc.neighbor_face)
end

# Accessor functions (TODO: Deprecate? We are so inconsistent with `getxx` vs `xx`...)
getnodes(cc::CellCache) = cc.nodes
get_cell_coordinates(cc::CellCache) = cc.coords
celldofs(cc::CellCache) = cc.dofs
cellid(cc::CellCache) = cc.cellid[]

# TODO: This can definitely be deprecated
celldofs!(v::Vector, cc::CellCache) = copyto!(v, cc.dofs) # celldofs!(v, cc.dh, cc.cellid[])

# TODO: These should really be replaced with something better...
nfaces(cc::CellCache) = nfaces(cc.grid.cells[cc.cellid[]])
onboundary(cc::CellCache, face::Int) = cc.grid.boundary_matrix[face, cc.cellid[]]

##################
## CellIterator ##
##################

const IntegerCollection = Union{AbstractSet{<:Integer}, AbstractVector{<:Integer}}

"""
    CellIterator(grid::Grid, cellset=1:getncells(grid))
    CellIterator(dh::AbstractDofHandler, cellset=1:getncells(dh))

Create a `CellIterator` to conveniently iterate over all, or a subset, of the cells in a
grid. The elements of the iterator are [`CellCache`](@ref)s which are properly
`reinit!`ialized. See [`CellCache`](@ref) for more details.

Looping over a `CellIterator`, i.e.:
```julia
for cc in CellIterator(grid, cellset)
    # ...
end
```
is thus simply convenience for the following equivalent snippet:
```julia
cc = CellCache(grid)
for idx in cellset
    reinit!(cc, idx)
    # ...
end
```
!!! warning
    `CellIterator` is stateful and should not be used for things other than `for`-looping
    (e.g. broadcasting over, or collecting the iterator may yield unexpected results).
"""
struct CellIterator{CC<:CellCache, IC<:IntegerCollection}
    cc::CC
    set::IC
end

function CellIterator(gridordh::Union{Grid,AbstractDofHandler},
                      set::Union{IntegerCollection,Nothing}=nothing,
                      flags::UpdateFlags=UpdateFlags())
    if set === nothing
        grid = gridordh isa AbstractDofHandler ? get_grid(gridordh) : gridordh
        set = 1:getncells(grid)
    end
    if gridordh isa DofHandler && !isconcretetype(getcelltype(get_grid(gridordh)))
        # TODO: Since the CellCache is resizeable this is not really necessary to check
        #       here, but might be useful to catch slow code paths?
        _check_same_celltype(get_grid(gridordh), set)
    end
    return CellIterator(CellCache(gridordh, flags), set)
end
function CellIterator(gridordh::Union{Grid,AbstractDofHandler}, flags::UpdateFlags)
    return CellIterator(gridordh, nothing, flags)
end

# Iterator interface
function Base.iterate(ci::CellIterator, state_in...)
    it = iterate(ci.set, state_in...)
    it === nothing && return nothing
    cellid, state_out = it
    reinit!(ci.cc, cellid)
    return (ci.cc, state_out)
end
Base.IteratorSize(::Type{<:CellIterator}) = Base.HasLength()
Base.IteratorEltype(::Type{<:CellIterator}) = Base.HasEltype()
Base.eltype(::Type{<:CellIterator{CC}}) where CC = CC
Base.length(ci::CellIterator) = length(ci.set)

##################
## FaceIterator ##
##################

"""
    FaceIterator(grid::Grid, interfaces_set=1:length(topology.face_skeleton), topology::ExclusiveTopology)
    FaceIterator(dh::AbstractDofHandler, interfaces_set=1:length(topology.face_skeleton), topology::ExclusiveTopology)

Create an `FaceIterator` to conveniently iterate over all, or a subset, of the faces in a
face skeleton. The elements of the iterator are [`FaceCache`](@ref)s which are properly
`reinit!`ialized. See [`FaceCache`](@ref) for more details.

Looping over an `FaceIterator`, i.e.:
```julia
for fc in FaceIterator(grid, cellset, topology)
    # ...
end
```
is thus simply convenience for the following equivalent snippet:
```julia
fc = FaceCache(grid, topology)
for face in topology.face_skeleton
    neighbor_face = topology.face_neighbor[face[1], face[2]][1]
    reinit!(fc, face, neighbor_face)
    # ...
end
```
!!! warning
    `FaceIterator` is stateful and should not be used for things other than `for`-looping
    (e.g. broadcasting over, or collecting the iterator may yield unexpected results).
"""
struct FaceIterator{Cache<:FaceCache, IC<:IntegerCollection}
    cache::Cache
    set::IC
end

function FaceIterator(gridordh::Union{Grid,AbstractDofHandler},
                      set::Union{IntegerCollection,Nothing},
                      topology::ExclusiveTopology)
    if set === nothing
        set = 1:length(topology.face_skeleton)
    end
    return FaceIterator(FaceCache(gridordh, topology), set)
end
function FaceIterator(gridordh::Union{Grid,AbstractDofHandler}, topology::ExclusiveTopology)
    return FaceIterator(gridordh, nothing, topology)
end

# Iterator interface
function Base.iterate(fi::FaceIterator, state_in...)
    it = iterate(fi.set, state_in...)
    it === nothing && return nothing
    face_id, state_out = it
    this_face = fi.cache.topology.face_skeleton[face_id]
    face_neighbors = fi.cache.topology.face_neighbor[this_face[1], this_face[2]]
    isempty(face_neighbors) && reinit!(fi.cache, this_face, nothing) && return (fi.cache, state_out)
    neighbor_face = face_neighbors[1]
    reinit!(fi.cache, this_face, neighbor_face)
    return (fi.cache, state_out)
end
Base.IteratorSize(::Type{<:FaceIterator}) = Base.HasLength()
Base.IteratorEltype(::Type{<:FaceIterator}) = Base.HasEltype()
Base.eltype(::Type{<:FaceIterator{Cache}}) where Cache = Cache
Base.length(fi::FaceIterator) = length(fi.set)


function _check_same_celltype(grid::AbstractGrid, cellset)
    celltype = getcelltype(grid, first(cellset))
    if !all(getcelltype(grid, i) == celltype for i in cellset)
        error("The cells in the cellset are not all of the same celltype.")
    end
end
