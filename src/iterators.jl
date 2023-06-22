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

####################
## InterfaceCache ##
####################

"""
    InterfaceCache(grid::Grid, topology::ExclusiveTopology)
    InterfaceCache(dh::AbstractDofHandler, topology::ExclusiveTopology)

Create a cache object with pre-allocated memory for the coordinates and facets of an
interface. The cache is updated for a new cell by calling `reinit!(cache, this_face, neighbor_face)` where
`this_face::FaceIndex` and `neighbor_face::FaceIndex` are the interface facets.

**Struct fields of `InterfaceCache`**
 - `ic.this_coords :: Vector{<:Vec}`: current cell node coordinates
 - `ic.neighbor_coords :: Vector{<:Vec}`: neighbor cell node coordinates
 - `ic.this_face :: Vector{<:Vec}`: local face index for current cell
 - `ic.neighbor_face :: Vector{<:Vec}`: local face index for neighbor cell
 - `ic.orientation_info :: InterfaceOrientationInfo`: whether the neighbor orientation info relative to current face
 - `ic.grid :: AbstractGrid`: grid information used in iteration
 - `ic.topology :: ExclusiveTopology`: topology information used in iteration

**Methods with `InterfaceCache`**
 - `reinit!(cache::InterfaceCache, this_face::FaceIndex, neighbor_face::FaceIndex)`: reinitialize [`InterfaceCache`](@ref)

See also [`InterfaceIterator`](@ref).
"""
struct InterfaceCache{CC<:CellCache}
    this_cell::CC
    neighbor_cell::CC
    this_face::ScalarWrapper{Int}
    neighbor_face::ScalarWrapper{Int}
    orientation_info::InterfaceOrientationInfo
    # Topology information needed for iteration
    topology::ExclusiveTopology
end

function InterfaceCache(gridordh::Union{AbstractGrid, AbstractDofHandler}, topology::ExclusiveTopology)
    this_cell = CellCache(gridordh)
    neighbor_cell = CellCache(gridordh)
    return InterfaceCache(this_cell, neighbor_cell, ScalarWrapper(0), ScalarWrapper(0), InterfaceOrientationInfo(false, 0), topology)
end

function reinit!(cache::InterfaceCache, this_face::FaceIndex, neighbor_face::FaceIndex)
    reinit!(cache.this_cell,this_face[1])
    reinit!(cache.neighbor_cell,neighbor_face[1])
    cache.this_face[] = this_face[2]
    cache.neighbor_face[] = neighbor_face[2]
    cache.orientation_info = InterfaceOrientationInfo(cache.grid, this_face, neighbor_face)
    return cache
end

# reinit! FEValues with CellCache
reinit!(cv::CellValues, cc::CellCache) = reinit!(cv, cc.coords)
reinit!(fv::FaceValues, cc::CellCache, f::Int) = reinit!(fv, cc.coords, f)
# TODOL enable this after InterfaceValues are merges
# reinit!(iv::InterfaceValues, ic::InterfaceCache) = begin
#     reinit!(iv.face_values,ic.this_cell.coords,ic.this_face)
#     reinit!(iv.face_values_neighbor,ic.neighbor_cell.coords,ic.neighbor_face)
#     @assert getnquadpoints(iv.face_values) == getnquadpoints(iv.face_values_neighbor)
# end

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

#######################
## InterfaceIterator ##
#######################

"""
    InterfaceIterator(grid::Grid, interfaces_set=1:length(topology.face_skeleton), topology::ExclusiveTopology)
    InterfaceIterator(dh::AbstractDofHandler, interfaces_set=1:length(topology.face_skeleton), topology::ExclusiveTopology)

Create an `InterfaceIterator` to conveniently iterate over all, or a subset, of the interfaces in a
grid. The elements of the iterator are [`InterfaceCache`](@ref)s which are properly
`reinit!`ialized. See [`InterfaceCache`](@ref) for more details.

Looping over an `InterfaceIterator`, i.e.:
```julia
for ic in InterfaceIterator(grid, cellset, topology)
    # ...
end
```
is thus simply convenience for the following equivalent snippet:
```julia
ic = InterfaceCache(grid, topology)
for face in topology.face_skeleton
    neighbor_face = topology.face_neighbor[face[1], face[2]][1]
    reinit!(ic, face, neighbor_face)
    # ...
end
```
!!! warning
    `InterfaceIterator` is stateful and should not be used for things other than `for`-looping
    (e.g. broadcasting over, or collecting the iterator may yield unexpected results).
"""
struct InterfaceIterator{Cache<:InterfaceCache, IC<:IntegerCollection}
    cache::Cache
    set::IC
end

function InterfaceIterator(gridordh::Union{Grid,AbstractDofHandler},
                      set::Union{IntegerCollection,Nothing},
                      topology::ExclusiveTopology)
    if set === nothing
        set = findall(face -> !isempty(topology.face_neighbor[face[1], face[2]]), topology.face_skeleton)
    elseif !isempty(findall(face -> isempty(topology.face_neighbor[face[1], face[2]]), topology.face_skeleton[set]))
        error("set passed to InterfaceIterator contains boundary faces")
    end
    return InterfaceIterator(InterfaceCache(gridordh, topology), set)
end
function InterfaceIterator(gridordh::Union{Grid,AbstractDofHandler}, topology::ExclusiveTopology)
    return InterfaceIterator(gridordh, nothing, topology)
end

const GridIterators{C} = Union{CellIterator{C},InterfaceIterator{C}}

# Iterator interface
function Base.iterate(ii::InterfaceIterator, state_in...)
    it = iterate(ii.set, state_in...)
    it === nothing && return nothing
    interface_id, state_out = it
    this_face = ii.cache.topology.face_skeleton[interface_id]
    neighbor_face = ii.cache.topology.face_neighbor[this_face[1], this_face[2]][1]
    reinit!(ii.cache, this_face, neighbor_face)
    return (ii.cache, state_out)
end
Base.IteratorSize(::Type{<:GridIterators}) = Base.HasLength()
Base.IteratorEltype(::Type{<:GridIterators}) = Base.HasEltype()
Base.eltype(::Type{<:GridIterators{CC}}) where CC = CC
Base.length(gi::GridIterators) = length(gi.set)

function _check_same_celltype(grid::AbstractGrid, cellset)
    celltype = getcelltype(grid, first(cellset))
    if !all(getcelltype(grid, i) == celltype for i in cellset)
        error("The cells in the cellset are not all of the same celltype.")
    end
end
