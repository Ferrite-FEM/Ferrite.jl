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

function CellCache(sdh::SubDofHandler, flags::UpdateFlags=UpdateFlags())
    CellCache(flags, sdh.dh.grid, ScalarWrapper(-1), Int[], Vec{2,Float64}[], sdh, Int[])
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

# reinit! FEValues with CellCache
reinit!(cv::CellValues, cc::CellCache) = reinit!(cv, cc.coords)
reinit!(fv::FaceValues, cc::CellCache, f::Int) = reinit!(fv, cc.coords, f) # TODO: Deprecate?
# TODOL enable this after InterfaceValues are merges
# reinit!(iv::InterfaceValues, ic::InterfaceCache) = reinit!(iv, FaceIndex(cellid(ic.face_a), ic.face_a.current_faceid[]), get_cell_coordinates(ic.face_a),
#     FaceIndex(cellid(ic.face_b), ic.face_b.current_faceid[]), get_cell_coordinates(ic.face_b), ic.face_a.cc.grid)

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


# TODO: Currently excluded from the docstring below. Should they be public?
# - `Ferrite.faceindex(fc)`: get the `FaceIndex` of the currently cached face
# - `Ferrite.faceid(fc)`: get the current faceid (`faceindex(fc)[2]`)

"""
    FaceCache(grid::Grid)
    FaceCache(dh::AbstractDofHandler)

Create a cache object with pre-allocated memory for the nodes, coordinates, and dofs of a
cell suitable for looping over *faces* in a grid. The cache is updated for a new face by
calling `reinit!(cache, fi::FaceIndex)`.

**Methods with `fc::FaceCache`**
 - `reinit!(fc, fi)`: reinitialize the cache for face `fi::FaceIndex`
 - `cellid(fc)`: get the current cellid (`faceindex(fc)[1]`)
 - `getnodes(fc)`: get the global node ids of the *cell*
 - `get_cell_coordinates(fc)`: get the coordinates of the *cell*
 - `celldofs(fc)`: get the global dof ids of the *cell*
 - `reinit!(fv, fc)`: reinitialize [`FaceValues`](@ref)

See also [`FaceIterator`](@ref).
"""
struct FaceCache{CC<:CellCache}
    cc::CC  # const for julia > 1.8
    current_faceid::ScalarWrapper{Int}
end
FaceCache(args...) = FaceCache(CellCache(args...), ScalarWrapper(0))

function reinit!(fc::FaceCache, face::FaceIndex)
    cellid, faceid = face
    reinit!(fc.cc, cellid)
    fc.current_faceid[] = faceid
    return nothing
end

# Delegate methods to the cell cache
for op = (:getnodes, :get_cell_coordinates, :cellid, :celldofs)
    @eval begin
        function Ferrite.$op(fc::FaceCache, args...)
            return Ferrite.$op(fc.cc, args...)
        end
    end
end
# @inline faceid(fc::FaceCache) = fc.current_faceid[]
@inline celldofs!(v::Vector, fc::FaceCache) = celldofs!(v, fc.cc)
# @inline onboundary(fc::FaceCache) = onboundary(fc.cc, faceid(fc))
# @inline faceindex(fc::FaceCache) = FaceIndex(cellid(fc), faceid(fc))
@inline function reinit!(fv::FaceValues, fc::FaceCache)
    reinit!(fv, fc.cc, fc.current_faceid[])
end

"""
    InterfaceCache(grid::Grid)
    InterfaceCache(dh::AbstractDofHandler)
Create a cache object with pre-allocated memory for the coordinates and faces of an
interface. The cache is updated for a new cell by calling `reinit!(cache, face_a, face_b)` where
`face_a::FaceIndex` and `face_b::FaceIndex` are the interface faces.
**Struct fields of `InterfaceCache`**
 - `ic.face_a :: FaceCache`: current cell node coordinates
 - `ic.face_b :: FaceCache`: neighbor cell node coordinates
**Methods with `InterfaceCache`**
 - `reinit!(cache::InterfaceCache, face_a::FaceIndex, face_b::FaceIndex)`: reinitialize [`InterfaceCache`](@ref)
 - `cellid(ic)`: get cell ids of the current interface
 - `getnodes(ic, use_cell_a = true)`: get the global node ids of cell A or cell B of the interface
 - `get_cell_coordinates(ic, use_cell_a = true)`: get the coordinates of cell A or cell B of the interface
 - `interfacedofs(ic)`: get the global dof ids of the interface cells
 - `interfacedofranges(ic)`: get the interface dof ranges of the interface cells
See also [`InterfaceIterator`](@ref).
"""
struct InterfaceCache{FC<:FaceCache}
    face_a::FC
    face_b::FC
end

function InterfaceCache(gridordh::Union{AbstractGrid, AbstractDofHandler})
    face_a = FaceCache(gridordh)
    face_b = FaceCache(gridordh)
    return InterfaceCache(face_a, face_b)
end

function reinit!(cache::InterfaceCache, face_a::FaceIndex, face_b::FaceIndex)
    reinit!(cache.face_a, face_a)
    reinit!(cache.face_b, face_b)
    return cache
end

getnodes(ic::InterfaceCache, use_cell_a::Bool = true) = getnodes(use_cell_a ? ic.face_a : ic.face_b)
get_cell_coordinates(ic::InterfaceCache, use_cell_a::Bool = true) = get_cell_coordinates(use_cell_a ? ic.face_a : ic.face_b)
cellid(ic::InterfaceCache) = (cellid(ic.face_a), cellid(ic.face_b))
interfacedofs(ic::InterfaceCache) = vcat(celldofs(ic.face_a), celldofs(ic.face_b))
interfacedofranges(ic::InterfaceCache) = (1 : length(celldofs(ic.face_a)), length(celldofs(ic.face_a)) + 1 : length(celldofs(ic.face_a))  + length(celldofs(ic.face_b)))

####################
## Grid iterators ##
####################

## CellIterator ##

const IntegerCollection = Union{Set{<:Integer}, AbstractVector{<:Integer}}

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

function CellIterator(gridordh::Union{Grid,DofHandler},
                      set::Union{IntegerCollection,Nothing}=nothing,
                      flags::UpdateFlags=UpdateFlags())
    if set === nothing
        grid = gridordh isa DofHandler ? get_grid(gridordh) : gridordh
        set = 1:getncells(grid)
    end
    if gridordh isa DofHandler
        # TODO: Since the CellCache is resizeable this is not really necessary to check
        #       here, but might be useful to catch slow code paths?
        _check_same_celltype(get_grid(gridordh), set)
    end
    return CellIterator(CellCache(gridordh, flags), set)
end
function CellIterator(gridordh::Union{Grid,DofHandler}, flags::UpdateFlags)
    return CellIterator(gridordh, nothing, flags)
end
function CellIterator(sdh::SubDofHandler, flags::UpdateFlags=UpdateFlags())
    CellIterator(sdh.dh, sdh.cellset, flags)
end

@inline _getset(ci::CellIterator) = ci.set
@inline _getcache(ci::CellIterator) = ci.cc


## FaceIterator ##

# Leaving flags undocumented as for CellIterator
"""
    FaceIterator(gridordh::Union{Grid,AbstractDofHandler}, faceset::Set{FaceIndex})

Create a `FaceIterator` to conveniently iterate over the faces in `faceset`. The elements of
the iterator are [`FaceCache`](@ref)s which are properly `reinit!`ialized. See
[`FaceCache`](@ref) for more details.

Looping over a `FaceIterator`, i.e.:
```julia
for fc in FaceIterator(grid, faceset)
    # ...
end
```
is thus simply convenience for the following equivalent snippet:
```julia
fc = FaceCache(grid)
for faceindex in faceset
    reinit!(fc, faceindex)
    # ...
end
"""
struct FaceIterator{FC<:FaceCache}
    fc::FC
    set::Set{FaceIndex}
end

function FaceIterator(gridordh::Union{Grid,AbstractDofHandler},
                      set, flags::UpdateFlags=UpdateFlags())
    if gridordh isa DofHandler
        # Keep here to maintain same settings as for CellIterator
        _check_same_celltype(get_grid(gridordh), set)
    end
    return FaceIterator(FaceCache(gridordh, flags), set)
end

@inline _getcache(fi::FaceIterator) = fi.fc
@inline _getset(fi::FaceIterator) = fi.set

"""
    InterfaceIterator(grid::Grid, interfaces_set=1:length(topology.face_skeleton), topology::ExclusiveTopology)
    InterfaceIterator(dh::AbstractDofHandler, interfaces_set=1:length(topology.face_skeleton), topology::ExclusiveTopology)
Create an `InterfaceIterator` to conveniently iterate over all, or a subset, of the interfaces in a
grid. The elements of the iterator are [`InterfaceCache`](@ref)s which are properly
`reinit!`ialized. See [`InterfaceCache`](@ref) for more details.
Looping over an `InterfaceIterator`, i.e.:
```julia
for ic in InterfaceIterator(grid, cellset)
    # ...
end
```
is thus simply convenience for the following equivalent snippet for grids of dimensions > 1:
```julia
ic = InterfaceCache(grid, topology)
for face in topology.face_skeleton
    neighborhood = topology.face_face_neighbor[face[1], face[2]]
    isempty(neighborhood) && continue
    neighbor_face = neighborhood[1]
    reinit!(ic, face, neighbor_face)
    # ...
end
```
!!! warning
    `InterfaceIterator` is stateful and should not be used for things other than `for`-looping
    (e.g. broadcasting over, or collecting the iterator may yield unexpected results).
"""
struct InterfaceIterator{Cache<:InterfaceCache}
    cache::Cache
    set::Set{NTuple{2, FaceIndex}}
end

function InterfaceIterator(gridordh::Union{Grid,AbstractDofHandler},
                      set::Union{Set{NTuple{2, FaceIndex}},Nothing},
                      topology::ExclusiveTopology = ExclusiveTopology(gridordh))
    
    if isnothing(set)
        # Maybe move this to faceskeleton and buffer it in topology?
        grid = gridordh isa Grid ? gridordh : get_grid(gridordh)
        i = 1;
        grid_dim = getdim(grid)
        neighborhood = grid_dim == 1 ? pairs(topology.vertex_vertex_neighbor) : pairs(topology.face_face_neighbor)
        interface_skeleton = Array{NTuple{2, FaceIndex}}(undef, count(pair -> !isempty(pair[2]) && pair[2].neighbor_info[][1] > pair[1][1], neighborhood))
        for (idx, face) in neighborhood
            !isempty(face.neighbor_info) && face.neighbor_info[][1] > idx[1] || continue
            interface_skeleton[i] = (face.neighbor_info[], FaceIndex(idx[1], idx[2])) #Assumes one neighbor only
            i+=1
        end
        set = Set(interface_skeleton)
    end
    return InterfaceIterator(InterfaceCache(gridordh), set)
end

function InterfaceIterator(gridordh::Union{Grid,AbstractDofHandler}, topology::ExclusiveTopology = ExclusiveTopology(gridordh))
    return InterfaceIterator(gridordh, nothing, topology)
end

# Iterator interface
function Base.iterate(ii::InterfaceIterator, state_in...)
    it = iterate(ii.set, state_in...)
    it === nothing && return nothing
    interface, state_out = it
    face_a = interface[1]
    face_b = interface[2]
    reinit!(ii.cache, face_a, face_b)
    return (ii.cache, state_out)
end


# Iterator interface for CellIterator/FaceIterator
const GridIterators{C} = Union{CellIterator{C}, FaceIterator{C}, InterfaceIterator{C}}

function Base.iterate(iterator::GridIterators, state_in...)
    it = iterate(_getset(iterator), state_in...)
    it === nothing && return nothing
    item, state_out = it
    cache = _getcache(iterator)
    reinit!(cache, item)
    return (cache, state_out)
end
Base.IteratorSize(::Type{<:GridIterators{C}}) where C = Base.IteratorSize(C)
Base.IteratorEltype(::Type{<:GridIterators}) = Base.HasEltype()
Base.eltype(::Type{<:GridIterators{C}}) where C = C
Base.length(iterator::GridIterators) = length(_getset(iterator))


function _check_same_celltype(grid::AbstractGrid, cellset::IntegerCollection)
    isconcretetype(getcelltype(grid)) && return nothing # Short circuit check
    celltype = getcelltype(grid, first(cellset))
    if !all(getcelltype(grid, i) == celltype for i in cellset)
        error("The cells in the cellset are not all of the same celltype.")
    end
end

function _check_same_celltype(grid::AbstractGrid, faceset::Set{FaceIndex})
    isconcretetype(getcelltype(grid)) && return nothing # Short circuit check
    celltype = getcelltype(grid, first(faceset)[1])
    if !all(getcelltype(grid, face[1]) == celltype for face in faceset)
        error("The cells in the faceset are not all of the same celltype.")
    end
end
