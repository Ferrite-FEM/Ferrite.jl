# This file defines iterators used for looping over a grid

struct UpdateFlags
    nodes::Bool
    coords::Bool
    dofs::Bool
end

UpdateFlags(; nodes::Bool = true, coords::Bool = true, dofs::Bool = true) =
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

**Methods with `CellCache`**
 - `reinit!(cc, i)`: reinitialize the cache for cell `i`
 - `cellid(cc)`: get the cell id of the currently cached cell
 - `getnodes(cc)`: get the global node ids of the cell
 - `getcoordinates(cc)`: get the coordinates of the cell
 - `celldofs(cc)`: get the global dof ids of the cell
 - `reinit!(fev, cc)`: reinitialize [`CellValues`](@ref) or [`FacetValues`](@ref)

See also [`CellIterator`](@ref).
"""
mutable struct CellCache{X, G <: AbstractGrid, DH <: Union{AbstractDofHandler, Nothing}}
    const flags::UpdateFlags
    const grid::G
    # Pretty useless to store this since you have it already for the reinit! call, but
    # needed for the CellIterator(...) workflow since the user doesn't necessarily control
    # the loop order in the cell subset.
    cellid::Int
    const nodes::Vector{Int}
    const coords::Vector{X}
    const dh::DH
    const dofs::Vector{Int}
end

function CellCache(grid::Grid{dim, C, T}, flags::UpdateFlags = UpdateFlags()) where {dim, C, T}
    N = nnodes_per_cell(grid, 1) # nodes and coords will be resized in `reinit!`
    nodes = zeros(Int, N)
    coords = zeros(Vec{dim, T}, N)
    return CellCache(flags, grid, -1, nodes, coords, nothing, Int[])
end

function CellCache(dh::DofHandler{dim}, flags::UpdateFlags = UpdateFlags()) where {dim}
    n = ndofs_per_cell(dh.subdofhandlers[1]) # dofs and coords will be resized in `reinit!`
    N = nnodes_per_cell(get_grid(dh), 1)
    nodes = zeros(Int, N)
    coords = zeros(Vec{dim, get_coordinate_eltype(get_grid(dh))}, N)
    celldofs = zeros(Int, n)
    return CellCache(flags, get_grid(dh), -1, nodes, coords, dh, celldofs)
end

function CellCache(sdh::SubDofHandler{<:DofHandler{dim}}, flags::UpdateFlags = UpdateFlags()) where {dim}
    n = ndofs_per_cell(sdh) # dofs and coords will be resized in `reinit!`
    N = nnodes_per_cell(get_grid(sdh.dh), sdh.cellset[1])
    nodes = zeros(Int, N)
    coords = zeros(Vec{dim, get_coordinate_eltype(get_grid(sdh.dh))}, N)
    celldofs = zeros(Int, n)
    return CellCache(flags, sdh.dh.grid, -1, nodes, coords, sdh, celldofs)
end

# TODO: Find a better way to make AllocCheck.jl happy
function reinit!(cc::CellCache, i::Int)
    cc.cellid = i
    if cc.flags.nodes
        resize!(cc.nodes, nnodes_per_cell(cc.grid, i))
        cellnodes!(cc.nodes, cc.grid, i)
    end
    if cc.flags.coords
        resize!(cc.coords, nnodes_per_cell(cc.grid, i))
        getcoordinates!(cc.coords, cc.grid, i)
    end
    if cc.dh !== nothing && cc.flags.dofs
        resize!(cc.dofs, ndofs_per_cell(cc.dh, i))
        celldofs!(cc.dofs, cc.dh, i)
    end
    return cc
end

function reinit!(cc::CellCache{<:Any, <:AbstractGrid, <:SubDofHandler{<:Any, CT}}, i::Int) where {CT}
    # If we have a DofHandler the cells must be of the same type -> no need to resize
    cc.cellid = i
    if cc.flags.nodes
        cell = getcells(cc.grid, i)::CT
        _cellnodes!(cc.nodes, cell)
    end
    if cc.flags.coords
        cell = getcells(cc.grid, i)::CT
        getcoordinates!(cc.coords, cc.grid, cell)
    end
    if cc.dh !== nothing && cc.flags.dofs
        celldofs!(cc.dofs, cc.dh, i)
    end
    return cc
end

# reinit! FEValues with CellCache
reinit!(cv::CellValues, cc::CellCache) = reinit!(cv, cc.coords)
reinit!(fv::FacetValues, cc::CellCache, f::Int) = reinit!(fv, cc.coords, f)

# Accessor functions
getnodes(cc::CellCache) = cc.nodes
getcoordinates(cc::CellCache) = cc.coords
celldofs(cc::CellCache) = cc.dofs
cellid(cc::CellCache) = cc.cellid

# TODO: These should really be replaced with something better...
nfacets(cc::CellCache) = nfacets(getcells(cc.grid, cc.cellid))


"""
    FacetCache(grid::Grid)
    FacetCache(dh::AbstractDofHandler)

Create a cache object with pre-allocated memory for the nodes, coordinates, and dofs of a
cell suitable for looping over *faces* in a grid. The cache is updated for a new face by
calling `reinit!(cache, fi::FacetIndex)`.

**Methods with `fc::FacetCache`**
 - `reinit!(fc, fi)`: reinitialize the cache for face `fi::FacetIndex`
 - `cellid(fc)`: get the current cellid
 - `getnodes(fc)`: get the global node ids of the *cell*
 - `getcoordinates(fc)`: get the coordinates of the *cell*
 - `celldofs(fc)`: get the global dof ids of the *cell*
 - `reinit!(fv, fc)`: reinitialize [`FacetValues`](@ref)

See also [`FacetIterator`](@ref).
"""
mutable struct FacetCache{CC <: CellCache}
    const cc::CC  # const for julia > 1.8
    const dofs::Vector{Int} # aliasing cc.dofs
    current_facet_id::Int
end
function FacetCache(args...)
    cc = CellCache(args...)
    return FacetCache(cc, cc.dofs, 0)
end

function reinit!(fc::FacetCache, facet::BoundaryIndex)
    cellid, facetid = facet
    reinit!(fc.cc, cellid)
    fc.current_facet_id = facetid
    return nothing
end

# Delegate methods to the cell cache
for op in (:getnodes, :getcoordinates, :cellid, :celldofs)
    @eval begin
        function $op(fc::FacetCache, args...)
            return $op(fc.cc, args...)
        end
    end
end

@inline function reinit!(fv::FacetValues, fc::FacetCache)
    return reinit!(fv, fc.cc, fc.current_facet_id)
end

"""
    InterfaceCache(grid::Grid)
    InterfaceCache(dh::AbstractDofHandler)

Create a cache object with pre-allocated memory for the nodes, coordinates, and dofs of an
interface. The cache is updated for a new cell by calling `reinit!(cache, facet_a, facet_b)` where
`facet_a::FacetIndex` and `facet_b::FacetIndex` are the two interface faces.

**Struct fields of `InterfaceCache`**
 - `ic.a :: FacetCache`: face cache for the first face of the interface
 - `ic.b :: FacetCache`: face cache for the second face of the interface
 - `ic.dofs :: Vector{Int}`: global dof ids for the interface (union of `ic.a.dofs` and `ic.b.dofs`)

**Methods with `InterfaceCache`**
 - `reinit!(cache::InterfaceCache, facet_a::FacetIndex, facet_b::FacetIndex)`: reinitialize the cache for a new interface
 - `interfacedofs(ic)`: get the global dof ids of the interface

See also [`InterfaceIterator`](@ref).
"""
struct InterfaceCache{FC_HERE <: FacetCache, FC_THERE <: FacetCache}
    a::FC_HERE
    b::FC_THERE
    dofs::Vector{Int}
end

function InterfaceCache(gridordh::Union{AbstractGrid, AbstractDofHandler})
    fc_a = FacetCache(gridordh)
    fc_b = FacetCache(gridordh)
    return InterfaceCache(fc_a, fc_b, zeros(Int, length(celldofs(fc_a)) + length(celldofs(fc_b))))
end

function InterfaceCache(sdh_here::SubDofHandler, sdh_there::SubDofHandler)
    fc_a = FacetCache(sdh_here)
    fc_b = FacetCache(sdh_there)
    return InterfaceCache(fc_a, fc_b, zeros(Int, length(celldofs(fc_a)) + length(celldofs(fc_b))))
end

function reinit!(cache::InterfaceCache, interface::InterfaceIndex)
    reinit!(cache.a, FacetIndex(interface.idx[1], interface.idx[2]))
    reinit!(cache.b, FacetIndex(interface.idx[3], interface.idx[4]))
    for (i, d) in pairs(cache.a.dofs)
        cache.dofs[i] = d
    end
    for (i, d) in pairs(cache.b.dofs)
        cache.dofs[i + length(cache.a.dofs)] = d
    end
    return cache
end

function reinit!(iv::InterfaceValues, ic::InterfaceCache)
    return reinit!(
        iv,
        getcells(ic.a.cc.grid, cellid(ic.a)),
        getcoordinates(ic.a),
        ic.a.current_facet_id[],
        getcells(ic.b.cc.grid, cellid(ic.b)),
        getcoordinates(ic.b),
        ic.b.current_facet_id[],
    )
end

interfacedofs(ic::InterfaceCache) = ic.dofs
dof_range(ic::InterfaceCache, field::Symbol) = (dof_range(ic.a.cc.dh, field), dof_range(ic.b.cc.dh, field) .+ length(celldofs(ic.a)))
getcoordinates(ic::InterfaceCache) = (getcoordinates(ic.a), getcoordinates(ic.b))

####################
## Grid iterators ##
####################

## CellIterator ##
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
struct CellIterator{CC <: CellCache, IC <: IntegerCollection}
    cc::CC
    set::IC
end

function CellIterator(
        gridordh::Union{Grid, DofHandler},
        set::Union{IntegerCollection, Nothing} = nothing,
        flags::UpdateFlags = UpdateFlags()
    )
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
function CellIterator(gridordh::Union{Grid, DofHandler}, flags::UpdateFlags)
    return CellIterator(gridordh, nothing, flags)
end
function CellIterator(sdh::SubDofHandler, flags::UpdateFlags = UpdateFlags())
    return CellIterator(sdh, sdh.cellset, flags)
end

function CellIterator(sdh::SubDofHandler, set::IntegerCollection, flags::UpdateFlags = UpdateFlags())
    return CellIterator(CellCache(sdh, flags), set)
end
@inline _getset(ci::CellIterator) = ci.set
@inline _getcache(ci::CellIterator) = ci.cc


## FacetIterator ##
FaceIterator(args...) = error("FaceIterator is deprecated, use FacetIterator instead")

# Leaving flags undocumented as for CellIterator
"""
    FacetIterator(gridordh::Union{Grid,AbstractDofHandler}, facetset::AbstractVecOrSet{FacetIndex})

Create a `FacetIterator` to conveniently iterate over the faces in `facestet`. The elements of
the iterator are [`FacetCache`](@ref)s which are properly `reinit!`ialized. See
[`FacetCache`](@ref) for more details.

Looping over a `FacetIterator`, i.e.:
```julia
for fc in FacetIterator(grid, facetset)
    # ...
end
```
is thus simply convenience for the following equivalent snippet:
```julia
fc = FacetCache(grid)
for faceindex in facetset
    reinit!(fc, faceindex)
    # ...
end
"""
struct FacetIterator{FC <: FacetCache, SetType <: AbstractVecOrSet{FacetIndex}}
    fc::FC
    set::SetType
end

function FacetIterator(
        gridordh::Union{Grid, AbstractDofHandler},
        set::AbstractVecOrSet{FacetIndex}, flags::UpdateFlags = UpdateFlags()
    )
    if gridordh isa DofHandler
        # Keep here to maintain same settings as for CellIterator
        _check_same_celltype(get_grid(gridordh), set)
    end
    return FacetIterator(FacetCache(gridordh, flags), set)
end

function FacetIterator(
        gridordh::Union{<:AbstractGrid, <:DofHandler}, topology::ExclusiveTopology,
        flags::UpdateFlags = UpdateFlags()
    )
    grid = gridordh isa DofHandler ? get_grid(gridordh) : gridordh
    set = Set(create_boundaryfacetset(grid, topology, _ -> true))
    return FacetIterator(gridordh, set, flags)
end

function FacetIterator(
        sdh::SubDofHandler, topology::ExclusiveTopology,
        flags::UpdateFlags = UpdateFlags()
    )
    grid = get_grid(sdh.dh)
    set_unfiltered = create_boundaryfacetset(grid, topology, _ -> true)
    set = Set(filter!(facet -> facet[1] ∈ sdh.cellset, set_unfiltered))
    return FacetIterator(sdh, set, flags)
end

@inline _getcache(fi::FacetIterator) = fi.fc
@inline _getset(fi::FacetIterator) = fi.set


"""
    InterfaceIterator(grid::Grid, [topology::ExclusiveTopology])
    InterfaceIterator(dh::AbstractDofHandler, [topology::ExclusiveTopology])

Create an `InterfaceIterator` to conveniently iterate over all the interfaces in a
grid. The elements of the iterator are [`InterfaceCache`](@ref)s which are properly
`reinit!`ialized. See [`InterfaceCache`](@ref) for more details.
Looping over an `InterfaceIterator`, i.e.:
```julia
for ic in InterfaceIterator(grid, topology)
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
struct InterfaceIterator{IC <: InterfaceCache, SetType <: AbstractSet{InterfaceIndex}}
    cache::IC
    set::SetType
end

@inline _getcache(ii::InterfaceIterator) = ii.cache
@inline _getset(ii::InterfaceIterator) = ii.set

function InterfaceIterator(
        gridordh::Union{Grid, DofHandler},
        set::AbstractVecOrSet{InterfaceIndex}
    )
    grid = gridordh isa Grid ? gridordh : get_grid(gridordh)
    if gridordh isa DofHandler
        # Keep here to maintain same settings as for CellIterator
        _check_same_celltype(grid, set)
    end
    return InterfaceIterator(InterfaceCache(gridordh), set)
end
function InterfaceIterator(
        sdh_here::SubDofHandler,
        sdh_there::SubDofHandler,
        set::AbstractVecOrSet{InterfaceIndex}
    )
    grid = get_grid(sdh_here.dh)
    _check_same_celltype(grid, set, getcelltype(grid, first(sdh_here.cellset)), getcelltype(grid, first(sdh_there.cellset)))
    return InterfaceIterator(InterfaceCache(sdh_here, sdh_there), set)
end
function InterfaceIterator(
        gridordh::Union{Grid, DofHandler},
        topology::ExclusiveTopology = ExclusiveTopology(gridordh isa Grid ? gridordh : get_grid(gridordh))
    )
    grid = gridordh isa Grid ? gridordh : get_grid(gridordh)
    if gridordh isa DofHandler
        # Keep here to maintain same settings as for CellIterator
        _check_same_celltype(grid, 1:getncells(grid))
        @assert length(gridordh.subdofhandlers) == 1 "Use InterfaceIterator(::SubDofHandler, ::SubDofHandler) for subdomain support"
    end
    neighborhood = get_facet_facet_neighborhood(topology, grid)
    fs = facetskeleton(topology, grid)
    ninterfaces = count(facet -> !isempty(neighborhood[facet[1], facet[2]]), fs)
    set = Set{InterfaceIndex}()
    sizehint!(set, ninterfaces)
    for facet in fs
        isempty(neighborhood[facet[1], facet[2]]) && continue
        push!(set, InterfaceIndex(facet[1], facet[2], neighborhood[facet[1], facet[2]][][1], neighborhood[facet[1], facet[2]][][2]))
    end
    return InterfaceIterator(gridordh, set)
end

function InterfaceIterator(
        sdh_here::SubDofHandler,
        sdh_there::SubDofHandler,
        topology::ExclusiveTopology = ExclusiveTopology(get_grid(sdh_here.dh)),
        # TODO: better name?
        allow_all::Bool = true # allow interfaces that belong to the other cell according to the cell numbering to be iterated over as if they were "here" not "there"
    )
    grid = get_grid(sdh_here.dh)
    neighborhood = get_facet_facet_neighborhood(topology, grid)
    fs = facetskeleton(topology, grid)
    ninterfaces = 0
    for facet in fs
        if facet[1] ∈ sdh_here.cellset
            neighbors = neighborhood[facet[1], facet[2]]
            isempty(neighbors) && continue
            neighbors[][1] ∈ sdh_there.cellset || continue
        elseif allow_all && facet[1] ∈ sdh_there.cellset
            neighbors = neighborhood[facet[1], facet[2]]
            isempty(neighbors) && continue
            neighbors[][1] ∈ sdh_here.cellset || continue
        else
            continue
        end
        ninterfaces += 1
    end
    set = Set{InterfaceIndex}()
    sizehint!(set, ninterfaces)
    for facet in fs
        if facet[1] ∈ sdh_here.cellset
            neighbors = neighborhood[facet[1], facet[2]]
            isempty(neighbors) && continue
            neighbors[][1] ∈ sdh_there.cellset || continue
            push!(set, InterfaceIndex(facet[1], facet[2], neighborhood[facet[1], facet[2]][][1], neighborhood[facet[1], facet[2]][][2]))
        elseif allow_all && facet[1] ∈ sdh_there.cellset
            neighbors = neighborhood[facet[1], facet[2]]
            isempty(neighbors) && continue
            neighbors[][1] ∈ sdh_here.cellset || continue
            push!(set, InterfaceIndex(neighborhood[facet[1], facet[2]][][1], neighborhood[facet[1], facet[2]][][2], facet[1], facet[2]))
        else
            continue
        end
    end
    return InterfaceIterator(sdh_here, sdh_there, set)
end

# Iterator interface for CellIterator/FacetIterator
const GridIterators{C} = Union{CellIterator{C}, FacetIterator{C}, InterfaceIterator{C}}

function Base.iterate(iterator::GridIterators, state_in...)
    it = iterate(_getset(iterator), state_in...)
    it === nothing && return nothing
    item, state_out = it
    cache = _getcache(iterator)
    reinit!(cache, item)
    return (cache, state_out)
end
Base.IteratorSize(::Type{<:GridIterators{C}}) where {C} = Base.IteratorSize(C)
Base.IteratorEltype(::Type{<:GridIterators}) = Base.HasEltype()
Base.eltype(::Type{<:GridIterators{C}}) where {C} = C
Base.length(iterator::GridIterators) = length(_getset(iterator))

function _check_same_celltype(grid::AbstractGrid, cellset::IntegerCollection)
    isconcretetype(getcelltype(grid)) && return nothing # Short circuit check
    celltype = getcelltype(grid, first(cellset))
    if !all(getcelltype(grid, i) == celltype for i in cellset)
        error("The cells in the cellset are not all of the same celltype.")
    end
    return
end

function _check_same_celltype(grid::AbstractGrid, facetset::AbstractVecOrSet{<:BoundaryIndex})
    isconcretetype(getcelltype(grid)) && return nothing # Short circuit check
    celltype = getcelltype(grid, first(facetset)[1])
    if !all(getcelltype(grid, facet[1]) == celltype for facet in facetset)
        error("The cells in the set (set of $(eltype(facetset))) are not all of the same celltype.")
    end
    return
end

function _check_same_celltype(
        grid::AbstractGrid, interfaceset::AbstractVecOrSet{InterfaceIndex},
        celltype_here::Type{<:AbstractCell} = getcelltype(grid, first(interfaceset)[1]),
        celltype_there::Type{<:AbstractCell} = getcelltype(grid, first(interfaceset)[3])
    )
    isconcretetype(getcelltype(grid)) && return nothing # Short circuit check
    if !all(getcelltype(grid, interface[1]) == celltype_here && getcelltype(grid, interface[3]) == celltype_there for interface in interfaceset)
        error("The cells in the set (set of InterfaceIndex) are not all of the same celltype on each side.")
    end
    return
end
