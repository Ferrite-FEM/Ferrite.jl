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
struct CellCache{X, G <: AbstractGrid, DH <: Union{AbstractDofHandler, Nothing}, CID <: AbstractArray, NID <: AbstractArray, DID <: AbstractArray, TX <: AbstractArray{X}}
    flags::UpdateFlags
    grid::G
    # Pretty useless to store this since you have it already for the reinit! call, but
    # needed for the CellIterator(...) workflow since the user doesn't necessarily control
    # the loop order in the cell subset.
    cellid::CID
    nodes::NID
    coords::TX
    dh::DH
    dofs::DID
end

function CellCache(grid::Grid{dim, C, T}, flags::UpdateFlags = UpdateFlags()) where {dim, C, T}
    N = nnodes_per_cell(grid, 1) # nodes and coords will be resized in `reinit!`
    nodes = zeros(Int, N)
    coords = zeros(Vec{dim, T}, N)
    return CellCache(flags, grid, [-1], nodes, coords, nothing, Int[])
end

function CellCache(dh::DofHandler{dim}, flags::UpdateFlags = UpdateFlags()) where {dim}
    n = ndofs_per_cell(dh.subdofhandlers[1]) # dofs and coords will be resized in `reinit!`
    N = nnodes_per_cell(get_grid(dh), 1)
    nodes = zeros(Int, N)
    coords = zeros(Vec{dim, get_coordinate_eltype(get_grid(dh))}, N)
    celldofs = zeros(Int, n)
    return CellCache(flags, get_grid(dh), [-1], nodes, coords, dh, celldofs)
end

function CellCache(sdh::SubDofHandler, flags::UpdateFlags = UpdateFlags())
    Tv = get_coordinate_type(sdh.dh.grid)
    return CellCache(flags, sdh.dh.grid, [-1], Int[], Tv[], sdh, Int[])
end

_isresizable(::AbstractArray) = false
_isresizable(::Vector) = true
_isresizable(::SubArray) = false
function reinit!(cc::CellCache, i::Integer)
    cc.cellid[1] = i
    if cc.flags.nodes
        _isresizable(cc.nodes) && resize!(cc.nodes, nnodes_per_cell(cc.grid, i))
        cellnodes!(cc.nodes, cc.grid, i)
    end
    if cc.flags.coords
        _isresizable(cc.coords) && resize!(cc.coords, nnodes_per_cell(cc.grid, i))
        getcoordinates!(cc.coords, cc.grid, i)
    end
    if cc.dh !== nothing && cc.flags.dofs
        _isresizable(cc.dofs) && resize!(cc.dofs, ndofs_per_cell(cc.dh, i))
        celldofs!(cc.dofs, cc.dh, i)
    end
    return cc
end

# reinit! FEValues with CellCache
function reinit!(cv::AbstractCellValues, cc::CellCache)
    cell = reinit_needs_cell(cv) ? getcells(cc.grid, cellid(cc)) : nothing
    return reinit!(cv, cell, cc.coords)
end
function reinit!(fv::FacetValues, cc::CellCache, f::Integer)
    cell = reinit_needs_cell(fv) ? getcells(cc.grid, cellid(cc)) : nothing
    return reinit!(fv, cell, cc.coords, f)
end

# Accessor functions
getnodes(cc::CellCache) = cc.nodes
getcoordinates(cc::CellCache) = cc.coords
celldofs(cc::CellCache) = cc.dofs
cellid(cc::CellCache) = only(cc.cellid)

# TODO: These should really be replaced with something better...
nfacets(cc::CellCache) = nfacets(getcells(cc.grid, cellid(cc)))


"""
    FacetCache(grid::Grid)
    FacetCache(dh::AbstractDofHandler)

Create a cache object with pre-allocated memory for the nodes, coordinates, and dofs of a
cell suitable for looping over *facets* in a grid. The cache is updated for a new facet by
calling `reinit!(cache, fi::FacetIndex)`.

**Methods with `fc::FacetCache`**
 - `reinit!(fc, fi)`: reinitialize the cache for facet `fi::FacetIndex`
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
`facet_a::FacetIndex` and `facet_b::FacetIndex` are the two interface facets.

**Struct fields of `InterfaceCache`**
 - `ic.a :: FacetCache`: facet cache for the first facet of the interface
 - `ic.b :: FacetCache`: facet cache for the second facet of the interface
 - `ic.dofs :: Vector{Int}`: global dof ids for the interface in the *stacked* layout
   `[celldofs(ic.a); celldofs(ic.b)]`. Note that this is a concatenation, not a union: a
   dof carried by both cells (e.g. for an interpolation with dofs on the shared facet)
   appears once per side.

**Methods with `InterfaceCache`**
 - `reinit!(cache::InterfaceCache, facet_a::FacetIndex, facet_b::FacetIndex)`: reinitialize the cache for a new interface
 - `interfacedofs(ic)`: get the global dof ids of the interface in the stacked layout
 - `unique_interfacedofs(ic)`: get the duplicate-free global dof ids of the interface
 - `nstacked_interface_dofs(ic)` / `nunique_interface_dofs(ic)`: sizes of the two layouts
 - `dof_range(ic, field)`: positions of a field's dofs in the stacked layout
 - `condense_interface!(buf, ic, Ke, fe)`: condense a stacked local system onto the unique
   dofs for assembly

!!! note "Lifetime"
    Vectors returned by the accessors above are *borrowed* cache storage: they are valid
    only until the next `reinit!` of the cache (e.g. the next iteration of an
    [`InterfaceIterator`](@ref)), must not be mutated, and must be copied before storing.

See also [`InterfaceIterator`](@ref).
"""
mutable struct InterfaceCache{FC <: FacetCache}
    const a::FC
    const b::FC
    const dofs::Vector{Int}               # stacked layout: [celldofs(a); celldofs(b)]
    const unique_dofs::Vector{Int}        # first-occurrence order: celldofs(a) ++ (b-only)
    const stacked_to_unique::Vector{Int}  # length(dofs): stacked index -> unique index
    unique_map_valid::Bool                # unique_dofs/stacked_to_unique/any_shared current?
    any_shared::Bool                      # length(unique_dofs) != length(dofs)
    sdh_index_a::Int                      # SubDofHandler index of each side, 0 if unknown
    sdh_index_b::Int
end

function InterfaceCache(gridordh::Union{AbstractGrid, AbstractDofHandler})
    fc_a = FacetCache(gridordh)
    fc_b = FacetCache(gridordh)
    return InterfaceCache(fc_a, fc_b, Int[], Int[], Int[], false, false, 0, 0)
end

# The DofHandler backing the cache (`nothing` for grid-only caches)
_interface_dofhandler(ic::InterfaceCache) = _parent_dofhandler(ic.a.cc.dh)
_parent_dofhandler(::Nothing) = nothing
_parent_dofhandler(dh::DofHandler) = dh
_parent_dofhandler(sdh::SubDofHandler) = sdh.dh

function reinit!(cache::InterfaceCache, facet_a::BoundaryIndex, facet_b::BoundaryIndex)
    reinit!(cache.a, facet_a)
    reinit!(cache.b, facet_b)
    dofs_a = celldofs(cache.a)
    dofs_b = celldofs(cache.b)
    n_a = length(dofs_a)
    n_b = length(dofs_b)
    resize!(cache.dofs, n_a + n_b)
    for (i, d) in pairs(dofs_a)
        cache.dofs[i] = d
    end
    for (i, d) in pairs(dofs_b)
        cache.dofs[n_a + i] = d
    end
    # The stacked -> unique dof map is built lazily by the accessors that need it
    # (unique_interfacedofs, nunique_interface_dofs, condense_interface!, ...), so that
    # loops which never use it -- e.g. existing raw discontinuous-Galerkin assembly and
    # the sparsity pattern construction -- do not pay for building it.
    cache.unique_map_valid = false
    # Record which SubDofHandler each side belongs to (0 when constructed from a Grid or
    # when the DofHandler is not defined on the cell)
    dh = _interface_dofhandler(cache)
    if dh === nothing
        cache.sdh_index_a = 0
        cache.sdh_index_b = 0
    else
        cache.sdh_index_a = dh.cell_to_subdofhandler[cellid(cache.a)]
        cache.sdh_index_b = dh.cell_to_subdofhandler[cellid(cache.b)]
    end
    return cache
end

# Build unique_dofs / stacked_to_unique / any_shared for the current interface, at most
# once per reinit!.
function _ensure_unique_interface_map!(cache::InterfaceCache)
    cache.unique_map_valid && return cache
    dofs_a = celldofs(cache.a)
    dofs_b = celldofs(cache.b)
    n_a = length(dofs_a)
    n_b = length(dofs_b)
    resize!(cache.unique_dofs, n_a + n_b)
    resize!(cache.stacked_to_unique, n_a + n_b)
    # Here side: first-occurrence ordering makes this an identity prefix
    for i in 1:n_a
        cache.unique_dofs[i] = dofs_a[i]
        cache.stacked_to_unique[i] = i
    end
    # There side: dedup against the here side by actual global dof equality (each side is
    # internally unique, so only cross-side comparisons are needed). Note that whether dofs
    # can be shared is a property of dof placement (which entities carry dofs), not of the
    # interpolation's conformity, so this scan can not be gated on e.g. `conformity`:
    # nonconforming interpolations such as `CrouzeixRaviart` share facet dofs too.
    n_unique = n_a
    for (i, d) in pairs(dofs_b)
        shared_at = 0
        for j in 1:n_a
            if dofs_a[j] == d
                shared_at = j
                break
            end
        end
        if shared_at == 0
            n_unique += 1
            cache.unique_dofs[n_unique] = d
            cache.stacked_to_unique[n_a + i] = n_unique
        else
            cache.stacked_to_unique[n_a + i] = shared_at
        end
    end
    resize!(cache.unique_dofs, n_unique)
    cache.any_shared = n_unique != n_a + n_b
    cache.unique_map_valid = true
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

"""
    interfacedofs(ic::InterfaceCache) -> Vector{Int}

Return the global dof ids of the interface in the *stacked* layout
`[celldofs(ic.a); celldofs(ic.b)]`. This is a concatenation, not a union: a dof carried by
both cells appears once per side, matching the basis layout of [`InterfaceValues`](@ref).
Local interface matrices/vectors computed in this layout must be condensed onto the unique
dofs before assembly when any dof is shared, see [`condense_interface!`](@ref).

The returned vector is borrowed cache storage, valid until the next `reinit!` of the cache.
"""
interfacedofs(ic::InterfaceCache) = ic.dofs

"""
    unique_interfacedofs(ic::InterfaceCache) -> Vector{Int}

Return the duplicate-free global dof ids of the interface, ordered by first occurrence in
[`interfacedofs`](@ref) (i.e. `celldofs(ic.a)` followed by the dofs unique to
`celldofs(ic.b)`). When no dof is shared between the two cells this is `interfacedofs(ic)`
itself (the identical vector).

Valid after `reinit!` of the cache; the first call after a `reinit!` builds the
stacked-to-unique dof map. The returned vector is borrowed cache storage, valid until the
next `reinit!`.
"""
function unique_interfacedofs(ic::InterfaceCache)
    _ensure_unique_interface_map!(ic)
    return ic.any_shared ? ic.unique_dofs : ic.dofs
end

"""
    nstacked_interface_dofs(ic::InterfaceCache) -> Int

Return the size of the *stacked* interface dof layout, `length(interfacedofs(ic))`. This is
the size of the local matrix/vector that interface kernels fill, matching the number of
basis functions of [`InterfaceValues`](@ref). Valid after `reinit!` of the cache. See also
[`nunique_interface_dofs`](@ref), [`max_nstacked_interface_dofs`](@ref).
"""
nstacked_interface_dofs(ic::InterfaceCache) = length(ic.dofs)

"""
    nunique_interface_dofs(ic::InterfaceCache) -> Int

Return the number of unique global dofs of the interface,
`length(unique_interfacedofs(ic))`. This is the size of the condensed system scattered by
`assemble!`. Equal to [`nstacked_interface_dofs`](@ref) when no dof is shared between the
two cells.

Valid after `reinit!` of the cache; the first call after a `reinit!` builds the
stacked-to-unique dof map.
"""
function nunique_interface_dofs(ic::InterfaceCache)
    _ensure_unique_interface_map!(ic)
    return length(ic.unique_dofs)
end

"""
    is_shared(ic::InterfaceCache, i::Int) -> Bool

Return whether the dof at stacked index `i` (cf. [`interfacedofs`](@ref)) is also carried
by the cell on the other side of the interface. Valid after `reinit!` of the cache.
Intended for checks and tests, not for hot loops.
"""
function is_shared(ic::InterfaceCache, i::Int)
    d = ic.dofs[i]
    other = i <= length(celldofs(ic.a)) ? celldofs(ic.b) : celldofs(ic.a)
    return d in other
end

getcoordinates(ic::InterfaceCache) = (getcoordinates(ic.a), getcoordinates(ic.b))

"""
    InterfaceDofRange

Return type of [`dof_range(::InterfaceCache, ::Symbol)`](@ref): the positions of a field's
dofs in the *stacked* interface layout, as an `AbstractVector{Int}`. Index `i` runs over
the field's basis functions in the [`InterfaceValues`](@ref) ordering (here side first,
then there side); the value is the corresponding row/column of the stacked local matrix.

The two sides are also available as contiguous ranges through the fields `r.here` and
`r.there` (the latter shifted to point into the stacked layout), which is the form the
`function_value`/`function_gradient` evaluation methods accept.
"""
struct InterfaceDofRange <: AbstractVector{Int}
    here::UnitRange{Int}    # the field's block in celldofs(a)
    there::UnitRange{Int}   # the field's block in celldofs(b), shifted by ndofs_per_cell(a)
end
Base.size(r::InterfaceDofRange) = (length(r.here) + length(r.there),)
Base.IndexStyle(::Type{InterfaceDofRange}) = IndexLinear()
@inline function Base.getindex(r::InterfaceDofRange, i::Int)
    @boundscheck checkbounds(r, i)
    nh = length(r.here)
    return i <= nh ? @inbounds(r.here[i]) : @inbounds(r.there[i - nh])
end

"""
    dof_range(ic::InterfaceCache, field::Symbol) -> InterfaceDofRange

Return the positions of `field`'s dofs in the *stacked* interface layout (cf.
[`interfacedofs`](@ref)), as an [`InterfaceDofRange`](@ref). Valid after `reinit!` of the
cache, also when the two cells belong to different `SubDofHandler`s.

Fields that exist on only one side of the interface are not supported and throw an error.

!!! note
    This method previously returned a `Tuple` of the two side-local ranges. The two ranges
    are now available as `r.here` and `r.there` of the returned object.
"""
function dof_range(ic::InterfaceCache, field::Symbol)
    dh = _interface_dofhandler(ic)
    if dh === nothing
        error("this InterfaceCache was constructed from a Grid: no dof information available")
    end
    if ic.sdh_index_a == 0 || ic.sdh_index_b == 0
        error("the DofHandler is not defined on both cells of the interface")
    end
    sdh_a = dh.subdofhandlers[ic.sdh_index_a]
    sdh_b = dh.subdofhandlers[ic.sdh_index_b]
    if _find_field(sdh_a, field) === nothing || _find_field(sdh_b, field) === nothing
        error(
            "field :$field is not present on both cells of the interface (fields on the " *
                "here side: $(sdh_a.field_names), on the there side: $(sdh_b.field_names)). " *
                "Interfaces where a field exists on only one side are not supported."
        )
    end
    here = dof_range(sdh_a, field)
    there = dof_range(sdh_b, field) .+ ndofs_per_cell(sdh_a)
    return InterfaceDofRange(here, there)
end

# Evaluation methods accepting an InterfaceDofRange: unpack into the existing two-range
# methods (`InterfaceValues` stays dof-agnostic; the range is just an index container).
for func in (:function_value, :function_gradient, :function_symmetric_gradient)
    @eval begin
        function $(func)(iv::InterfaceValues, q_point::Int, u::AbstractVector, r::InterfaceDofRange; here::Bool)
            return $(func)(iv, q_point, u, r.here, r.there; here = here)
        end
    end
end
for func in (:function_value_average, :function_gradient_average, :function_value_jump, :function_gradient_jump)
    @eval begin
        function $(func)(iv::InterfaceValues, qp::Int, u::AbstractVector, r::InterfaceDofRange)
            return $(func)(iv, qp, u, r.here, r.there)
        end
    end
end

####################
## Grid iterators ##
####################

## CellIterator ##
"""
    CellIterator(grid::Grid, cellset = 1:getncells(grid))
    CellIterator(dh::AbstractDofHandler, cellset = 1:getncells(dh))

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
    return CellIterator(CellCache(sdh, flags), sdh.cellset)
end

@inline _getset(ci::CellIterator) = ci.set
@inline _getcache(ci::CellIterator) = ci.cc


## FacetIterator ##
FaceIterator(args...) = error("FaceIterator is deprecated, use FacetIterator instead")

# Leaving flags undocumented as for CellIterator
"""
    FacetIterator(gridordh::Union{Grid, AbstractDofHandler}, facetset::AbstractVecOrSet{FacetIndex})

Create a `FacetIterator` to conveniently iterate over the faces in `facetset`. The elements of
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
```
"""
struct FacetIterator{FC <: FacetCache}
    fc::FC
    set::OrderedSet{FacetIndex}
end

function FacetIterator(
        gridordh::Union{Grid, AbstractDofHandler},
        set::AbstractVecOrSet{FacetIndex}, flags::UpdateFlags = UpdateFlags()
    )
    if gridordh isa DofHandler
        # Keep here to maintain same settings as for CellIterator
        _check_same_celltype(get_grid(gridordh), set)
    end
    return FacetIterator(FacetCache(gridordh, flags), convert_to_orderedset(set))
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
ic = InterfaceCache(grid)
neighborhood = Ferrite.get_facet_facet_neighborhood(topology, grid)
for facet in facetskeleton(topology, grid)
    neighbors = neighborhood[facet[1], facet[2]]
    isempty(neighbors) && continue
    neighbor_facet = neighbors[1]
    reinit!(ic, facet, neighbor_facet)
    # ...
end
```
!!! warning
    `InterfaceIterator` is stateful and should not be used for things other than `for`-looping
    (e.g. broadcasting over, or collecting the iterator may yield unexpected results).
"""
struct InterfaceIterator{IC <: InterfaceCache, G <: AbstractGrid, TopologyType <: AbstractTopology}
    cache::IC
    grid::G
    topology::TopologyType
end

function InterfaceIterator(
        gridordh::Union{Grid, AbstractDofHandler},
        topology::ExclusiveTopology = ExclusiveTopology(gridordh isa Grid ? gridordh : get_grid(gridordh))
    )
    grid = gridordh isa Grid ? gridordh : get_grid(gridordh)
    return InterfaceIterator(InterfaceCache(gridordh), grid, topology)
end

# Iterator interface
@inline function Base.iterate(ii::InterfaceIterator, i::Integer)
    neighborhood = get_facet_facet_neighborhood(ii.topology, ii.grid) # TODO: This could be moved to InterfaceIterator constructor (potentially type-instable for non-union or mixed grids)
    skeleton = facetskeleton(ii.topology, ii.grid)
    while i <= length(skeleton)
        facet_a = skeleton[i]; i += 1
        neighbors = neighborhood[facet_a[1], facet_a[2]]
        isempty(neighbors) && continue
        length(neighbors) > 1 && error("multiple neighboring facets not supported yet")
        facet_b = neighbors[1]
        reinit!(ii.cache, facet_a, facet_b)
        return ii.cache, i
    end
    return nothing
end
@inline Base.iterate(ii::InterfaceIterator) = iterate(ii, 1)

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
