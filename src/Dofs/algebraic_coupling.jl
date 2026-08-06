# Coupling descriptors and augmented local dof layouts for algebraic variables.
#
# A coupling descriptor states, structurally, where weak-form terms couple algebraic
# variables (see `AlgebraicVariable`) to spatial fields: on which entities, between which
# test/trial variables. Descriptors contain no physics -- user code supplies the weak-form
# kernel -- and are consumed by sparsity generation (`add_coupling_entries!`) and by
# `local_dofs`, which builds the augmented local dof layout used for square local assembly.

########################
# Local layout metadata #
########################

# Immutable name/range metadata for the augmented local layout of one descriptor on one
# SubDofHandler (or, for an AlgebraicCoupling, its single algebraic layout). Prepared once
# at descriptor construction and reused for every entity.
struct LocalLayoutInfo
    names::Vector{Symbol}          # sdh field names followed by the descriptor's algebraic variables
    ranges::Vector{UnitRange{Int}} # local dof range for each entry in `names`
    n_cell_dofs::Int               # number of ordinary cell dofs (0 for AlgebraicCoupling)
    n_total::Int                   # total number of local dofs
    algebraic_indices::Vector{Int} # indices into the DofHandler's algebraic registries, in descriptor field order
    dof_mask::Matrix{Bool}         # n_total × n_total expansion of the descriptor's coupling matrix
end

# Number of rows/columns a field occupies in a component-level coupling matrix: all
# components for a spatial field, the active components for an algebraic variable.
function _participating_ncomponents(dh::DofHandler, name::Symbol)
    vidx = _find_algebraic_variable(dh, name)
    vidx === nothing || return n_algebraic_dofs(dh.algebraic_variables[vidx])
    return n_components(dh, name)
end

# Expand the descriptor's field- or component-level coupling matrix to a dof-pair level
# mask over the augmented local layout. Note that this expansion is keyed to the
# descriptor's selected `fields` (the existing cell-`coupling` expander assumes all handler
# fields and cannot be reused for subsets). Rows are test dofs and columns trial dofs.
# Local dofs of spatial fields not selected by the descriptor keep zero rows and columns.
function _expand_coupling_mask(
        dh::DofHandler, fields::Tuple{Vararg{Symbol}}, coupling::Matrix{Bool},
        field_level::Bool, names::Vector{Symbol}, ranges::Vector{UnitRange{Int}}, n_total::Int,
    )
    mask = zeros(Bool, n_total, n_total)
    field_ranges = map(fields) do fname
        i = findfirst(==(fname), names)
        @assert i !== nothing # validated during descriptor construction
        return ranges[i]
    end
    if field_level
        for (j, rj) in pairs(field_ranges), (i, ri) in pairs(field_ranges)
            coupling[i, j] || continue
            mask[ri, rj] .= true
        end
    else
        # Component-level: within a field's local range the component index cycles fastest
        # (matching the celldofs layout of vectorized interpolations); for an algebraic
        # variable each active component owns exactly one local dof.
        ncomps = Int[_participating_ncomponents(dh, f) for f in fields]
        offsets = pushfirst!(cumsum(ncomps), 0)
        for (jf, rj) in pairs(field_ranges), (j, J) in pairs(rj)
            cj = offsets[jf] + mod1(j, ncomps[jf])
            for (i_f, ri) in pairs(field_ranges), (i, I) in pairs(ri)
                ci = offsets[i_f] + mod1(i, ncomps[i_f])
                mask[I, J] = coupling[ci, cj]
            end
        end
    end
    return mask
end

# Build the layout metadata for one SubDofHandler (`sdh === nothing` for the algebraic-only
# layout of an AlgebraicCoupling). The layout contains all ordinary cell dofs in celldofs
# order followed by the active dofs of each algebraic variable named by the descriptor, in
# descriptor field order and then active-component order.
function _local_layout_info(
        dh::DofHandler, sdh::Union{SubDofHandler, Nothing},
        fields::Tuple{Vararg{Symbol}}, coupling::Matrix{Bool}, field_level::Bool,
    )
    names = Symbol[]
    ranges = UnitRange{Int}[]
    n_cell_dofs = 0
    if sdh !== nothing
        n_cell_dofs = ndofs_per_cell(sdh)
        for (i, fname) in pairs(sdh.field_names)
            push!(names, fname)
            push!(ranges, dof_range(sdh, i))
        end
    end
    offset = n_cell_dofs
    algebraic_indices = Int[]
    for fname in fields
        vidx = _find_algebraic_variable(dh, fname)
        vidx === nothing && continue
        push!(algebraic_indices, vidx)
        n = n_algebraic_dofs(dh.algebraic_variables[vidx])
        push!(names, fname)
        push!(ranges, (offset + 1):(offset + n))
        offset += n
    end
    n_total = offset
    dof_mask = _expand_coupling_mask(dh, fields, coupling, field_level, names, ranges, n_total)
    return LocalLayoutInfo(names, ranges, n_cell_dofs, n_total, algebraic_indices, dof_mask)
end

##############################
# Descriptor validation      #
##############################

function _validate_coupling_fields(dh::DofHandler, fields)
    isclosed(dh) || error("coupling descriptors require a closed DofHandler")
    fields isa Union{Tuple, AbstractVector} || error("`fields` must be a tuple or vector of `Symbol`s, got $(repr(fields))")
    isempty(fields) && error("`fields` must name at least one variable")
    all(x -> x isa Symbol, fields) || error("`fields` must be a collection of `Symbol`s, got $(repr(fields))")
    fields_t = Tuple(fields)
    allunique(fields_t) || error("duplicate names in `fields = $(fields_t)`")
    for name in fields_t
        if _find_algebraic_variable(dh, name) === nothing && name ∉ getfieldnames(dh)
            error("no spatial field or algebraic variable named :$name in the DofHandler (fields: $(getfieldnames(dh)), algebraic variables: $(dh.algebraic_names))")
        end
    end
    return fields_t
end

# Classify the coupling matrix as field level or component level based on its size, see
# the CellCoupling docstring. Returns the (copied) matrix and `field_level::Bool`.
function _classify_coupling_matrix(dh::DofHandler, fields::Tuple{Vararg{Symbol}}, coupling::AbstractMatrix{Bool})
    size(coupling, 1) == size(coupling, 2) || error("the coupling matrix must be square, got size $(size(coupling))")
    nf = length(fields)
    ncomps = sum(f -> _participating_ncomponents(dh, f), fields)
    sz = size(coupling, 1)
    if sz == nf
        return Matrix{Bool}(coupling), true
    elseif sz == ncomps
        return Matrix{Bool}(coupling), false
    else
        error(
            "could not interpret the coupling matrix of size $(size(coupling)) for fields $(fields): " *
                "expected either a field-level matrix of size ($nf, $nf) or a component-level matrix " *
                "of size ($ncomps, $ncomps) (all components for spatial fields, active components for algebraic variables)"
        )
    end
end

# Normalize the `algebraic_coupling` specification of a descriptor to
# `(fields, coupling_matrix, field_level)`. Two forms are supported, see the CellCoupling
# docstring: a collection of pair/tuple entries (field level, `fields` derived from the
# entries), or a `Bool` matrix together with the `fields` keyword.

function _process_algebraic_coupling(dh::DofHandler, fields, spec::AbstractMatrix{Bool})
    fields === nothing && error("the matrix form of `algebraic_coupling` requires the participating variables to be named with the `fields` keyword argument")
    fields_t = _validate_coupling_fields(dh, fields)
    coupling_mat, field_level = _classify_coupling_matrix(dh, fields_t, spec)
    return fields_t, coupling_mat, field_level
end

function _process_algebraic_coupling(dh::DofHandler, fields, spec)
    fields === nothing || error("`fields` can only be passed together with the matrix form of `algebraic_coupling`; in the pair form the participating variables are derived from the entries")
    # A bare entry is accepted as a one-entry specification
    if spec isa Pair || spec isa Tuple{Symbol, Symbol}
        spec = (spec,)
    end
    if !(spec isa Union{Tuple, AbstractVector})
        error("cannot interpret `algebraic_coupling = $(repr(spec))`: expected a collection of `:a => :b` pairs and/or `(:a, :b)` tuples, or a `Bool` matrix together with `fields`")
    end
    isempty(spec) && error("`algebraic_coupling` must declare at least one coupling entry")
    # Expand each entry to its directed (test, trial) pairs: `:a => :b` couples one way,
    # `(:a, :b)` couples both ways.
    directed = Tuple{Symbol, Symbol}[]
    for entry in spec
        if entry isa Pair{Symbol, Symbol}
            push!(directed, (entry.first, entry.second))
        elseif entry isa Tuple{Symbol, Symbol}
            push!(directed, entry)
            entry[1] == entry[2] || push!(directed, (entry[2], entry[1]))
        else
            error("cannot interpret the `algebraic_coupling` entry $(repr(entry)): expected a directional pair `:a => :b` (test `:a`, trial `:b`) or a symmetric tuple `(:a, :b)` (both directions)")
        end
    end
    if !allunique(directed)
        dup = directed[findfirst(i -> directed[i] in view(directed, 1:(i - 1)), eachindex(directed))]
        error("`algebraic_coupling` declares the coupling (test :$(dup[1]), trial :$(dup[2])) more than once")
    end
    # The participating variables, in order of first appearance
    names = Symbol[]
    for (a, b) in directed
        a in names || push!(names, a)
        b in names || push!(names, b)
    end
    fields_t = _validate_coupling_fields(dh, Tuple(names))
    for (a, b) in directed
        if _find_algebraic_variable(dh, a) === nothing && _find_algebraic_variable(dh, b) === nothing
            error(
                "the `algebraic_coupling` entry (:$a, :$b) couples two spatial fields, but every entry must " *
                    "involve at least one algebraic variable (entries between spatial fields on the same cell " *
                    "are governed by the `coupling` keyword of `allocate_matrix`)"
            )
        end
    end
    coupling_mat = zeros(Bool, length(fields_t), length(fields_t))
    for (a, b) in directed
        coupling_mat[findfirst(==(a), fields_t), findfirst(==(b), fields_t)] = true
    end
    return fields_t, coupling_mat, true
end

# Verify that every spatial field selected by the descriptor exists on the SubDofHandler
# that `entity` belongs to.
function _check_spatial_fields_on_sdh(dh::DofHandler, sdh::SubDofHandler, fields::Tuple{Vararg{Symbol}}, entity::String)
    for name in fields
        _find_algebraic_variable(dh, name) === nothing || continue
        if name ∉ sdh.field_names
            error("the field :$name does not exist on $entity (fields on the corresponding SubDofHandler: $(sdh.field_names))")
        end
    end
    return
end

##########################
# Coupling descriptors   #
##########################

"""
    CellCoupling(dh::DofHandler, cells; algebraic_coupling)
    CellCoupling(dh::DofHandler, cells; fields, algebraic_coupling::AbstractMatrix{Bool})

Structural descriptor for weak-form terms integrated over the cells in `cells`, coupling
algebraic variables to spatial fields (and/or to other algebraic variables). Descriptors
carry only structural metadata -- the entity set, the participating variables, and the
allowed test/trial blocks -- and are used by [`allocate_matrix`](@ref) /
[`add_sparsity_entries!`](@ref) through the `algebraic_couplings` keyword, and by
[`local_dofs`](@ref) during assembly. The weak-form kernel itself remains user code.

In the first (pair) form, `algebraic_coupling` is a collection of entries where each
entry involves at least one algebraic variable:
 - `:u => :σ̄` declares a directional coupling: test dofs of `:u` may couple to trial dofs
   of `:σ̄`;
 - `(:u, :σ̄)` declares both directions at once.

For example, `algebraic_coupling = ((:u, :σ̄), :σ̄ => :σ̄)` declares the `u`-`σ̄`, `σ̄`-`u`,
and `σ̄`-`σ̄` blocks. The participating variables are derived from the entries, in order of
first appearance.

In the second (matrix) form, `fields` is a tuple of the participating variable names,
e.g. `fields = (:u, :σ̄)`, and `algebraic_coupling` is a square `Bool` matrix stating
which test variables (rows) may couple to which trial variables (columns), ordered as
`fields`. The matrix may be asymmetric. Either a field-level matrix of size
`length(fields) × length(fields)`, or a component-level matrix whose size is the sum of
the participating component counts: all components for spatial fields and active
components for algebraic variables.

Coupling declared by descriptors is *additive*: sparsity is the union of the ordinary cell
entries (controlled by the `coupling` keyword of `allocate_matrix`) and all descriptor
entries, and an entry *not* declared by a descriptor only means that *this descriptor*
adds nothing for that block.

See also [`FacetCoupling`](@ref) and [`AlgebraicCoupling`](@ref).
"""
struct CellCoupling{N, DH <: DofHandler} <: AbstractCoupling
    dh::DH
    cells::OrderedSet{Int}
    fields::NTuple{N, Symbol}
    coupling::Matrix{Bool}
    field_level::Bool
    layout_infos::Vector{Union{Nothing, LocalLayoutInfo}} # indexed by SubDofHandler index
end

function CellCoupling(dh::DofHandler, cells::IntegerCollection; algebraic_coupling, fields = nothing)
    fields_t, coupling_mat, field_level = _process_algebraic_coupling(dh, fields, algebraic_coupling)
    # The descriptor owns its entity set (mutating the input set after construction must
    # not desynchronize it from the precomputed layout metadata)
    cellset = OrderedSet{Int}(cells)
    ncells = getncells(get_grid(dh))
    layout_infos = Vector{Union{Nothing, LocalLayoutInfo}}(nothing, length(dh.subdofhandlers))
    for cellid in cellset
        1 <= cellid <= ncells || error("cell index $cellid is out of bounds (the grid has $ncells cells)")
        sdhidx = dh.cell_to_subdofhandler[cellid]
        sdhidx == 0 && error("cell $cellid does not belong to any SubDofHandler")
        layout_infos[sdhidx] === nothing || continue
        sdh = dh.subdofhandlers[sdhidx]
        _check_spatial_fields_on_sdh(dh, sdh, fields_t, "cell $cellid")
        layout_infos[sdhidx] = _local_layout_info(dh, sdh, fields_t, coupling_mat, field_level)
    end
    return CellCoupling(dh, cellset, fields_t, coupling_mat, field_level, layout_infos)
end

"""
    FacetCoupling(dh::DofHandler, facets; algebraic_coupling)
    FacetCoupling(dh::DofHandler, facets; fields, algebraic_coupling::AbstractMatrix{Bool})

Structural descriptor for weak-form terms integrated over the facets in `facets`
(a set of `FacetIndex`). See [`CellCoupling`](@ref) for the meaning of the keyword
arguments and the additive coupling semantics.

The local layout of a facet term contains *all* dofs of the adjacent cell, including dofs
without support on the facet itself, since facet integrals may evaluate cell gradients on
the facet. Consequently only cells adjacent to the selected facets couple to the algebraic
variables, but they do so with all dofs of the selected spatial fields.
"""
struct FacetCoupling{N, DH <: DofHandler} <: AbstractCoupling
    dh::DH
    facets::OrderedSet{FacetIndex}
    adjacent_cells::OrderedSet{Int} # unique cells adjacent to `facets` (internal, used by sparsity generation)
    fields::NTuple{N, Symbol}
    coupling::Matrix{Bool}
    field_level::Bool
    layout_infos::Vector{Union{Nothing, LocalLayoutInfo}} # indexed by SubDofHandler index
end

function FacetCoupling(dh::DofHandler, facets::AbstractVecOrSet{FacetIndex}; algebraic_coupling, fields = nothing)
    fields_t, coupling_mat, field_level = _process_algebraic_coupling(dh, fields, algebraic_coupling)
    # The descriptor owns its entity set, see the CellCoupling constructor
    facetset = OrderedSet{FacetIndex}(facets)
    grid = get_grid(dh)
    ncells = getncells(grid)
    layout_infos = Vector{Union{Nothing, LocalLayoutInfo}}(nothing, length(dh.subdofhandlers))
    adjacent_cells = OrderedSet{Int}()
    for (cellid, facetid) in facetset
        1 <= cellid <= ncells || error("facet ($cellid, $facetid): cell index $cellid is out of bounds (the grid has $ncells cells)")
        cell = getcells(grid, cellid)
        1 <= facetid <= nfacets(cell) || error("facet ($cellid, $facetid): facet index $facetid is out of bounds for a cell with $(nfacets(cell)) facets")
        sdhidx = dh.cell_to_subdofhandler[cellid]
        sdhidx == 0 && error("facet ($cellid, $facetid): cell $cellid does not belong to any SubDofHandler")
        push!(adjacent_cells, cellid)
        layout_infos[sdhidx] === nothing || continue
        sdh = dh.subdofhandlers[sdhidx]
        _check_spatial_fields_on_sdh(dh, sdh, fields_t, "the cell of facet ($cellid, $facetid)")
        layout_infos[sdhidx] = _local_layout_info(dh, sdh, fields_t, coupling_mat, field_level)
    end
    return FacetCoupling(dh, facetset, adjacent_cells, fields_t, coupling_mat, field_level, layout_infos)
end

"""
    AlgebraicCoupling(dh::DofHandler; algebraic_coupling)
    AlgebraicCoupling(dh::DofHandler; fields, algebraic_coupling::AbstractMatrix{Bool})

Structural descriptor for terms involving only algebraic variables, e.g. the tangent of a
coupled 0D problem. All participating names must refer to algebraic variables. See
[`CellCoupling`](@ref) for the meaning of the keyword arguments and the additive coupling
semantics. The corresponding local layout (see [`local_dofs`](@ref)) contains only the
selected algebraic variables, in field order.

Note that the diagonal entries of the global matrix are always allocated, so an
`AlgebraicCoupling` is only needed for off-diagonal algebraic--algebraic entries (dense
self-coupling of a multi-component variable or cross-coupling between variables).
"""
struct AlgebraicCoupling{N, DH <: DofHandler} <: AbstractCoupling
    dh::DH
    fields::NTuple{N, Symbol}
    coupling::Matrix{Bool}
    field_level::Bool
    layout_info::LocalLayoutInfo
end

function AlgebraicCoupling(dh::DofHandler; algebraic_coupling, fields = nothing)
    fields_t, coupling_mat, field_level = _process_algebraic_coupling(dh, fields, algebraic_coupling)
    for name in fields_t
        if _find_algebraic_variable(dh, name) === nothing
            error(":$name is a spatial field, but an AlgebraicCoupling can only couple algebraic variables; use a CellCoupling or FacetCoupling for terms involving spatial fields")
        end
    end
    layout_info = _local_layout_info(dh, nothing, fields_t, coupling_mat, field_level)
    return AlgebraicCoupling(dh, fields_t, coupling_mat, field_level, layout_info)
end

########################
# Descriptor interface #
########################

"""
    Ferrite.entities(coupling::Union{CellCoupling, FacetCoupling})

Return the entity set of the coupling descriptor: the cell set of a [`CellCoupling`](@ref)
or the facet set of a [`FacetCoupling`](@ref). An [`AlgebraicCoupling`](@ref) has no
entity set. The returned set is a copy; descriptors are immutable after construction.
"""
entities(c::CellCoupling) = copy(c.cells)
entities(c::FacetCoupling) = copy(c.facets)

"""
    Ferrite.fields(coupling::Ferrite.AbstractCoupling)

Return the ordered tuple of variable names participating in the coupling descriptor.
"""
fields(c::AbstractCoupling) = c.fields

"""
    Ferrite.coupling_matrix(coupling::Ferrite.AbstractCoupling)

Return the `Bool` coupling matrix of the descriptor, where rows are test variables and
columns trial variables, ordered as [`Ferrite.fields`](@ref). The matrix is
either field level or component level, see [`CellCoupling`](@ref). The returned matrix is
a copy; descriptors are immutable after construction.
"""
coupling_matrix(c::AbstractCoupling) = copy(c.coupling)

function _show_coupling(io::IO, c::AbstractCoupling, name::String, entity_str::Union{String, Nothing})
    println(io, name, ":")
    println(io, "  Fields: ", join(map(repr, c.fields), ", "))
    entity_str === nothing || println(io, "  Entities: ", entity_str)
    print(io, "  Coupling matrix: ", size(c.coupling, 1), "×", size(c.coupling, 2), " (", c.field_level ? "field level" : "component level", ")")
    return
end

function Base.show(io::IO, ::MIME"text/plain", c::CellCoupling)
    return _show_coupling(io, c, "CellCoupling", string(length(c.cells), " cells"))
end
function Base.show(io::IO, ::MIME"text/plain", c::FacetCoupling)
    return _show_coupling(io, c, "FacetCoupling", string(length(c.facets), " facets (", length(c.adjacent_cells), " adjacent cells)"))
end
function Base.show(io::IO, ::MIME"text/plain", c::AlgebraicCoupling)
    return _show_coupling(io, c, "AlgebraicCoupling", nothing)
end

# Normalization helper for the `algebraic_couplings` keyword: iterate descriptors stored
# in a named tuple, or any other iterable (tuple, vector, generator, ...), or a single
# descriptor. Elements are validated to be descriptors by the caller.
_iterate_algebraic_couplings(c::AbstractCoupling) = (c,)
_iterate_algebraic_couplings(cs::NamedTuple) = values(cs)
function _iterate_algebraic_couplings(cs)
    if !applicable(Base.iterate, cs)
        error("cannot interpret `algebraic_couplings = $(repr(cs))`: expected coupling descriptors (`CellCoupling`, `FacetCoupling`, `AlgebraicCoupling`) or an iterable of them")
    end
    return cs
end

##################
# LocalDofLayout #
##################

"""
    LocalDofLayout

Read-only vector of global dof indices for the augmented local system of one coupling
descriptor on one entity, filled by [`local_dofs`](@ref) or in place by
[`local_dofs!`](@ref). The layout contains the ordinary cell dofs (in `celldofs` order)
followed by the active dofs of the descriptor's algebraic variables (in descriptor field
order and then active-component order), and can be passed directly to [`assemble!`](@ref)
and [`apply_assemble!`](@ref) together with a square local matrix. Local ranges are
queried by name with [`dof_range(::LocalDofLayout, ::Symbol)`](@ref).

The layout owns its dof vector: it remains valid after the iterator that produced the
entity advances. `LocalDofLayout()` constructs an empty layout to be filled with
[`local_dofs!`](@ref), the typical pattern for hoisting the layout out of an assembly
loop.
"""
mutable struct LocalDofLayout <: AbstractVector{Int}
    const dofs::Vector{Int}
    # `names`/`ranges` are shared with the descriptor's layout metadata and must not be
    # mutated; `local_dofs!` swaps them when the layout is reused for an entity with a
    # different augmented layout.
    names::Vector{Symbol}
    ranges::Vector{UnitRange{Int}}
end

LocalDofLayout() = LocalDofLayout(Int[], Symbol[], UnitRange{Int}[])

Base.size(l::LocalDofLayout) = size(l.dofs)
Base.IndexStyle(::Type{LocalDofLayout}) = IndexLinear()
Base.@propagate_inbounds Base.getindex(l::LocalDofLayout, i::Int) = l.dofs[i]

"""
    dof_range(layout::LocalDofLayout, name::Symbol)

Return the local dof range of the field or algebraic variable `name` in the augmented
local system described by `layout`. In contrast to `dof_range(dh, name)` this includes
the algebraic variables appended by the coupling descriptor.
"""
function dof_range(l::LocalDofLayout, name::Symbol)
    i = findfirst(==(name), l.names)
    if i === nothing
        error("no field or algebraic variable named :$name in this layout (available: $(l.names))")
    end
    return l.ranges[i]
end

# Resolve the owning DofHandler of a cache created from either a DofHandler or a
# SubDofHandler.
function _owning_dof_handler(cc::CellCache)
    dh = cc.dh
    dh === nothing && error("`local_dofs` requires a cache created from a DofHandler, not from a Grid")
    return dh isa SubDofHandler ? dh.dh : dh
end

function _update_local_dofs!(layout::LocalDofLayout, dh::DofHandler, cell_dofs::AbstractVector{Int}, info::LocalLayoutInfo)
    if length(cell_dofs) != info.n_cell_dofs
        error("expected $(info.n_cell_dofs) cell dofs but the cache holds $(length(cell_dofs)) (was the cache created with `UpdateFlags(dofs = true)`?)")
    end
    # The layout owns its dof vector: copy the cell dofs instead of viewing the iterator's
    # scratch storage, which is overwritten when the iterator advances.
    dofs = layout.dofs
    resize!(dofs, info.n_total)
    copyto!(dofs, 1, cell_dofs, 1, info.n_cell_dofs)
    k = info.n_cell_dofs
    for vidx in info.algebraic_indices
        for d in dh.algebraic_dofs[vidx]
            k += 1
            dofs[k] = d
        end
    end
    @assert k == info.n_total
    layout.names = info.names
    layout.ranges = info.ranges
    return layout
end

function _build_local_dofs(dh::DofHandler, cell_dofs::AbstractVector{Int}, info::LocalLayoutInfo)
    return _update_local_dofs!(LocalDofLayout(), dh, cell_dofs, info)
end

"""
    local_dofs(cell::CellCache, coupling::CellCoupling)
    local_dofs(facet::FacetCache, coupling::FacetCoupling)
    local_dofs(coupling::AlgebraicCoupling)

Return the [`LocalDofLayout`](@ref) for the augmented local system of the coupling
descriptor on the given entity: the ordinary dofs of the (adjacent) cell, in `celldofs`
order, followed by the active dofs of each algebraic variable named by the descriptor, in
descriptor field order and then active-component order. For an
[`AlgebraicCoupling`](@ref) the layout contains only the selected algebraic variables.

The entity must belong to the descriptor's entity set and to the same `DofHandler`. Note
that facet layouts contain *all* dofs of the adjacent cell, see [`FacetCoupling`](@ref).
Each call allocates a fresh layout; [`local_dofs!`](@ref) is the non-allocating in-place
variant for hot assembly loops.

# Examples
```julia
# `boundary` is the facet set the descriptor was constructed with
for facet in FacetIterator(dh, boundary)
    layout = local_dofs(facet, descriptor)
    range_u = dof_range(layout, :u)
    range_p0 = dof_range(layout, :p0)
    # ... compute the local contribution Ke, fe ...
    assemble!(assembler, layout, Ke, fe)
end
```
"""
function _entity_layout_info(cc::CellCache, c::CellCoupling)
    if _owning_dof_handler(cc) !== c.dh
        error("the cell cache and the coupling descriptor belong to different DofHandlers")
    end
    cid = cellid(cc)
    if cid ∉ c.cells
        error("cell $cid is not in the cell set of this CellCoupling")
    end
    return c.layout_infos[c.dh.cell_to_subdofhandler[cid]]::LocalLayoutInfo
end

function _entity_layout_info(fc::FacetCache, c::FacetCoupling)
    if _owning_dof_handler(fc.cc) !== c.dh
        error("the facet cache and the coupling descriptor belong to different DofHandlers")
    end
    facet = FacetIndex(cellid(fc), fc.current_facet_id)
    if facet ∉ c.facets
        error("facet $(facet.idx) is not in the facet set of this FacetCoupling")
    end
    return c.layout_infos[c.dh.cell_to_subdofhandler[cellid(fc)]]::LocalLayoutInfo
end

function local_dofs(cc::CellCache, c::CellCoupling)
    return _build_local_dofs(c.dh, celldofs(cc), _entity_layout_info(cc, c))
end

function local_dofs(fc::FacetCache, c::FacetCoupling)
    return _build_local_dofs(c.dh, celldofs(fc), _entity_layout_info(fc, c))
end

function local_dofs(c::AlgebraicCoupling)
    return _build_local_dofs(c.dh, Int[], c.layout_info)
end

"""
    local_dofs!(layout::LocalDofLayout, cell::CellCache, coupling::CellCoupling)
    local_dofs!(layout::LocalDofLayout, facet::FacetCache, coupling::FacetCoupling)

Update `layout` in place with the augmented local dofs of the coupling descriptor on the
given entity and return it: the non-allocating counterpart of [`local_dofs`](@ref) for
assembly loops. The layout is hoisted out of the loop (like the local matrix and vector
buffers, each task needs its own in threaded assembly) and resized as needed:

```julia
layout = LocalDofLayout()
for facet in FacetIterator(dh, boundary)
    local_dofs!(layout, facet, descriptor)
    # ... compute the local contribution Ke, fe ...
    assemble!(assembler, layout, Ke, fe)
end
```
"""
function local_dofs!(layout::LocalDofLayout, cc::CellCache, c::CellCoupling)
    return _update_local_dofs!(layout, c.dh, celldofs(cc), _entity_layout_info(cc, c))
end

function local_dofs!(layout::LocalDofLayout, fc::FacetCache, c::FacetCoupling)
    return _update_local_dofs!(layout, c.dh, celldofs(fc), _entity_layout_info(fc, c))
end

##################################
# Sparsity pattern contributions #
##################################

"""
    add_coupling_entries!(
        sp::AbstractSparsityPattern, coupling::Ferrite.AbstractCoupling,
        ch::Union{ConstraintHandler, Nothing} = nothing;
        keep_constrained::Bool = true,
    )

Add the entries declared by the coupling descriptor (see [`CellCoupling`](@ref),
[`FacetCoupling`](@ref), and [`AlgebraicCoupling`](@ref)) to the sparsity pattern `sp`:
for every entity in the descriptor's set, entries for every (test dof, trial dof) pair of
the augmented local layout allowed by the coupling matrix.

This is the low-level mutation entry point; prefer passing descriptors to the
`algebraic_couplings` keyword of [`allocate_matrix`](@ref) or [`add_sparsity_entries!`](@ref),
which guarantees that coupling entries are added *before* the constraint entries (see
[`add_constraint_entries!`](@ref)).

# Keyword arguments
 - `keep_constrained`: whether or not entries for constrained DoFs should be kept
   (`keep_constrained = true`) or eliminated (`keep_constrained = false`) from the
   sparsity pattern. `keep_constrained = false` requires passing the ConstraintHandler
   `ch`. Eliminated entries touching an affine-constrained dof are distributed to the
   constraint's master dofs -- the entries that condensed assembly
   ([`apply_assemble!`](@ref)) writes to.
"""
function add_coupling_entries!(
        sp::AbstractSparsityPattern, coupling::AbstractCoupling,
        ch::Union{ConstraintHandler, Nothing} = nothing;
        keep_constrained::Bool = true,
    )
    dh = coupling.dh
    if getnrows(sp) < ndofs(dh) || getncols(sp) < ndofs(dh)
        error("number of rows ($(getnrows(sp))) or columns ($(getncols(sp))) in the sparsity pattern is smaller than number of dofs ($(ndofs(dh)))")
    end
    if !keep_constrained
        ch === nothing && error("must pass ConstraintHandler when `keep_constrained = false`")
        isclosed(ch) || error("the ConstraintHandler must be closed")
        ch.dh === dh || error("the DofHandler and the ConstraintHandler's DofHandler must be the same")
    end
    return _add_coupling_entries!(sp, coupling, ch, keep_constrained)
end

function _add_coupling_entries!(sp::AbstractSparsityPattern, c::CellCoupling, ch::Union{ConstraintHandler, Nothing}, keep_constrained::Bool)
    return _add_cell_coupling_entries!(sp, c.dh, c.cells, c.layout_infos, ch, keep_constrained)
end

function _add_coupling_entries!(sp::AbstractSparsityPattern, c::FacetCoupling, ch::Union{ConstraintHandler, Nothing}, keep_constrained::Bool)
    # A facet term uses all dofs of the adjacent cell, so the entries added for a facet
    # equal the entries added for its adjacent cell and it suffices to visit each adjacent
    # cell once.
    return _add_cell_coupling_entries!(sp, c.dh, c.adjacent_cells, c.layout_infos, ch, keep_constrained)
end

function _add_coupling_entries!(sp::AbstractSparsityPattern, c::AlgebraicCoupling, ch::Union{ConstraintHandler, Nothing}, keep_constrained::Bool)
    layout = local_dofs(c)
    _add_masked_entries!(sp, layout, c.layout_info.dof_mask, ch, keep_constrained)
    return sp
end

function _add_cell_coupling_entries!(
        sp::AbstractSparsityPattern, dh::DofHandler, cells::OrderedSet{Int},
        layout_infos::Vector{Union{Nothing, LocalLayoutInfo}},
        ch::Union{ConstraintHandler, Nothing}, keep_constrained::Bool,
    )
    cc = CellCache(dh, UpdateFlags(nodes = false, coords = false, dofs = true))
    for cellid in cells
        reinit!(cc, cellid)
        info = layout_infos[dh.cell_to_subdofhandler[cellid]]::LocalLayoutInfo
        layout = _build_local_dofs(dh, celldofs(cc), info)
        _add_masked_entries!(sp, layout, info.dof_mask, ch, keep_constrained)
    end
    return sp
end

function _add_masked_entries!(
        sp::AbstractSparsityPattern, dofs::AbstractVector{Int}, mask::Matrix{Bool},
        ch::Union{ConstraintHandler, Nothing}, keep_constrained::Bool,
    )
    if keep_constrained
        for (i, row) in pairs(dofs), (j, col) in pairs(dofs)
            mask[i, j] || continue
            add_entry!(sp, row, col)
        end
        return
    end
    # With `keep_constrained = false`, eliminated entries are never stored in the pattern,
    # so the later constraint expansion (`add_constraint_entries!`), which only sees
    # stored entries, cannot discover them. Entries touching an affine-constrained dof
    # are therefore distributed to the constraint's master dofs already at insertion --
    # the entries that condensed assembly (`apply_assemble!`) writes to -- and entries
    # touching other prescribed dofs are dropped.
    for (i, row) in pairs(dofs)
        rows′ = _eliminated_entry_targets(ch, row)
        isempty(rows′) && continue
        for (j, col) in pairs(dofs)
            mask[i, j] || continue
            for col′ in _eliminated_entry_targets(ch, col), row′ in rows′
                add_entry!(sp, row′, col′)
            end
        end
    end
    return
end

# Dofs that an entry involving `dof` maps to when constrained dofs are eliminated from
# the pattern: the dof itself when unconstrained, the (unconstrained) master dofs of an
# affine constraint, and no dofs at all for other prescribed dofs.
function _eliminated_entry_targets(ch::ConstraintHandler, dof::Int)
    coeffs = coefficients_for_dof(ch.dofmapping, ch.dofcoefficients, dof)
    if coeffs !== nothing
        return Int[m for (m, _) in coeffs if !haskey(ch.dofmapping, m)]
    elseif haskey(ch.dofmapping, dof)
        return ()
    else
        return (dof,)
    end
end
