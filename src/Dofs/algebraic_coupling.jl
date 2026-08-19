# Coupling descriptors for algebraic variables.
#
# Descriptors are purely declarative: an entity set plus the coupled variable pairs. They
# are independent of any DofHandler; validation against a DofHandler and the local layout
# computation happen when the descriptor is used (`add_coupling_entries!`).

########################
# Local layout metadata #
########################

# Sparsity metadata for the augmented local layout (cell dofs followed by the
# descriptor's algebraic dofs) of one descriptor on one SubDofHandler, computed when the
# descriptor is applied to a sparsity pattern. The coupled (test-range, trial-range) block
# pairs are split into blocks that involve cell dofs and blocks between algebraic dofs
# only (the latter reference the same global dofs for every entity and only need to be
# added to a sparsity pattern once).
struct LocalLayoutInfo
    n_cell_dofs::Int               # number of ordinary cell dofs (0 for AlgebraicCoupling)
    n_total::Int                   # total number of local dofs
    algebraic_indices::Vector{Int} # indices into the DofHandler's algebraic registries, in descriptor field order
    cell_blocks::Vector{NTuple{2, UnitRange{Int}}}      # coupled blocks involving cell dofs
    algebraic_blocks::Vector{NTuple{2, UnitRange{Int}}} # coupled blocks between algebraic dofs
end

# Collect the coupled block pairs declared by the field coupling matrix.
function _coupled_blocks(
        fields::Vector{Symbol}, coupling::Matrix{Bool},
        names::Vector{Symbol}, ranges::Vector{UnitRange{Int}}, n_cell_dofs::Int,
    )
    field_ranges = map(fields) do fname
        i = findfirst(==(fname), names)
        @assert i !== nothing # validated before the layout is computed
        return ranges[i]
    end
    cell_blocks = NTuple{2, UnitRange{Int}}[]
    algebraic_blocks = NTuple{2, UnitRange{Int}}[]
    for (j, rj) in pairs(field_ranges), (i, ri) in pairs(field_ranges)
        coupling[i, j] || continue
        # Ranges after the cell dofs belong to algebraic variables
        onlyalgebraic = first(ri) > n_cell_dofs && first(rj) > n_cell_dofs
        push!(onlyalgebraic ? algebraic_blocks : cell_blocks, (ri, rj))
    end
    return cell_blocks, algebraic_blocks
end

# Build the layout metadata for one SubDofHandler (`sdh === nothing` for the algebraic-only
# layout of an AlgebraicCoupling): all cell dofs in celldofs order followed by the
# dofs of the descriptor's algebraic variables.
function _local_layout_info(
        dh::DofHandler, sdh::Union{SubDofHandler, Nothing},
        fields::Vector{Symbol}, coupling::Matrix{Bool},
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
    cell_blocks, algebraic_blocks = _coupled_blocks(fields, coupling, names, ranges, n_cell_dofs)
    return LocalLayoutInfo(n_cell_dofs, n_total, algebraic_indices, cell_blocks, algebraic_blocks)
end

##################################
# Coupling specification parsing #
##################################

# Normalize pair/tuple entries to a field list and coupling matrix. The specification is
# `@nospecialize`d: user code is free to pass any tuple/vector shape without triggering a
# fresh compilation of the parsing, and the descriptors store the normalized
# (shape-independent) form. Only the shape is validated here; the names are checked
# against a DofHandler when the descriptor is used.
function _process_algebraic_coupling(@nospecialize(spec))
    # A bare entry is accepted as a one-entry specification
    if spec isa Pair || spec isa Tuple{Symbol, Symbol}
        spec = (spec,)
    end
    if !(spec isa Union{Tuple, AbstractVector})
        error("cannot interpret `algebraic_coupling = $(repr(spec))`: expected a collection of `:a => :b` pairs and/or `(:a, :b)` tuples")
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
    coupling_mat = zeros(Bool, length(names), length(names))
    for (a, b) in directed
        coupling_mat[findfirst(==(a), names), findfirst(==(b), names)] = true
    end
    return names, coupling_mat
end

##########################
# Coupling descriptors   #
##########################

"""
    CellCoupling(cells; algebraic_coupling)

Structural descriptor for weak-form terms integrated over the cells in `cells`, coupling
algebraic variables to spatial fields or other algebraic variables.

`algebraic_coupling` accepts entries involving at least one algebraic variable:
 - `:u => :σ̄` declares a directional coupling: test dofs of `:u` may couple to trial dofs
   of `:σ̄`;
 - `(:u, :σ̄)` declares both directions at once.

A single entry may be passed bare, without the surrounding collection.

The descriptor is independent of any `DofHandler`: it only declares the coupling. The
names are resolved when the descriptor is used, e.g. passed to the `algebraic_couplings`
keyword argument of [`allocate_matrix`](@ref).

See also [`FacetCoupling`](@ref) and [`AlgebraicCoupling`](@ref).
"""
struct CellCoupling <: AbstractCoupling
    cells::OrderedSet{Int}
    fields::Vector{Symbol}
    coupling::Matrix{Bool}
end

function CellCoupling(cells::IntegerCollection; algebraic_coupling)
    fields, coupling_mat = _process_algebraic_coupling(algebraic_coupling)
    # Own a copy so later mutation of `cells` cannot affect the descriptor
    return CellCoupling(OrderedSet{Int}(cells), fields, coupling_mat)
end

"""
    FacetCoupling(facets; algebraic_coupling)

Structural descriptor for weak-form terms integrated over the facets in `facets`
(a set of `FacetIndex`). See [`CellCoupling`](@ref) for the keywords.

The local layout contains all dofs of each adjacent cell, since facet terms may use cell
gradients.
"""
struct FacetCoupling <: AbstractCoupling
    facets::OrderedSet{FacetIndex}
    fields::Vector{Symbol}
    coupling::Matrix{Bool}
end

function FacetCoupling(facets::AbstractVecOrSet{FacetIndex}; algebraic_coupling)
    fields, coupling_mat = _process_algebraic_coupling(algebraic_coupling)
    # The descriptor owns its entity set, see the CellCoupling constructor
    return FacetCoupling(OrderedSet{FacetIndex}(facets), fields, coupling_mat)
end

"""
    AlgebraicCoupling(; algebraic_coupling)

Descriptor for terms involving only algebraic variables. See [`CellCoupling`](@ref) for
the keywords. Since diagonal matrix entries are always allocated, this is only needed for
off-diagonal entries.
"""
struct AlgebraicCoupling <: AbstractCoupling
    fields::Vector{Symbol}
    coupling::Matrix{Bool}
end

function AlgebraicCoupling(; algebraic_coupling)
    fields, coupling_mat = _process_algebraic_coupling(algebraic_coupling)
    return AlgebraicCoupling(fields, coupling_mat)
end

##############################
# Descriptor validation      #
##############################

# Validate the descriptor's variable names against `dh`. Called when the descriptor is
# used, since the descriptor itself is DofHandler-independent.
function _validate_coupling_fields(dh::DofHandler, c::AbstractCoupling)
    isclosed(dh) || error("coupling descriptors require a closed DofHandler")
    for name in c.fields
        if _find_algebraic_variable(dh, name) === nothing && name ∉ getfieldnames(dh)
            error("no spatial field or algebraic variable named :$name in the DofHandler (fields: $(getfieldnames(dh)), algebraic variables: $(dh.algebraic_names))")
        end
    end
    if c isa AlgebraicCoupling
        for name in c.fields
            if _find_algebraic_variable(dh, name) === nothing
                error(":$name is a spatial field, but an AlgebraicCoupling can only couple algebraic variables; use a CellCoupling or FacetCoupling for terms involving spatial fields")
            end
        end
    end
    for j in axes(c.coupling, 2), i in axes(c.coupling, 1)
        c.coupling[i, j] || continue
        a, b = c.fields[i], c.fields[j]
        if _find_algebraic_variable(dh, a) === nothing && _find_algebraic_variable(dh, b) === nothing
            error(
                "the `algebraic_coupling` entry (:$a, :$b) couples two spatial fields, but every entry must " *
                    "involve at least one algebraic variable (entries between spatial fields on the same cell " *
                    "are governed by the `coupling` keyword of `allocate_matrix`)"
            )
        end
    end
    return
end

# Verify that the descriptor's spatial fields exist on the SubDofHandler of `entity`.
function _check_spatial_fields_on_sdh(dh::DofHandler, sdh::SubDofHandler, fields::Vector{Symbol}, cellid::Int)
    for name in fields
        _find_algebraic_variable(dh, name) === nothing || continue
        if name ∉ sdh.field_names
            error("the field :$name does not exist on cell $cellid (fields on the corresponding SubDofHandler: $(sdh.field_names))")
        end
    end
    return
end

########################
# Descriptor interface #
########################

"""
    Ferrite.entities(coupling::Union{CellCoupling, FacetCoupling})

Return a copy of the cell or facet set of `coupling`.
"""
entities(c::CellCoupling) = copy(c.cells)
entities(c::FacetCoupling) = copy(c.facets)

"""
    Ferrite.fields(coupling::Ferrite.AbstractCoupling)

Return the variable names participating in `coupling`.
"""
fields(c::AbstractCoupling) = Tuple(c.fields)

function _show_coupling(io::IO, c::AbstractCoupling, name::String, entity_str::Union{String, Nothing})
    println(io, name, ":")
    println(io, "  Fields: ", join(map(repr, c.fields), ", "))
    entity_str === nothing || println(io, "  Entities: ", entity_str)
    print(io, "  Coupling matrix: ", size(c.coupling, 1), "×", size(c.coupling, 2))
    return
end

function Base.show(io::IO, ::MIME"text/plain", c::CellCoupling)
    return _show_coupling(io, c, "CellCoupling", string(length(c.cells), " cells"))
end
function Base.show(io::IO, ::MIME"text/plain", c::FacetCoupling)
    return _show_coupling(io, c, "FacetCoupling", string(length(c.facets), " facets"))
end
function Base.show(io::IO, ::MIME"text/plain", c::AlgebraicCoupling)
    return _show_coupling(io, c, "AlgebraicCoupling", nothing)
end

# Normalize the `algebraic_couplings` keyword (a single descriptor, a named tuple, or any
# other iterable) to an iterable of descriptors; the caller validates the elements.
_iterate_algebraic_couplings(c::AbstractCoupling) = (c,)
_iterate_algebraic_couplings(cs::NamedTuple) = values(cs)
function _iterate_algebraic_couplings(cs)
    if !applicable(Base.iterate, cs)
        error("cannot interpret `algebraic_couplings = $(repr(cs))`: expected coupling descriptors (`CellCoupling`, `FacetCoupling`, `AlgebraicCoupling`) or an iterable of them")
    end
    return cs
end

# Fill `dofs` with the cell dofs followed by the dofs of the descriptor's
# algebraic variables: the augmented local dof vector whose entries the sparsity blocks
# below index into.
function _augmented_dofs!(dofs::Vector{Int}, dh::DofHandler, cell_dofs::AbstractVector{Int}, info::LocalLayoutInfo)
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
    return dofs
end

##################################
# Sparsity pattern contributions #
##################################

"""
    add_coupling_entries!(
        sp::AbstractSparsityPattern, dh::DofHandler, coupling::Ferrite.AbstractCoupling,
        ch::Union{ConstraintHandler, Nothing} = nothing;
        keep_constrained::Bool = true,
    )

Add the entries declared by `coupling` to `sp`. Usually descriptors are passed through the
`algebraic_couplings` keyword of [`allocate_matrix`](@ref) or
[`add_sparsity_entries!`](@ref) instead.

# Keyword arguments
 - `keep_constrained`: whether or not entries for constrained DoFs should be kept
   (`keep_constrained = true`) or eliminated (`keep_constrained = false`) from the
   sparsity pattern. `keep_constrained = false` requires passing the ConstraintHandler
   `ch`.
"""
function add_coupling_entries!(
        sp::AbstractSparsityPattern, dh::DofHandler, coupling::AbstractCoupling,
        ch::Union{ConstraintHandler, Nothing} = nothing;
        keep_constrained::Bool = true,
    )
    if getnrows(sp) < ndofs(dh) || getncols(sp) < ndofs(dh)
        error("number of rows ($(getnrows(sp))) or columns ($(getncols(sp))) in the sparsity pattern is smaller than number of dofs ($(ndofs(dh)))")
    end
    if !keep_constrained
        ch === nothing && error("must pass ConstraintHandler when `keep_constrained = false`")
        isclosed(ch) || error("the ConstraintHandler must be closed")
        ch.dh === dh || error("the DofHandler and the ConstraintHandler's DofHandler must be the same")
    end
    return _add_coupling_entries!(sp, dh, coupling, ch, keep_constrained)
end

function _add_coupling_entries!(sp::AbstractSparsityPattern, dh::DofHandler, c::AbstractCoupling, ch::Union{ConstraintHandler, Nothing}, keep_constrained::Bool)
    _validate_coupling_fields(dh, c)
    isconstrained = keep_constrained ? nothing : _isconstrained_by_dof(ch, getnrows(sp))
    if c isa AlgebraicCoupling
        info = _local_layout_info(dh, nothing, c.fields, c.coupling)
        dofs = _augmented_dofs!(Int[], dh, Int[], info)
        _add_block_entries!(sp, dofs, info.algebraic_blocks, isconstrained)
        return sp
    end
    # A facet term uses all dofs of the adjacent cell, so it suffices to visit each
    # adjacent cell once.
    cells = c isa FacetCoupling ? _adjacent_cells(dh, c.facets) : c.cells
    return _add_cell_coupling_entries!(sp, dh, cells, c.fields, c.coupling, isconstrained)
end

# The unique cells adjacent to `facets`, with bounds checking.
function _adjacent_cells(dh::DofHandler, facets::OrderedSet{FacetIndex})
    grid = get_grid(dh)
    ncells = getncells(grid)
    adjacent_cells = OrderedSet{Int}()
    for (cellid, facetid) in facets
        1 <= cellid <= ncells || error("facet ($cellid, $facetid): cell index $cellid is out of bounds (the grid has $ncells cells)")
        cell = getcells(grid, cellid)
        1 <= facetid <= nfacets(cell) || error("facet ($cellid, $facetid): facet index $facetid is out of bounds for a cell with $(nfacets(cell)) facets")
        push!(adjacent_cells, cellid)
    end
    return adjacent_cells
end

function _add_cell_coupling_entries!(
        sp::AbstractSparsityPattern, dh::DofHandler, cells::OrderedSet{Int},
        fields::Vector{Symbol}, coupling::Matrix{Bool},
        isconstrained::Union{Nothing, Vector{Bool}},
    )
    isempty(cells) && return sp
    ncells = getncells(get_grid(dh))
    # CellCache construction requires a SubDofHandler; fail like the per-cell check would
    if isempty(dh.subdofhandlers)
        cellid = first(cells)
        1 <= cellid <= ncells || error("cell index $cellid is out of bounds (the grid has $ncells cells)")
        error("cell $cellid does not belong to any SubDofHandler")
    end
    # Layout metadata is computed for each SubDofHandler on first encounter
    layout_infos = Vector{Union{Nothing, LocalLayoutInfo}}(nothing, length(dh.subdofhandlers))
    cc = CellCache(dh, UpdateFlags(nodes = false, coords = false, dofs = true))
    dofs = Int[]
    # Rows of algebraic test dofs receive columns from every visited cell and can grow
    # dense; their columns are collected across the cell loop and inserted in bulk (sorted
    # and deduplicated), since eager sorted insertion of scattered columns is quadratic in
    # the row length.
    collected_cols = Dict{Int, Vector{Int}}()
    firstcell = true
    for cellid in cells
        1 <= cellid <= ncells || error("cell index $cellid is out of bounds (the grid has $ncells cells)")
        sdhidx = dh.cell_to_subdofhandler[cellid]
        sdhidx == 0 && error("cell $cellid does not belong to any SubDofHandler")
        info = layout_infos[sdhidx]
        if info === nothing
            sdh = dh.subdofhandlers[sdhidx]
            _check_spatial_fields_on_sdh(dh, sdh, fields, cellid)
            info = layout_infos[sdhidx] = _local_layout_info(dh, sdh, fields, coupling)
        end
        reinit!(cc, cellid)
        _augmented_dofs!(dofs, dh, celldofs(cc), info)
        for (ri, rj) in info.cell_blocks
            if first(ri) > info.n_cell_dofs
                for i in ri
                    cols = get!(Vector{Int}, collected_cols, dofs[i])
                    for j in rj
                        push!(cols, dofs[j])
                    end
                end
            else
                _add_block_entries!(sp, dofs, (ri, rj), isconstrained)
            end
        end
        if firstcell
            # The algebraic-only blocks reference the same global dofs for every cell, so
            # adding them for one cell suffices
            _add_block_entries!(sp, dofs, info.algebraic_blocks, isconstrained)
            firstcell = false
        end
    end
    for (row, cols) in collected_cols
        isconstrained !== nothing && isconstrained[row] && continue
        sort!(cols)
        unique!(cols)
        for col in cols
            isconstrained !== nothing && isconstrained[col] && continue
            add_entry!(sp, row, col)
        end
    end
    return sp
end

# Add the entries of the given (test-range, trial-range) blocks of `dofs` to `sp`.
function _add_block_entries!(
        sp::AbstractSparsityPattern, dofs::AbstractVector{Int},
        blocks::Vector{NTuple{2, UnitRange{Int}}},
        isconstrained::Union{Nothing, Vector{Bool}},
    )
    for block in blocks
        _add_block_entries!(sp, dofs, block, isconstrained)
    end
    return
end

function _add_block_entries!(
        sp::AbstractSparsityPattern, dofs::AbstractVector{Int},
        (ri, rj)::NTuple{2, UnitRange{Int}},
        isconstrained::Union{Nothing, Vector{Bool}},
    )
    for i in ri
        row = dofs[i]
        isconstrained !== nothing && isconstrained[row] && continue
        for j in rj
            col = dofs[j]
            isconstrained !== nothing && isconstrained[col] && continue
            add_entry!(sp, row, col)
        end
    end
    return
end
