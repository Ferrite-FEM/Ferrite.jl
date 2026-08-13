###########################
# AbstractSparsityPattern #
###########################

"""
    Ferrite.AbstractSparsityPattern

Supertype for sparsity pattern implementations, e.g. [`SparsityPattern`](@ref) and
[`BlockSparsityPattern`](@ref).
"""
abstract type AbstractSparsityPattern end

"""
    getnrows(sp::AbstractSparsityPattern)

Return the number of rows in the sparsity pattern `sp`.
"""
getnrows(sp::AbstractSparsityPattern)

"""
    getncols(sp::AbstractSparsityPattern)

Return the number of columns in the sparsity pattern `sp`.
"""
getncols(sp::AbstractSparsityPattern)

"""
    add_entry!(sp::AbstractSparsityPattern, row::Int, col::Int)

Add an entry to the sparsity pattern `sp` at row `row` and column `col`.
"""
add_entry!(sp::AbstractSparsityPattern, row::Int, col::Int)

# This is necessary to avoid warning about not importing Base.eachrow when
# adding docstring before the definitions further down.
function eachrow end

"""
    eachrow(sp::AbstractSparsityPattern)

Return an iterator over the rows of the sparsity pattern `sp`.
Each element of the iterator iterates the indices of the stored *columns* for that row, in
sorted order.
"""
eachrow(sp::AbstractSparsityPattern)

"""
    eachrow(sp::AbstractSparsityPattern, row::Int)

Return an iterator over the stored *column* indices in row `row` of the sparsity pattern,
in sorted order.

Conceptually this is equivalent to [`eachrow(sp)[row]`](@ref
eachrow(::AbstractSparsityPattern)). However, the iterator `eachrow(sp)` isn't always
indexable. This method should be used when a specific row needs to be "random access"d.
"""
eachrow(sp::AbstractSparsityPattern, row::Int)


###################
# SparsityPattern #
###################

"""
    struct SparsityPattern <: AbstractSparsityPattern

Data structure representing non-zero entries in the eventual sparse matrix.

See the constructor [`SparsityPattern(::Int, ::Int)`](@ref) for the user-facing
documentation.

# Struct fields
 - `nrows::Int`: number of rows.
 - `ncols::Int`: number of columns.
 - `buffer::CollectionsOfViews.ConstructionBuffer{Int, 1}`: contiguous storage where each row's
   column indices live as a slice. Backing all rows with a single growable buffer (plus an
   isbits metadata array) keeps GC pressure low.
 - `sorted::Bool`: whether each row's column indices are currently sorted. Rows are filled unsorted
   by the fast path and sorted lazily on demand (see `eachrow`/`add_entry!`).

!!! warning "Internal struct"
    The specific implementation of this struct, such as struct fields, type layout and type
    parameters, are internal and should not be relied upon.
"""
mutable struct SparsityPattern <: AbstractSparsityPattern
    const nrows::Int
    const ncols::Int
    buffer::CollectionsOfViews.ConstructionBuffer{Int, 1}
    sorted::Bool
end

"""
    SparsityPattern(nrows::Int, ncols::Int; nnz_per_row::Int = 8)

Create an empty [`SparsityPattern`](@ref) with `nrows` rows and `ncols` columns.
`nnz_per_row` is used as a memory hint for the number of non zero entries per row: it is the
initial reservation for a row that receives its first entry through [`add_entry!`](@ref
Ferrite.add_entry!), and the minimum increment when a full row has to grow.

`SparsityPattern` is the default sparsity pattern type for the standard DofHandler and is
therefore commonly constructed using [`init_sparsity_pattern`](@ref) instead of with this
constructor.

# Examples
```julia
# Create a sparsity pattern for an 100 x 100 matrix, hinting at 10 entries per row
sparsity_pattern = SparsityPattern(100, 100; nnz_per_row = 10)
```

# Methods
The following methods apply to `SparsityPattern` (see their respective documentation for
more details):
 - [`add_sparsity_entries!`](@ref): convenience method for calling
   [`add_cell_entries!`](@ref), [`add_interface_entries!`](@ref), and
   [`add_constraint_entries!`](@ref).
 - [`add_cell_entries!`](@ref): add entries corresponding to DoF couplings within the cells.
 - [`add_interface_entries!`](@ref): add entries corresponding to DoF couplings on the
   interface between cells.
 - [`add_constraint_entries!`](@ref): add entries resulting from constraints.
 - [`allocate_matrix`](@ref allocate_matrix(::SparsityPattern)): instantiate a matrix from
   the pattern. The default matrix type is `SparseMatrixCSC{Float64, Int}`.
"""
function SparsityPattern(nrows::Int, ncols::Int; nnz_per_row::Int = 8)
    buffer = ConstructionBuffer(Int[], (nrows,), nnz_per_row)
    return SparsityPattern(nrows, ncols, buffer, true)
end

function Base.show(io::IO, ::MIME"text/plain", sp::SparsityPattern)
    iob = IOBuffer()
    println(iob, "$(getnrows(sp))×$(getncols(sp)) $(sprint(show, typeof(sp))):")
    # Collect min/max/avg entries per row
    min_entries = typemax(Int)
    max_entries = typemin(Int)
    stored_entries = 0
    for r in eachrow(sp)
        l = length(r)
        stored_entries += l
        min_entries = min(min_entries, l)
        max_entries = max(max_entries, l)
    end
    # Print sparsity
    sparsity_pct = round(
        (getnrows(sp) * getncols(sp) - stored_entries) / (getnrows(sp) * getncols(sp)) * 100 * 1000
    ) / 1000
    println(iob, " - Sparsity: $(sparsity_pct)% ($(stored_entries) stored entries)")
    # Print row stats
    avg_entries = round(stored_entries / getnrows(sp) * 10) / 10
    println(iob, " - Entries per row (min, max, avg): $(min_entries), $(max_entries), $(avg_entries)")
    # Compute memory estimate
    meta_bytes = sizeof(sp.buffer.indices)
    bytes_used = meta_bytes + stored_entries * sizeof(Int)
    bytes_allocated = meta_bytes + length(sp.buffer.data) * sizeof(Int)
    print(iob, " - Memory estimate: $(Base.format_bytes(bytes_used)) used, $(Base.format_bytes(bytes_allocated)) allocated")
    write(io, seekstart(iob))
    return
end

getnrows(sp::SparsityPattern) = sp.nrows
getncols(sp::SparsityPattern) = sp.ncols

## Storage helpers ##

# The sorted, de-duplicating insert (with geometric growth on overflow) is
# `CollectionsOfViews.insert_sorted_at_index!`. A row that overflows its reservation relocates to
# the end of `data`, leaving a transient hole.

# Lay out `data` so each row gets its own exact block [start, start+cap): one allocation, no holes.
function _presize_buffer(rowlen::Vector{Int}, sizehint::Int)
    n = length(rowlen)
    indices = Vector{AdaptiveRange}(undef, n)
    total = 0
    @inbounds for r in 1:n
        total += max(rowlen[r], 1)
    end
    data = Vector{Int}(undef, total)
    start = 1
    @inbounds for r in 1:n
        cap = max(rowlen[r], 1)
        indices[r] = AdaptiveRange(start, 0, cap)
        start += cap
    end
    return ConstructionBuffer{Int, 1}(indices, data, sizehint)
end

# Hot-path guard: kept tiny so that it inlines into callers; the actual sorting is out of
# line in _sort_pattern! and only runs when the pattern is unsorted.
@inline function _ensure_sorted!(sp::SparsityPattern)
    sp.sorted || _sort_pattern!(sp)
    return sp
end

# Sort each row's slice in place (rows are already deduplicated). Rows are disjoint slices of
# `data`, so they can be sorted in parallel by chunking the rows over tasks.
@noinline function _sort_pattern!(sp::SparsityPattern)
    nrows = getnrows(sp)
    # One chunk per thread (rows have similar cost so no load balancing is needed), but at
    # least 1000 rows per chunk so that small patterns don't spawn useless tasks.
    chunksize = max(1000, cld(nrows, Threads.nthreads()))
    @sync for rowrange in Iterators.partition(1:nrows, chunksize)
        Threads.@spawn _sort_rows!(sp, rowrange)
    end
    sp.sorted = true
    return sp
end

# QuickSort explicitly since the default sorting algorithm (radix) allocates.
function _sort_rows!(sp::SparsityPattern, rowrange::UnitRange{Int})
    b = sp.buffer
    data = b.data
    @inbounds for row in rowrange
        r = b.indices[row]
        r.ncurrent > 1 && sort!(data, r.start, r.start + r.ncurrent - 1, QuickSort, Base.Order.Forward)
    end
    return
end

# Fast path: count exact per-row sizes with a column marker, presize the buffer, then marker-dedup
# fill each row UNSORTED (sorted lazily). Always includes the diagonal. Non-full `coupling` and
# `keep_constrained = false` are handled by applying the expanded coupling masks and the
# constrained-dof filter identically in the count and the fill pass (see _visit_row_candidates!).
function _fast_fill_cells!(
        sp::SparsityPattern, dh::DofHandler,
        couplings::Union{Nothing, Vector{Matrix{Bool}}} = nothing,
        isconstrained::Union{Nothing, Vector{Bool}} = nothing,
        neighbor_cells::Union{Nothing, ArrayOfVectorViews} = nothing,
        interface_couplings::Union{Nothing, Vector{Matrix{Bool}}} = nothing;
        fill_interfaces::Bool = false,
    )
    nrows = getnrows(sp)
    #...
    cell_dofs = create_celldofs(dh)
    # The local-index map only exists to index the expanded coupling masks; skip building it
    # when no mask is in play (`_visit_row_candidates!` never reads it then).
    row_to_cells, row_to_localidx = create_row_to_cells(
        cell_dofs, nrows, couplings !== nothing || interface_couplings !== nothing
    )
    rowlen = zeros(Int, nrows)
    marker = zeros(Int, getncols(sp))
    cell_to_sdh = dh.cell_to_subdofhandler
    _visit_row_candidates!(
        rowlen, nothing, nothing, marker, row_to_cells, row_to_localidx, cell_dofs,
        cell_to_sdh, couplings, isconstrained, neighbor_cells, interface_couplings
    )
    buffer = _presize_buffer(rowlen, sp.buffer.sizehint)
    fill!(marker, 0)
    _visit_row_candidates!(
        rowlen, buffer.data, buffer.indices, marker, row_to_cells, row_to_localidx, cell_dofs,
        cell_to_sdh, couplings, isconstrained,
        fill_interfaces ? neighbor_cells : nothing, fill_interfaces ? interface_couplings : nothing
    )
    sp.buffer = buffer
    sp.sorted = false
    return sp
end

@inline function _row_view(sp::SparsityPattern, row::Int)
    r = @inbounds sp.buffer.indices[row]
    r.ncurrent == 0 && return view(sp.buffer.data, 1:0)
    return view(sp.buffer.data, r.start:(r.start + r.ncurrent - 1))
end

## AbstractSparsityPattern interface ##

@inline function add_entry!(sp::SparsityPattern, row::Int, col::Int)
    @boundscheck (1 <= row <= getnrows(sp) && 1 <= col <= getncols(sp)) || throw(BoundsError(sp, (row, col)))
    _ensure_sorted!(sp) # a sorted insert needs the row sorted
    insert_sorted_at_index!(sp.buffer, col, row)
    return
end

function eachrow(sp::SparsityPattern, row::Int)
    _ensure_sorted!(sp)
    return _row_view(sp, row)
end
function eachrow(sp::SparsityPattern)
    _ensure_sorted!(sp)
    return (_row_view(sp, row) for row in 1:getnrows(sp))
end

# Internal variant of `eachrow` with no ordering guarantee, for consumers that do not need
# sorted rows. The default is `eachrow`; `SparsityPattern` hands out its raw row slices
# without triggering the lazy sort (this is what keeps the common
# pattern-to-SparseMatrixCSC path sort-free).
_eachrow_anyorder(sp::AbstractSparsityPattern) = eachrow(sp)
_eachrow_anyorder(sp::SparsityPattern) = (_row_view(sp, row) for row in 1:getnrows(sp))


################################################
## Adding entries to AbstractSparsityPatterns ##
################################################

"""
    init_sparsity_pattern(dh::DofHandler; nnz_per_row::Int)

Initialize an empty [`SparsityPattern`](@ref) with `ndofs(dh)` rows and `ndofs(dh)` columns.

# Keyword arguments
 - `nnz_per_row`: memory optimization hint for the number of non-zero entries per row that
   will be added to the pattern.
"""
function init_sparsity_pattern(
        dh::DofHandler;
        # TODO: What is a good estimate for nnz_per_row?
        nnz_per_row::Int = 2 * ndofs_per_cell(dh.subdofhandlers[1]), # FIXME
    )
    sp = SparsityPattern(ndofs(dh), ndofs(dh); nnz_per_row = nnz_per_row)
    return sp
end

"""
    add_sparsity_entries!(
        sp::AbstractSparsityPattern,
        dh::DofHandler,
        ch::Union{ConstraintHandler, Nothing} = nothing;
        topology = nothing,
        keep_constrained::Bool = true,
        coupling = nothing,
        interface_coupling = nothing,
    )

Convenience method for doing the common task of calling [`add_cell_entries!`](@ref),
[`add_interface_entries!`](@ref), and [`add_constraint_entries!`](@ref), depending on what
arguments are passed:
 - `add_cell_entries!` is always called
 - `add_interface_entries!` is called if `interface_coupling` is provided (i.e. not
   `nothing`). Passing `topology` is optional, but recommended for performance reasons
   (see [`add_interface_entries!`](@ref)).
 - `add_constraint_entries!` is called if the ConstraintHandler is provided

For more details about arguments and keyword arguments, see the respective functions.
"""
function add_sparsity_entries!(
        sp::AbstractSparsityPattern, dh::DofHandler,
        ch::Union{ConstraintHandler, Nothing} = nothing;
        keep_constrained::Bool = true,
        coupling::Union{AbstractMatrix{Bool}, Nothing} = nothing,
        interface_coupling::Union{AbstractMatrix{Bool}, Nothing} = nothing,
        topology = nothing,
    )
    # Argument checking
    isclosed(dh) || error("the DofHandler must be closed")
    if getnrows(sp) < ndofs(dh) || getncols(sp) < ndofs(dh)
        error("number of rows ($(getnrows(sp))) or columns ($(getncols(sp))) in the sparsity pattern is smaller than number of dofs ($(ndofs(dh)))")
    end
    # Add all entries
    add_diagonal_entries!(sp)
    add_cell_entries!(sp, dh, ch; keep_constrained, coupling)
    if interface_coupling !== nothing
        add_interface_entries!(sp, dh, ch; topology, keep_constrained, interface_coupling)
    end
    if ch !== nothing
        add_constraint_entries!(sp, ch; keep_constrained)
    end
    return sp
end

# SparsityPattern takes a fast path whenever it is empty: cell entries (including the diagonal,
# and respecting `coupling` and `keep_constrained`) are exactly counted, the buffer is presized in
# one allocation, and rows are filled with a marker-dedup append. A non-empty pattern falls back
# to the generic per-entry path. Interface entries (interface_coupling) and constraints always
# layer on afterwards, so they are compatible with the fast path.
function add_sparsity_entries!(
        sp::SparsityPattern, dh::DofHandler, ch::Union{ConstraintHandler, Nothing} = nothing;
        keep_constrained::Bool = true,
        coupling::Union{AbstractMatrix{Bool}, Nothing} = nothing,
        interface_coupling::Union{AbstractMatrix{Bool}, Nothing} = nothing,
        topology = nothing,
    )
    isclosed(dh) || error("the DofHandler must be closed")
    if getnrows(sp) < ndofs(dh) || getncols(sp) < ndofs(dh)
        error("number of rows ($(getnrows(sp))) or columns ($(getncols(sp))) in the sparsity pattern is smaller than number of dofs ($(ndofs(dh)))")
    end
    interfaces_filled = false
    if isempty(sp.buffer.data)
        keep_constrained || _check_keep_constrained_args(dh, ch)
        couplings = _coupling_to_local_dof_coupling(dh, coupling)
        isconstrained = keep_constrained ? nothing : _isconstrained_by_dof(ch, getnrows(sp))
        # With interface entries coming (added below), reserve row space for them up front so
        # they land in place instead of relocating rows. When the reservation enumeration is
        # provably exact it doubles as the insertion itself (see _visit_row_candidates!).
        if interface_coupling === nothing
            neighbor_cells = interface_couplings = nothing
        else
            if topology === nothing
                topology = ExclusiveTopology(get_grid(dh))
            end
            neighbor_cells = create_cell_to_neighbors(dh.grid, topology)
            interface_couplings = _coupling_to_local_dof_coupling(dh, interface_coupling)
            interfaces_filled = _can_fill_interfaces_directly(dh)
        end
        _fast_fill_cells!(
            sp, dh, couplings, isconstrained, neighbor_cells, interface_couplings;
            fill_interfaces = interfaces_filled
        )
    else
        add_diagonal_entries!(sp)
        add_cell_entries!(sp, dh, ch; keep_constrained, coupling)
    end
    interface_coupling !== nothing && !interfaces_filled && add_interface_entries!(sp, dh, ch; topology, keep_constrained, interface_coupling)
    ch !== nothing && add_constraint_entries!(sp, ch; keep_constrained)
    return sp
end

"""
    add_cell_entries!(
        sp::AbstractSparsityPattern,
        dh::DofHandler,
        ch::Union{ConstraintHandler, Nothing} = nothing;
        keep_constrained::Bool = true,
        coupling::Union{AbstractMatrix{Bool}, Nothing}, = nothing
    )

Add entries to the sparsity pattern `sp` corresponding to DoF couplings within the cells as
described by the DofHandler `dh`.

# Keyword arguments
 - `keep_constrained`: whether or not entries for constrained DoFs should be kept
   (`keep_constrained = true`) or eliminated (`keep_constrained = false`) from the sparsity
   pattern. `keep_constrained = false` requires passing the ConstraintHandler `ch`.
 - `coupling`: the coupling between fields/components within each cell. By default
   (`coupling = nothing`) it is assumed that all DoFs in each cell couple with each other.
   `coupling[f1, f2] == true` means that the equation for field `f1` (i.e. the rows
   corresponding to the test functions of `f1`) contains the unknown field `f2` (i.e. the
   columns corresponding to the trial functions of `f2`).
"""
function add_cell_entries!(
        sp::AbstractSparsityPattern,
        dh::DofHandler, ch::Union{ConstraintHandler, Nothing} = nothing;
        keep_constrained::Bool = true, coupling::Union{AbstractMatrix{Bool}, Nothing} = nothing,
    )
    # Expand coupling from nfields x nfields to ndofs_per_cell x ndofs_per_cell
    # (nothing and full masks expand to nothing = no restriction).
    # TODO: Perhaps this can be done in the loop over SubDofHandlers instead.
    coupling = _coupling_to_local_dof_coupling(dh, coupling)
    keep_constrained || _check_keep_constrained_args(dh, ch)
    return _add_cell_entries!(sp, dh, ch, keep_constrained, coupling)
end

# Argument checking for methods with `keep_constrained = false`
function _check_keep_constrained_args(dh::DofHandler, ch::Union{ConstraintHandler, Nothing})
    ch === nothing && error("must pass ConstraintHandler when `keep_constrained = true`")
    isclosed(ch) || error("the ConstraintHandler must be closed")
    ch.dh === dh || error("the DofHandler and the ConstraintHandler's DofHandler must be the same")
    return
end

# For now interfaces can not be fast-filled with multiple SubDofHandlers (the expanded
# coupling masks are per-sdh and do not apply to a neighbor with a different dof layout);
# the enumeration still (over)counts cross-sdh neighbors so the reservation covers what
# add_interface_entries! inserts afterwards.
function _can_fill_interfaces_directly(dh::DofHandler)
    return length(dh.subdofhandlers) == 1
end

# Boolean mask for whether the row is constrained or not
function _isconstrained_by_dof(ch::ConstraintHandler, nrows::Int)
    isconstrained = zeros(Bool, nrows)
    for d in keys(ch.dofmapping)
        isconstrained[d] = true
    end
    return isconstrained
end

"""
    add_interface_entries!(
        sp::SparsityPattern, dh::DofHandler, ch::Union{ConstraintHandler, Nothing};
        topology::Union{ExclusiveTopology, Nothing} = nothing, keep_constrained::Bool = true,
        interface_coupling::AbstractMatrix{Bool},
    )

Add entries to the sparsity pattern `sp` corresponding to DoF couplings on the interface
between cells as described by the DofHandler `dh`.

# Keyword arguments
 - `topology`: the topology corresponding to the grid. If not passed it is constructed
   from the grid. Since the construction is relatively expensive it is recommended to
   pass an existing topology, in particular since one is typically needed for e.g.
   [`InterfaceIterator`](@ref) in the assembly loop anyway.
 - `keep_constrained`: whether or not entries for constrained DoFs should be kept
   (`keep_constrained = true`) or eliminated (`keep_constrained = false`) from the sparsity
   pattern. `keep_constrained = false` requires passing the ConstraintHandler `ch`.
 - `interface_coupling`: the coupling between fields/components in interface integrals.
   `interface_coupling[i, j] = true` means that, for every interface, entries exist for
   *every* pair of (test DoF of field/component `i`, trial DoF of field/component `j`)
   within the union of the DoFs of the two cells sharing the interface. As for cell
   `coupling`, rows correspond to test functions and columns to trial functions.
"""
function add_interface_entries!(
        sp::SparsityPattern, dh::DofHandler, ch::Union{ConstraintHandler, Nothing} = nothing;
        topology::Union{ExclusiveTopology, Nothing} = nothing, keep_constrained::Bool = true,
        interface_coupling::AbstractMatrix{Bool},
    )
    keep_constrained || _check_keep_constrained_args(dh, ch)
    if topology === nothing
        topology = ExclusiveTopology(get_grid(dh))
    end
    return _add_interface_entries!(sp, dh, ch, topology, keep_constrained, interface_coupling)
end

"""
    add_constraint_entries!(
        sp::AbstractSparsityPattern, ch::ConstraintHandler;
        keep_constrained::Bool = true,
    )

Add all entries resulting from constraints in the ConstraintHandler `ch` to the sparsity
pattern. Note that, since this operation depends on existing entries in the pattern, this
function must be called as the *last* step when creating the sparsity pattern.

# Keyword arguments
 - `keep_constrained`: whether or not entries for constrained DoFs should be kept
   (`keep_constrained = true`) or eliminated (`keep_constrained = false`) from the sparsity
   pattern.
"""
function add_constraint_entries!(
        sp::AbstractSparsityPattern, ch::ConstraintHandler;
        keep_constrained::Bool = true,
    )
    return _add_constraint_entries!(sp, ch.dofcoefficients, ch.dofmapping, keep_constrained)
end

function add_diagonal_entries!(sp::AbstractSparsityPattern)
    for d in 1:min(getnrows(sp), getncols(sp))
        add_entry!(sp, d, d)
    end
    return sp
end


############################################################
# Sparse matrix instantiation from AbstractSparsityPattern #
############################################################

"""
    allocate_matrix(::Type{SparseMatrixCSC{Tv, Ti}}, sp::SparsityPattern)

Allocate a sparse matrix of type `SparseMatrixCSC{Tv, Ti}` from the sparsity pattern `sp`.
"""
function allocate_matrix(::Type{S}, sp::AbstractSparsityPattern) where {Tv, Ti, S <: SparseMatrixCSC{Tv, Ti}}
    return _allocate_matrix(S, sp, #=sym=# false)
end

"""
    allocate_matrix(::Type{Symmetric{Tv, SparseMatrixCSC{Tv, Ti}}}, sp::SparsityPattern)

Instantiate a sparse matrix of type `Symmetric{Tv, SparseMatrixCSC{Tv, Ti}}`, i.e. a
`LinearAlgebra.Symmetric`-wrapped `SparseMatrixCSC`, from the sparsity pattern `sp`. The
resulting matrix will only store entries above, and including, the diagonal.
"""
function allocate_matrix(::Type{Symmetric{Tv, S}}, sp::AbstractSparsityPattern) where {Tv, Ti, S <: SparseMatrixCSC{Tv, Ti}}
    return Symmetric(_allocate_matrix(S, sp, #=sym=# true))
end

"""
    allocate_matrix(sp::SparsityPattern)

Allocate a sparse matrix of type `SparseMatrixCSC{Float64, Int}` from the sparsity pattern
`sp`.

This method is a shorthand for the equivalent
[`allocate_matrix(SparseMatrixCSC{Float64, Int}, sp)`]
(@ref allocate_matrix(::Type{S}, sp::Ferrite.AbstractSparsityPattern) where {Tv, Ti, S <: SparseMatrixCSC{Tv, Ti}}).
"""
allocate_matrix(sp::SparsityPattern) = allocate_matrix(SparseMatrixCSC{Float64, Int}, sp)

"""
    allocate_matrix(MatrixType, dh::DofHandler, args...; kwargs...)

Allocate a matrix of type `MatrixType` from the DofHandler `dh`.

This is a convenience method and is equivalent to:

```julia
sp = init_sparsity_pattern(dh)
add_sparsity_entries!(sp, dh, args...; kwargs...)
allocate_matrix(MatrixType, sp)
````

Refer to [`allocate_matrix`](@ref allocate_matrix(::Type{<:Any}, ::SparsityPattern)) for
supported matrix types, and to [`init_sparsity_pattern`](@ref) for details about supported
arguments `args` and keyword arguments `kwargs`.

!!! note
    If more than one sparse matrix is needed (e.g. a stiffness and a mass matrix) it is more
    efficient to explicitly create the sparsity pattern instead of using this method, i.e.
    use
    ```julia
    sp = init_sparsity_pattern(dh)
    add_sparsity_entries!(sp, dh)
    K = allocate_matrix(sp)
    M = allocate_matrix(sp)
    ```
    instead of
    ```julia
    K = allocate_matrix(dh)
    M = allocate_matrix(dh)
    ```
    Note that for some matrix types it is possible to `copy` the instantiated matrix (`M =
    copy(K)`) instead.
"""
function allocate_matrix(::Type{MatrixType}, dh::DofHandler, args...; kwargs...) where {MatrixType}
    sp = init_sparsity_pattern(dh)
    add_sparsity_entries!(sp, dh, args...; kwargs...)
    return allocate_matrix(MatrixType, sp)
end

"""
    allocate_matrix(dh::DofHandler, args...; kwargs...)

Allocate a matrix of type `SparseMatrixCSC{Float64, Int}` from the DofHandler `dh`.

This method is a shorthand for the equivalent [`allocate_matrix(SparseMatrixCSC{Float64, Int},
dh, args...; kwargs...)`](@ref allocate_matrix(::Type{MatrixType}, ::DofHandler, args...;
kwargs...) where {MatrixType}) -- refer to that method for details.
"""
function allocate_matrix(dh::DofHandler, args...; kwargs...)
    return allocate_matrix(SparseMatrixCSC{Float64, Int}, dh, args...; kwargs...)
end


##############################
# Sparsity pattern internals #
##############################

# Compute a coupling matrix of size (ndofs_per_cell × ndofs_per_cell) based on the input
# coupling which can be of size i) (nfields × nfields) specifying coupling between fields,
# ii) (ncomponents × ncomponents) specifying coupling between components, or iii)
# (ndofs_per_cell × ndofs_per_cell) specifying coupling between all local dofs, i.e. a
# "template" local matrix.
# Validate the dimensions of a user-provided coupling mask without expanding it (the same
# checks the expansion below performs). Used before the full-mask normalization in
# add_sparsity_entries!, which skips the expansion where validation normally happens.
function _check_coupling_dims(dh::DofHandler, coupling::AbstractMatrix{Bool})
    sz = size(coupling, 1)
    sz == size(coupling, 2) || error("coupling not square")
    sz == length(dh.field_names) && return # coupling by fields
    sz == sum(fieldname -> n_components(dh, fieldname), dh.field_names) && return # by components
    all(sdh -> sz == ndofs_per_cell(sdh), dh.subdofhandlers) && return # template local matrix
    error("could not create coupling")
end

_coupling_to_local_dof_coupling(::DofHandler, ::Nothing) = nothing
function _coupling_to_local_dof_coupling(dh::DofHandler, coupling::AbstractMatrix{Bool})
    _check_coupling_dims(dh, coupling)
    # A full mask does not restrict anything: normalize to `nothing`, which all consumers
    # treat as every-pair-passes, so that mask evaluation is skipped entirely.
    all(coupling) && return nothing
    # Return one matrix per (potential) sub-domain
    return Matrix{Bool}[
        _coupling_to_local_dof_coupling(dh, coupling, sdh, sdh) for sdh in dh.subdofhandlers
    ]
end

# Compute a rectangular coupling matrix of size
# (ndofs_per_cell(sdh_row) × ndofs_per_cell(sdh_col)) where the rows are local dof indices
# in `sdh_row`'s layout and the columns local dof indices in `sdh_col`'s layout. With
# `sdh_row === sdh_col` this is the local coupling matrix of a cell; with different
# subdofhandlers it is used for the coupling across an interface between the two.
function _coupling_to_local_dof_coupling(
        dh::DofHandler, coupling::AbstractMatrix{Bool},
        sdh_row::SubDofHandler, sdh_col::SubDofHandler,
    )
    sz = size(coupling, 1)
    sz == size(coupling, 2) || error("coupling not square")

    field_dims = map(fieldname -> n_components(dh, fieldname), dh.field_names)

    out = zeros(Bool, ndofs_per_cell(sdh_row), ndofs_per_cell(sdh_col))

    dof_ranges_row = [dof_range(sdh_row, f) for f in sdh_row.field_names]
    dof_ranges_col = [dof_range(sdh_col, f) for f in sdh_col.field_names]
    global_idxs_row = [findfirst(x -> x === f, dh.field_names) for f in sdh_row.field_names]
    global_idxs_col = [findfirst(x -> x === f, dh.field_names) for f in sdh_col.field_names]

    if sz == length(dh.field_names) # Coupling given by fields
        for (j, jrange) in pairs(dof_ranges_col), (i, irange) in pairs(dof_ranges_row)
            out[irange, jrange] .= coupling[global_idxs_row[i], global_idxs_col[j]]
        end
    elseif sz == sum(field_dims) # Coupling given by components
        component_offsets = pushfirst!(cumsum(field_dims), 0)
        for (jf, jrange) in pairs(dof_ranges_col), (j, J) in pairs(jrange)
            jc = mod1(j, field_dims[global_idxs_col[jf]]) + component_offsets[global_idxs_col[jf]]
            for (i_f, irange) in pairs(dof_ranges_row), (i, I) in pairs(irange)
                ic = mod1(i, field_dims[global_idxs_row[i_f]]) + component_offsets[global_idxs_row[i_f]]
                out[I, J] = coupling[ic, jc]
            end
        end
    elseif sz == ndofs_per_cell(sdh_row) == ndofs_per_cell(sdh_col)
        # Coupling given by template local matrix. Note that this assumes matching local
        # dof layouts in `sdh_row` and `sdh_col`.
        # TODO: coupling[fieldhandler_idx] if different template per subddomain
        out .= coupling
    else
        error("could not create coupling")
    end
    return out
end

function _add_cell_entries!(
        sp::AbstractSparsityPattern, dh::DofHandler, ch::Union{ConstraintHandler, Nothing},
        keep_constrained::Bool, coupling::Union{Vector{<:AbstractMatrix{Bool}}, Nothing},
    )
    # Add all connections between dofs for every cell while filtering based
    # on a) constraints, and b) field/dof coupling.
    cc = CellCache(dh)
    for (sdhi, sdh) in pairs(dh.subdofhandlers)
        set = BitSet(sdh.cellset)
        coupling === nothing || (coupling_sdh = coupling[sdhi])
        for cell_id in set
            reinit!(cc, cell_id)
            for (i, row) in pairs(cc.dofs)
                # a) check constraint for row
                !keep_constrained && haskey(ch.dofmapping, row) && continue
                # TODO: Extracting the row here and reinserting after the j-loop
                #       should give some nice speedup
                for (j, col) in pairs(cc.dofs)
                    # b) check coupling between (local) dofs i and j
                    coupling === nothing || coupling_sdh[i, j] || continue
                    # a) check constraint for col
                    !keep_constrained && haskey(ch.dofmapping, col) && continue
                    # Insert col as a non zero index for this row
                    add_entry!(sp, row, col)
                end
            end
        end
    end
    return sp
end

# Sorted, de-duplicating insert into a plain Vector (used for the constraint scratch below).
@inline function insert_sorted!(v::Vector{Int}, item::Int)
    k = searchsortedfirst(v, item)
    (k == length(v) + 1 || @inbounds(v[k]) != item) && insert!(v, k, item)
    return v
end

function _add_constraint_entries!(
        sp::AbstractSparsityPattern, dofcoefficients::Vector{Union{DofCoefficients{T, Ti}, Nothing}},
        dofmapping::Dict{Int, Int}, keep_constrained::Bool,
    ) where {T, Ti}

    # Return early if there are no non-trivial affine constraints
    any(i -> !(i === nothing || isempty(i)), dofcoefficients) || return

    # New entries tracked separately and inserted after since it is not possible to modify
    # the datastructure while looping over it.
    sp′ = Dict{Int, Vector{Int}}()

    for (row, colidxs) in zip(1:getnrows(sp), eachrow(sp)) # pairs(eachrow(sp))
        row_coeffs = coefficients_for_dof(dofmapping, dofcoefficients, row)
        if row_coeffs === nothing
            # This row is _not_ constrained, check columns of this row...
            !keep_constrained && haskey(dofmapping, row) && continue
            for col in colidxs
                col_coeffs = coefficients_for_dof(dofmapping, dofcoefficients, col)
                if col_coeffs === nothing
                    # ... this column is _not_ constrained, done.
                    continue
                else
                    # ... this column _is_ constrained, distribute to columns.
                    for (col′, _) in col_coeffs
                        insert_sorted!(get!(() -> Int[], sp′, row), col′)
                    end
                end
            end
        else
            # This row _is_ constrained, check columns of this row...
            for col in colidxs
                col_coeffs = coefficients_for_dof(dofmapping, dofcoefficients, col)
                if col_coeffs === nothing
                    # ... this column is _not_ constrained, distribute to rows.
                    !keep_constrained && haskey(dofmapping, col) && continue
                    for (row′, _) in row_coeffs
                        insert_sorted!(get!(() -> Int[], sp′, row′), col)
                    end
                else
                    # ... this column _is_ constrained, double-distribute to columns/rows.
                    for (row′, _) in row_coeffs
                        !keep_constrained && haskey(dofmapping, row′) && continue
                        for (col′, _) in col_coeffs
                            !keep_constrained && haskey(dofmapping, col′) && continue
                            insert_sorted!(get!(() -> Int[], sp′, row′), col′)
                        end
                    end
                end
            end
        end
    end

    # Insert new entries into the sparsity pattern
    for (row, colidxs) in sp′
        # TODO: Extract row here and just insert_sorted
        for col in colidxs
            add_entry!(sp, row, col)
        end
    end

    return sp
end

function _add_interface_entry(
        sp::SparsityPattern,
        cell_field_dofs::Union{Vector{Int}, SubArray}, neighbor_field_dofs::Union{Vector{Int}, SubArray},
        i::Int, j::Int, keep_constrained::Bool, ch::Union{ConstraintHandler, Nothing}
    )
    dofi = cell_field_dofs[i]
    dofj = neighbor_field_dofs[j]
    # sym && (dofj > dofi && return cnt)
    !keep_constrained && (haskey(ch.dofmapping, dofi) || haskey(ch.dofmapping, dofj)) && return
    add_entry!(sp, dofi, dofj)
    return
end

function _add_interface_entries!(
        sp::SparsityPattern, dh::DofHandler, ch::Union{ConstraintHandler, Nothing},
        topology::ExclusiveTopology, keep_constrained::Bool,
        interface_coupling::AbstractMatrix{Bool},
    )
    # Expanded coupling masks, lazily created per (row cell sdh, column cell sdh) pair
    couplings = Dict{NTuple{2, Int}, Matrix{Bool}}()
    for ic in InterfaceIterator(dh, topology)
        # TODO: This looks like it can be optimized for the common case where
        #       the cells are in the same subdofhandler
        sdhs_idx = dh.cell_to_subdofhandler[cellid.([ic.a, ic.b])]
        sdhs = dh.subdofhandlers[sdhs_idx]
        # An interface integral is assembled over the stacked dof vector
        # [dofs(ic.a); dofs(ic.b)] and its local matrix can be dense within every field
        # block allowed by the mask (including the same-side blocks, since jump/average
        # terms expand into same-side products). The pattern must therefore contain every
        # (test dof, trial dof) pair allowed by the mask from the union of the two cells.
        # This is realized by looping over all four (row cell, column cell) combinations:
        # the two cross combinations and the two same-side combinations. Each pair is
        # gated by the coupling mask with the row cell as the first (test function) index
        # and the column cell as the second (trial function) index. Pairs that are also
        # produced by the cell pass, or by neighboring interfaces, are deduplicated by
        # `add_entry!`.
        for row_i in 1:2, col_i in 1:2
            sdh = sdhs[row_i]
            sdh2 = sdhs[col_i]
            row_dofs = celldofs(row_i == 1 ? ic.a : ic.b)
            col_dofs = celldofs(col_i == 1 ? ic.a : ic.b)
            coupling_sdh = get!(couplings, (sdhs_idx[row_i], sdhs_idx[col_i])) do
                _coupling_to_local_dof_coupling(dh, interface_coupling, sdh, sdh2)
            end
            for row_field in sdh.field_names
                dofrange1 = dof_range(sdh, row_field)
                row_field_dofs = @view row_dofs[dofrange1]
                for col_field in sdh2.field_names
                    dofrange2 = dof_range(sdh2, col_field)
                    col_field_dofs = @view col_dofs[dofrange2]

                    for (j, dof_j) in enumerate(dofrange2)
                        for (i, dof_i) in enumerate(dofrange1)
                            coupling_sdh[dof_i, dof_j] || continue
                            _add_interface_entry(sp, row_field_dofs, col_field_dofs, i, j, keep_constrained, ch)
                        end
                    end
                end
            end
        end
    end
    return sp
end

# Internal matrix instantiation for SparseMatrixCSC and Symmetric{SparseMatrixCSC}
# The counting transpose does not need sorted rows: per-column rowval order comes from the
# ascending outer row loop, and the `sym` filter is a per-entry predicate. Rows are
# therefore read through _eachrow_anyorder, which for SparsityPattern skips the lazy sort.
function _allocate_matrix(::Type{SparseMatrixCSC{Tv, Ti}}, sp::AbstractSparsityPattern, sym::Bool) where {Tv, Ti}
    # 1. Setup colptr
    colptr = zeros(Ti, getncols(sp) + 1)
    colptr[1] = 1
    for (row, colidxs) in enumerate(_eachrow_anyorder(sp))
        for col in colidxs
            sym && row > col && continue
            colptr[col + 1] += 1
        end
    end
    cumsum!(colptr, colptr)
    nnz = colptr[end] - 1
    # 2. Allocate rowval and nzval now that nnz is known
    rowval = Vector{Ti}(undef, nnz)
    nzval = zeros(Tv, nnz)
    # 3. Populate rowval. Since patterns are row-based we need to allocate an extra
    #    work buffer here to keep track of the next index into rowval
    nextinds = copy(colptr)
    for (row, colidxs) in zip(1:getnrows(sp), _eachrow_anyorder(sp))
        for col in colidxs
            sym && row > col && continue
            k = nextinds[col]
            rowval[k] = row
            nextinds[col] = k + 1
        end
    end
    @assert all(i -> nextinds[i] == colptr[i + 1], 1:getncols(sp))
    S = SparseMatrixCSC(getnrows(sp), getncols(sp), colptr, rowval, nzval)
    return S
end

# Build the cell -> global dofs map as an ArrayOfVectorViews (a compact copy of the
# DofHandler's cell dof storage).
function create_celldofs(dh::DofHandler)
    isclosed(dh) || throw(ArgumentError("DofHandler must be closed"))
    ncells = getncells(dh.grid)
    indices = similar(dh.cell_dofs_offset, ncells + 1)
    cell_dofs = similar(dh.cell_dofs)
    n = 1
    for cell_idx in 1:ncells
        indices[cell_idx] = n
        num = ndofs_per_cell(dh, cell_idx)
        num == 0 && continue
        # celldofs!(view(cell_dofs, r), dh, cell_idx), but ~4x faster without the view
        # (Vector-to-Vector copyto! and no re-validation per cell)
        soffs = dh.cell_dofs_offset[cell_idx]
        copyto!(cell_dofs, n, dh.cell_dofs, soffs, num)
        n += num
    end
    indices[end] = n
    return ArrayOfVectorViews(indices, cell_dofs, LinearIndices((ncells,)))
end

# For each cell, the cells sharing a facet with it (used by the fast path to reserve row
# space for interface entries).
function create_cell_to_neighbors(grid::AbstractGrid, topology)
    ncells = getncells(grid)
    counts = zeros(Int, ncells)
    for cell in 1:ncells
        for facet in 1:nfacets(getcells(grid, cell))
            counts[cell] += length(getneighborhood(topology, grid, FacetIndex(cell, facet)))
        end
    end
    indices = Vector{Int}(undef, ncells + 1)
    indices[1] = 1
    for cell in 1:ncells
        indices[cell + 1] = indices[cell] + counts[cell]
    end
    data = Vector{Int}(undef, indices[end] - 1)
    fill!(counts, 0)
    for cell in 1:ncells
        for facet in 1:nfacets(getcells(grid, cell))
            for nb in getneighborhood(topology, grid, FacetIndex(cell, facet))
                data[indices[cell] + counts[cell]] = nb.idx[1]
                counts[cell] += 1
            end
        end
    end
    return ArrayOfVectorViews(indices, data, LinearIndices((ncells,)))
end

# Inverse of cell_dofs: for each row (dof), the connected cells and, in parallel, the row's
# local dof index within each such cell. The local index differs per incident cell (a dof
# shared by two cells generally has different local numbers in them) and is only needed to
# apply the (expanded, locally indexed) coupling masks. With full coupling the second map
# is skipped (`nothing`).
function create_row_to_cells(cell_dofs::ArrayOfVectorViews, nrows::Int, build_localidx::Bool)
    num_cells = zeros(Int, nrows)
    # 1: Figure out how many cells are connected to each dof
    n_connected = 0
    @inbounds for rows in cell_dofs # dof = row
        for row in rows
            num_cells[row] += 1
            n_connected += 1
        end
    end
    # 2: Create the correct datastructure
    cells = Vector{Int}(undef, n_connected)
    localidx = build_localidx ? Vector{Int}(undef, n_connected) : nothing
    indices = Vector{Int}(undef, nrows + 1)
    indices[1] = 1
    @inbounds for row in 1:nrows
        indices[row + 1] = indices[row] + num_cells[row]
    end
    # Reset and reuse `num_cells` as a per-row write cursor: in the loop below
    # `num_cells[row]` counts how many incident cells have been recorded for `row` so far,
    # making `indices[row] + num_cells[row]` the next free slot in `row`'s block.
    fill!(num_cells, 0)
    @inbounds for (cellnr, rows) in enumerate(cell_dofs)
        for (li, row) in enumerate(rows)
            k = indices[row] + num_cells[row]
            cells[k] = cellnr
            localidx === nothing || (localidx[k] = li)
            num_cells[row] += 1
        end
    end
    lin = LinearIndices((nrows,))
    row_to_cells = ArrayOfVectorViews(indices, cells, lin)
    row_to_localidx = localidx === nothing ? nothing : ArrayOfVectorViews(indices, localidx, lin)
    return row_to_cells, row_to_localidx
end

# Count (data === nothing) or fill (data/indices from the presized buffer) the per-row
# candidate columns: the diagonal,
# plus for every incident cell the cell's dofs, filtered by the (expanded) coupling mask and, for
# keep_constrained = false, the constrained-dof filter. The count and fill passes MUST enumerate
# identically; both go through this function to guarantee that.
function _visit_row_candidates!(
        rowlen::Vector{Int}, data::Union{Nothing, Vector{Int}}, indices::Union{Nothing, Vector{AdaptiveRange}},
        marker::Vector{Int}, row_to_cells::ArrayOfVectorViews, row_to_localidx::Union{Nothing, ArrayOfVectorViews},
        cell_dofs::ArrayOfVectorViews, cell_to_sdh::Vector{Int},
        couplings::Union{Nothing, Vector{Matrix{Bool}}}, isconstrained::Union{Nothing, Vector{Bool}},
        neighbor_cells::Union{Nothing, ArrayOfVectorViews} = nothing,
        interface_couplings::Union{Nothing, Vector{Matrix{Bool}}} = nothing,
    )
    counting = data === nothing
    ncols = length(marker) # marker has one slot per column
    @inbounds for row in 1:length(rowlen)
        start = counting ? 0 : indices[row].start
        p = 0
        # The diagonal is always stored when the column exists (the pattern may have more
        # rows than columns)
        if row <= ncols
            marker[row] = row
            counting || (data[start] = row)
            p = 1
        end
        # For keep_constrained = false a constrained row stores only the diagonal (included
        # in the pattern just above)
        if !(isconstrained !== nothing && isconstrained[row])
            cells = row_to_cells[row]
            # `row_to_localidx` is `nothing` exactly when no mask restricts the candidates:
            # no coupling/interface_coupling passed, or explicitly full masks (normalized to
            # `nothing` in add_sparsity_entries!). Then `li` is never consulted and the
            # placeholder below is dead code.
            lidxs = row_to_localidx === nothing ? nothing : row_to_localidx[row]
            for k in 1:length(cells)
                cnr = cells[k]
                dofs = cell_dofs[cnr]
                mask = couplings === nothing ? nothing : couplings[cell_to_sdh[cnr]]
                # Same-side interface blocks: for a cell that participates in an interface,
                # add_interface_entries! also inserts the pairs the interface mask allows
                # within the cell itself, where `interface_couplings === nothing` means no
                # restriction (nothing/full mask), i.e. every pair. The own cell is always in
                # its own subdofhandler, so the square mask applies even with multiple
                # subdofhandlers. A candidate therefore passes if either the cell coupling or
                # the interface coupling allows it.
                on_interface = neighbor_cells !== nothing && !isempty(neighbor_cells[cnr])
                imask = on_interface && interface_couplings !== nothing ?
                    interface_couplings[cell_to_sdh[cnr]] : nothing
                li = lidxs === nothing ? 0 : lidxs[k]
                for j in 1:length(dofs)
                    mask === nothing || mask[li, j] || (on_interface && (imask === nothing || imask[li, j])) || continue
                    col = dofs[j]
                    isconstrained !== nothing && isconstrained[col] && continue
                    if marker[col] != row
                        marker[col] = row
                        counting || (data[start + p] = col)
                        p += 1
                    end
                end
            end
            # Interface entries, cross blocks: visit the dofs of facet-neighbor cells, filtered
            # by the expanded interface_coupling mask. add_interface_entries! inserts (row, col)
            # across an interface iff the mask couples them with the row cell as the test
            # (first) index -- exactly `imask[li, j]` seen from this row; both orientations of
            # every interface are covered here because every row visits all facet-neighbors of
            # all its cells (the same-side blocks are covered by the own-cell loop above). For a
            # neighbor in a different subdofhandler the square mask's local indices do not apply
            # (other layout, possibly other size), so all of its dofs are counted unmasked -- a
            # deliberate over-count that only costs reservation slack (the direct fill is gated
            # to a single subdofhandler by _can_fill_interfaces_directly). In the count pass this
            # reserves what add_interface_entries! inserts (loose only under the unmasked
            # cross-sdh fallback); over-reservation only leaves slack, never wrong entries. The
            # fill pass receives `neighbor_cells` only when the enumeration is provably exact
            # (see _can_fill_interfaces_directly), in which case the interface entries are
            # emitted here directly and add_interface_entries! is skipped.
            if neighbor_cells !== nothing
                for k in 1:length(cells)
                    cnr = cells[k]
                    li = lidxs === nothing ? 0 : lidxs[k]
                    for nbc in neighbor_cells[cnr]
                        nbdofs = cell_dofs[nbc]
                        imask = if interface_couplings !== nothing && cell_to_sdh[nbc] == cell_to_sdh[cnr]
                            interface_couplings[cell_to_sdh[cnr]]
                        else
                            nothing
                        end
                        for j in 1:length(nbdofs)
                            imask === nothing || imask[li, j] || continue
                            col = nbdofs[j]
                            isconstrained !== nothing && isconstrained[col] && continue
                            if marker[col] != row
                                marker[col] = row
                                counting || (data[start + p] = col)
                                p += 1
                            end
                        end
                    end
                end
            end
        end
        if counting
            rowlen[row] = p
        else
            r = indices[row]
            indices[row] = AdaptiveRange(r.start, p, r.nmax)
        end
    end
    return
end
