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
Each element of the iterator iterates indices of the stored *columns* for that row.
"""
eachrow(sp::AbstractSparsityPattern)

"""
    eachrow(sp::AbstractSparsityPattern, row::Int)

Return an iterator over *column* indices in row `row` of the sparsity pattern.

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
 - `nrows::Int`: number of rows
 - `ncols::Int`: number of column
 - `rows::Vector{Vector{Int}}`: vector of length `nrows`, where `rows[i]` is a
   *sorted* vector of column indices for non zero entries in row `i`.

!!! warning "Internal struct"
    The specific implementation of this struct, such as struct fields, type layout and type
    parameters, are internal and should not be relied upon.
"""
struct SparsityPattern <: AbstractSparsityPattern
    nrows::Int
    ncols::Int
    mempool::PoolAllocator.MemoryPool{Int}
    rows::Vector{PoolAllocator.PoolVector{Int}}
end

"""
    SparsityPattern(nrows::Int, ncols::Int; nnz_per_row::Int = 8)

Create an empty [`SparsityPattern`](@ref) with `nrows` rows and `ncols` columns.
`nnz_per_row` is used as a memory hint for the number of non zero entries per
row.

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
    mempool = PoolAllocator.MemoryPool{Int}()
    rows = Vector{PoolAllocator.PoolVector{Int}}(undef, nrows)
    for i in 1:nrows
        rows[i] = PoolAllocator.resize(PoolAllocator.malloc(mempool, nnz_per_row), 0)
    end
    sp = SparsityPattern(nrows, ncols, mempool, rows)
    return sp
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
    @assert getnrows(sp) * sizeof(eltype(sp.rows)) == sizeof(sp.rows)
    bytes_used      = sizeof(sp.rows) + stored_entries * sizeof(Int)
    bytes_allocated = sizeof(sp.rows) + PoolAllocator.mempool_stats(sp.mempool)[2]
    print(iob,   " - Memory estimate: $(Base.format_bytes(bytes_used)) used, $(Base.format_bytes(bytes_allocated)) allocated")
    write(io, seekstart(iob))
    return
end

getnrows(sp::SparsityPattern) = sp.nrows
getncols(sp::SparsityPattern) = sp.ncols

@inline function add_entry!(sp::SparsityPattern, row::Int, col::Int)
    @boundscheck (1 <= row <= getnrows(sp) && 1 <= col <= getncols(sp)) || throw(BoundsError(sp, (row, col)))
    r = @inbounds sp.rows[row]
    r = insert_sorted(r, col)
    @inbounds sp.rows[row] = r
    return
end

@inline function insert_sorted(x::PoolAllocator.PoolVector{Int}, item::Int)
    k = searchsortedfirst(x, item)
    if k == length(x) + 1 || @inbounds(x[k]) != item
        x = PoolAllocator.insert(x, k, item)
    end
    return x
end

eachrow(sp::SparsityPattern)           = sp.rows
eachrow(sp::SparsityPattern, row::Int) = sp.rows[row]


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
 - `add_interface_entries!` is called if `topology` is provided (i.e. not `nothing`)
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
    if topology !== nothing
        add_interface_entries!(sp, dh, ch; topology, keep_constrained, interface_coupling)
    end
    if ch !== nothing
        add_constraint_entries!(sp, ch; keep_constrained)
    end
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
"""
function add_cell_entries!(
        sp::AbstractSparsityPattern,
        dh::DofHandler, ch::Union{ConstraintHandler, Nothing} = nothing;
        keep_constrained::Bool = true, coupling::Union{AbstractMatrix{Bool}, Nothing} = nothing,
    )
    # Expand coupling from nfields × nfields to ndofs_per_cell × ndofs_per_cell
    # TODO: Perhaps this can be done in the loop over SubDofHandlers instead.
    if coupling !== nothing
        coupling = _coupling_to_local_dof_coupling(dh, coupling)
    end
    if !keep_constrained
        ch === nothing && error("must pass ConstraintHandler when `keep_constrained = true`")
        isclosed(ch) || error("the ConstraintHandler must be closed")
        ch.dh === dh || error("the DofHandler and the ConstraintHandler's DofHandler must be the same")
    end
    return _add_cell_entries!(sp, dh, ch, keep_constrained, coupling)
end

"""
    add_interface_entries!(
        sp::SparsityPattern, dh::DofHandler, ch::Union{ConstraintHandler, Nothing};
        topology::ExclusiveTopology, keep_constrained::Bool = true,
        interface_coupling::AbstractMatrix{Bool},
    )

Add entries to the sparsity pattern `sp` corresponding to DoF couplings on the interface
between cells as described by the DofHandler `dh`.

# Keyword arguments
 - `topology`: the topology corresponding to the grid.
 - `keep_constrained`: whether or not entries for constrained DoFs should be kept
   (`keep_constrained = true`) or eliminated (`keep_constrained = false`) from the sparsity
   pattern. `keep_constrained = false` requires passing the ConstraintHandler `ch`.
 - `interface_coupling`: the coupling between fields/components across the interface.
"""
function add_interface_entries!(
        sp::SparsityPattern, dh::DofHandler, ch::Union{ConstraintHandler, Nothing} = nothing;
        topology::ExclusiveTopology, keep_constrained::Bool = true,
        interface_coupling::AbstractMatrix{Bool},
    )
    if !keep_constrained
        ch === nothing && error("must pass ConstraintHandler when `keep_constrained = true`")
        isclosed(ch) || error("the ConstraintHandler must be closed")
        ch.dh === dh || error("the DofHandler and the ConstraintHandler's DofHandler must be the same")
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
supported matrix types, and to [`create_sparsity_pattern`](@ref) for details about supported
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
function _coupling_to_local_dof_coupling(dh::DofHandler, coupling::AbstractMatrix{Bool})
    sz = size(coupling, 1)
    sz == size(coupling, 2) || error("coupling not square")

    # Return one matrix per (potential) sub-domain
    outs = Matrix{Bool}[]
    field_dims = map(fieldname -> n_components(dh, fieldname), dh.field_names)

    for sdh in dh.subdofhandlers
        out = zeros(Bool, ndofs_per_cell(sdh), ndofs_per_cell(sdh))
        push!(outs, out)

        dof_ranges = [dof_range(sdh, f) for f in sdh.field_names]
        global_idxs = [findfirst(x -> x === f, dh.field_names) for f in sdh.field_names]

        if sz == length(dh.field_names) # Coupling given by fields
            for (j, jrange) in pairs(dof_ranges), (i, irange) in pairs(dof_ranges)
                out[irange, jrange] .= coupling[global_idxs[i], global_idxs[j]]
            end
        elseif sz == sum(field_dims) # Coupling given by components
            component_offsets = pushfirst!(cumsum(field_dims), 0)
            for (jf, jrange) in pairs(dof_ranges), (j, J) in pairs(jrange)
                jc = mod1(j, field_dims[global_idxs[jf]]) + component_offsets[global_idxs[jf]]
                for (i_f, irange) in pairs(dof_ranges), (i, I) in pairs(irange)
                    ic = mod1(i, field_dims[global_idxs[i_f]]) + component_offsets[global_idxs[i_f]]
                    out[I, J] = coupling[ic, jc]
                end
            end
        elseif sz == ndofs_per_cell(sdh) # Coupling given by template local matrix
            # TODO: coupling[fieldhandler_idx] if different template per subddomain
            out .= coupling
        else
            error("could not create coupling")
        end
    end
    return outs
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

function _add_constraint_entries!(
        sp::AbstractSparsityPattern, dofcoefficients::Vector{Union{DofCoefficients{T}, Nothing}},
        dofmapping::Dict{Int,Int}, keep_constrained::Bool,
    ) where {T}

    # Return early if there are no non-trivial affine constraints
    any(i -> !(i === nothing || isempty(i)), dofcoefficients) || return

    # New entries tracked separately and inserted after since it is not possible to modify
    # the datastructure while looping over it.
    mempool = PoolAllocator.MemoryPool{Int}()
    sp′ = Dict{Int, PoolAllocator.PoolVector{Int}}()

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
                        r = get(sp′, row) do
                            PoolAllocator.resize(PoolAllocator.malloc(mempool, 8), 0)
                        end
                        r = insert_sorted(r, col′)
                        sp′[row] = r
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
                        r = get(sp′, row′) do
                            PoolAllocator.resize(PoolAllocator.malloc(mempool, 8), 0)
                        end
                        r = insert_sorted(r, col)
                        sp′[row′] = r
                    end
                else
                    # ... this column _is_ constrained, double-distribute to columns/rows.
                    for (row′, _) in row_coeffs
                        !keep_constrained && haskey(dofmapping, row′) && continue
                        for (col′, _) in col_coeffs
                            !keep_constrained && haskey(dofmapping, col′) && continue
                            r = get(sp′, row′) do
                                PoolAllocator.resize(PoolAllocator.malloc(mempool, 8), 0)
                            end
                            r = insert_sorted(r, col′)
                            sp′[row′] = r
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

function _add_interface_entry(sp::SparsityPattern, coupling_sdh::Matrix{Bool}, dof_i::Int, dof_j::Int,
        cell_field_dofs::Union{Vector{Int}, SubArray}, neighbor_field_dofs::Union{Vector{Int}, SubArray},
        i::Int, j::Int, keep_constrained::Bool, ch::Union{ConstraintHandler, Nothing})

    coupling_sdh[dof_i, dof_j] || return
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
    couplings = _coupling_to_local_dof_coupling(dh, interface_coupling)
    for ic in InterfaceIterator(dh, topology)
        # TODO: This looks like it can be optimized for the common case where
        #       the cells are in the same subdofhandler
        sdhs_idx = dh.cell_to_subdofhandler[cellid.([ic.a, ic.b])]
        sdhs = dh.subdofhandlers[sdhs_idx]
        for (i, sdh) in pairs(sdhs)
            sdh_idx = sdhs_idx[i]
            coupling_sdh = couplings[sdh_idx]
            for cell_field in sdh.field_names
                dofrange1 = dof_range(sdh, cell_field)
                cell_dofs = celldofs(sdh_idx == 1 ? ic.a : ic.b)
                cell_field_dofs = @view cell_dofs[dofrange1]
                for neighbor_field in sdh.field_names
                    sdh2 = sdhs[i == 1 ? 2 : 1]
                    neighbor_field ∈ sdh2.field_names || continue
                    dofrange2 = dof_range(sdh2, neighbor_field)
                    neighbor_dofs = celldofs(sdh_idx == 2 ? ic.a : ic.b)
                    neighbor_field_dofs = @view neighbor_dofs[dofrange2]
                    # Typical coupling procedure
                    for (j, dof_j) in pairs(dofrange2), (i, dof_i) in pairs(dofrange1)
                        # This line to avoid coupling the shared dof in continuous interpolations as cross-element. They're coupled in the local coupling matrix.
                        (cell_field_dofs[i] ∈ neighbor_dofs || neighbor_field_dofs[j] ∈ cell_dofs) && continue
                        _add_interface_entry(sp, coupling_sdh, dof_i, dof_j, cell_field_dofs, neighbor_field_dofs, i, j, keep_constrained, ch)
                        _add_interface_entry(sp, coupling_sdh, dof_j, dof_i, neighbor_field_dofs, cell_field_dofs, j, i, keep_constrained, ch)
                    end
                end
            end
        end
    end
    return sp
end

# Internal matrix instantiation for SparseMatrixCSC and Symmetric{SparseMatrixCSC}
function _allocate_matrix(::Type{SparseMatrixCSC{Tv, Ti}}, sp::AbstractSparsityPattern, sym::Bool) where {Tv, Ti}
    # 1. Setup colptr
    colptr = zeros(Ti, getncols(sp) + 1)
    colptr[1] = 1
    for (row, colidxs) in enumerate(eachrow(sp))
        for col in colidxs
            sym && row > col && continue
            colptr[col+1] += 1
        end
    end
    cumsum!(colptr, colptr)
    nnz = colptr[end] - 1
    # 2. Allocate rowval and nzval now that nnz is known
    rowval = Vector{Ti}(undef, nnz)
    nzval = zeros(Tv, nnz)
    # 3. Populate rowval. Since SparsityPattern is row-based we need to allocate an extra
    #    work buffer here to keep track of the next index into rowval
    nextinds = copy(colptr)
    for (row, colidxs) in zip(1:getnrows(sp), eachrow(sp)) # pairs(eachrow(sp))
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
