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
    bytes_used = sizeof(sp.rows) + stored_entries * sizeof(Int)
    bytes_allocated = sizeof(sp.rows) + PoolAllocator.mempool_stats(sp.mempool)[2]
    print(iob, " - Memory estimate: $(Base.format_bytes(bytes_used)) used, $(Base.format_bytes(bytes_allocated)) allocated")
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

eachrow(sp::SparsityPattern) = sp.rows
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
 - `interface_coupling`: the coupling between fields/components in interface integrals.
   `interface_coupling[i, j] = true` means that, for every interface, entries exist for
   *every* pair of (test DoF of field/component `i`, trial DoF of field/component `j`)
   within the union of the DoFs of the two cells sharing the interface. As for cell
   `coupling`, rows correspond to test functions and columns to trial functions.
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

"""
    add_system_variable_entires!(sp::AbstractSparsityPattern, dh::DofHandler, cells, name::Symbol)

Add full coupling between the degrees of freedom associated with the cells in `cells` and
all degrees of freedom in the global dof block named `name`.

The resulting sparsity pattern contains both `(dof, gdof)` and `(gdof, dof)` entries for
all cell dofs `dof` and global dofs `gdof` in the selected cells and global dof block.
"""
function add_system_variable_entries!(sp::AbstractSparsityPattern, dh::DofHandler, cellset #=::OrderedSet=#; cell_fields::Vector{Symbol}, system_variable::Symbol)
    isclosed(dh) || error("the DofHandler must be closed")
    system_variable ∈ dh.system_variables_names || throw(KeyError(system_variable))
    system_dofs = system_variable_dofs(dh, system_variable)
    
    #The given cellset might extend multiple subdofhandlers.
    for sdh in dh.subdofhandlers
        filtered_cells = intersect(sdh.cellset, cellset)
        length(filtered_cells) == 0 && continue
        for cell_field in cell_fields
            field_idx = _find_field(sdh, cell_field)
            field_idx === nothing && throw(ArgumentError("One of the elements in the cellset (e.g. cellid=$(first(filtered_cells))) does not have field $cell_field."))
            dofrange = dof_range(sdh, field_idx)

            # Loop over the filtered sets and couple the celldofs with the global field
            for celldata in CellIterator(dh, filtered_cells)
                dofs = celldofs(celldata)
                for k in dofrange, gdof in system_dofs
                    dof = dofs[k]
                    add_entry!(sp, dof, gdof)
                    add_entry!(sp, gdof, dof)
                end
            end
        end
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
    _get_Ti(::Type{<:AbstractMatrix}) = Int
    _get_Ti(::Type{<:AbstractSparseMatrix{<:Any, Ti}}) where {Ti} = Ti
    if _can_use_fastsp(MatrixType, args...; kwargs...)
        fsp = FastSparsityPattern(_get_Ti(MatrixType), dh, args...; kwargs...)
        return allocate_matrix(MatrixType, fsp)
    end
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

function _add_constraint_entries!(
        sp::AbstractSparsityPattern, dofcoefficients::Vector{Union{DofCoefficients{T}, Nothing}},
        dofmapping::Dict{Int, Int}, keep_constrained::Bool,
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
function _allocate_matrix(::Type{SparseMatrixCSC{Tv, Ti}}, sp::AbstractSparsityPattern, sym::Bool) where {Tv, Ti}
    # 1. Setup colptr
    colptr = zeros(Ti, getncols(sp) + 1)
    colptr[1] = 1
    for (row, colidxs) in enumerate(eachrow(sp))
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

## ================= ##
# FastSparsityPattern #
## ================= ##

"""
    FastSparsityPattern([Ti = Int64], dh::DofHandler)

This sparsity does not currently support the full `AbstractSparsityPattern` interface,
but is used as an internal fast-path for `allocate_matrix(MatrixType, dh)` for some
supported `MatrixType`s. It can be extended in the future or potentially be merged
with `SparsityPattern`.
See [#1302](https://github.com/Ferrite-FEM/Ferrite.jl/pull/1302) for details.

!!! warning "Internal"
    `FastSparsityPattern` is strictly internal and its interface and implementation
    may change at any time.

"""
mutable struct FastSparsityPattern{Ti} <: AbstractSparsityPattern
    const rowlen::Vector{Ti} # Number of stored entries in each row
    const marker::Vector{Ti} # Marker if column has been "visited" by certain row
    const rowptr::Vector{Ti} # Index of stored entries at the start of each row
    const colidx::Vector{Ti} # colidx[i] gives the column number of the ith stored entry
    is_colidx_sorted::Bool   # Is colidx sorted for each row
end
function FastSparsityPattern(::Type{Ti}, ncols, nrows) where {Ti <: Integer}
    rowlen = zeros(Ti, nrows)
    marker = zeros(Ti, ncols)
    rowptr = Vector{Ti}(undef, nrows + 1)
    colidx = Vector{Ti}(undef, 0) # To be resized later
    return FastSparsityPattern(rowlen, marker, rowptr, colidx, false)
end

# _can_use_fastsp(MatrixType, args...; kwargs...) where args and kwargs are those passed to
# `allocate_matrix`. See `add_sparsity_entries!` for a description of args/kwargs.
function _can_use_fastsp(
        ::Type{MatrixType},
        ch = nothing;
        topology = nothing,
        keep_constrained = true,
        coupling = nothing,
        interface_coupling = nothing
    ) where {MatrixType}
    if ch === topology === coupling === interface_coupling === nothing
        if MatrixType <: AbstractSparseMatrix # Symmetric/Block matrices not supported
            return keep_constrained
        end
    end
    return false
end

FastSparsityPattern(dh::DofHandler) = FastSparsityPattern(Int, dh)
function FastSparsityPattern(::Type{Ti}, dh::DofHandler) where {Ti}
    sp = FastSparsityPattern(Ti, ndofs(dh), ndofs(dh))
    # Step 1: Create cell_dof_views::ArrayOfVectorViews (would be nice in `DofHandler` directly)
    cell_dofs_views = create_celldofs(dh)
    # Step 2: Define mapping rownr to cells
    row_to_cells = create_row_to_cells(cell_dofs_views, sp)
    # Step 3: Count how many cols stored for each row
    count_row_sizes!(sp, row_to_cells, cell_dofs_views)
    # Step 4: Build the rowptr (indices for s)
    build_rowptr!(sp)
    fill_colidx!(sp, row_to_cells, cell_dofs_views)
    return sp
end

getncols(sp::FastSparsityPattern) = length(sp.marker)
getnrows(sp::FastSparsityPattern) = length(sp.rowlen)

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
        r = n:(n + num - 1)
        #celldofs!(view(cell_dofs, r), dh, cell_idx), but faster without view:
        soffs = dh.cell_dofs_offset[cell_idx]
        copyto!(cell_dofs, n, dh.cell_dofs, soffs, num)
        n = last(r) + 1
    end
    indices[end] = n
    return ArrayOfVectorViews(indices, cell_dofs, LinearIndices((ncells,)))
end

function create_row_to_cells(cell_dofs::ArrayOfVectorViews, sp)
    nrows = getnrows(sp)
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
    data = Vector{Int}(undef, n_connected)
    indices = Vector{Int}(undef, nrows + 1)
    indices[1] = 1
    @inbounds for row in 1:nrows
        indices[row + 1] = indices[row] + num_cells[row]
    end
    fill!(num_cells, 0) # Now we use this to count how many have been added
    @inbounds for (cellnr, rows) in enumerate(cell_dofs)
        for row in rows
            data[indices[row] + num_cells[row]] = cellnr
            num_cells[row] += 1
        end
    end
    return ArrayOfVectorViews(indices, data, LinearIndices((nrows,)))
end

function count_row_sizes!(sp::FastSparsityPattern, row_to_cells::AbstractVector, cell_dofs::ArrayOfVectorViews)
    @inbounds for row in 1:getnrows(sp)
        for cnr in row_to_cells[row]
            for col in cell_dofs[cnr]
                if sp.marker[col] != row
                    sp.marker[col] = row
                    sp.rowlen[row] += 1
                end
            end
        end
    end
    return sp
end

function build_rowptr!(sp)
    sp.rowptr[1] = 1
    @inbounds for row in 1:getnrows(sp)
        sp.rowptr[row + 1] = sp.rowptr[row] + sp.rowlen[row]
    end
    return sp
end

function fill_colidx!(sp::FastSparsityPattern, row_to_cells::AbstractVector, cell_dofs::ArrayOfVectorViews)
    resize!(sp.colidx, sp.rowptr[end] - 1) # nnz
    fill!(sp.marker, 0)
    @inbounds for row in 1:getnrows(sp)
        pos = sp.rowptr[row]
        for cnr in row_to_cells[row]
            for col in cell_dofs[cnr]
                if sp.marker[col] != row
                    sp.marker[col] = row
                    sp.colidx[pos] = col
                    pos += 1
                end
            end
        end
    end
    return sp
end

allocate_matrix(sp::FastSparsityPattern) = allocate_matrix(SparseMatrixCSC, sp)
allocate_matrix(::Type{SparseMatrixCSC}, sp::FastSparsityPattern{Int}) = allocate_matrix(SparseMatrixCSC{Float64, Int}, sp)
function allocate_matrix(::Type{<:SparseMatrixCSC{Tv, Ti}}, sp::FastSparsityPattern{Ti}) where {Ti, Tv}
    nnz = length(sp.colidx)
    ncols = getncols(sp)
    nrows = getnrows(sp)

    # Number of stored entries per column
    collen = zeros(Ti, ncols)
    @inbounds for col in sp.colidx
        collen[col] += 1
    end

    # Index of stored entries at the start of each column
    colptr = Vector{Ti}(undef, ncols + 1)
    colptr[1] = 1
    @inbounds for col in 1:ncols
        colptr[col + 1] = colptr[col] + collen[col]
    end

    # Build rowidx[i] giving the row number of the ith stored entry
    rowidx = Vector{Ti}(undef, nnz)
    next = copy(colptr)
    @inbounds for row in 1:nrows
        for p in sp.rowptr[row]:(sp.rowptr[row + 1] - 1)
            col = sp.colidx[p]
            q = next[col]
            rowidx[q] = row
            next[col] = q + 1
            # For a given col, next[col] is increasing, and row is
            # increasing in the outer loop -> rowidx sorted for each col
        end
    end
    nzval = zeros(Tv, nnz)
    return SparseMatrixCSC(nrows, ncols, colptr, rowidx, nzval)
end
