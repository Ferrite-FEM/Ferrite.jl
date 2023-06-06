###########################
# AbstractSparsityPattern #
###########################

"""
    AbstractSparsityPattern

Supertype for sparsity pattern implementations, e.g. [`SparsityPattern`](@ref) and
[`BlockSparsityPattern`](@ref).
"""
abstract type AbstractSparsityPattern end

"""
    n_rows(sp::AbstractSparsityPattern)

Return the number of rows in the sparsity pattern `sp`.
"""
n_rows(sp::AbstractSparsityPattern)

"""
    n_cols(sp::AbstractSparsityPattern)

Return the number of columns in the sparsity pattern `sp`.
"""
n_cols(sp::AbstractSparsityPattern)

"""
    add_entry!(sp::AbstractSparsityPattern, row::Int, col::Int)

Add an entry to the sparsity pattern `sp` at row `row` and column `col`.
"""
add_entry!(sp::AbstractSparsityPattern, row::Int, col::Int)

# Necessary to avoid warning about not importing Base.eachrow when adding docstring before
# the definitions further down.
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
    rows::Vector{Vector{Int}}
end

"""
    SparsityPattern(nrows::Int, ncols::Int; nnz_per_row::Int = 0)

Create an empty [`SparsityPattern`](@ref) with `nrows` rows and `ncols` columns.
`nnz_per_row` is used as a memory optimization hint for the number of non zero entries per
row.

`SparsityPattern` is the default sparsity pattern type for the standard DofHandler and is
therefore commonly constructed using [`create_sparsity_pattern`](@ref) instead of with this
constructor.

# Examples
```julia
# Create a sparsity pattern for an 100 x 100 matrix, hinting at 10 entries per row
sparsity_pattern = SparsityPattern(100, 100; nnz_per_row = 10)
```

# Methods
The following methods apply to `SparsityPattern` (see their respective documentation for
more details):
 - [`create_sparsity_pattern!`](@ref): add entries corresponding to DoF couplings.
 - [`condense_sparsity_pattern!`](@ref): add entries resulting from constraints.
 - [`create_matrix`](@ref create_matrix(::SparsityPattern)): instantiate a matrix from the pattern. The default matrix type
   is `SparseMatrixCSC{Float64, Int}`.
"""
function SparsityPattern(nrows::Int, ncols::Int; nnz_per_row::Int = 0)
    # Note: Empirical testing suggest resize!(Vector{Int}(undef, nnz_per_row), 0) is better
    # than sizehint!(Vector{Int}(undef, 0), nnz_per_row).
    rows = Vector{Int}[resize!(Vector{Int}(undef, nnz_per_row), 0) for _ in 1:nrows]
    return SparsityPattern(nrows, ncols, rows)
end

n_rows(sp::SparsityPattern) = sp.nrows
n_cols(sp::SparsityPattern) = sp.ncols

@inline function add_entry!(sp::SparsityPattern, row::Int, col::Int)
    @boundscheck 1 <= row <= n_rows(sp) && 1 <= col <= n_cols(sp)
    @inbounds insert_sorted!(sp.rows[row], col)
    return
end

eachrow(sp::SparsityPattern)           = sp.rows
eachrow(sp::SparsityPattern, row::Int) = sp.rows[row]


########################
# BlockSparsityPattern #
########################

# This is implemented as an extension in ext/FerriteBlockArrays.jl. This method exist to
# give a nice error message. Keep the signature in sync with the constructor in
# ext/FerriteBlockArrays.jl (just a tad less specific here).
function BlockSparsityPattern(::AbstractVector{<:Integer})
    msg = "Ferrite's block matrix functionality depends on the package BlockArrays to " *
          "be installed and loaded into the session. Install BlockArrays " *
          "(`pkg> add BlockArrays`) and load it (`using BlockArrays`) to enable."
    error(msg)
end


################################################
## Adding entries to AbstractSparsityPatterns ##
################################################

"""
    create_sparsity_pattern(
        dh::DofHandler,
        args...; kwargs...
    )

Create a [`SparsityPattern`](@ref) and add entries corresponding to the degree-of-freedom
distribution in the DofHandler `dh`. See [`create_sparsity_pattern!`](@ref) for more details
and documentation of supported arguments `args...` and keyword arguments `kwargs...`.
"""
function create_sparsity_pattern(
        dh::DofHandler, args...;
        # TODO: What is a good estimate for nnz_per_row?
        nnz_per_row::Int = 2 * ndofs_per_cell(dh),
        kwargs...,
    )
    sp = SparsityPattern(ndofs(dh), ndofs(dh); nnz_per_row = nnz_per_row)
    return create_sparsity_pattern!(sp, dh, args...; kwargs...)
end

"""
    create_sparsity_pattern!(
        sp::AbstractSparsityPattern,
        dh::DofHandler,
        ch::Union{ConstraintHandler, Nothing} = nothing;
        nnz_per_row::Int = 2 * ndofs_per_cell(dh),
        keep_constrained::Bool = true,
        coupling = nothing,
    )

Add entries corresponding to DoF couplings in the DofHandler `dh` to the sparsity pattern
`sp`.

# Arguments
 - `dh`: the DofHandler for which to generate the pattern.
 - `ch`: the ConstraintHandler. Used to add the non-zero entries resulting from
   `AffineConstraint`s (and `PeriodicDirichlet` constraints, which uses `AffineConstraint`s
   internally). See [`condense_sparsity_pattern!`](@ref).
# Keyword arguments
 - `nnz_per_row`: memory optimization hint for the number of non-zero entries per row.
 - `keep_constrained`: whether or not entries for constrained DoFs should be kept
   (`keep_constrained = true`) or eliminated (`keep_constrained = false`) from the sparsity
   pattern. `keep_constrained = false` requires passing the ConstraintHandler.
 - `coupling`: the coupling between fields/components within each cell. By default
   (`coupling = nothing`) it is assumed that all DoFs in each cell couple with each other.
"""
function create_sparsity_pattern!(
        sp::AbstractSparsityPattern, dh::DofHandler, ch::Union{ConstraintHandler, Nothing} = nothing;
        keep_constrained::Bool = true,
        coupling::Union{AbstractMatrix{Bool}, Nothing} = nothing,
    )
    # Argument checking
    isclosed(dh) || error("the DofHandler must be closed")
    if n_rows(sp) < ndofs(dh) || n_cols(sp) < ndofs(dh)
        error("number of rows ($(n_rows(sp))) or columns ($(n_cols(sp))) in the sparsity pattern is smaller than number of dofs ($(ndofs(dh)))")
    end
    if !keep_constrained
        ch === nothing && error("must pass ConstraintHandler when `keep_constrained = true`")
        isclosed(ch) || error("the ConstraintHandler must be closed")
        ch.dh === dh || error("the DofHandler and the ConstraintHandler's DofHandler must be the same")
    end
    # Expand coupling from nfields × nfields to ndofs_per_cell × ndofs_per_cell
    # TODO: Perhaps this can be done in the loop over SubDofHandlers instead.
    if coupling !== nothing
        coupling = _coupling_to_local_dof_coupling(dh, coupling)
    end
    return _create_sparsity_pattern!(sp, dh, ch, keep_constrained, coupling)
end

"""
    condense_sparsity_pattern!(
        sp::AbstractSparsityPattern,
        ch::ConstraintHandler;
        keep_constrained::Bool = true,
    )

TBW
"""
function condense_sparsity_pattern!(
        sp::AbstractSparsityPattern, ch::ConstraintHandler;
        keep_constrained::Bool = true,
)
    return _condense_sparsity_pattern!(sp, ch.dofcoefficients, ch.dofmapping, keep_constrained)
end


############################################################
# Sparse matrix instantiation from AbstractSparsityPattern #
############################################################

"""
    create_matrix(::Type{SparseMatrixCSC{Tv, Ti}}, sp::SparsityPattern)

Instantiate a sparse matrix of type `SparseMatrixCSC{Tv, Ti}` from the sparsity pattern
`sp`.
"""
function create_matrix(::Type{S}, sp::AbstractSparsityPattern) where {Tv, Ti, S <: SparseMatrixCSC{Tv, Ti}}
    return _create_matrix(S, sp, #=sym=# false)
end

"""
    create_matrix(::Type{Symmetric{Tv, SparseMatrixCSC{Tv, Ti}}}, sp::SparsityPattern)

Instantiate a sparse matrix of type `Symmetric{Tv, SparseMatrixCSC{Tv, Ti}}`, i.e. a
`LinearAlgebra.Symmetric`-wrapped `SparseMatrixCSC`, from the sparsity pattern `sp`. The
resulting matrix will only store entries above, and including, the diagonal.
"""
function create_matrix(::Type{Symmetric{Tv, S}}, sp::AbstractSparsityPattern) where {Tv, Ti, S <: SparseMatrixCSC{Tv, Ti}}
    return Symmetric(_create_matrix(S, sp, #=sym=# true))
end

"""
    create_matrix(sp::SparsityPattern)

Instantiate a sparse matrix of type `SparseMatrixCSC{Float64, Int}` from the sparsity
pattern `sp`.

This method is a shorthand for the equivalent
[`create_matrix(SparseMatrixCSC{Float64, Int}, sp)`]
(@ref create_matrix(::Type{S}, sp::AbstractSparsityPattern) where {Tv, Ti, S <: SparseMatrixCSC{Tv, Ti}}).
"""
create_matrix(sp::SparsityPattern) = create_matrix(SparseMatrixCSC{Float64, Int}, sp)

"""
    create_matrix(MatrixType, dh::DofHandler, args...; kwargs...)

Instantiate a matrix of type `MatrixType` from the DofHandler `dh`.

This method is a convenience shorthand for the equivalent `create_matrix(MatrixType,
create_sparsity_pattern(dh, args...; kwargs...))` -- refer to [`create_matrix`](@ref
create_matrix(::Type{<:Any}, ::SparsityPattern)) for supported matrix types, and to
[`create_sparsity_pattern`](@ref) for details about supported arguments `args` and keyword
arguments `kwargs`.

!!! note
    If more than one sparse matrix is needed (e.g. a stiffness and a mass matrix) it is more
    efficient to explicitly create the sparsity pattern instead of using this method, i.e.
    use
    ```julia
    sp = create_sparsity_pattern(dh)
    K = create_matrix(sp)
    M = create_matrix(sp)
    ```
    instead of
    ```julia
    K = create_matrix(dh)
    M = create_matrix(dh)
    ```
"""
function create_matrix(::Type{MatrixType}, dh::DofHandler, args...; kwargs...) where {MatrixType}
    sp = create_sparsity_pattern(dh, args...; kwargs...)
    return create_matrix(MatrixType, sp)
end

"""
    create_matrix(dh::DofHandler, args...; kwargs...)

Instantiate a matrix of type `SparseMatrixCSC{Float64, Int}` from the DofHandler `dh`.

This method is a shorthand for the equivalent [`create_matrix(SparseMatrixCSC{Float64, Int},
dh, args...; kwargs...)`](@ref create_matrix(::Type{MatrixType}, ::DofHandler, args...;
kwargs...) where {MatrixType}) -- refer to that method for details.
"""
function create_matrix(dh::DofHandler, args...; kwargs...)
    return create_matrix(SparseMatrixCSC{Float64, Int}, dh, args...; kwargs...)
end


##############################
# Sparsity pattern internals #
##############################

@inline function insert_sorted!(x::Vector{Int}, item::Int)
    k = searchsortedfirst(x, item)
    if k == lastindex(x) + 1 || item != x[k]
        insert!(x, k, item)
    end
    return x
end

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
    field_dims = map(fieldname -> getfielddim(dh, fieldname), dh.field_names)

    for fh in dh.fieldhandlers
        out = zeros(Bool, ndofs_per_cell(fh), ndofs_per_cell(fh))
        push!(outs, out)

        dof_ranges = [dof_range(fh, f) for f in fh.field_names]
        global_idxs = [findfirst(x -> x === f, dh.field_names) for f in fh.field_names]

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
        elseif sz == ndofs_per_cell(fh) # Coupling given by template local matrix
            # TODO: coupling[fieldhandler_idx] if different template per subddomain
            out .= coupling
        else
            error("could not create coupling")
        end
    end
    return outs
end

function _create_sparsity_pattern!(
        sp::AbstractSparsityPattern, dh::DofHandler, ch::Union{ConstraintHandler, Nothing},
        keep_constrained::Bool,
        coupling::Union{Vector{<:AbstractMatrix{Bool}}, Nothing},
    )
    # 1. Add all connections between dofs for every cell while filtering based
    #    on a) constraints, and b) field/dof coupling.
    cc = CellCache(dh)
    for (sdhi, sdh) in pairs(dh.fieldhandlers)
        set = BitSet(sdh.cellset)
        coupling === nothing || (coupling_sdh = coupling[sdhi])
        for cell_id in set
            reinit!(cc, cell_id)
            for (i, row) in pairs(cc.dofs)
                # a) check constraint for row
                !keep_constrained && haskey(ch.dofmapping, row) && continue
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
    # 2. Make sure diagonal entries are included
    for d in 1:ndofs(dh)
        add_entry!(sp, d, d)
    end
    # 3. Insert entries necessary for handling affine constraints
    if ch !== nothing
        condense_sparsity_pattern!(sp, ch; keep_constrained = keep_constrained)
    end
    return sp
end

function _condense_sparsity_pattern!(
        sp::AbstractSparsityPattern, dofcoefficients::Vector{Union{DofCoefficients{T}, Nothing}},
        dofmapping::Dict{Int,Int}, keep_constrained::Bool,
    ) where {T}

    # Return early if there are no non-trivial affine constraints
    any(i -> !(i === nothing || isempty(i)), dofcoefficients) || return

    # New entries tracked separately and inserted after since it is not possible to modify
    # the datastructure while looping over it.
    sp′ = Dict{Int, Vector{Int}}()

    for (row, colidxs) in pairs(eachrow(sp))
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
                        insert_sorted!(get!(Vector{Int}, sp′, row), col′)
                    end
                end
            end
        else
            # This row _is_ constrained, check columnss of this row...
            for col in colidxs
                col_coeffs = coefficients_for_dof(dofmapping, dofcoefficients, col)
                if col_coeffs === nothing
                    # ... this column is _not_ constrained, distribute to rows.
                    !keep_constrained && haskey(dofmapping, col) && continue
                    for (row′, _) in row_coeffs
                        insert_sorted!(get!(Vector{Int}, sp′, row′), col)
                    end
                else
                    # ... this column _is_ constrained, double-distribute to columns/rows.
                    for (row′, _) in row_coeffs
                        !keep_constrained && haskey(dofmapping, row′) && continue
                        for (col′, _) in col_coeffs
                            !keep_constrained && haskey(dofmapping, col′) && continue
                            insert_sorted!(get!(Vector{Int}, sp′, row′), col′)
                        end
                    end
                end
            end
        end
    end

    # Insert new entries into the sparsity pattern
    for (row, colidxs) in sp′
        for col in colidxs
            add_entry!(sp, row, col)
        end
    end

    return sp
end

# Internal matrix instantiation for SparseMatrixCSC and Symmetric{SparseMatrixCSC}
function _create_matrix(::Type{SparseMatrixCSC{Tv, Ti}}, sp::AbstractSparsityPattern, sym::Bool) where {Tv, Ti}
    # 1. Setup colptr
    colptr = zeros(Ti, n_cols(sp) + 1)
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
    for (row, colidxs) in pairs(eachrow(sp))
        for col in colidxs
            sym && row > col && continue
            k = nextinds[col]
            rowval[k] = row
            nextinds[col] = k + 1
        end
    end
    @assert all(i -> nextinds[i] == colptr[i + 1], 1:n_cols(sp))
    S = SparseMatrixCSC(n_rows(sp), n_cols(sp), colptr, rowval, nzval)
    return S
end
