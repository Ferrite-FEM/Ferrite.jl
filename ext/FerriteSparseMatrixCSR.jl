module FerriteSparseMatrixCSR

using Ferrite, SparseArrays, SparseMatricesCSR
import Ferrite: AbstractSparsityPattern, CSRAssembler, getnrows, getncols
import Base: @propagate_inbounds

# Could be generalized if https://github.com/JuliaSparse/SparseArrays.jl/pull/546 is merged
# See src/assembler.jl for the `@constprop` annotation (concrete return type for literal `atomic`).
Base.@constprop :aggressive function Ferrite.start_assemble(K::SparseMatrixCSR{<:Any, T, Ti}, f::Vector = T[]; fillzero::Bool = true, maxcelldofs_hint::Int = 0, atomic::Bool = false) where {T, Ti}
    Ferrite._check_atomic_eltype(atomic, T)
    fillzero && (Ferrite.fillzero!(K); Ferrite.fillzero!(f))
    return CSRAssembler{T, Ti, typeof(K), atomic}(K, f, zeros(Int, maxcelldofs_hint), zeros(Int, maxcelldofs_hint), zeros(Int, maxcelldofs_hint), zeros(Int, maxcelldofs_hint))
end

@propagate_inbounds function Ferrite._assemble_inner!(
        K::SparseMatrixCSR, Ke::AbstractMatrix,
        rowdofs::AbstractVector, sortedrowdofs::AbstractVector, rowpermutation::AbstractVector,
        coldofs::AbstractVector, sortedcoldofs::AbstractVector, colpermutation::AbstractVector,
        sym::Bool, atomic::Val = Val(false)
    )
    current_row = 1
    ld = length(coldofs)
    ncols = size(K, 2)
    threshold = Ferrite.SPARSE_COLUMN_SEARCH_RATIO * ld
    return @inbounds for Krow in sortedrowdofs
        maxlookups = sym ? current_row : ld
        Kerow = rowpermutation[current_row]
        ci = 1 # col index pointer for the local matrix
        Ci = 1 # col index pointer for the global matrix
        nzr = nzrange(K, Krow)
        # Fast paths for rows holding many entries per local column, mirroring the
        # corresponding column fast paths in `Ferrite._assemble_inner!` for
        # `SparseMatrixCSC` in src/assembler.jl
        if length(nzr) == ncols
            offset = first(nzr) - 1
            for ci in 1:maxlookups
                val = Ke[Kerow, colpermutation[ci]]
                iszero(val) || Ferrite.addindex!(K.nzval, val, offset + sortedcoldofs[ci], atomic)
            end
            current_row += 1
            continue
        end
        if length(nzr) > (sym ? Ferrite.SPARSE_COLUMN_SEARCH_RATIO * maxlookups : threshold)
            lo = first(nzr)
            hi = last(nzr)
            for ci in 1:maxlookups
                Kecol_dof = sortedcoldofs[ci]
                C = searchsortedfirst(K.colval, Kecol_dof, lo, hi, Base.Order.Forward)
                if C <= hi && K.colval[C] == Kecol_dof
                    val = Ke[Kerow, colpermutation[ci]]
                    iszero(val) || Ferrite.addindex!(K.nzval, val, C, atomic)
                    lo = C + 1
                else
                    # No entry exists in the global matrix for this column, which is
                    # allowed as long as the value which would have been inserted is zero.
                    iszero(Ke[Kerow, colpermutation[ci]]) || Ferrite._missing_sparsity_pattern_error(Krow, Kecol_dof)
                    lo = C
                end
            end
            current_row += 1
            continue
        end
        while Ci <= length(nzr) && ci <= maxlookups
            C = nzr[Ci]
            Kcol = K.colval[C]
            Kecol_dof = sortedcoldofs[ci]
            if Kcol == Kecol_dof
                # Match: add the value (if non-zero) and advance the pointers
                val = Ke[Kerow, colpermutation[ci]]
                if !iszero(val)
                    Ferrite.addindex!(K.nzval, val, C, atomic)
                end
                ci += 1
                Ci += 1
            elseif Kcol < Kecol_dof
                # No match yet: advance the global matrix row pointer
                Ci += 1
            else # Kcol > Kecol_dof
                # No match: no entry exist in the global matrix for this row. This is
                # allowed as long as the value which would have been inserted is zero.
                iszero(Ke[Kerow, colpermutation[ci]]) || Ferrite._missing_sparsity_pattern_error(Krow, Kecol_dof)
                # Advance the local matrix row pointer
                ci += 1
            end
        end
        # Make sure that remaining entries in this column of the local matrix are all zero
        for i in ci:maxlookups
            if !iszero(Ke[Kerow, colpermutation[i]])
                Ferrite._missing_sparsity_pattern_error(Krow, sortedcoldofs[i])
            end
        end
        current_row += 1
    end
end

function Ferrite.zero_out_rows!(K::SparseMatrixCSR, ch::ConstraintHandler)
    @debug @assert issorted(ch.prescribed_dofs)
    for row in ch.prescribed_dofs
        r = nzrange(K, row)
        K.nzval[r] .= 0.0
    end
    return
end

function Ferrite.zero_out_columns!(K::SparseMatrixCSR, ch::ConstraintHandler)
    @boundscheck checkbounds(ch.isconstrained, axes(K, 2))
    colval = K.colval
    nzval = K.nzval
    return @inbounds for (i, col) in pairs(colval)
        if ch.isconstrained[col]
            nzval[i] = 0
        end
    end
end

function Ferrite.allocate_matrix(::Type{SparseMatrixCSR}, sp::AbstractSparsityPattern)
    return Ferrite.allocate_matrix(SparseMatrixCSR{1, Float64, Int64}, sp)
end

function Ferrite.allocate_matrix(::Type{SparseMatrixCSR{1, Tv, Ti}}, sp::AbstractSparsityPattern) where {Tv, Ti}
    return _allocate_matrix(SparseMatrixCSR{1, Tv, Ti}, sp, false)
end

# Copy one pattern row into `dest` starting at `k` (only used by _allocate_matrix below):
# bulk copy for `AbstractVector` rows (e.g. `SparsityPattern`), iteration fallback for other
# iterables (e.g. `BlockSparsityPattern`'s lazy rows).
function _copyto!(dest::Vector, k::Int, colidxs::AbstractVector)
    copyto!(dest, k, colidxs, 1, length(colidxs))
    return k + length(colidxs)
end
function _copyto!(dest::Vector, k::Int, colidxs)
    for col in colidxs
        dest[k] = col
        k += 1
    end
    return k
end

# The pattern rows are exactly CSR's rows: `eachrow` hands out the sorted column indices
# (for `SparsityPattern` this sorts the rows lazily, a no-op if already sorted), so `rowptr`
# follows from the row lengths and each row is copied straight into `colval`.
function _allocate_matrix(::Type{SparseMatrixCSR{1, Tv, Ti}}, sp::AbstractSparsityPattern, sym::Bool) where {Tv, Ti}
    sym && throw(ArgumentError("Symmetric SparseMatrixCSR is not supported"))
    nrows = Ferrite.getnrows(sp)
    # 1. Setup rowptr
    rowptr = Vector{Ti}(undef, nrows + 1)
    rowptr[1] = 1
    for (row, colidxs) in enumerate(Ferrite.eachrow(sp))
        rowptr[row + 1] = rowptr[row] + length(colidxs)
    end
    nnz = Int(rowptr[end]) - 1
    # 2. Allocate colval and nzval now that nnz is known
    colval = Vector{Ti}(undef, nnz)
    nzval = zeros(Tv, nnz)
    # 3. Populate colval
    _fill_colval!(colval, rowptr, sp)
    return SparseMatrixCSR{1}(nrows, Ferrite.getncols(sp), rowptr, colval, nzval)
end

# Generic AbstractSparsityPattern: the interface only promises whole-pattern row iteration
# (rows may be lazy generators, and implementations need not support concurrent row
# access), so the rows are copied serially in iteration order.
function _fill_colval!(colval::Vector, rowptr::Vector, sp::AbstractSparsityPattern)
    k = 1
    for colidxs in Ferrite.eachrow(sp)
        k = _copyto!(colval, k, colidxs)
    end
    # The copy pass must consume exactly the row lengths that built rowptr
    @assert k == rowptr[end]
    return
end

# SparsityPattern: the rows are random-access views of disjoint slices, so each chunk of
# rows is copied concurrently into its disjoint slice of colval.
function _fill_colval!(colval::Vector{Ti}, rowptr::Vector{Ti}, sp::Ferrite.SparsityPattern) where {Ti}
    Ferrite._ensure_sorted!(sp)
    @sync for rowrange in Ferrite._task_chunks(Ferrite.getnrows(sp))
        Threads.@spawn _fill_colval_chunk!(colval, rowptr, sp, rowrange)
    end
    return
end

function _fill_colval_chunk!(colval::Vector{Ti}, rowptr::Vector{Ti}, sp::Ferrite.SparsityPattern, rowrange::UnitRange{Int}) where {Ti}
    @inbounds for row in rowrange
        colidxs = Ferrite._row_view(sp, row)
        copyto!(colval, Int(rowptr[row]), colidxs, 1, length(colidxs))
    end
    return
end

end
