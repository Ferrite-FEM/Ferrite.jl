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
                iszero(val) || Ferrite._addindex!(K.nzval, offset + sortedcoldofs[ci], val, atomic)
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
                    iszero(val) || Ferrite._addindex!(K.nzval, C, val, atomic)
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

function _allocate_matrix(::Type{SparseMatrixCSR{1, Tv, Ti}}, sp::AbstractSparsityPattern, sym::Bool) where {Tv, Ti}
    # 1. Setup rowptr
    rowptr = zeros(Ti, Ferrite.getnrows(sp) + 1)
    rowptr[1] = 1
    for (row, colidxs) in enumerate(Ferrite.eachrow(sp))
        for col in colidxs
            sym && row > col && continue
            rowptr[row + 1] += 1
        end
    end
    cumsum!(rowptr, rowptr)
    nnz = rowptr[end] - 1
    # 2. Allocate colval and nzval now that nnz is known
    colval = Vector{Ti}(undef, nnz)
    nzval = zeros(Tv, nnz)
    # 3. Populate colval.
    k = 1
    for (row, colidxs) in zip(1:Ferrite.getnrows(sp), Ferrite.eachrow(sp)) # pairs(eachrow(sp))
        for col in colidxs
            sym && row > col && continue
            colval[k] = col
            k += 1
        end
    end
    S = SparseMatrixCSR{1}(Ferrite.getnrows(sp), Ferrite.getncols(sp), rowptr, colval, nzval)
    return S
end

# Specialized for SparsityPattern: the pattern is row-based like CSR, so after sorting the rows
# (lazy, no-op if already sorted) the raw row slices can be copied straight into colval, skipping
# the view-based double iteration of the generic method above.
function _allocate_matrix(::Type{SparseMatrixCSR{1, Tv, Ti}}, sp::Ferrite.SparsityPattern, sym::Bool) where {Tv, Ti}
    sym && throw(ArgumentError("Symmetric SparseMatrixCSR is not supported"))
    Ferrite._ensure_sorted!(sp) # CSR requires sorted colval within each row
    b = sp.buffer
    nrows = getnrows(sp)
    rowptr = Vector{Ti}(undef, nrows + 1)
    rowptr[1] = 1
    @inbounds for row in 1:nrows
        rowptr[row + 1] = rowptr[row] + b.indices[row].ncurrent
    end
    nnz = rowptr[end] - 1
    colval = Vector{Ti}(undef, nnz)
    k = 1
    @inbounds for row in 1:nrows
        r = b.indices[row]
        copyto!(colval, k, b.data, r.start, r.ncurrent)
        k += r.ncurrent
    end
    nzval = zeros(Tv, nnz)
    return SparseMatrixCSR{1}(nrows, getncols(sp), rowptr, colval, nzval)
end

end
