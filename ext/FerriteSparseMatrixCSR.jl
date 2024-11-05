module FerriteSparseMatrixCSR

using Ferrite, SparseArrays, SparseMatricesCSR
import Ferrite: AbstractSparsityPattern, CSRAssembler
import Base: @propagate_inbounds

#FIXME https://github.com/JuliaSparse/SparseArrays.jl/pull/546
function Ferrite.start_assemble(K::SparseMatrixCSR{<:Any, T}, f::Vector = T[]; fillzero::Bool = true, maxcelldofs_hint::Int = 0) where {T}
    fillzero && (Ferrite.fillzero!(K); Ferrite.fillzero!(f))
    return CSRAssembler(K, f, zeros(Int, maxcelldofs_hint), zeros(Int, maxcelldofs_hint))
end

@propagate_inbounds function Ferrite._assemble_inner!(K::SparseMatrixCSR, Ke::AbstractMatrix, dofs::AbstractVector, sorteddofs::AbstractVector, permutation::AbstractVector, sym::Bool)
    current_row = 1
    ld = length(dofs)
    return @inbounds for Krow in sorteddofs
        maxlookups = sym ? current_row : ld
        Kerow = permutation[current_row]
        ci = 1 # col index pointer for the local matrix
        Ci = 1 # col index pointer for the global matrix
        nzr = nzrange(K, Krow)
        while Ci <= length(nzr) && ci <= maxlookups
            C = nzr[Ci]
            Kcol = K.colval[C]
            Kecol = permutation[ci]
            val = Ke[Kerow, Kecol]
            if Kcol == dofs[Kecol]
                # Match: add the value (if non-zero) and advance the pointers
                if !iszero(val)
                    K.nzval[C] += val
                end
                ci += 1
                Ci += 1
            elseif Kcol < dofs[Kecol]
                # No match yet: advance the global matrix row pointer
                Ci += 1
            else # Kcol > dofs[Kecol]
                # No match: no entry exist in the global matrix for this row. This is
                # allowed as long as the value which would have been inserted is zero.
                iszero(val) || Ferrite._missing_sparsity_pattern_error(Krow, Kcol)
                # Advance the local matrix row pointer
                ci += 1
            end
        end
        # Make sure that remaining entries in this column of the local matrix are all zero
        for i in ci:maxlookups
            if !iszero(Ke[Kerow, permutation[i]])
                Ferrite._missing_sparsity_pattern_error(Krow, sorteddofs[i])
            end
        end
        current_row += 1
    end
end

function Ferrite.zero_out_rows!(K::SparseMatrixCSR, ch::ConstraintHandler) # can be removed in 0.7 with #24711 merged
    @debug @assert issorted(ch.prescribed_dofs)
    for row in ch.prescribed_dofs
        r = nzrange(K, row)
        K.nzval[r] .= 0.0
    end
    return
end

function Ferrite.zero_out_columns!(K::SparseMatrixCSR, ch::ConstraintHandler)
    colval = K.colval
    nzval = K.nzval
    return @inbounds for i in eachindex(colval, nzval)
        if haskey(ch.dofmapping, colval[i])
            nzval[i] = 0
        end
    end
end

function Ferrite.allocate_matrix(::Type{SparseMatrixCSR}, sp::AbstractSparsityPattern)
    return _allocate_matrix(SparseMatrixCSR{1, Float64, Int64}, sp)
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

end
