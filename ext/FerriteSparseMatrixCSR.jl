module FerriteSparseMatrixCSR

using Ferrite, SparseArrays, SparseMatricesCSR
import Base: @propagate_inbounds
import Ferrite: Symmetric

@propagate_inbounds function Ferrite._assemble_inner!(K::SparseMatrixCSR, Ke::AbstractMatrix, dofs::AbstractVector, sorteddofs::AbstractVector, permutation::AbstractVector, sym::Bool)
    current_row = 1
    ld = length(dofs)
    @inbounds for Krow in sorteddofs
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
                iszero(val) || _missing_sparsity_pattern_error(Krow, Kcol)
                # Advance the local matrix row pointer
                ci += 1
            end
        end
        # Make sure that remaining entries in this column of the local matrix are all zero
        for i in ci:maxlookups
            if !iszero(Ke[Kerow, permutation[i]])
                _missing_sparsity_pattern_error(Krow, sorteddofs[i])
            end
        end
        current_row += 1
    end
end

function Ferrite.add_inhomogeneities!(K::SparseMatrixCSR, f::AbstractVector, inhomogeneities::AbstractVector, prescribed_dofs::AbstractVector, dofmapping, sym::Bool)
    @inbounds for i in 1:length(inhomogeneities)
        d = prescribed_dofs[i]
        for j in nzrange(K, d)
            c = K.colval[j]
            sym && c > d && break # don't look below diagonal
            if (i = get(dofmapping, c, 0); i != 0)
                f[d] -= inhomogeneities[i] * K.nzval[j]
            end
        end
    end
    if sym
        error("Symmetric inhomogeneities not supported for SparseMatrixCSR")
        # In the symmetric case, for a constrained dof `d`, we handle the contribution
        # from `K[1:d, d]` in the loop above, but we are still missing the contribution
        # from `K[(d+1):size(K,1), d]`. These values are not stored, but since the
        # matrix is symmetric we can instead use `K[d, (d+1):size(K,1)]`. Looping over
        # rows is slow, so loop over all columns again, and check if the row is a
        # constrained row.
        # @inbounds for col in 1:size(K, 2)
        #     for ri in nzrange(K, col)
        #         row = K.rowval[ri]
        #         row >= col && break
        #         if (i = get(dofmapping, row, 0); i != 0)
        #             f[col] -= inhomogeneities[i] * K.nzval[ri]
        #         end
        #     end
        # end
    end
end

function Ferrite.zero_out_rows!(K::SparseMatrixCSR, ch::ConstraintHandler) # can be removed in 0.7 with #24711 merged
    @debug @assert issorted(ch.prescribed_dofs)
    for row in ch.prescribed_dofs
        r = nzrange(K, row)
        K.nzval[r] .= 0.0
    end
end

function Ferrite.zero_out_columns!(K::SparseMatrixCSR, ch::ConstraintHandler)
    colval = K.colval
    nzval = K.nzval
    @inbounds for i in eachindex(colval, nzval)
        if haskey(ch.dofmapping, colval[i])
            nzval[i] = 0
        end
    end
end

end
