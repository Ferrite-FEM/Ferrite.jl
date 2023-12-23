module FerriteSparseMatrixCSR

using Ferrite, SparseMatricesCSR
import Base: @propagate_inbounds

@propagate_inbounds function Ferrite._assemble_inner!(K::SparseMatrixCSR, Ke::AbstractMatrix, dofs::AbstractVector, sorteddofs::AbstractVector, permutation::AbstractVector, sym::Bool)
    current_row = 1
    @inbounds for Krow in sorteddofs
        maxlookups = sym ? current_row : ld
        Kerow = permutation[current_row]
        ci = 1 # col index pointer for the local matrix
        Ci = 1 # col index pointer for the global matrix
        nzr = nzrange(K, Krow)
        while Ci <= length(nzr) && ci <= maxlookups
            C = nzr[Ci]
            Kcol = K.colval[C]
            Kecol = permutation[ri]
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
        for i in ri:maxlookups
            if !iszero(Ke[Kerow, permutation[i]])
                _missing_sparsity_pattern_error(Krow, sorteddofs[i])
            end
        end
        current_row += 1
    end
end

end
