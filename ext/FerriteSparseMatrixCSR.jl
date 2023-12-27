module FerriteSparseMatrixCSR

using Ferrite, SparseArrays, SparseMatricesCSR
import Base: @propagate_inbounds

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
