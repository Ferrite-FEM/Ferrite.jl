module FerriteSparseMatrixCSR

using Ferrite, SparseArrays, SparseMatricesCSR
import Ferrite: AbstractSparsityPattern, CSRAssembler, DofCoefficients, getnrows, getncols
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

###########################################################
## Constraint application (see the devdocs on assembly)  ##
###########################################################

# CSR is the mirror image of CSC: the rows are stored contiguously ("the majors") and the
# stored column indices are the minors. That is all the shared kernels in Ferrite need to know,
# so every method below is a one-liner picking the major or the minor variant.
#
# Restricted to `SparseMatrixCSR{1}` throughout: `colvals` returns the raw index array, which
# for `Bi = 0` holds 0-based column indices, and the kernels index it as a 1-based dof number.
Ferrite.minor_indices(K::SparseMatrixCSR{1}) = colvals(K)

function Ferrite.zero_out_rows!(K::SparseMatrixCSR{1}, rows::AbstractVector{<:Integer}, ::AbstractVector{Bool})
    return Ferrite._zero_out_majors!(K, rows)
end

function Ferrite.zero_out_columns!(K::SparseMatrixCSR{1}, ::AbstractVector{<:Integer}, mask::AbstractVector{Bool})
    @boundscheck checkbounds(mask, axes(K, 2))
    return Ferrite._zero_out_minors!(K, mask)
end

# The prescribed columns are minors here, so they are spread over all rows and every stored
# entry has to be visited; the inhomogeneities are scattered into a dense vector for lookup.
function Ferrite.add_inhomogeneities!(f::AbstractVector, K::SparseMatrixCSR{1}, columns::AbstractVector{<:Integer}, inhomogeneities::AbstractVector)
    g = Ferrite._dense_inhomogeneities(eltype(K), columns, inhomogeneities, size(K, 2))
    return Ferrite._add_inhomogeneities_minors!(f, K, g)
end

function Ferrite.condense_into!(
        Kdst::AbstractMatrix, K::SparseMatrixCSR{1}, rowoffset::Int, coloffset::Int,
        dofcoefficients::Vector, dofmapping::Dict{<:Integer, <:Integer},
    )
    return Ferrite._condense_majors!(Kdst, K, Ferrite.MajorIsRow(), rowoffset, coloffset, dofcoefficients, dofmapping)
end

function Ferrite._condense!(K::SparseMatrixCSR{1}, f::AbstractVector, dofcoefficients::Vector{Union{Nothing, DofCoefficients{T, Ti}}}, dofmapping::Dict{<:Integer, <:Integer}, sym::Bool = false) where {T, Ti}
    return Ferrite._condense_sparse!(K, f, dofcoefficients, dofmapping, sym)
end

# Mirror of `Ferrite.addindex!(::SparseMatrixCSC, ...)` in src/arrayutils.jl. Needed to write
# the affine contributions of `condense_into!`/`_condense_local!` into a CSR matrix, and to
# assemble into CSR blocks of a blocked matrix.
function Ferrite.addindex!(A::SparseMatrixCSR{1, Tv}, v::Tv, i::Int, j::Int, ::Val{atomic} = Val(false)) where {Tv, atomic}
    @boundscheck checkbounds(A, i, j)
    # Return early if v is 0
    iszero(v) && return A
    # Search row i for column j
    nzr = nzrange(A, i)
    searchk = searchsortedfirst(colvals(A), j, first(nzr), last(nzr), Base.Order.Forward)
    if searchk <= last(nzr) && colvals(A)[searchk] == j
        # Row i contains entry A[i,j]. Update and return.
        Ferrite.addindex!(nonzeros(A), v, searchk, Val{atomic}())
        return A
    else
        # (i, j) not stored. Throw.
        throw(Ferrite.SparsityError())
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
    nnz = rowptr[end] - 1
    # 2. Allocate colval and nzval now that nnz is known
    colval = Vector{Ti}(undef, nnz)
    nzval = zeros(Tv, nnz)
    # 3. Populate colval row by row
    k = 1
    for colidxs in Ferrite.eachrow(sp)
        k = _copyto!(colval, k, colidxs)
    end
    return SparseMatrixCSR{1}(nrows, Ferrite.getncols(sp), rowptr, colval, nzval)
end

end
