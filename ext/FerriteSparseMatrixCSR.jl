module FerriteSparseMatrixCSR

using Ferrite, SparseArrays, SparseMatricesCSR
import Ferrite: AbstractSparsityPattern, CSRAssembler, DofCoefficients, getnrows, getncols
import Base: @propagate_inbounds

# Could be generalized if https://github.com/JuliaSparse/SparseArrays.jl/pull/546 is merged
# See src/assembler.jl for the `@constprop` annotation (concrete return type for literal `atomic`).
Base.@constprop :aggressive function Ferrite.start_assemble(K::SparseMatrixCSR{<:Any, T, Ti}, f::Vector = T[]; fillzero::Bool = true, maxcelldofs_hint::Int = 0, atomic::Bool = false) where {T, Ti}
    Ferrite._check_atomic_eltype(atomic, T)
    fillzero && (Ferrite.fillzero!(K); Ferrite.fillzero!(f))
    return CSRAssembler{T, Ti, typeof(K), atomic, typeof(f), Vector{Int}}(K, f, zeros(Int, maxcelldofs_hint), zeros(Int, maxcelldofs_hint), zeros(Int, maxcelldofs_hint), zeros(Int, maxcelldofs_hint))
end

# The row/column roles are swapped compared to CSC: the row is the major index and the
# column the minor one, so the local matrix is handed to the shared kernel transposed
# (lazily; `PermutedDimsArray` would allocate on Julia 1.10). `sym` is always `false` since there is no symmetric CSR assembler; note that
# the shared kernel stores the `minor <= major` triangle, which is the *lower* triangle for
# CSR storage. Only `SparseMatrixCSR{1}` is supported, since the kernel indexes `rowptr`
# and `colval` directly (`colval` holds 1-based column indices only for `Bi == 1`).
@propagate_inbounds function Ferrite._assemble_inner!(
        K::SparseMatrixCSR{1}, Ke::AbstractMatrix,
        rowdofs::AbstractVector, sortedrowdofs::AbstractVector, rowpermutation::AbstractVector,
        coldofs::AbstractVector, sortedcoldofs::AbstractVector, colpermutation::AbstractVector,
        sym::Bool, atomic::Val = Val(false), rowoffset::Int = 0, coloffset::Int = 0
    )
    return Ferrite._assemble_compressed!(
        Ferrite.MajorIsRow(), K.rowptr, K.colval, K.nzval, size(K, 2), transpose(Ke),
        sortedrowdofs, rowpermutation, sortedcoldofs, colpermutation, false, atomic, rowoffset, coloffset
    )
end

@propagate_inbounds function Ferrite._assemble_inner_unsorted!(
        K::SparseMatrixCSR{1}, Ke::AbstractMatrix,
        rowdofs::AbstractVector, coldofs::AbstractVector,
        sym::Bool, atomic::Val = Val(false)
    )
    return Ferrite._assemble_compressed_unsorted!(
        Ferrite.MajorIsRow(), K.rowptr, K.colval, K.nzval, transpose(Ke),
        rowdofs, coldofs, false, atomic
    )
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
# entry has to be visited.
function Ferrite.add_inhomogeneities!(f::AbstractVector, K::SparseMatrixCSR{1}, columns::AbstractVector{<:Integer}, inhomogeneities::AbstractVector)
    return Ferrite._add_inhomogeneities_minors!(f, K, columns, inhomogeneities)
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
function Ferrite.addindex!(A::SparseMatrixCSR{Bi, Tv}, v::Tv, i::Int, j::Int, ::Val{atomic} = Val(false)) where {Bi, Tv, atomic}
    @boundscheck checkbounds(A, i, j)
    # Return early if v is 0
    iszero(v) && return A
    # Search row i for column j
    nzr = nzrange(A, i)
    stored_j = j - (1 - Bi)
    searchk = searchsortedfirst(A.colval, stored_j, first(nzr), last(nzr), Base.Order.Forward)
    if searchk <= last(nzr) && A.colval[searchk] == stored_j
        # Row i contains entry A[i,j]. Update and return.
        Ferrite.addindex!(A.nzval, v, searchk, Val{atomic}())
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
