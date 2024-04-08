# This file contains utiltiies for working with (sparse) matrices and vectors.
# These methods can be overloaded by other array types.

struct SparsityError{IDX<:Tuple}
    indices::IDX
end
SparsityError(args::Int...) = SparsityError(args)

function Base.showerror(io::IO, se::SparsityError)
    print(io, "SparsityError: writing to index ", se.indices, ", which is outside the sparsity pattern, is not allowed")
    return
end

"""
    addindex!(A::AbstractMatrix{T}, v::T, i::Int, j::Int)
    addindex!(b::AbstractVector{T}, v::T, i::Int)

Equivalent to `A[i, j] += v` but more efficient.

`A[i, j] += v` is lowered to `A[i, j] = A[i, j] + v` which requires a double lookup of the
memory location for index `(i, j)` -- one time for the read, and one time for the write.
This method avoids the double lookup.

Zeros are ignored (i.e. if `iszero(v)`) by returning early. If the index `(i, j)` is not
existing in the sparsity pattern of `A` this method throws a `SparsityError`.

Fallback: `A[i, j] += v`.
"""
addindex!

function addindex!(A::AbstractMatrix{T}, v, i::Integer, j::Integer) where T
    return addindex!(A, T(v), Int(i), Int(j))
end
function addindex!(A::AbstractMatrix{T}, v::T, i::Int, j::Int) where T
    iszero(v) && return A
    A[i, j] += v
    return A
end

function addindex!(b::AbstractVector{T}, v, i::Integer) where T
    return addindex!(b, T(v), Int(i))
end
function addindex!(b::AbstractVector{T}, v::T, i::Int) where T
    iszero(v) && return b
    b[i] += v
    return b
end

"""
    fillzero!(A::AbstractVecOrMat{T})

Fill the (stored) entries of the vector or matrix `A` with zeros.

Fallback: `fill!(A, zero(T))`.
"""
fillzero!(A)

function fillzero!(A::AbstractVecOrMat{T}) where T
    return fill!(A, zero(T))
end

##################################
## SparseArrays.SparseMatrixCSC ##
##################################

function addindex!(A::SparseMatrixCSC{Tv}, v::Tv, i::Int, j::Int) where Tv
    @boundscheck checkbounds(A, i, j)
    # Return early if v is 0
    iszero(v) && return A
    # Search column j for row i
    coljfirstk = Int(SparseArrays.getcolptr(A)[j])
    coljlastk = Int(SparseArrays.getcolptr(A)[j+1] - 1)
    searchk = searchsortedfirst(rowvals(A), i, coljfirstk, coljlastk, Base.Order.Forward)
    if searchk <= coljlastk && rowvals(A)[searchk] == i
        # Column j contains entry A[i,j]. Update and return.
        nonzeros(A)[searchk] += v
        return A
    else
        # (i, j) not stored. Throw.
        throw(SparsityError(i, j))
    end
end

function fillzero!(A::SparseMatrixCSC{T}) where T
    fill!(nonzeros(A), zero(T))
    return A
end
function fillzero!(A::Symmetric{T,<:SparseMatrixCSC}) where T
    fillzero!(A.data)
    return A
end

# Compat and convenience layer around SparseArrays.spzeros! (SparseArrays.jl#284, SparseArrays.jl#315)
function spzeros!!(::Type{Tv}, I::Vector{Ti}, J::Vector{Ti}, m::Int, n::Int) where {Tv, Ti <: Integer}
    @assert length(I) == length(J)
    @static if isdefined(SparseArrays, :spzeros!)
        klasttouch = Vector{Ti}(undef, n)
        csrrowptr  = Vector{Ti}(undef, m + 1)
        csrcolval  = Vector{Ti}(undef, length(I))
        S = SparseArrays.spzeros!(Tv, I, J, m, n, klasttouch, csrrowptr, csrcolval, I, J)
    else
        S = sparse(I, J, zeros(Tv, length(I)), m, n)
    end
    return S
end
