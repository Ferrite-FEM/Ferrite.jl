# This file contains utiltiies for working with (sparse) matrices and vectors.
# These methods can be overloaded by other array types.

struct SparsityError end
function Base.showerror(io::IO, ::SparsityError)
    print(io, "SparsityError: writing to an index outside the sparsity pattern is not allowed")
    return
end

"""
    addindex!(A::AbstractMatrix{T}, v::T, i::Integer, j::Integer, ::Val{atomic} = Val(false))
    addindex!(b::AbstractVector{T}, v::T, i::Integer, ::Val{atomic} = Val(false))

Equivalent to `A[i, j] += v` but more efficient. The optional `atomic` input controls
whether the operation should be performed atomically (i.e. concurrency-safe) or not.

`A[i, j] += v` is lowered to `A[i, j] = A[i, j] + v` which requires a double lookup of the
memory location for index `(i, j)` -- one time for the read, and one time for the write.
This method avoids the double lookup.

Zeros are ignored (i.e. if `iszero(v)`) by returning early. If the index `(i, j)` is not
existing in the sparsity pattern of `A` this method throws a `SparsityError`.

Fallback: `A[i, j] += v`.
"""
addindex!

function addindex!(A::AbstractMatrix{T}, v, i::Integer, j::Integer, ::Val{atomic} = Val(false)) where {T, atomic}
    return addindex!(A, T(v), Int(i), Int(j), Val{atomic}())
end
function addindex!(A::AbstractMatrix{T}, v::T, i::Int, j::Int, ::Val{atomic}) where {T, atomic}
    iszero(v) && return A
    if atomic
        A[i, j] += v
    else
        error("Atomic addindex! not supported for matrices.")
    end
    return A
end

function addindex!(b::AbstractVector{T}, v, i::Integer, ::Val{atomic} = Val(false)) where {T, atomic}
    return addindex!(b, T(v), Int(i), Val(atomic))
end

# Atomic accumulation primitive used for assembling with `atomic = true`.
#
# This is written with `llvmcall` since the built-in alternatives in Julia currently
# generate bad code for floating point addition: `Core.Intrinsics.atomic_pointermodify`
# (and thus `@atomic`) lowers `+` on floats to a compare-exchange loop with a non-inlined
# call to `+` inside, whereas this generates a single `atomicrmw fadd` instruction.
#
# Monotonic ordering is sufficient since no other memory is synchronized through these
# additions -- the task join at the end of a threaded assembly loop is the synchronization
# point that makes the accumulated values visible.
for (T, llvmT) in ((Float64, "double"), (Float32, "float"))
    ir = if VERSION >= v"1.12.0-DEV"
        """
        %rv = atomicrmw fadd ptr %0, $llvmT %1 monotonic
        ret void
        """
    else
        """
        %p = inttoptr i$(Sys.WORD_SIZE) %0 to $(llvmT)*
        %rv = atomicrmw fadd $(llvmT)* %p, $llvmT %1 monotonic
        ret void
        """
    end
    @eval @propagate_inbounds function _atomic_add!(x::Vector{$T}, v::$T, i::Int)
        @boundscheck checkbounds(x, i)
        GC.@preserve x begin
            p = pointer(x, i)
            Base.llvmcall($ir, Cvoid, Tuple{Ptr{$T}, $T}, p, v)
        end
        return
    end
end

# Accumulate `v` into `x[i]`, atomically if `atomic` is `Val(true)`. This is the only
# point where the atomic and non-atomic matrix assembly kernels differ.
@propagate_inbounds function addindex!(x::AbstractVector, v, i::Int, ::Val{atomic} = Val(false)) where {atomic}
    if atomic
        _atomic_add!(x, v, i)
    else
        x[i] += v
    end
    return
end

"""
    fillzero!(A::AbstractVecOrMat{T})

Fill the (stored) entries of the vector or matrix `A` with zeros.

Fallback: `fill!(A, zero(T))`.
"""
fillzero!(A)

function fillzero!(A::AbstractVecOrMat{T}) where {T}
    return fill!(A, zero(T))
end

##################################
## SparseArrays.SparseMatrixCSC ##
##################################

function addindex!(A::SparseMatrixCSC{Tv}, v::Tv, i::Int, j::Int, ::Val{atomic} = Val(false)) where {Tv, atomic}
    @boundscheck checkbounds(A, i, j)
    # Return early if v is 0
    iszero(v) && return A
    # Search column j for row i
    coljfirstk = Int(SparseArrays.getcolptr(A)[j])
    coljlastk = Int(SparseArrays.getcolptr(A)[j + 1] - 1)
    searchk = searchsortedfirst(rowvals(A), i, coljfirstk, coljlastk, Base.Order.Forward)
    if searchk <= coljlastk && rowvals(A)[searchk] == i
        # Column j contains entry A[i,j]. Update and return.
        nzs = nonzeros(A)
        addindex!(nzs, v, searchk, Val{atomic}())
        return A
    else
        # (i, j) not stored. Throw.
        throw(SparsityError())
    end
end

function fillzero!(A::AbstractSparseMatrix{T}) where {T}
    fill!(nonzeros(A), zero(T))
    return A
end
function fillzero!(A::Symmetric{T, <:AbstractSparseMatrix}) where {T}
    fillzero!(A.data)
    return A
end
