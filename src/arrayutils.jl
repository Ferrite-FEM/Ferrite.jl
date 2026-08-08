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
        error("Atomic addindex! not supported for matrices.")
    else
        A[i, j] += v
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
# call to `+` inside, whereas this generates a single `atomicrmw fadd` instruction. (For
# Float16 there is no atomic add instruction on typical CPUs and LLVM legalizes the
# `atomicrmw fadd` to a compare-exchange loop.)
#
# Monotonic ordering is sufficient since no other memory is synchronized through these
# additions -- the task join at the end of a threaded assembly loop is the synchronization
# point that makes the accumulated values visible.
for (T, llvmT) in ((Float64, "double"), (Float32, "float"), (Float16, "half"))
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
    @eval @inline function _atomic_fadd!(p::Ptr{$T}, v::$T)
        Base.llvmcall($ir, Cvoid, Tuple{Ptr{$T}, $T}, p, v)
        return
    end
end

# Eltypes supported by atomic assembly (see `start_assemble`).
const AtomicEltypes = Union{Float16, Float32, Float64, Complex{Float16}, Complex{Float32}, Complex{Float64}}

@propagate_inbounds function _atomic_add!(x::Vector{T}, v::T, i::Int) where {T <: Union{Float16, Float32, Float64}}
    @boundscheck checkbounds(x, i)
    GC.@preserve x _atomic_fadd!(pointer(x, i), v)
    return
end

# Complex addition is component-wise and assembly only accumulates -- no task reads the
# values until after the task join -- so the real and imaginary parts can be accumulated
# with two independent atomic additions.
@propagate_inbounds function _atomic_add!(x::Vector{Complex{T}}, v::Complex{T}, i::Int) where {T <: Union{Float16, Float32, Float64}}
    @boundscheck checkbounds(x, i)
    GC.@preserve x begin
        p = convert(Ptr{T}, pointer(x, i))
        _atomic_fadd!(p, real(v))
        _atomic_fadd!(p + sizeof(T), imag(v))
    end
    return
end

# Accumulate `v` into `x[i]`, atomically if `atomic` is `Val(true)`. This is the only
# point where the atomic and non-atomic assembly kernels differ.
@propagate_inbounds function addindex!(x::AbstractVector, v, i::Int, ::Val{atomic} = Val(false)) where {atomic}
    if atomic
        _atomic_add!(x, convert(eltype(x), v), i)
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
