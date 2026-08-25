# Assembly into sparse matrices living on a device. The assemblers created here carry no
# sorting scratch (`ST === Nothing`), so a single instance can be shared by all workers and
# the kernels look up every entry with a binary search, see
# `Ferrite._assemble_compressed_unsorted!`.

# `KA.@atomic` accumulates into a single element, so there is no way to update the real and
# the imaginary part of a complex number in one atomic operation.
function _check_device_atomic_eltype(atomic::Bool, ::Type{Tv}) where {Tv}
    Ferrite._check_atomic_eltype(atomic, Tv)
    if atomic && Tv <: Complex
        throw(ArgumentError("atomic assembly on device is not supported for complex value types, got $Tv"))
    end
    return
end

# The `@constprop :aggressive` propagates a literal `atomic` keyword argument into the
# `atomic` type parameter, see the corresponding host methods in src/assembler.jl.
Base.@constprop :aggressive function Ferrite.start_assemble(
        K::AbstractGPUSparseMatrixCSC{Tv, Ti},
        f::AbstractGPUVector{Tv} = KA.zeros(get_backend(nonzeros(K)), Tv, 0);
        fillzero::Bool = true, atomic::Bool = false
    ) where {Tv, Ti}
    _check_device_atomic_eltype(atomic, Tv)
    fillzero && (Ferrite.fillzero!(K); Ferrite.fillzero!(f))
    return Ferrite.CSCAssembler{Tv, Ti, typeof(K), atomic, typeof(f), Nothing}(K, f, nothing, nothing, nothing, nothing)
end

Base.@constprop :aggressive function Ferrite.start_assemble(
        K::AbstractGPUSparseMatrixCSR{Tv, Ti},
        f::AbstractGPUVector{Tv} = KA.zeros(get_backend(nonzeros(K)), Tv, 0);
        fillzero::Bool = true, atomic::Bool = false
    ) where {Tv, Ti}
    _check_device_atomic_eltype(atomic, Tv)
    fillzero && (Ferrite.fillzero!(K); Ferrite.fillzero!(f))
    return Ferrite.CSRAssembler{Tv, Ti, typeof(K), atomic, typeof(f), Nothing}(K, f, nothing, nothing, nothing, nothing)
end

@propagate_inbounds function Ferrite._assemble_inner_unsorted!(
        K::GPUArrays.GPUSparseDeviceMatrixCSC, Ke::AbstractMatrix,
        rowdofs::AbstractVector, coldofs::AbstractVector,
        sym::Bool, atomic::Val = Val(false)
    )
    return Ferrite._assemble_compressed_unsorted!(
        Ferrite.MajorIsColumn(), SparseArrays.getcolptr(K), rowvals(K), nonzeros(K), Ke,
        coldofs, rowdofs, sym, atomic
    )
end

# Row and column roles are swapped compared to CSC, so the local matrix is handed to the
# shared kernel transposed (a lazy view), mirroring the host method in
# ext/FerriteSparseMatrixCSR.jl. That one uses `PermutedDimsArray`, which does not compile
# for CUDA, so `transpose` is used here (equivalent for the scalar `Ke` of an assembler).
@propagate_inbounds function Ferrite._assemble_inner_unsorted!(
        K::GPUArrays.GPUSparseDeviceMatrixCSR, Ke::AbstractMatrix,
        rowdofs::AbstractVector, coldofs::AbstractVector,
        sym::Bool, atomic::Val = Val(false)
    )
    return Ferrite._assemble_compressed_unsorted!(
        Ferrite.MajorIsRow(), K.rowPtr, K.colVal, K.nzVal, transpose(Ke),
        rowdofs, coldofs, false, atomic
    )
end

# Atomic accumulation for device vectors, and for `Array` under the `KA.CPU` backend. Plain
# `Vector`s keep the more specific `llvmcall` based method from src/arrayutils.jl.
@propagate_inbounds function Ferrite._atomic_add!(x::AbstractVector{T}, v::T, i::Int) where {T <: Union{Float16, Float32, Float64}}
    @boundscheck checkbounds(x, i)
    KA.@atomic x[i] += v
    return
end
