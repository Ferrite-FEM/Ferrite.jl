# Assembly into sparse matrices living on a device, see `Ferrite._device_assembler`.

# The `@constprop :aggressive` propagates a literal `atomic` keyword argument into the
# `atomic` type parameter, see the corresponding host methods in src/assembler.jl.
Base.@constprop :aggressive function Ferrite.start_assemble(
        K::AbstractGPUSparseMatrixCSC{Tv, Ti},
        f::AbstractGPUVector{Tv} = KA.zeros(get_backend(nonzeros(K)), Tv, 0);
        fillzero::Bool = true, atomic::Bool = false
    ) where {Tv, Ti}
    return Ferrite._device_assembler(Ferrite.CSCAssembler, K, f, fillzero, Val(atomic))
end

Base.@constprop :aggressive function Ferrite.start_assemble(
        K::AbstractGPUSparseMatrixCSR{Tv, Ti},
        f::AbstractGPUVector{Tv} = KA.zeros(get_backend(nonzeros(K)), Tv, 0);
        fillzero::Bool = true, atomic::Bool = false
    ) where {Tv, Ti}
    return Ferrite._device_assembler(Ferrite.CSRAssembler, K, f, fillzero, Val(atomic))
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
