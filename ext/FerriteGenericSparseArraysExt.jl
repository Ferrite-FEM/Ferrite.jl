module FerriteGenericSparseArraysExt

# GenericSparseArrays.jl provides sparse matrices whose storage vectors can live on any
# backend. This makes device assembly available for backends without a sparse matrix library
# of their own (Metal, in particular), and it is exercised on the `KernelAbstractions.CPU`
# backend in the regular test suite.

using Ferrite, GenericSparseArrays, SparseArrays

import Base: @propagate_inbounds

import Adapt: Adapt, adapt
import GPUArrays
import GenericSparseArrays: colvals, getrowptr
import KernelAbstractions as KA
import KernelAbstractions: get_backend

# ---------------- matrix allocation ------------------------------------

# The matrix is allocated on the host; move it to a device with `adapt(backend, K)`.
function Ferrite.allocate_matrix(::Type{<:GenericSparseMatrixCSC{Tv, Ti}}, dh::DofHandler) where {Tv, Ti}
    return GenericSparseMatrixCSC(allocate_matrix(SparseMatrixCSC{Tv, Ti}, dh))
end

function Ferrite.allocate_matrix(::Type{<:GenericSparseMatrixCSR{Tv, Ti}}, dh::DofHandler) where {Tv, Ti}
    return GenericSparseMatrixCSR(allocate_matrix(SparseMatrixCSC{Tv, Ti}, dh))
end

# ---------------- assembly ---------------------------------------------

# These always create a scratch free assembler (see `Ferrite._device_assembler`), also when
# the storage vectors are plain `Vector`s: these matrices exist to be assembled from many
# concurrent workers, and serial host assembly is better served by `SparseMatrixCSC`.
# The `@constprop :aggressive` propagates a literal `atomic` keyword argument into the
# `atomic` type parameter, see the corresponding host methods in src/assembler.jl.
Base.@constprop :aggressive function Ferrite.start_assemble(
        K::GenericSparseMatrixCSC{Tv, Ti},
        f::AbstractVector{Tv} = KA.zeros(get_backend(nonzeros(K)), Tv, 0);
        fillzero::Bool = true, atomic::Bool = false
    ) where {Tv, Ti}
    return Ferrite._device_assembler(Ferrite.CSCAssembler, K, f, fillzero, Val(atomic))
end

Base.@constprop :aggressive function Ferrite.start_assemble(
        K::GenericSparseMatrixCSR{Tv, Ti},
        f::AbstractVector{Tv} = KA.zeros(get_backend(nonzeros(K)), Tv, 0);
        fillzero::Bool = true, atomic::Bool = false
    ) where {Tv, Ti}
    return Ferrite._device_assembler(Ferrite.CSRAssembler, K, f, fillzero, Val(atomic))
end

@propagate_inbounds function Ferrite._assemble_inner_unsorted!(
        K::GenericSparseMatrixCSC, Ke::AbstractMatrix,
        rowdofs::AbstractVector, coldofs::AbstractVector,
        sym::Bool, atomic::Val = Val(false)
    )
    return Ferrite._assemble_compressed_unsorted!(
        Ferrite.MajorIsColumn(), SparseArrays.getcolptr(K), rowvals(K), nonzeros(K), Ke,
        coldofs, rowdofs, sym, atomic
    )
end

# Row and column roles are swapped compared to CSC, so the local matrix is handed to the
# shared kernel transposed, see the CSR methods in ext/FerriteKAExt/assembler.jl.
@propagate_inbounds function Ferrite._assemble_inner_unsorted!(
        K::GenericSparseMatrixCSR, Ke::AbstractMatrix,
        rowdofs::AbstractVector, coldofs::AbstractVector,
        sym::Bool, atomic::Val = Val(false)
    )
    return Ferrite._assemble_compressed_unsorted!(
        Ferrite.MajorIsRow(), getrowptr(K), colvals(K), nonzeros(K), transpose(Ke),
        rowdofs, coldofs, false, atomic
    )
end

# ---------------- kernel arguments -------------------------------------

# `Adapt.adapt_structure` of a `GenericSparseMatrix*` rebuilds it through its constructor,
# which calls `get_backend` on the storage vectors -- that is not defined for the device
# arrays a kernel adaptor produces. The kernels only need the raw storage, so the assembler
# hands them the GPUArrays device view, which FerriteKAExt assembles into. (Its last type
# parameter is the address space of the storage on the vendor packages, unused here.)
function Adapt.adapt_structure(to, a::Ferrite.CSCAssembler{Tv, Ti, <:GenericSparseMatrixCSC, atomic, <:Any, Nothing}) where {Tv, Ti, atomic}
    colptr, rowval, nzval = adapt(to, SparseArrays.getcolptr(a.K)), adapt(to, rowvals(a.K)), adapt(to, nonzeros(a.K))
    K = GPUArrays.GPUSparseDeviceMatrixCSC{Tv, Ti, typeof(colptr), typeof(nzval), Nothing}(colptr, rowval, nzval, size(a.K), Ti(nnz(a.K)))
    f = adapt(to, a.f)
    return Ferrite.CSCAssembler{Tv, Ti, typeof(K), atomic, typeof(f), Nothing}(K, f, nothing, nothing, nothing, nothing)
end

function Adapt.adapt_structure(to, a::Ferrite.CSRAssembler{Tv, Ti, <:GenericSparseMatrixCSR, atomic, <:Any, Nothing}) where {Tv, Ti, atomic}
    rowptr, colval, nzval = adapt(to, getrowptr(a.K)), adapt(to, colvals(a.K)), adapt(to, nonzeros(a.K))
    K = GPUArrays.GPUSparseDeviceMatrixCSR{Tv, Ti, typeof(rowptr), typeof(nzval), Nothing}(rowptr, colval, nzval, size(a.K), Ti(nnz(a.K)))
    f = adapt(to, a.f)
    return Ferrite.CSRAssembler{Tv, Ti, typeof(K), atomic, typeof(f), Nothing}(K, f, nothing, nothing, nothing, nothing)
end

end
