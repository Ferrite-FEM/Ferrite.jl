module FerriteOneAPIExt

using Ferrite, oneAPI, SparseArrays

import Base: @propagate_inbounds

import Adapt: Adapt, adapt
import GPUArrays
import KernelAbstractions as KA
# `@device_override` comes from SPIRVIntrinsics (imported into oneAPI) and overlays the
# `method_table` that is visible in the module it is used from, hence the import below.
import oneAPI: oneVector, oneDeviceVector, @device_override, method_table
import oneAPI.oneMKL: oneSparseMatrixCSC, oneSparseMatrixCSR

# The oneMKL sparse matrices do not subtype the GPUArrays sparse types, so the methods that
# FerriteKAExt provides for those have to be repeated here. Everything below the assembler,
# i.e. the kernel side, is shared: the adaptors turn the matrices into the GPUArrays device
# views that FerriteKAExt assembles into.

# ---------------- custom dispatches for error paths --------------------

# GPUs cannot interpolate strings, so the device error messages are constant.
@device_override @noinline Ferrite.throw_detJ_not_pos(detJ) = throw(ArgumentError("det(J) is not positive. Please check the value on CPU."))
@device_override @noinline function Ferrite.throw_incompatible_dof_length(length_ue, n_base_funcs)
    throw(ArgumentError("the number of base functions does not match the length of the vector. Perhaps you passed the global vector, or forgot to pass a dof_range? Please check the values on CPU."))
end
@device_override @noinline function Ferrite.throw_incompatible_coord_length(length_x, n_base_funcs)
    throw(ArgumentError("the number of (geometric) base functions does not match the number of coordinates in the vector. Perhaps you forgot to use an appropriate geometric interpolation when creating FE values? See https://github.com/Ferrite-FEM/Ferrite.jl/issues/265 for more details. Please check the values on CPU."))
end

@device_override @noinline function Ferrite._missing_sparsity_pattern_error(Krow::Integer, Kcol::Integer)
    throw(ErrorException("You are trying to assemble values in to K, but the entry is missing in the sparsity pattern. Make sure you have called `K = allocate_matrix(dh)` or `K = allocate_matrix(dh, ch)` if you have affine constraints. This error might also happen if you are using the assembler in a threaded assembly loop (you need to create one `assembler` for each task)."))
end

# ---------------- matrix allocation ------------------------------------

function Ferrite.allocate_matrix(::Type{oneSparseMatrixCSC{Tv, Ti}}, dh::DofHandler) where {Tv, Ti}
    return oneSparseMatrixCSC(allocate_matrix(SparseMatrixCSC{Tv, Ti}, dh))
end

function Ferrite.allocate_matrix(::Type{oneSparseMatrixCSR{Tv, Ti}}, dh::DofHandler) where {Tv, Ti}
    return oneSparseMatrixCSR(allocate_matrix(SparseMatrixCSC{Tv, Ti}, dh))
end

# oneMKL only implements `nonzeros` for its sparse matrices; `Ferrite.apply!` also needs
# the structure of the CSC storage.
SparseArrays.getcolptr(K::oneSparseMatrixCSC) = K.colPtr
SparseArrays.rowvals(K::oneSparseMatrixCSC) = K.rowVal

# ---------------- kernel arguments -------------------------------------

# The last type parameter of the GPUArrays device views is the address space of the storage.
_addrspace(::Type{<:oneAPI.oneDeviceArray{T, N, A}}) where {T, N, A} = A

# Type piracy on oneAPI.jl's matrix types.
function Adapt.adapt_structure(to::oneAPI.KernelAdaptor, K::oneSparseMatrixCSC{Tv, Ti}) where {Tv, Ti}
    colPtr, rowVal, nzVal = adapt(to, K.colPtr), adapt(to, K.rowVal), adapt(to, K.nzVal)
    return GPUArrays.GPUSparseDeviceMatrixCSC{Tv, Ti, typeof(colPtr), typeof(nzVal), _addrspace(typeof(nzVal))}(colPtr, rowVal, nzVal, size(K), K.nnz)
end

function Adapt.adapt_structure(to::oneAPI.KernelAdaptor, K::oneSparseMatrixCSR{Tv, Ti}) where {Tv, Ti}
    rowPtr, colVal, nzVal = adapt(to, K.rowPtr), adapt(to, K.colVal), adapt(to, K.nzVal)
    return GPUArrays.GPUSparseDeviceMatrixCSR{Tv, Ti, typeof(rowPtr), typeof(nzVal), _addrspace(typeof(nzVal))}(rowPtr, colVal, nzVal, size(K), K.nnz)
end

# ---------------- assembly ---------------------------------------------

# The `@constprop :aggressive` propagates a literal `atomic` keyword argument into the
# `atomic` type parameter, see the corresponding host methods in src/assembler.jl.
Base.@constprop :aggressive function Ferrite.start_assemble(
        K::oneSparseMatrixCSC{Tv, Ti}, f::oneVector{Tv} = oneAPI.zeros(Tv, 0);
        fillzero::Bool = true, atomic::Bool = false
    ) where {Tv, Ti}
    return Ferrite._device_assembler(Ferrite.CSCAssembler, K, f, fillzero, Val(atomic))
end

Base.@constprop :aggressive function Ferrite.start_assemble(
        K::oneSparseMatrixCSR{Tv, Ti}, f::oneVector{Tv} = oneAPI.zeros(Tv, 0);
        fillzero::Bool = true, atomic::Bool = false
    ) where {Tv, Ti}
    return Ferrite._device_assembler(Ferrite.CSRAssembler, K, f, fillzero, Val(atomic))
end

# ---------------- atomic assembly --------------------------------------

# See `Ferrite._atomic_add!` in ext/FerriteKAExt/assembler.jl.
@propagate_inbounds function Ferrite._atomic_add!(x::oneDeviceVector{T}, v::T, i::Int) where {T <: Union{Float16, Float32, Float64}}
    @boundscheck checkbounds(x, i)
    KA.@atomic x[i] += v
    return
end

end
