module FerriteAMDGPUExt

using Ferrite, AMDGPU, SparseArrays

import AMDGPU: ROCVector
import AMDGPU.Device: @device_override
import AMDGPU.rocSPARSE: ROCSparseMatrixCSC, ROCSparseMatrixCSR

# The device assembler itself is backend agnostic and lives in FerriteKAExt, which is loaded
# together with this extension since AMDGPU.jl depends on all of its triggers.

# ---------------- custom dispatches for error paths --------------------

# GPUs cannot interpolate strings out of the box, so we use a reduced error message for now.
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

function Ferrite.allocate_matrix(::Type{ROCSparseMatrixCSC{Tv, Ti}}, dh::DofHandler) where {Tv, Ti}
    K = allocate_matrix(SparseMatrixCSC{Tv, Ti}, dh)
    return ROCSparseMatrixCSC{Tv, Ti}(ROCVector{Ti}(K.colptr), ROCVector{Ti}(K.rowval), ROCVector{Tv}(K.nzval), size(K))
end

function Ferrite.allocate_matrix(::Type{ROCSparseMatrixCSR{Tv, Ti}}, dh::DofHandler) where {Tv, Ti}
    # The CSR storage of K is the CSC storage of transpose(K). The transposition is done on
    # the host, since the conversion routines in AMDGPU.jl force `Ti = Cint`.
    Kt = SparseMatrixCSC(transpose(allocate_matrix(SparseMatrixCSC{Tv, Ti}, dh)))
    return ROCSparseMatrixCSR{Tv, Ti}(ROCVector{Ti}(Kt.colptr), ROCVector{Ti}(Kt.rowval), ROCVector{Tv}(Kt.nzval), reverse(size(Kt)))
end

end
