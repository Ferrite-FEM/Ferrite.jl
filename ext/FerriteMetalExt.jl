module FerriteMetalExt

# Metal.jl has no sparse matrix type, so this extension only makes the error paths of the
# element routines compilable. Global assembly on a Metal device works through
# GenericSparseArrays.jl, see ext/FerriteGenericSparseArraysExt.jl. Note also that Metal
# does not support `Float64`.

using Ferrite, Metal

import Metal: @device_override

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

end
