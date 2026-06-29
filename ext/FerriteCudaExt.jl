# TODO's
# * `update!` for ConstraintHandler on GPU.

module FerriteCudaExt

using Ferrite, CUDA, SparseArrays

import Adapt: Adapt, adapt, adapt_structure

import Ferrite: get_grid, AbstractGrid, AbstractDofHandler, get_coordinate_eltype
import Ferrite: get_substruct
import Ferrite: meandiag, nnodes_per_cell
import Ferrite: CellCache

import CUDA: CUDA.CUSPARSE.CuSparseMatrixCSC, CUDA.CUSPARSE.CuSparseMatrixCSR, @device_override
import CUDACore: @gputhrow
import KernelAbstractions as KA
import KernelAbstractions: get_backend, @kernel, @index

# ---------------- custom dispatches for error paths --------------------

# GPUs cannot interpolate strings out of the box, so we use a reduced error message for now.
@device_override @noinline Ferrite.throw_detJ_not_pos(detJ) = @gputhrow("ArgumentError", "det(J) is not positive. Please check the value on CPU.")
@device_override @noinline function Ferrite.throw_incompatible_dof_length(length_ue, n_base_funcs)
    @gputhrow("ArgumentError", "the number of base functions does not match the length of the vector. Perhaps you passed the global vector, or forgot to pass a dof_range? Please check the values on CPU.")
end
@device_override @noinline function Ferrite.throw_incompatible_coord_length(length_x, n_base_funcs)
    @gputhrow("ArgumentError", "the number of (geometric) base functions does not match the number of coordinates in the vector. Perhaps you forgot to use an appropriate geometric interpolation when creating FE values? See https://github.com/Ferrite-FEM/Ferrite.jl/issues/265 for more details. Please check the values on CPU.")
end

# -------------------- assembler ----------------------

struct DeviceCSCAssembler{Tv, Ti, KType <: AbstractSparseArray{Tv, Ti, 2}, FType <: AbstractVector{Tv}} <: Ferrite.AbstractAssembler{Tv}
    K::KType
    f::FType
end
Adapt.@adapt_structure DeviceCSCAssembler

# FIXME buffer
function Ferrite.start_assemble(K::CuSparseMatrixCSC, f::Union{CuVector, Nothing} = nothing; fillzero::Bool = true)
    fillzero && fill!(nonzeros(K), zero(eltype(K)))
    f !== nothing && fillzero && fill!(f, zero(eltype(f)))
    return DeviceCSCAssembler(K, f)
end

function Ferrite.assemble!(A::DeviceCSCAssembler, dofs::AbstractVector{<:Integer}, Ke::AbstractMatrix, fe::AbstractVector)
    Ferrite.assemble!(A, dofs, Ke)
    Ferrite.assemble!(A, dofs, fe)
    return nothing
end

function Ferrite.assemble!(A::DeviceCSCAssembler, dofs::AbstractVector{<:Integer}, fe::AbstractVector)
    for (i, dof) in enumerate(dofs)
        A.f[dof] += fe[i]
    end
    return nothing
end

function Ferrite.assemble!(A::DeviceCSCAssembler, dofs::AbstractVector{<:Integer}, Ke::AbstractMatrix)
    colptr = SparseArrays.getcolptr(A.K)
    rowval = rowvals(A.K)
    nzval = nonzeros(A.K)

    ndofs = length(dofs)
    for j in 1:ndofs
        col = dofs[j]
        r1 = colptr[col]
        r2 = colptr[col + 1] - 1
        for i in 1:ndofs
            val = Ke[i, j]
            iszero(val) && continue
            row = dofs[i]
            # Linear search for the row in this column's nonzeros
            for idx in r1:r2
                if rowval[idx] == row
                    nzval[idx] += val
                    break
                end
            end
        end
    end
    return nothing
end

function Ferrite.allocate_matrix(::Type{CuSparseMatrixCSC{Tv, Ti}}, dh::DofHandler) where {Tv, Ti}
    return CuSparseMatrixCSC(allocate_matrix(SparseMatrixCSC{Tv, Ti}, dh))
end

function Ferrite.allocate_matrix(::Type{CuSparseMatrixCSR{Tv, Ti}}, dh::DofHandler) where {Tv, Ti}
    return CuSparseMatrixCSR(allocate_matrix(SparseMatrixCSC{Tv, Ti}, dh))
end

end
