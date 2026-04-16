# TODO's
# * `update!` for ConstraintHandler on GPU.

module FerriteCudaExt

using Ferrite, CUDA, SparseArrays

import Adapt: Adapt, adapt, adapt_structure

import Ferrite: get_grid, AbstractGrid, AbstractDofHandler, get_coordinate_eltype
import Ferrite: as_structure_of_arrays, get_substruct
import Ferrite: meandiag, nnodes_per_cell
import Ferrite: CellCacheContainer, CellValuesContainer, CellCache

import CUDA: CUDA.CUSPARSE.CuSparseMatrixCSC, CUDA.CUSPARSE.CuSparseMatrixCSR, @gputhrow, @device_override
import KernelAbstractions as KA
import KernelAbstractions: get_backend

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

struct DeviceCSCAssembler{KType, FType} <: Ferrite.AbstractAssembler
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

# -------------------- constraints ----------------------

struct GPUConstraintHandler{Tv, Ti, PD <: AbstractVector{Ti}, IH <: AbstractVector{Tv}, IP <: AbstractVector{Bool}}
    prescribed_dofs::PD
    inhomogeneities::IH
    is_prescribed::IP
end

function adapt_structure(backend, ch::ConstraintHandler)
    @assert Ferrite.isclosed(ch)
    n = Ferrite.ndofs(ch.dh)
    is_prescribed = zeros(Bool, n)
    for d in ch.prescribed_dofs
        is_prescribed[d] = true
    end
    return GPUConstraintHandler(
        adapt(backend, ch.prescribed_dofs),
        adapt(backend, ch.inhomogeneities),
        adapt(backend, is_prescribed),
    )
end

function meandiag_kernel!(diag, colptr, rowval, nzval, n)
    j = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    j > n && return
    for k in colptr[j]:(colptr[j + 1] - 1)
        if rowval[k] == j
            diag[j] = abs(nzval[k])
            break
        end
    end
    return nothing
end

# TODO The code below must be refactored a bit to use KernelAbstractions directly.
#      The limiting factor right now is not having a enough sparse matrix base types
#      and related to this no interfaces to query the fields.

function meandiag(K::CuSparseMatrixCSC{T}) where {T}
    n = size(K, 1)
    backend = get_backend(nonzeros(K))
    diag = KA.zeros(backend, T, n)
    threads = min(256, n, CUDA.attribute(device(), CUDA.DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK))
    @cuda threads = threads blocks = cld(n, threads) meandiag_kernel!(
        diag, SparseArrays.getcolptr(K), rowvals(K), nonzeros(K), n
    )
    return sum(diag) / n
end

# Combined: f -= K[:, d] * inhom[d], then zero column d
function apply_inhom_zero_cols_kernel!(f, colptr, rowval, nzval, prescribed_dofs, inhomogeneities, n_prescribed, applyzero)
    idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    idx > n_prescribed && return
    d = prescribed_dofs[idx]
    v = inhomogeneities[idx]
    for k in colptr[d]:(colptr[d + 1] - 1)
        if !applyzero && !iszero(v)
            CUDA.@atomic f[rowval[k]] -= v * nzval[k]
        end
        nzval[k] = zero(eltype(nzval))
    end
    return nothing
end

# Zero out prescribed rows
function apply_zero_rows_kernel!(rowval, nzval, is_prescribed, nnz)
    idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    idx > nnz && return
    if is_prescribed[rowval[idx]]
        nzval[idx] = zero(eltype(nzval))
    end
    return nothing
end

# Set K[d,d] = m and f[d] = inhom[d] * m
function apply_set_diag_kernel!(colptr, rowval, nzval, f, prescribed_dofs, inhomogeneities, m, n_prescribed, applyzero)
    idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    idx > n_prescribed && return
    d = prescribed_dofs[idx]
    for k in colptr[d]:(colptr[d + 1] - 1)
        if rowval[k] == d
            nzval[k] = m
            break
        end
    end
    f[d] = (applyzero ? zero(eltype(f)) : inhomogeneities[idx]) * m
    return nothing
end

function Ferrite.apply!(K::CuSparseMatrixCSC{T}, f::CuVector{T}, ch::GPUConstraintHandler{T}, applyzero::Bool = false) where {T}
    n_prescribed = length(ch.prescribed_dofs)
    n_prescribed == 0 && return

    colptr = SparseArrays.getcolptr(K)
    rv = rowvals(K)
    nz = nonzeros(K)
    nnz = length(nz)
    m = meandiag(K)
    threads_p = min(256, n_prescribed, CUDA.attribute(device(), CUDA.DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK))
    blocks_p = cld(n_prescribed, threads_p)

    # TODO fuze into one kernel
    # Step 1+2: f -= K[:,d]*v, then zero prescribed columns
    @cuda threads = threads_p blocks = blocks_p apply_inhom_zero_cols_kernel!(
        f, colptr, rv, nz, ch.prescribed_dofs, ch.inhomogeneities, n_prescribed, applyzero
    )

    # Step 3: zero prescribed rows
    threads_z = min(256, nnz, CUDA.attribute(device(), CUDA.DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK))
    @cuda threads = threads_z blocks = cld(nnz, threads_z) apply_zero_rows_kernel!(
        rv, nz, ch.is_prescribed, nnz
    )

    # Step 4: set diagonal and rhs
    @cuda threads = threads_p blocks = blocks_p apply_set_diag_kernel!(
        colptr, rv, nz, f, ch.prescribed_dofs, ch.inhomogeneities, T(m), n_prescribed, applyzero
    )

    return
end

function Ferrite.apply_zero!(K::CuSparseMatrixCSC{T}, f::CuVector{T}, ch::GPUConstraintHandler{T}) where {T}
    return Ferrite.apply!(K, f, ch, true)
end

end
