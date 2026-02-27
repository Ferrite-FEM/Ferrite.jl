module FerriteCudaExt

using Ferrite, CUDA, SparseArrays

import Adapt: Adapt, adapt, adapt_structure

import Ferrite: get_grid, AbstractGrid, AbstractDofHandler, get_coordinate_eltype, TaskDescriptor, get_worker_part

import CUDA: CUDA.CUSPARSE.CuSparseMatrixCSC, CUDA.CUSPARSE.CuSparseMatrixCSR, @allowscalar
import KernelAbstractions: KernelAbstractions, get_backend


# ----------------- grid --------------------

struct GPUGrid{
        dim, C <: Ferrite.AbstractCell, T <: Real,
        CA <: AbstractArray{C, 1}, NA <: AbstractArray{Node{dim, T}, 1},
    } <: Ferrite.AbstractGrid{dim}
    cells::CA
    nodes::NA
end

function Adapt.adapt_structure(to, grid::GPUGrid)
    return GPUGrid(Adapt.adapt_structure(to, grid.cells), Adapt.adapt_structure(to, grid.nodes))
end

function GPUGrid(backend, grid::AbstractGrid)
    return GPUGrid(adapt(backend, getcells(grid)), adapt(backend, getnodes(grid)))
end


# ----------------- dofs --------------------

struct GPUSubDofHandler{
        CS <: AbstractVector{Int}, CD <: AbstractVector{Int},
        CO <: AbstractVector{Int}, FN, DR <: Tuple, G,
    }
    cellset::CS
    cell_dofs::CD
    cell_dofs_offset::CO
    ndofs_per_cell::Int
    # Vector{Symbol} on CPU, Nothing on GPU — Symbol is not a bitstype and cannot
    # be stored in GPU memory. Use the integer index overload of dof_range on GPU.
    field_names::FN
    dof_ranges::DR
    grid::G
end

function Adapt.adapt_structure(to, sdh::GPUSubDofHandler)
    return GPUSubDofHandler(
        Adapt.adapt_structure(to, sdh.cellset),
        Adapt.adapt_structure(to, sdh.cell_dofs),
        Adapt.adapt_structure(to, sdh.cell_dofs_offset),
        sdh.ndofs_per_cell,
        nothing,
        sdh.dof_ranges,
        Adapt.adapt_structure(to, sdh.grid),
    )
end

Ferrite.ndofs_per_cell(sdh::GPUSubDofHandler) = sdh.ndofs_per_cell

function Ferrite.dof_range(sdh::GPUSubDofHandler, field_idx::Int)
    return sdh.dof_ranges[field_idx]
end
function Ferrite.dof_range(sdh::GPUSubDofHandler, field_name::Symbol)
    idx = findfirst(==(field_name), sdh.field_names)
    idx === nothing && error("Field $field_name not found in GPUSubDofHandler")
    return sdh.dof_ranges[idx]
end

function Ferrite.celldofs!(global_dofs::AbstractVector, sdh::GPUSubDofHandler, i::Integer)
    offset = sdh.cell_dofs_offset[i] - 1
    n = sdh.ndofs_per_cell
    @inbounds for j in 1:n
        global_dofs[j] = sdh.cell_dofs[offset + j]
    end
    return global_dofs
end

# CPU-only container — not sent to the GPU
struct GPUDofHandler{dim, G <: GPUGrid{dim}, SDH <: GPUSubDofHandler} <: Ferrite.AbstractDofHandler
    subdofhandlers::Vector{SDH}
    grid::G
end

function GPUDofHandler(backend, dh::DofHandler)
    gpu_grid = GPUGrid(backend, dh.grid)
    cell_dofs = adapt(backend, dh.cell_dofs)
    cell_dofs_offset = adapt(backend, dh.cell_dofs_offset)
    subdofhandlers = map(dh.subdofhandlers) do sdh
        dof_ranges = Tuple(Ferrite.dof_range(sdh, i) for i in 1:length(sdh.field_names))
        GPUSubDofHandler(
            adapt(backend, collect(Int, sdh.cellset)), cell_dofs, cell_dofs_offset,
            sdh.ndofs_per_cell, copy(sdh.field_names), dof_ranges, gpu_grid
        )
    end
    return GPUDofHandler(subdofhandlers, gpu_grid)
end

function Adapt.adapt_structure(to::TaskDescriptor, dh::DofHandler)
    return GPUDofHandler(to.device, dh)
end

Ferrite.get_grid(dh::GPUDofHandler) = dh.grid


# ----------------- Adapt.jl parts --------------------

import Adapt: Adapt, adapt, adapt_structure
Adapt.@adapt_structure CellValues
Adapt.@adapt_structure Ferrite.GeometryMapping
Adapt.@adapt_structure Ferrite.FunctionValues

adapt(to, ip::Ferrite.Interpolation) = ip

function adapt(to::TaskDescriptor, fv::Ferrite.FunctionValues)
    N = to.num_workers
    d = to.device
    return Ferrite.FunctionValues(
        adapt(to, fv.ip),
        KernelAbstractions.zeros(d, eltype(fv.Nx), N, size(fv.Nx, 1), size(fv.Nx, 2)),
        adapt(d, collect(fv.Nξ)),
        KernelAbstractions.zeros(d, eltype(fv.dNdx), N, size(fv.dNdx, 1), size(fv.dNdx, 2)),
        adapt(d, collect(fv.dNdξ)),
        fv.d2Ndx2 === nothing ? nothing : KernelAbstractions.zeros(d, eltype(fv.d2Ndx2), N, size(fv.d2Ndx2, 1), size(fv.d2Ndx2, 2)),
        fv.d2Ndξ2 === nothing ? nothing : adapt(d, collect(fv.d2Ndξ2)),
    )
end

function adapt(to::TaskDescriptor, fv::Ferrite.GeometryMapping)
    N = to.num_workers
    d = to.device
    return Ferrite.GeometryMapping(
        adapt(to, fv.ip),
        KernelAbstractions.zeros(d, eltype(fv.M), N, size(fv.M, 1), size(fv.M, 2)),
        fv.dMdξ === nothing ? nothing : adapt(d, collect(fv.dMdξ)),
        fv.d2Mdξ2 === nothing ? nothing : adapt(d, collect(fv.d2Mdξ2)),
    )
end

function adapt(to::TaskDescriptor, cv::CellValues)
    N = to.num_workers
    d = to.device
    return CellValues(
        adapt(to, cv.fun_values),
        adapt(to, cv.geo_mapping),
        adapt(to, cv.qr),
        KernelAbstractions.zeros(d, eltype(cv.detJdV), N, length(cv.detJdV)),
    )
end

function adapt(to::TaskDescriptor, qr::QuadratureRule{shape}) where {shape}
    d = to.device
    return QuadratureRule{shape}(adapt(d, collect(qr.weights)), adapt(d, collect(qr.points)))
end

function adapt(to, qr::QuadratureRule{shape}) where {shape}
    return QuadratureRule{shape}(adapt(to, qr.weights), adapt(to, qr.points))
end

function adapt_structure(to, qr::QuadratureRule{shape}) where {shape}
    return QuadratureRule{shape}(adapt_structure(to, qr.weights), adapt_structure(to, qr.points))
end


# -------------------- iterator ----------------------

struct GPUCellCache{G <: AbstractGrid, SDH, IVT, VX}
    flags::UpdateFlags
    grid::G
    cellid::Int
    nodes::IVT
    coords::VX
    sdh::SDH
    dofs::IVT
end
Adapt.@adapt_structure GPUCellCache

function GPUCellCache(
        sdh::GPUSubDofHandler, grid::GPUGrid{dim}, descriptor::TaskDescriptor,
        flags::UpdateFlags = UpdateFlags()
    ) where {dim}
    d = descriptor.device
    @allowscalar begin
        n = sdh.ndofs_per_cell
        N = Ferrite.nnodes_per_cell(grid, 1)
        W = descriptor.num_workers
        nodes = KernelAbstractions.zeros(d, Int, W, N)
        coords = KernelAbstractions.zeros(d, Vec{dim, get_coordinate_eltype(grid)}, W, N)
        dofs = KernelAbstractions.zeros(d, Int, W, n)
    end
    return GPUCellCache(flags, grid, -1, nodes, coords, sdh, dofs)
end

Ferrite.celldofs(cc::GPUCellCache) = cc.dofs

# TODO I do not like this design.
function Ferrite.CellCache(descriptor::TaskDescriptor, sdh::GPUSubDofHandler, flags::UpdateFlags = UpdateFlags())
    return GPUCellCache(sdh, sdh.grid, descriptor, flags)
end

function get_worker_part(i, cc::GPUCellCache, cellid)
    return GPUCellCache(
        cc.flags, cc.grid, cellid,
        view(cc.nodes, i, :), view(cc.coords, i, :), cc.sdh, view(cc.dofs, i, :)
    )
end


# -------------------- assembler ----------------------

struct GPUCSCAssembler{CP <: AbstractVector, RV <: AbstractVector, NZ <: AbstractVector}
    colptr::CP
    rowval::RV
    nzval::NZ
end
Adapt.@adapt_structure GPUCSCAssembler

function Ferrite.start_assemble(K::CuSparseMatrixCSC; fillzero::Bool = true)
    fillzero && fill!(nonzeros(K), zero(eltype(K)))
    return GPUCSCAssembler(SparseArrays.getcolptr(K), rowvals(K), nonzeros(K))
end

function Ferrite.assemble!(A::GPUCSCAssembler, dofs::AbstractVector{<:Integer}, Ke::AbstractMatrix, fe::Nothing = nothing)
    ndofs = length(dofs)
    for j in 1:ndofs
        col = dofs[j]
        r1 = A.colptr[col]
        r2 = A.colptr[col + 1] - 1
        for i in 1:ndofs
            val = Ke[i, j]
            iszero(val) && continue
            row = dofs[i]
            # Linear search for the row in this column's nonzeros
            for idx in r1:r2
                if A.rowval[idx] == row
                    A.nzval[idx] += val
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

function adapt(backend, ch::ConstraintHandler)
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

function meandiag(K::CuSparseMatrixCSC{T}) where {T}
    n = size(K, 1)
    backend = get_backend(nonzeros(K))
    diag = KernelAbstractions.zeros(backend, T, n)
    threads = min(256, n)
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
    threads_p = min(256, n_prescribed)
    blocks_p = cld(n_prescribed, threads_p)

    # Step 1+2: f -= K[:,d]*v, then zero prescribed columns
    @cuda threads = threads_p blocks = blocks_p apply_inhom_zero_cols_kernel!(
        f, colptr, rv, nz, ch.prescribed_dofs, ch.inhomogeneities, n_prescribed, applyzero
    )

    # Step 3: zero prescribed rows
    threads_z = min(256, nnz)
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
