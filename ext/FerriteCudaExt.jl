# TODO's
# * `update!` for ConstraintHandler on GPU.

module FerriteCudaExt

using Ferrite, CUDA, SparseArrays

import Adapt: Adapt, adapt, adapt_structure

import Ferrite: get_grid, AbstractGrid, AbstractDofHandler, get_coordinate_eltype
import Ferrite: as_structure_of_arrays, get_substruct
import Ferrite: meandiag, nnodes_per_cell
import Ferrite: CellCacheContainer, CellValuesContainer, CellCache

import CUDA: CUDA.CUSPARSE.CuSparseMatrixCSC, CUDA.CUSPARSE.CuSparseMatrixCSR, @allowscalar
import KernelAbstractions as KA
import KernelAbstractions: get_backend


# ----------------- grid --------------------

struct DeviceGrid{
        dim, C <: Ferrite.AbstractCell, T <: Real,
        CA <: AbstractArray{C, 1}, NA <: AbstractArray{Node{dim, T}, 1},
    } <: Ferrite.AbstractGrid{dim}
    cells::CA
    nodes::NA
end

function Adapt.adapt_structure(to, grid::DeviceGrid)
    return DeviceGrid(Adapt.adapt_structure(to, grid.cells), Adapt.adapt_structure(to, grid.nodes))
end

function DeviceGrid(backend, grid::AbstractGrid)
    return DeviceGrid(adapt(backend, getcells(grid)), adapt(backend, getnodes(grid)))
end

Ferrite.get_coordinate_eltype(::DeviceGrid{<:Any, <:Any, T}) where {T} = T

# ----------------- dofs --------------------

struct DeviceSubDofHandler{
        dim,
        CS <: AbstractVector{Int}, CD <: AbstractVector{Int},
        CO <: AbstractVector{Int}, FN, DR <: Tuple, G <: Ferrite.AbstractGrid{dim},
    } <: AbstractDofHandler
    cellset::CS
    cell_dofs::CD
    cell_dofs_offset::CO
    ndofs_per_cell::Int
    nnodes_per_cell::Int
    # Vector{Symbol} on host, Nothing on device — Symbol is not a bitstype and cannot
    # be stored in device memory. Use the integer index overload of dof_range on device.
    field_names::FN
    dof_ranges::DR
    grid::G
end

Ferrite.get_grid(dh::DeviceSubDofHandler) = dh.grid

function Adapt.adapt_structure(to, sdh::DeviceSubDofHandler)
    return DeviceSubDofHandler(
        Adapt.adapt_structure(to, sdh.cellset),
        Adapt.adapt_structure(to, sdh.cell_dofs),
        Adapt.adapt_structure(to, sdh.cell_dofs_offset),
        sdh.ndofs_per_cell,
        sdh.nnodes_per_cell,
        nothing,
        sdh.dof_ranges,
        Adapt.adapt_structure(to, sdh.grid),
    )
end

Ferrite.nnodes_per_cell(sdh::DeviceSubDofHandler) = sdh.nnodes_per_cell
Ferrite.ndofs_per_cell(sdh::DeviceSubDofHandler) = sdh.ndofs_per_cell

function Ferrite.dof_range(sdh::DeviceSubDofHandler, field_idx::Int)
    return sdh.dof_ranges[field_idx]
end
function Ferrite.dof_range(sdh::DeviceSubDofHandler, field_name::Symbol)
    idx = findfirst(==(field_name), sdh.field_names)
    idx === nothing && error("Field $field_name not found in DeviceSubDofHandler")
    return sdh.dof_ranges[idx]
end

function Ferrite.celldofs!(global_dofs::AbstractVector, sdh::DeviceSubDofHandler, i::Integer)
    copyto!(global_dofs, 1, sdh.cell_dofs, sdh.cell_dofs_offset[i], length(global_dofs))
    return global_dofs
end

# Host-only container — not sent to the device!
struct HostDofHandler{dim, G <: DeviceGrid{dim}, SDH <: DeviceSubDofHandler, DH <: AbstractDofHandler} <: AbstractDofHandler
    subdofhandlers::Vector{SDH}
    grid::G
    original_dh::DH
end

function HostDofHandler(backend, dh::DofHandler)
    gpu_grid = DeviceGrid(backend, dh.grid)
    cell_dofs = adapt(backend, dh.cell_dofs)
    cell_dofs_offset = adapt(backend, dh.cell_dofs_offset)
    subdofhandlers = map(dh.subdofhandlers) do sdh
        dof_ranges = Tuple(Ferrite.dof_range(sdh, i) for i in 1:length(sdh.field_names))
        DeviceSubDofHandler(
            adapt(backend, collect(Int, sdh.cellset)), cell_dofs, cell_dofs_offset,
            sdh.ndofs_per_cell, nnodes_per_cell(dh.grid, first(sdh.cellset)), copy(sdh.field_names), dof_ranges, gpu_grid
        )
    end
    return HostDofHandler(subdofhandlers, gpu_grid, dh)
end

function Adapt.adapt_structure(to::KA.Backend, dh::DofHandler)
    return HostDofHandler(to, dh)
end

Ferrite.get_grid(dh::HostDofHandler) = dh.grid


# ----------------- Adapt.jl parts --------------------

import Adapt: Adapt, adapt, adapt_structure
Adapt.@adapt_structure CellValues
Adapt.@adapt_structure Ferrite.GeometryMapping
Adapt.@adapt_structure Ferrite.FunctionValues
function adapt_structure(to, ccc::Ferrite.CellValuesContainer)
    inner_values = adapt(to, ccc.values)
    return Ferrite.CellValuesContainer{typeof(get_substruct(1, inner_values)), typeof(inner_values)}(inner_values)
end

adapt(to, ip::Ferrite.Interpolation) = ip

function adapt(d, cv::CellValues)
    return CellValues(
        adapt(d, cv.fun_values),
        adapt(d, cv.geo_mapping),
        adapt(d, cv.qr),
        adapt(d, cv.detJdV),
    )
end

function as_structure_of_arrays(d, outer_dim, ::Type{CellValues}, args...; kwargs...)
    cv = CellValues(args...; kwargs...)
    return as_structure_of_arrays(d, outer_dim, cv)
end

function as_structure_of_arrays(d, N, cv::CellValues)
    return CellValues(
        as_structure_of_arrays(d, N, cv.fun_values),
        as_structure_of_arrays(d, N, cv.geo_mapping),
        adapt(d, cv.qr),
        KA.zeros(d, eltype(cv.detJdV), N, length(cv.detJdV)),
    )
end

function adapt(d, fv::Ferrite.FunctionValues)
    Nξ = adapt(d, fv.Nξ)
    return Ferrite.FunctionValues(
        adapt(d, fv.ip),
        fv.Nξ === fv.Nx ? Nξ : adapt(fv.Nx), # Ensure proper aliasing
        Nξ,
        adapt(d, fv.dNdx),
        adapt(d, fv.dNdξ),
        fv.d2Ndx2 === nothing ? nothing : as_shared_array(d, N, fv.d2Ndx2),
        fv.d2Ndξ2 === nothing ? nothing : adapt(d, collect(fv.d2Ndξ2)),
    )
end

function as_structure_of_arrays(d, N, fv::Ferrite.FunctionValues)
    Nξ = adapt(d, fv.Nξ)
    return Ferrite.FunctionValues(
        adapt(d, fv.ip),
        fv.Nξ === fv.Nx ? Nξ : KA.zeros(d, eltype(fv.Nx), N, size(fv.Nx, 1), size(fv.Nx, 2)), # Ensure proper aliasing,
        Nξ,
        fv.dNdx === nothing ? nothing : KA.zeros(d, eltype(fv.dNdx), N, size(fv.dNdx, 1), size(fv.dNdx, 2)),
        adapt(d, fv.dNdξ),
        fv.d2Ndx2 === nothing ? nothing : KA.zeros(d, eltype(fv.d2Ndx2), N, size(fv.d2Ndx2, 1), size(fv.d2Ndx2, 2)),
        fv.d2Ndξ2 === nothing ? nothing : adapt(d, fv.d2Ndξ2),
    )
end

function adapt(d, fv::Ferrite.GeometryMapping)
    return Ferrite.GeometryMapping(
        adapt(d, fv.ip),
        adapt(d, fv.M),
        fv.dMdξ === nothing ? nothing : adapt(d, fv.dMdξ),
        fv.d2Mdξ2 === nothing ? nothing : adapt(d, fv.d2Mdξ2),
    )
end

function as_structure_of_arrays(d, N, fv::Ferrite.GeometryMapping)
    return Ferrite.GeometryMapping(
        adapt(d, fv.ip),
        KA.zeros(d, eltype(fv.M), N, size(fv.M, 1), size(fv.M, 2)),
        fv.dMdξ === nothing ? nothing : adapt(d, fv.dMdξ),
        fv.d2Mdξ2 === nothing ? nothing : adapt(d, fv.d2Mdξ2),
    )
end

function adapt(to, qr::QuadratureRule{shape}) where {shape}
    return QuadratureRule{shape}(adapt(to, qr.weights), adapt(to, qr.points))
end

# Adapt.@adapt_structure QuadratureRule does not work here due to the type parameter ctor.
function adapt_structure(to, qr::QuadratureRule{shape}) where {shape}
    return QuadratureRule{shape}(adapt_structure(to, qr.weights), adapt_structure(to, qr.points))
end

# -------------------- iterator ----------------------

# NOTE CellCache is mutable and hence inherently incompatible with GPU. So here is the
# immutable variant. Making the CellCache immutable is considered breaking due to the reinit! API integration.
struct ImmutableCellCache{G <: AbstractGrid, SDH, IVT, VX}
    flags::UpdateFlags
    grid::G
    cellid::Int
    nodes::IVT
    coords::VX
    sdh::SDH
    dofs::IVT
end
(cc::ImmutableCellCache)(cellid::Int) = ImmutableCellCache(cc.flags, cc.grid, cellid, cc.nodes, cc.coords, cc.sdh, cc.dofs)
Adapt.@adapt_structure ImmutableCellCache
function adapt_structure(to, ccc::CellCacheContainer)
    inner_values = adapt(to, ccc.values)
    return CellCacheContainer{typeof(get_substruct(1, inner_values, -1)), typeof(inner_values)}(inner_values)
end

Ferrite.celldofs(cc::ImmutableCellCache) = cc.dofs

function as_structure_of_arrays(backend, outer_dim, ::Type{CellCache}, dh::HostDofHandler, flags::UpdateFlags = UpdateFlags())
    @assert length(dh.subdofhandlers) == 1 "ImmutableCellCache only works on HostDofHandler's with a single subdomain. Please call the ImmutableCellCache adaptation on the DeviceSubDofHandler."
    return as_structure_of_arrays(backend, outer_dim, CellCache, first(dh.subdofhandlers), flags)
end

function as_structure_of_arrays(backend, outer_dim, ::Type{CellCache}, sdh::DeviceSubDofHandler{dim}, flags::UpdateFlags = UpdateFlags()) where {dim}
    grid = get_grid(sdh)
    begin
        n = Ferrite.ndofs_per_cell(sdh)
        N = Ferrite.nnodes_per_cell(sdh)
        nodes = KA.zeros(backend, Int, outer_dim, N)
        coords = KA.zeros(backend, Vec{dim, get_coordinate_eltype(grid)}, outer_dim, N)
        dofs = KA.zeros(backend, Int, outer_dim, n)
    end
    return ImmutableCellCache(flags, grid, -1, nodes, coords, sdh, dofs)
end

function Ferrite.CellCache(backend, dh::HostDofHandler{dim}, flags::UpdateFlags = UpdateFlags()) where {dim}
    @assert length(dh.subdofhandlers) == 1 "ImmutableCellCache only works on HostDofHandler's with a single subdomain. Please call the ImmutableCellCache adaptation on the DeviceSubDofHandler."
    return CellCache(backend, first(dh.subdofhandlers), flags)
end

function Ferrite.CellCache(backend, sdh::DeviceSubDofHandler{dim}, flags::UpdateFlags = UpdateFlags()) where {dim}
    grid = get_grid(sdh)
    N = Ferrite.nnodes_per_cell(grid, first(sdh.cellset))
    nodes = KA.zeros(backend, Int, N)
    coords = KA.zeros(backend, Vec{dim, get_coordinate_eltype(grid)}, N)

    n = Ferrite.ndofs_per_cell(sdh)
    dofs = KA.zeros(backend, Int, n)
    return ImmutableCellCache(flags, grid, -1, nodes, coords, sdh, dofs)
end

function adapt(backend, cc::ImmutableCellCache)
    return ImmutableCellCache(
        cc.flags,
        adapt(backend, cc.grid),
        -1,
        adapt(backend, cc.nodes),
        adapt(backend, cc.coords),
        adapt(backend, cc.sdh),
        adapt(backend, cc.dofs),
    )
end

function get_substruct(i, cc::ImmutableCellCache, cellid)
    return ImmutableCellCache(
        cc.flags, cc.grid, cellid,
        view(cc.nodes, i, :), view(cc.coords, i, :), cc.sdh, view(cc.dofs, i, :)
    )
end

function Ferrite.reinit!(cc_i::ImmutableCellCache, cellid::Integer)
    cc_i.flags.nodes  && Ferrite.cellnodes!(cc_i.nodes, cc_i.grid, cellid)
    cc_i.flags.coords && Ferrite.getcoordinates!(cc_i.coords, cc_i.grid, cellid)
    cc_i.sdh !== nothing && cc_i.flags.dofs && Ferrite.celldofs!(cc_i.dofs, cc_i.sdh, cellid)
    return nothing
end

# -------------------- assembler ----------------------

struct DeviceCSCAssembler{KType, FType}
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
    diag = KA.zeros(backend, T, n)
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
