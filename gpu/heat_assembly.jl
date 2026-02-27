using Ferrite, CUDA
import CUDA: CUDA.CUSPARSE.CuSparseMatrixCSC, @allowscalar
# using IterativeSolvers, LinearAlgebra
import Ferrite: get_grid, AbstractGrid, AbstractDofHandler, get_coordinate_eltype

### TODO Extension
import Adapt: Adapt, adapt, adapt_structure
using StaticArrays, SparseArrays

struct GPUGrid{sdim,C<:Ferrite.AbstractCell,T<:Real, CELA<:AbstractArray{C,1}, NODA<:AbstractArray{Node{sdim,T},1}} <: Ferrite.AbstractGrid{sdim}
    cells::CELA
    nodes::NODA
    # TODO subdomains
end
function Adapt.adapt_structure(to, grid::GPUGrid)
    cells = Adapt.adapt_structure(to, grid.cells)
    nodes = Adapt.adapt_structure(to, grid.nodes)
    return GPUGrid(cells, nodes)
end
GPUGrid(grid::Grid{<:Any,<:Any,T}) where T = GPUGrid(T, grid)
function GPUGrid(::Type{T}, grid::Grid) where T
    GPUGrid(
        CuArray(getcells(grid)),
        CuArray(getnodes(grid))
    )
end

struct GPUSubDofHandler
    ndofs_per_cell::Int
end

struct GPUDofHandler{sdim, GRID <: GPUGrid{sdim}, CADOFS <: AbstractArray{Int,1}, CAOFFSETS <: AbstractArray{Int,1}, CASDHI <: AbstractArray{Int,1}, SDHS <: Tuple} <: Ferrite.AbstractDofHandler
    cell_dofs::CADOFS
    cell_dofs_offset::CAOFFSETS
    cell_to_subdofhandler::CASDHI
    subdofhandlers::SDHS
    grid::GRID
end
function GPUDofHandler(dh::DofHandler)
    sdhs = Tuple(GPUSubDofHandler(sdh.ndofs_per_cell) for sdh in dh.subdofhandlers)
    GPUDofHandler(
        CuArray(dh.cell_dofs),
        CuArray(dh.cell_dofs_offset),
        CuArray(dh.cell_to_subdofhandler),
        sdhs,
        GPUGrid(dh.grid)
    )
end
function GPUDofHandler(::Type{T}, dh::DofHandler) where T
    sdhs = Tuple(GPUSubDofHandler(sdh.ndofs_per_cell) for sdh in dh.subdofhandlers)
    GPUDofHandler(
        CuArray(dh.cell_dofs),
        CuArray(dh.cell_dofs_offset),
        CuArray(dh.cell_to_subdofhandler),
        sdhs,
        GPUGrid(T, dh.grid)
    )
end
Ferrite.get_grid(dh::GPUDofHandler) = dh.grid
function Ferrite.ndofs_per_cell(dh::GPUDofHandler, i::Int)
    sdhidx = dh.cell_to_subdofhandler[i]
    return dh.subdofhandlers[sdhidx].ndofs_per_cell
end
function Ferrite.celldofs(dh::GPUDofHandler, cell_index::Int)
    n = Ferrite.ndofs_per_cell(dh, cell_index)
    offset = dh.cell_dofs_offset[cell_index] - 1
    return @view dh.cell_dofs[offset:(offset + n)]
end
function Ferrite.celldofs!(global_dofs::AbstractVector, dh::GPUDofHandler, i::Integer)
    offset = dh.cell_dofs_offset[i] - 1
    n = Ferrite.ndofs_per_cell(dh, i)
    @inbounds for j in 1:n
        global_dofs[j] = dh.cell_dofs[offset + j]
    end
    return global_dofs
end
Adapt.@adapt_structure GPUDofHandler

grid = generate_grid(Hexahedron, (10, 10, 10), Vec{3}((-1.f0,-1.f0,-1.f0)), Vec{3}((1.f0,1.f0,1.f0)));
colors = create_coloring(grid)

ip = Lagrange{RefHexahedron, 1}()
qr = QuadratureRule{RefHexahedron}(Float32, 2)
cv = CellValues(qr, ip);

dh = DofHandler(grid)
add!(dh, :u, ip)
close!(dh);

function assemble_element!(Ke::AbstractMatrix, cv::CellValues)
    n_basefuncs = getnbasefunctions(cv)
    fill!(Ke, 0)
    for q_point in 1:getnquadpoints(cv)
        dΩ = getdetJdV(cv, q_point)
        for i in 1:n_basefuncs
            δu  = shape_gradient(cv, q_point, i)
            for j in 1:n_basefuncs
                u = shape_gradient(cv, q_point, j)
                Ke[i, j] += (δu ⋅ u) * dΩ
            end
        end
    end
    return Ke
end

function assemble_kernel!(Ke, cell, cv, assembler)
    reinit!(cv, nothing, cell.coords)
    assemble_element!(Ke, cv)
    assemble!(assembler, celldofs(cell), Ke)
end

function assemble_global_cpu(cv::CellValues, K::SparseMatrixCSC, dh::DofHandler)
    n_basefuncs = getnbasefunctions(cv)
    Ke = zeros(Float32, n_basefuncs, n_basefuncs)
    assembler = start_assemble(K)
    for cell in CellIterator(dh)
        assemble_kernel!(Ke, cell, cv, assembler)
    end
    return nothing
end

K = allocate_matrix(SparseMatrixCSC{Float32, Int}, dh)

struct CudaTaskDescriptor
    num_workers::Int
end

adapt(to, ip::Ferrite.Interpolation) = ip

function adapt(to::CudaTaskDescriptor, fv::Ferrite.FunctionValues)
    N = to.num_workers
    return Ferrite.FunctionValues(
        adapt(to, fv.ip),
        CUDA.zeros(eltype(fv.Nx), N, size(fv.Nx, 1), size(fv.Nx, 2)),
        CuMatrix(fv.Nξ),
        CUDA.zeros(eltype(fv.dNdx), N, size(fv.dNdx, 1), size(fv.dNdx, 2)),
        CuMatrix(fv.dNdξ),
        fv.d2Ndx2 === nothing ? nothing : CUDA.zeros(eltype(fv.d2Ndx2), N, size(fv.d2Ndx2, 1), size(fv.d2Ndx2, 2)),
        fv.d2Ndξ2 === nothing ? nothing : CuMatrix(fv.d2Ndξ2),
    )
end

function adapt(to::CudaTaskDescriptor, fv::Ferrite.GeometryMapping)
    N = to.num_workers
    return Ferrite.GeometryMapping(
        adapt(to, fv.ip),
        CUDA.zeros(eltype(fv.M), N, size(fv.M, 1), size(fv.M, 2)),
        fv.dMdξ === nothing ? nothing : CuMatrix(fv.dMdξ),
        fv.d2Mdξ2 === nothing ? nothing : CuMatrix(fv.d2Mdξ2),
    )
end

function adapt(to::CudaTaskDescriptor, cv::CellValues)
    N = to.num_workers
    return CellValues(
        adapt(to, cv.fun_values),
        adapt(to, cv.geo_mapping),
        adapt(to, cv.qr),
        CUDA.zeros(eltype(cv.detJdV), N, length(cv.detJdV))
    )
end

function adapt(to::CudaTaskDescriptor, qr::QuadratureRule{shape}) where shape
    return QuadratureRule{shape}(
        # adapt(to, qr.weights),
        # adapt(to, qr.points),
        CuVector(qr.weights),
        CuVector(qr.points),
    )
end

function adapt(to, qr::QuadratureRule{shape}) where shape
    return QuadratureRule{shape}(
        adapt(to, qr.weights),
        adapt(to, qr.points),
    )
end

function adapt_structure(to, qr::QuadratureRule{shape}) where shape
    return QuadratureRule{shape}(
        adapt_structure(to, qr.weights),
        adapt_structure(to, qr.points),
    )
end

function extract_ith_item(i, cv::CellValues)
    return CellValues(
        extract_ith_item(i, cv.fun_values),
        extract_ith_item(i, cv.geo_mapping),
        cv.qr,
        view(cv.detJdV, i, :),
    )
end

function extract_ith_item(i, fv::Ferrite.FunctionValues)
    return Ferrite.FunctionValues(
        fv.ip,
        view(fv.Nx, i, :, :),
        fv.Nξ,
        view(fv.dNdx, i, :, :),
        fv.dNdξ,
        # fv.d2Ndx2 === nothing ? nothing : view(fv.d2Ndx2, i, :, :),
        # fv.d2Ndξ2,
        nothing,
        nothing,
    )
end

function extract_ith_item(i, fv::Ferrite.GeometryMapping)
    return Ferrite.GeometryMapping(
        fv.ip,
        view(fv.M, i, :, :),
        fv.dMdξ,
        fv.d2Mdξ2,
    )
end

Adapt.@adapt_structure CellValues
Adapt.@adapt_structure Ferrite.GeometryMapping
Adapt.@adapt_structure Ferrite.FunctionValues

struct GPUCellCache{G <: AbstractGrid, DH <: Union{AbstractDofHandler, Nothing}, IVT, VX}
    flags::UpdateFlags
    grid::G
    cellid::Int
    nodes::IVT
    coords::VX
    dh::DH
    dofs::IVT
end
Adapt.@adapt_structure GPUCellCache

function GPUCellCache(dh::GPUDofHandler{dim}, descriptor::CudaTaskDescriptor, flags::UpdateFlags = UpdateFlags()) where {dim}
    @allowscalar begin 
        n = Ferrite.ndofs_per_cell(dh, 1) # dofs and coords will be resized in `reinit!`
        N = Ferrite.nnodes_per_cell(get_grid(dh), 1)
        W = descriptor.num_workers
        nodes = CUDA.zeros(Int, W, N)
        coords = CUDA.zeros(Vec{dim, get_coordinate_eltype(get_grid(dh))}, W, N)
        celldofs = CUDA.zeros(Int, W, n)
    end
    return GPUCellCache(flags, get_grid(dh), -1, nodes, coords, dh, celldofs)
end

function extract_ith_item(i, cc::GPUCellCache, cellid)
    return GPUCellCache(
        cc.flags,
        cc.grid,
        cellid,
        view(cc.nodes, i, :),
        view(cc.coords, i, :),
        cc.dh,
        view(cc.dofs, i, :),
    )
end

Ferrite.celldofs(cc::GPUCellCache) = cc.dofs

# GPU CSC Assembler - stores the raw CSC arrays for kernel-compatible access
struct GPUCSCAssembler{ColPtr <: AbstractVector, RowVal <: AbstractVector, NzVal <: AbstractVector}
    colptr::ColPtr
    rowval::RowVal
    nzval::NzVal
end
Adapt.@adapt_structure GPUCSCAssembler

function Ferrite.start_assemble(K::CuSparseMatrixCSC; fillzero::Bool = true)
    if fillzero
        fill!(nonzeros(K), zero(eltype(K)))
    end
    return GPUCSCAssembler(SparseArrays.getcolptr(K), rowvals(K), nonzeros(K))
end

function Ferrite.assemble!(A::GPUCSCAssembler, dofs::AbstractVector{<:Integer}, Ke::AbstractMatrix, fe::Nothing = nothing)
    colptr = A.colptr
    rowval = A.rowval
    nzval = A.nzval
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

# GPU ConstraintHandler - stores constraint data for GPU apply!
struct GPUConstraintHandler{T, PD <: AbstractVector, IH <: AbstractVector{T}, IP <: AbstractVector{Bool}}
    prescribed_dofs::PD
    inhomogeneities::IH
    is_prescribed::IP
end

function GPUConstraintHandler(ch::ConstraintHandler{<:Any, T}, ::Type{Tv} = T) where {T, Tv}
    @assert Ferrite.isclosed(ch)
    n = Ferrite.ndofs(ch.dh)
    is_prescribed = zeros(Bool, n)
    for d in ch.prescribed_dofs
        is_prescribed[d] = true
    end
    return GPUConstraintHandler(
        CuArray(ch.prescribed_dofs),
        CuArray(Tv.(ch.inhomogeneities)),
        CuArray(is_prescribed),
    )
end

# --- GPU kernels for apply! ---

function _meandiag_kernel!(diag, colptr, rowval, nzval, n)
    j = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if j <= n
        for k in colptr[j]:(colptr[j + 1] - 1)
            if rowval[k] == j
                diag[j] = abs(nzval[k])
                break
            end
        end
    end
    return nothing
end

function _gpu_meandiag(K::CuSparseMatrixCSC{T}) where T
    n = size(K, 1)
    diag = CUDA.zeros(T, n)
    threads = min(256, n)
    blocks = cld(n, threads)
    @cuda threads=threads blocks=blocks _meandiag_kernel!(diag, SparseArrays.getcolptr(K), rowvals(K), nonzeros(K), n)
    return sum(diag) / n
end

# Combined: f -= K[:, d] * inhom[i], then zero column d
function _apply_inhom_zero_cols_kernel!(f, colptr, rowval, nzval, prescribed_dofs, inhomogeneities, n_prescribed, applyzero)
    idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if idx <= n_prescribed
        d = prescribed_dofs[idx]
        v = inhomogeneities[idx]
        for k in colptr[d]:(colptr[d + 1] - 1)
            if !applyzero && !iszero(v)
                CUDA.@atomic f[rowval[k]] -= v * nzval[k]
            end
            nzval[k] = zero(eltype(nzval))
        end
    end
    return nothing
end

# Zero out prescribed rows
function _apply_zero_rows_kernel!(rowval, nzval, is_prescribed, nnz)
    idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if idx <= nnz
        if is_prescribed[rowval[idx]]
            nzval[idx] = zero(eltype(nzval))
        end
    end
    return nothing
end

# Set K[d,d] = m and f[d] = inhom * m
function _apply_set_diag_kernel!(colptr, rowval, nzval, f, prescribed_dofs, inhomogeneities, m, n_prescribed, applyzero)
    idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if idx <= n_prescribed
        d = prescribed_dofs[idx]
        for k in colptr[d]:(colptr[d + 1] - 1)
            if rowval[k] == d
                nzval[k] = m
                break
            end
        end
        vz = applyzero ? zero(eltype(f)) : inhomogeneities[idx]
        f[d] = vz * m
    end
    return nothing
end

function Ferrite.apply!(K::CuSparseMatrixCSC{T}, f::CuVector{T}, ch::GPUConstraintHandler{T}, applyzero::Bool = false) where T
    n_prescribed = length(ch.prescribed_dofs)
    n_prescribed == 0 && return

    colptr = SparseArrays.getcolptr(K)
    rv = rowvals(K)
    nz = nonzeros(K)
    nnz_val = length(nz)

    m = _gpu_meandiag(K)

    # Step 1+2: f -= K[:,d]*v, then zero prescribed columns
    threads = min(256, n_prescribed)
    blocks = cld(n_prescribed, threads)
    @cuda threads=threads blocks=blocks _apply_inhom_zero_cols_kernel!(
        f, colptr, rv, nz, ch.prescribed_dofs, ch.inhomogeneities, n_prescribed, applyzero)

    # Step 3: zero prescribed rows
    threads_r = min(256, nnz_val)
    blocks_r = cld(nnz_val, threads_r)
    @cuda threads=threads_r blocks=blocks_r _apply_zero_rows_kernel!(
        rv, nz, ch.is_prescribed, nnz_val)

    # Step 4: set diagonal and f
    @cuda threads=threads blocks=blocks _apply_set_diag_kernel!(
        colptr, rv, nz, f, ch.prescribed_dofs, ch.inhomogeneities, T(m), n_prescribed, applyzero)

    return
end

function Ferrite.apply_zero!(K::CuSparseMatrixCSC{T}, f::CuVector{T}, ch::GPUConstraintHandler{T}) where T
    return Ferrite.apply!(K, f, ch, true)
end

function assemble_global_gpu(cv::CellValues, K::CuSparseMatrixCSC, dh::GPUDofHandler, ccs, colors::Vector, Kes)
    n_basefuncs = getnbasefunctions(cv)
    num_parallel_workers = maximum(length.(colors))
    Ke = zeros(n_basefuncs, n_basefuncs)
    assembler = start_assemble(K)
    for color in colors
        @cuda threads=num_parallel_workers cuda_kernel(assembler, dh, color, ccs, cv, Kes)
        CUDA.synchronize()
    end
    return nothing
end

function cuda_kernel(assemblers, dh, color, ccs, allcv, Kes)
    i = threadIdx().x
    if i > length(color) 
        return # do not go out of bounds
    end

    cellid = color[i]
    cv = extract_ith_item(i, allcv)
    cc = extract_ith_item(i, ccs, cellid)
    # reinit!(cc)
    if cc.flags.nodes
        Ferrite.cellnodes!(cc.nodes, cc.grid, cellid)
    end
    if cc.flags.coords
        Ferrite.getcoordinates!(cc.coords, cc.grid, cellid)
    end
    if cc.dh !== nothing && cc.flags.dofs
        Ferrite.celldofs!(cc.dofs, cc.dh, cellid)
    end

    Ferrite.reinit!(cv, nothing, cc.coords)

    assembler = assemblers # This is a hotfix... should become extract_ith_item(i, assemblers)
    Ke = view(Kes, i, :, :)
    assemble_kernel!(Ke, cc, cv, assembler)

    return nothing
end

dhgpu = GPUDofHandler(dh)
Kgpu = CuSparseMatrixCSC(K)
colorsgpu = CuVector.(colors)
num_parallel_workers = maximum(length.(colors))
cvgpu = adapt(CudaTaskDescriptor(num_parallel_workers), cv)
ccs = GPUCellCache(dhgpu, CudaTaskDescriptor(num_parallel_workers))
Kes = CUDA.zeros(num_parallel_workers, getnbasefunctions(cv), getnbasefunctions(cv))
assemble_global_gpu(cvgpu, Kgpu, dhgpu, ccs, colorsgpu, Kes)
assemble_global_cpu(cv, K, dh)

using Test
@test CuSparseMatrixCSC(K) ≈ Kgpu

# Test apply! with Dirichlet boundary conditions
ch = ConstraintHandler(dh)
∂Ω = union(getfacetset(grid, "left"), getfacetset(grid, "right"))
add!(ch, Dirichlet(:u, ∂Ω, (x, t) -> 0.0))
close!(ch)

chgpu = GPUConstraintHandler(ch, Float32)

f = zeros(Float32, ndofs(dh))
fgpu = CUDA.zeros(Float32, ndofs(dh))

apply!(K, f, ch)
apply!(Kgpu, fgpu, chgpu)

@test CuSparseMatrixCSC(K) ≈ Kgpu
@test CuVector(f) ≈ fgpu


ch2 = ConstraintHandler(dh)
∂Ω2 = union(
    getfacetset(grid, "left"),
    getfacetset(grid, "right"),
    getfacetset(grid, "top"),
    getfacetset(grid, "bottom"),
);
add!(ch2, Dirichlet(:u, ∂Ω2, (x, t) -> 1.0))
close!(ch2)

chgpu2 = GPUConstraintHandler(ch2, Float32)

f2 = zeros(Float32, ndofs(dh))
fgpu2 = CUDA.zeros(Float32, ndofs(dh))

apply!(K, f2, ch2)
apply!(Kgpu, fgpu2, chgpu2)

@test CuSparseMatrixCSC(K) ≈ Kgpu
@test CuVector(f2) ≈ fgpu2
