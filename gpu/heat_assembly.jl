using Ferrite, CUDA
import CUDA: CUDA.CUSPARSE.CuSparseMatrixCSC, @allowscalar
import Ferrite: get_grid, AbstractGrid, AbstractDofHandler, get_coordinate_eltype
import Adapt: Adapt, adapt, adapt_structure
using StaticArrays, SparseArrays

### GPUGrid

struct GPUGrid{dim, C <: Ferrite.AbstractCell, T <: Real,
               CA <: AbstractArray{C,1}, NA <: AbstractArray{Node{dim,T},1}} <: Ferrite.AbstractGrid{dim}
    cells::CA
    nodes::NA
end

function Adapt.adapt_structure(to, grid::GPUGrid)
    return GPUGrid(Adapt.adapt_structure(to, grid.cells), Adapt.adapt_structure(to, grid.nodes))
end

GPUGrid(grid::Grid{<:Any,<:Any,T}) where T = GPUGrid(T, grid)
function GPUGrid(::Type{T}, grid::Grid) where T
    return GPUGrid(CuArray(getcells(grid)), CuArray(getnodes(grid)))
end

### GPUSubDofHandler + GPUDofHandler

struct GPUSubDofHandler{CS <: AbstractVector{Int}, CD <: AbstractVector{Int},
                        CO <: AbstractVector{Int}, FN, DR <: Tuple}
    cellset::CS
    cell_dofs::CD
    cell_dofs_offset::CO
    ndofs_per_cell::Int
    # Vector{Symbol} on CPU, Nothing on GPU — Symbol is not a bitstype and cannot
    # be stored in GPU memory. Use the integer index overload of dof_range on GPU.
    field_names::FN
    dof_ranges::DR
end

function Adapt.adapt_structure(to, sdh::GPUSubDofHandler)
    return GPUSubDofHandler(
        Adapt.adapt_structure(to, sdh.cellset),
        Adapt.adapt_structure(to, sdh.cell_dofs),
        Adapt.adapt_structure(to, sdh.cell_dofs_offset),
        sdh.ndofs_per_cell,
        nothing,
        sdh.dof_ranges,
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

function GPUDofHandler(dh::DofHandler)
    cell_dofs        = CuArray(dh.cell_dofs)
    cell_dofs_offset = CuArray(dh.cell_dofs_offset)
    subdofhandlers = map(dh.subdofhandlers) do sdh
        dof_ranges = Tuple(Ferrite.dof_range(sdh, i) for i in 1:length(sdh.field_names))
        GPUSubDofHandler(CuArray(collect(Int, sdh.cellset)), cell_dofs, cell_dofs_offset,
                         sdh.ndofs_per_cell, copy(sdh.field_names), dof_ranges)
    end
    return GPUDofHandler(subdofhandlers, GPUGrid(dh.grid))
end

function GPUDofHandler(::Type{T}, dh::DofHandler) where T
    cell_dofs        = CuArray(dh.cell_dofs)
    cell_dofs_offset = CuArray(dh.cell_dofs_offset)
    subdofhandlers = map(dh.subdofhandlers) do sdh
        dof_ranges = Tuple(Ferrite.dof_range(sdh, i) for i in 1:length(sdh.field_names))
        GPUSubDofHandler(CuArray(collect(Int, sdh.cellset)), cell_dofs, cell_dofs_offset,
                         sdh.ndofs_per_cell, copy(sdh.field_names), dof_ranges)
    end
    return GPUDofHandler(subdofhandlers, GPUGrid(T, dh.grid))
end

Ferrite.get_grid(dh::GPUDofHandler) = dh.grid

### CellValues adaptation

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
        CUDA.zeros(eltype(cv.detJdV), N, length(cv.detJdV)),
    )
end

function adapt(to::CudaTaskDescriptor, qr::QuadratureRule{shape}) where shape
    return QuadratureRule{shape}(CuVector(qr.weights), CuVector(qr.points))
end

function adapt(to, qr::QuadratureRule{shape}) where shape
    return QuadratureRule{shape}(adapt(to, qr.weights), adapt(to, qr.points))
end

function adapt_structure(to, qr::QuadratureRule{shape}) where shape
    return QuadratureRule{shape}(adapt_structure(to, qr.weights), adapt_structure(to, qr.points))
end

Adapt.@adapt_structure CellValues
Adapt.@adapt_structure Ferrite.GeometryMapping
Adapt.@adapt_structure Ferrite.FunctionValues

# Extract the i-th worker's local slice from batched GPU data
function get_worker_part(i, cv::CellValues)
    return CellValues(get_worker_part(i, cv.fun_values), get_worker_part(i, cv.geo_mapping),
                      cv.qr, view(cv.detJdV, i, :))
end
function get_worker_part(i, fv::Ferrite.FunctionValues)
    return Ferrite.FunctionValues(fv.ip, view(fv.Nx, i, :, :), fv.Nξ,
                                  view(fv.dNdx, i, :, :), fv.dNdξ, nothing, nothing)
end
function get_worker_part(i, fv::Ferrite.GeometryMapping)
    return Ferrite.GeometryMapping(fv.ip, view(fv.M, i, :, :), fv.dMdξ, fv.d2Mdξ2)
end

### GPUCellCache

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

function GPUCellCache(sdh::GPUSubDofHandler, grid::GPUGrid{dim}, descriptor::CudaTaskDescriptor,
                      flags::UpdateFlags = UpdateFlags()) where dim
    @allowscalar begin
        n      = sdh.ndofs_per_cell
        N      = Ferrite.nnodes_per_cell(grid, 1)
        W      = descriptor.num_workers
        nodes  = CUDA.zeros(Int, W, N)
        coords = CUDA.zeros(Vec{dim, get_coordinate_eltype(grid)}, W, N)
        dofs   = CUDA.zeros(Int, W, n)
    end
    return GPUCellCache(flags, grid, -1, nodes, coords, sdh, dofs)
end

function get_worker_part(i, cc::GPUCellCache, cellid)
    return GPUCellCache(cc.flags, cc.grid, cellid,
                        view(cc.nodes, i, :), view(cc.coords, i, :), cc.sdh, view(cc.dofs, i, :))
end

Ferrite.celldofs(cc::GPUCellCache) = cc.dofs

### GPUCSCAssembler

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
        r1  = A.colptr[col]
        r2  = A.colptr[col + 1] - 1
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

### GPUConstraintHandler

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

function meandiag(K::CuSparseMatrixCSC{T}) where T
    n = size(K, 1)
    diag = CUDA.zeros(T, n)
    threads = min(256, n)
    @cuda threads=threads blocks=cld(n, threads) meandiag_kernel!(
        diag, SparseArrays.getcolptr(K), rowvals(K), nonzeros(K), n)
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

function Ferrite.apply!(K::CuSparseMatrixCSC{T}, f::CuVector{T}, ch::GPUConstraintHandler{T}, applyzero::Bool = false) where T
    n_prescribed = length(ch.prescribed_dofs)
    n_prescribed == 0 && return

    colptr    = SparseArrays.getcolptr(K)
    rv        = rowvals(K)
    nz        = nonzeros(K)
    nnz       = length(nz)
    m         = meandiag(K)
    threads_p = min(256, n_prescribed)
    blocks_p  = cld(n_prescribed, threads_p)

    # Step 1+2: f -= K[:,d]*v, then zero prescribed columns
    @cuda threads=threads_p blocks=blocks_p apply_inhom_zero_cols_kernel!(
        f, colptr, rv, nz, ch.prescribed_dofs, ch.inhomogeneities, n_prescribed, applyzero)

    # Step 3: zero prescribed rows
    threads_z = min(256, nnz)
    @cuda threads=threads_z blocks=cld(nnz, threads_z) apply_zero_rows_kernel!(
        rv, nz, ch.is_prescribed, nnz)

    # Step 4: set diagonal and rhs
    @cuda threads=threads_p blocks=blocks_p apply_set_diag_kernel!(
        colptr, rv, nz, f, ch.prescribed_dofs, ch.inhomogeneities, T(m), n_prescribed, applyzero)

    return
end

function Ferrite.apply_zero!(K::CuSparseMatrixCSC{T}, f::CuVector{T}, ch::GPUConstraintHandler{T}) where T
    return Ferrite.apply!(K, f, ch, true)
end

### Assembly

function assemble_element!(Ke::AbstractMatrix, cv::CellValues)
    n_basefuncs = getnbasefunctions(cv)
    fill!(Ke, 0)
    for q_point in 1:getnquadpoints(cv)
        dΩ = getdetJdV(cv, q_point)
        for i in 1:n_basefuncs
            δu = shape_gradient(cv, q_point, i)
            for j in 1:n_basefuncs
                Ke[i, j] += (δu ⋅ shape_gradient(cv, q_point, j)) * dΩ
            end
        end
    end
    return Ke
end

function assemble_cell!(Ke, cell, cv, assembler)
    reinit!(cv, nothing, cell.coords)
    assemble_element!(Ke, cv)
    assemble!(assembler, celldofs(cell), Ke)
end

function assembly_kernel!(assembler, color, cell_cache, cv, Ke)
    i = threadIdx().x
    i > length(color) && return

    cellid = color[i]
    cv_i   = get_worker_part(i, cv)
    cc_i   = get_worker_part(i, cell_cache, cellid)

    cc_i.flags.nodes  && Ferrite.cellnodes!(cc_i.nodes, cc_i.grid, cellid)
    cc_i.flags.coords && Ferrite.getcoordinates!(cc_i.coords, cc_i.grid, cellid)
    cc_i.sdh !== nothing && cc_i.flags.dofs && Ferrite.celldofs!(cc_i.dofs, cc_i.sdh, cellid)

    Ferrite.reinit!(cv_i, nothing, cc_i.coords)

    assemble_cell!(view(Ke, i, :, :), cc_i, cv_i, assembler)

    return nothing
end

function assemble_global!(cv::CellValues, K::CuSparseMatrixCSC, cell_cache, colors::Vector, Ke)
    assembler = start_assemble(K)
    for color in colors
        @cuda threads=length(color) assembly_kernel!(assembler, color, cell_cache, cv, Ke)
        CUDA.synchronize()
    end
    return nothing
end

function assemble_global!(cv::CellValues, K::SparseMatrixCSC, dh::DofHandler)
    n_basefuncs = getnbasefunctions(cv)
    Ke = zeros(Float32, n_basefuncs, n_basefuncs)
    assembler = start_assemble(K)
    for cell in CellIterator(dh)
        assemble_cell!(Ke, cell, cv, assembler)
    end
    return nothing
end

### --- Example / Tests ---

grid = generate_grid(Hexahedron, (10, 10, 10), Vec{3}((-1.f0,-1.f0,-1.f0)), Vec{3}((1.f0,1.f0,1.f0)))
ip   = Lagrange{RefHexahedron, 1}()
qr   = QuadratureRule{RefHexahedron}(Float32, 2)
cv   = CellValues(qr, ip)
dh   = DofHandler(grid)
add!(dh, :u, ip)
close!(dh)
K = allocate_matrix(SparseMatrixCSC{Float32, Int}, dh)

dh_gpu     = GPUDofHandler(dh)
sdh        = dh_gpu.subdofhandlers[1]
colors     = create_coloring(grid, collect(Int, dh.subdofhandlers[1].cellset))
K_gpu      = CuSparseMatrixCSC(K)
colors_gpu = CuVector.(colors)
n_workers  = maximum(length.(colors_gpu))
cv_gpu     = adapt(CudaTaskDescriptor(n_workers), cv)
cell_cache = GPUCellCache(sdh, get_grid(dh_gpu), CudaTaskDescriptor(n_workers))
Ke         = CUDA.zeros(n_workers, getnbasefunctions(cv), getnbasefunctions(cv))

assemble_global!(cv_gpu, K_gpu, cell_cache, colors_gpu, Ke)
assemble_global!(cv, K, dh)

using Test
@test CuSparseMatrixCSC(K) ≈ K_gpu

# Dirichlet BCs — homogeneous (left + right)
ch = ConstraintHandler(dh)
add!(ch, Dirichlet(:u, union(getfacetset(grid, "left"), getfacetset(grid, "right")), (x, t) -> 0.0))
close!(ch)

ch_gpu = GPUConstraintHandler(ch, Float32)
f      = zeros(Float32, ndofs(dh))
f_gpu  = CUDA.zeros(Float32, ndofs(dh))

apply!(K, f, ch)
apply!(K_gpu, f_gpu, ch_gpu)

@test CuSparseMatrixCSC(K) ≈ K_gpu
@test CuVector(f) ≈ f_gpu

# Dirichlet BCs — inhomogeneous (left + right + top + bottom)
ch2 = ConstraintHandler(dh)
add!(ch2, Dirichlet(:u, union(getfacetset(grid, "left"), getfacetset(grid, "right"),
                              getfacetset(grid, "top"),  getfacetset(grid, "bottom")), (x, t) -> 1.0))
close!(ch2)

ch_gpu2 = GPUConstraintHandler(ch2, Float32)
f2      = zeros(Float32, ndofs(dh))
f_gpu2  = CUDA.zeros(Float32, ndofs(dh))

apply!(K, f2, ch2)
apply!(K_gpu, f_gpu2, ch_gpu2)

@test CuSparseMatrixCSC(K) ≈ K_gpu
@test CuVector(f2) ≈ f_gpu2
