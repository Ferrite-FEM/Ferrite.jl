# # Matrix-free operator evaluation on the GPU: exploration
#
# This script explores how the matrix-free infrastructure in `src/matrix_free.jl` maps to
# the GPU, using KernelAbstractions.jl so the same kernel runs on any backend. Run it
# as-is for the CPU backend, or on macOS:
#
#     julia --project=experiments -e 'import Pkg; Pkg.add("Metal")'
#     julia --project=experiments experiments/matrix_free_gpu.jl metal
#
# (and analogously with CUDA.jl / `CUDABackend()` etc.)
#
# ## Portability of the pieces
#
# Taking stock of what `src/matrix_free.jl` defines and how each piece relates to a GPU:
#
# - The **contraction kernels** (`contract_1!/2!/3!`) are plain loops with compile-time
#   trip counts over `AbstractArray`s -- they compile for the GPU *unchanged* and are
#   called from inside the device kernel below, operating on stack-allocated `MArray`s
#   with the 1D matrices passed by value as `SMatrix`. (The only change this required in
#   Ferrite was relaxing their signature from `Matrix` to `AbstractMatrix`.)
# - The **`ConstrainedDofMap` / dofmap / `Dq` data** is plain, isbits-element arrays. This
#   is what `Adapt.jl` is for: `adapt(backend, x)` converts the storage to the backend's
#   array type (KernelAbstractions implements the adaptor for every backend). For the
#   fields of `ConstrainedDofMap` to be adaptable the struct would need its field types as
#   type parameters (they are currently hardcoded `Matrix{Int}`/`Vector`) plus an
#   `Adapt.adapt_structure` definition -- mechanical changes. In this script the arrays
#   are passed to the kernel individually, which needs no struct support at all.
# - The **`TensorProductEvaluator` should _not_ be adapted**. Its design -- heap-allocated
#   scratch shared across a serial cell loop -- is inherently the CPU model (one evaluator
#   per task). On the GPU the scratch must live in registers or workgroup-local memory,
#   private to the cell being processed, so the "evaluator" is *constructed inside the
#   kernel*. This mirrors deal.II, where `Portable::FEEvaluation` is a device-side object
#   created in the user's kernel, unrelated to the host-side `MatrixFree` data holder.
# - **Element type**: Apple GPUs (Metal) do not support `Float64`, which is why the
#   evaluator constructor takes `T` (`TensorProductEvaluator(ip, qr1d; T = Float32)`) and
#   why everything below is generic in `T`.
# - **Constraints**: the `ConstrainedDofMap` encoding (negative indices into a compressed
#   master/coefficient table) is exactly index lists + flat arrays, i.e. GPU friendly; the
#   gather/scatter branches port directly. Not exercised in this script.
#
# The kernel below uses the simplest possible parallelization: **one thread per cell**,
# with all scratch in thread-private `MArray`s and the scatter step using atomics (two
# cells sharing a dof may scatter concurrently). This is correctness-first; the
# established fast layout is one *workgroup* per cell (or batch of cells) with the
# contractions parallelized over the workgroup and scratch in `@localmem` -- see the notes
# at the end.

using Ferrite
using KernelAbstractions, Adapt, Atomix, StaticArrays
using LinearAlgebra, SparseArrays, Printf, Test

# ## Backend selection

backend, default_T = if "metal" in ARGS
    using Metal
    MetalBackend(), Float32
elseif "cuda" in ARGS
    using CUDA
    CUDABackend(), Float64
elseif "amdgpu" in ARGS
    using AMDGPU
    ROCBackend(), Float64
else
    CPU(), Float64
end
## Pass `f32` to force Float32 also on non-Metal backends (to test the Metal-realistic
## precision without Metal hardware)
"f32" in ARGS && (default_T = Float32)
@info "Using backend" backend default_T

# ## The device kernels
#
# The five step structure (gather, evaluate, pointwise, integrate, scatter) for one cell
# per thread. Note that the evaluate and integrate helpers call the *same*
# `Ferrite.contract_*!` functions as the CPU evaluator, and the pointwise steps use the
# same Tensors.jl types and operations -- all of it compiles for the device. The 1D
# matrices are passed by value as `SMatrix` (they are tiny), which also provides the
# compile-time sizes `NQ` and `N`. All scratch is thread-private `MArray`s ("the
# evaluator", constructed device-side); everything must inline so that the `MArray`s never
# escape (escape means heap allocation, which is fatal in device code).

@inline function device_scratch(::Val{NQ}, ::Val{N}, ::Type{T}) where {NQ, N, T}
    return (
        ue = MArray{Tuple{N, N, N}, T}(undef),
        ye = MArray{Tuple{N, N, N}, T}(undef),
        tmp = MArray{Tuple{N, N, N}, T}(undef),
        t1 = MArray{Tuple{NQ, N, N}, T}(undef),
        t2 = MArray{Tuple{NQ, N, N}, T}(undef),
        s1 = MArray{Tuple{NQ, NQ, N}, T}(undef),
        s2 = MArray{Tuple{NQ, NQ, N}, T}(undef),
        s3 = MArray{Tuple{NQ, NQ, N}, T}(undef),
    )
end

@inline function device_evaluate_gradients!(gx, gy, gz, ue, c, B, D, ::Val{N}, ::Val{NQ}) where {N, NQ}
    n, nq = Val(N), Val(NQ)
    Ferrite.contract_1!(c.t1, B, ue, n, nq)
    Ferrite.contract_1!(c.t2, D, ue, n, nq)
    Ferrite.contract_2!(c.s1, B, c.t1, n, nq)
    Ferrite.contract_2!(c.s2, D, c.t1, n, nq)
    Ferrite.contract_2!(c.s3, B, c.t2, n, nq)
    Ferrite.contract_3!(gx, B, c.s3, n, nq)
    Ferrite.contract_3!(gy, B, c.s2, n, nq)
    Ferrite.contract_3!(gz, D, c.s1, n, nq)
    return
end

@inline function device_integrate_gradients!(ye, gx, gy, gz, c, Bᵀ, Dᵀ, ::Val{N}, ::Val{NQ}) where {N, NQ}
    n, nq = Val(N), Val(NQ)
    Ferrite.contract_3!(c.s1, Bᵀ, gx, nq, n)
    Ferrite.contract_2!(c.t1, Bᵀ, c.s1, nq, n)
    Ferrite.contract_1!(ye, Dᵀ, c.t1, nq, n)
    Ferrite.contract_3!(c.s1, Bᵀ, gy, nq, n)
    Ferrite.contract_2!(c.t1, Dᵀ, c.s1, nq, n)
    Ferrite.contract_1!(c.tmp, Bᵀ, c.t1, nq, n)
    @inbounds for l in eachindex(ye)
        ye[l] += c.tmp[l]
    end
    Ferrite.contract_3!(c.s1, Dᵀ, gz, nq, n)
    Ferrite.contract_2!(c.t1, Bᵀ, c.s1, nq, n)
    Ferrite.contract_1!(c.tmp, Bᵀ, c.t1, nq, n)
    @inbounds for l in eachindex(ye)
        ye[l] += c.tmp[l]
    end
    return
end

# The heat kernel: scalar field, pointwise operation `D_q ⋅ ĝ`.

@kernel function heat_pa_kernel!(
        y, @Const(x), @Const(dofmap), @Const(Dq),
        B::SMatrix{NQ, N, T}, D::SMatrix{NQ, N, T},
    ) where {NQ, N, T}
    e = @index(Global, Linear)
    c = device_scratch(Val(NQ), Val(N), T)
    gx = MArray{Tuple{NQ, NQ, NQ}, T}(undef)
    gy = MArray{Tuple{NQ, NQ, NQ}, T}(undef)
    gz = MArray{Tuple{NQ, NQ, NQ}, T}(undef)
    n, nq = Val(N), Val(NQ)
    ## 1. Gather
    @inbounds for l in 1:(N * N * N)
        c.ue[l] = x[dofmap[l, e]]
    end
    ## 2. Evaluate: reference gradients in the quadrature points
    device_evaluate_gradients!(gx, gy, gz, c.ue, c, B, D, n, nq)
    ## 3. Pointwise application of the stored D_q
    @inbounds for q in 1:(NQ * NQ * NQ)
        h = Dq[q, e] ⋅ Vec(gx[q], gy[q], gz[q])
        gx[q] = h[1]
        gy[q] = h[2]
        gz[q] = h[3]
    end
    ## 4. Integrate: transposed contractions
    device_integrate_gradients!(c.ye, gx, gy, gz, c, transpose(B), transpose(D), n, nq)
    ## 5. Scatter, atomically since neighboring cells share dofs
    @inbounds for l in 1:(N * N * N)
        Atomix.@atomic y[dofmap[l, e]] += c.ye[l]
    end
end

# A variant with a plain (racy!) scatter, selected with the `noatomics` flag. The result is
# wrong at shared dofs, but if this kernel runs where the atomic one traps, the problem is
# the atomics support of the backend and not the kernel body.

@kernel function heat_pa_kernel_noatomics!(
        y, @Const(x), @Const(dofmap), @Const(Dq),
        B::SMatrix{NQ, N, T}, D::SMatrix{NQ, N, T},
    ) where {NQ, N, T}
    e = @index(Global, Linear)
    c = device_scratch(Val(NQ), Val(N), T)
    gx = MArray{Tuple{NQ, NQ, NQ}, T}(undef)
    gy = MArray{Tuple{NQ, NQ, NQ}, T}(undef)
    gz = MArray{Tuple{NQ, NQ, NQ}, T}(undef)
    n, nq = Val(N), Val(NQ)
    @inbounds for l in 1:(N * N * N)
        c.ue[l] = x[dofmap[l, e]]
    end
    device_evaluate_gradients!(gx, gy, gz, c.ue, c, B, D, n, nq)
    @inbounds for q in 1:(NQ * NQ * NQ)
        h = Dq[q, e] ⋅ Vec(gx[q], gy[q], gz[q])
        gx[q] = h[1]
        gy[q] = h[2]
        gz[q] = h[3]
    end
    device_integrate_gradients!(c.ye, gx, gy, gz, c, transpose(B), transpose(D), n, nq)
    @inbounds for l in 1:(N * N * N)
        y[dofmap[l, e]] += c.ye[l] # NOTE: racy, for isolating backend atomics issues only
    end
end

# ## The workgroup-per-cell kernel
#
# The layout deal.II and MFEM use: one *workgroup* (CUDA thread block / Metal threadgroup /
# AMDGPU wavefront group) per cell, with the scratch in `@localmem` -- the workgroup-shared
# on-chip memory (CUDA "shared memory", AMDGPU "LDS", Metal "threadgroup memory";
# KernelAbstractions abstracts all of them) -- and the contractions parallelized over the
# threads of the workgroup, with `@synchronize` barriers between the phases.
#
# For `p = 2` with a 3-point 1D rule every scratch tensor is 3×3×3 (`NQ == N == M`), so a
# workgroup of `M³ = 27` threads is a perfect fit: thread `(α, β, γ)` owns entry
# `(α, β, γ)` of every buffer and computes exactly one entry per contraction phase (a
# 3-term fma loop reading its neighbors' entries from local memory). Compared to the
# thread-per-cell kernel this removes the ~300 floats of register spill per thread (each
# thread now holds a couple of accumulators), and the gather is cooperative. The general
# `NQ != N` case needs a per-stage thread map; here we simply require square 1D matrices.
#
# Note for the CPU backend: thread-local variables do not survive `@synchronize` there, so
# each phase re-derives its indices from `@index` (which is always valid) instead of
# computing them once.

@inline function _thread_ijk(l, ::Val{M}) where {M}
    return ((l - 1) % M + 1, ((l - 1) ÷ M) % M + 1, (l - 1) ÷ (M * M) + 1)
end

@kernel function heat_pa_kernel_wgpc!(
        y, @Const(x), @Const(dofmap), @Const(Dq),
        B::SMatrix{M, M, T}, D::SMatrix{M, M, T},
    ) where {M, T}
    ue = @localmem T (M, M, M)
    t1 = @localmem T (M, M, M)
    t2 = @localmem T (M, M, M)
    s1 = @localmem T (M, M, M)
    s2 = @localmem T (M, M, M)
    s3 = @localmem T (M, M, M)
    gx = @localmem T (M, M, M)
    gy = @localmem T (M, M, M)
    gz = @localmem T (M, M, M)
    ## NOTE: on the CPU backend thread-local variables do not survive `@synchronize`, and
    ## `@index` calls must sit at the top level of each phase (not nested in other macro
    ## blocks) -- hence every phase re-derives its indices.
    ## Gather: one dof per thread (cooperative)
    l = @index(Local, Linear)
    gl = @index(Global, Linear)
    e = (gl - l) ÷ (M * M * M) + 1
    @inbounds ue[l] = x[dofmap[l, e]]
    @synchronize
    ## Contract dimension 1: t1 = C1(B, ue), t2 = C1(D, ue)
    l = @index(Local, Linear)
    α, β, γ = _thread_ijk(l, Val(M))
    @inbounds begin
        a1 = zero(T)
        a2 = zero(T)
        for i in 1:M
            v = ue[i, β, γ]
            a1 = muladd(B[α, i], v, a1)
            a2 = muladd(D[α, i], v, a2)
        end
        t1[l] = a1
        t2[l] = a2
    end
    @synchronize
    ## Contract dimension 2: s1 = C2(B, t1), s2 = C2(D, t1), s3 = C2(B, t2)
    l = @index(Local, Linear)
    α, β, γ = _thread_ijk(l, Val(M))
    @inbounds begin
        a1 = zero(T)
        a2 = zero(T)
        a3 = zero(T)
        for j in 1:M
            v1 = t1[α, j, γ]
            v2 = t2[α, j, γ]
            a1 = muladd(B[β, j], v1, a1)
            a2 = muladd(D[β, j], v1, a2)
            a3 = muladd(B[β, j], v2, a3)
        end
        s1[l] = a1
        s2[l] = a2
        s3[l] = a3
    end
    @synchronize
    ## Contract dimension 3 (gx = C3(B, s3) etc.), fused with the pointwise application of
    ## D_q: the thread that computes the gradient at quadrature point l = (α, β, γ) also
    ## owns that point's slot of gx/gy/gz, so no barrier is needed in between.
    l = @index(Local, Linear)
    gl = @index(Global, Linear)
    e = (gl - l) ÷ (M * M * M) + 1
    α, β, γ = _thread_ijk(l, Val(M))
    @inbounds begin
        a1 = zero(T)
        a2 = zero(T)
        a3 = zero(T)
        for k in 1:M
            a1 = muladd(B[γ, k], s3[α, β, k], a1)
            a2 = muladd(B[γ, k], s2[α, β, k], a2)
            a3 = muladd(D[γ, k], s1[α, β, k], a3)
        end
        h = Dq[l, e] ⋅ Vec(a1, a2, a3)
        gx[l] = h[1]
        gy[l] = h[2]
        gz[l] = h[3]
    end
    @synchronize
    ## Integrate, transposed contractions: contract dimension 3 with Bᵀ/Dᵀ
    l = @index(Local, Linear)
    α, β, γ = _thread_ijk(l, Val(M))
    @inbounds begin
        a1 = zero(T)
        a2 = zero(T)
        a3 = zero(T)
        for k in 1:M
            a1 = muladd(B[k, γ], gx[α, β, k], a1) # Bᵀ[γ, k] == B[k, γ]
            a2 = muladd(B[k, γ], gy[α, β, k], a2)
            a3 = muladd(D[k, γ], gz[α, β, k], a3)
        end
        s1[l] = a1
        s2[l] = a2
        s3[l] = a3
    end
    @synchronize
    ## Contract dimension 2 with Bᵀ/Dᵀ (t3 reuses the gx buffer)
    l = @index(Local, Linear)
    α, β, γ = _thread_ijk(l, Val(M))
    @inbounds begin
        a1 = zero(T)
        a2 = zero(T)
        a3 = zero(T)
        for j in 1:M
            a1 = muladd(B[j, β], s1[α, j, γ], a1)
            a2 = muladd(D[j, β], s2[α, j, γ], a2)
            a3 = muladd(B[j, β], s3[α, j, γ], a3)
        end
        t1[l] = a1
        t2[l] = a2
        gx[l] = a3
    end
    @synchronize
    ## Contract dimension 1 with Dᵀ/Bᵀ, sum the three components, and scatter. The result
    ## entry is owned by this thread, so it goes straight from registers to the atomic add.
    l = @index(Local, Linear)
    gl = @index(Global, Linear)
    e = (gl - l) ÷ (M * M * M) + 1
    α, β, γ = _thread_ijk(l, Val(M))
    @inbounds begin
        a = zero(T)
        for i in 1:M
            a = muladd(D[i, α], t1[i, β, γ], a)
            a = muladd(B[i, α], t2[i, β, γ] + gx[i, β, γ], a)
        end
        Atomix.@atomic y[dofmap[l, e]] += a
    end
end

# Host-side operator: the same data as the CPU `MatrixFreeOperator` (permuted dofmap and
# per-quadrature-point data) but with the storage adapted to the backend, plus the 1D
# matrices as `SMatrix`.

struct GPUHeatOperator{TB, TD <: AbstractMatrix, TQ <: AbstractMatrix, SB, SD}
    backend::TB
    dofmap::TD
    Dq::TQ
    B::SB
    D::SD
    ndofs::Int
end

const use_atomics = !("noatomics" in ARGS)

function LinearAlgebra.mul!(y::AbstractVector, A::GPUHeatOperator, x::AbstractVector)
    fill!(y, 0)
    kernel! = use_atomics ? heat_pa_kernel!(A.backend) : heat_pa_kernel_noatomics!(A.backend)
    kernel!(y, x, A.dofmap, A.Dq, A.B, A.D; ndrange = size(A.dofmap, 2))
    KernelAbstractions.synchronize(A.backend)
    return y
end
Base.:*(A::GPUHeatOperator, x::AbstractVector) = mul!(similar(x, A.ndofs), A, x)

# ## Problem setup (on the host, in Float64, like the CPU how-to)

function setup(n, ::Type{T}) where {T}
    grid = generate_grid(Hexahedron, (n, n, n))
    ip = Lagrange{RefHexahedron, 2}()
    dh = close!(add!(DofHandler(grid), :u, ip))
    qr1d = QuadratureRule{RefLine}(3)
    p1d = Ferrite.getpoints(qr1d)
    w1d = Ferrite.getweights(qr1d)
    qr = QuadratureRule{RefHexahedron}(
        vec([wx * wy * wz for wx in w1d, wy in w1d, wz in w1d]),
        vec([Vec(px[1], py[1], pz[1]) for px in p1d, py in p1d, pz in p1d]),
    )
    κ(x) = 2.0 + sinpi(x[1]) * cospi(2 * x[2]) * sinpi(x[3] / 2)
    dofmap = Ferrite.lexicographic_dofmap(dh, ip)
    Dq = Ferrite.quadrature_point_data(grid, qr) do x, J, w
        Jinv = inv(J)
        return det(J) * w * κ(x) * dott(Jinv)
    end
    ## Narrow the per-point data to T (SymmetricTensor{2, 3, T, 6}) for the device
    DqT = map(d -> convert(SymmetricTensor{2, 3, T}, d), Dq)
    ## 1D matrices in T, as SMatrix
    ev = Ferrite.TensorProductEvaluator(ip, qr1d; T = T)
    B = SMatrix{size(ev.B, 1), size(ev.B, 2), T}(ev.B)
    D = SMatrix{size(ev.D, 1), size(ev.D, 2), T}(ev.D)
    return grid, ip, dh, qr, κ, dofmap, DqT, B, D
end

T = default_T
grid, ip, dh, qr, κ, dofmap, Dq, B, D = setup(16, T)

# ## Moving the data to the device
#
# This is the `Adapt.jl` step: `adapt(backend, x)` returns `x` backed by the backend's
# array type (a no-op for the CPU backend). Note that the *element* types (the
# `SymmetricTensor`s in `Dq`) survive -- only the storage changes.

A = GPUHeatOperator(
    backend,
    adapt(backend, dofmap),
    adapt(backend, Dq),
    B, D,
    ndofs(dh),
)

x_h = rand(T, ndofs(dh))
x_d = adapt(backend, x_h)
y_d = adapt(backend, zeros(T, ndofs(dh)))

mul!(y_d, A, x_d)
y_h = Array(y_d);

# ## Verification against the assembled matrix (assembled in Float64 on the host)

function assemble_sparse_heat(dh, ip, qr, κ)
    cv = CellValues(qr, ip)
    K = allocate_matrix(dh)
    assembler = start_assemble(K)
    nbf = getnbasefunctions(cv)
    Ke = zeros(nbf, nbf)
    for cell in CellIterator(dh)
        reinit!(cv, cell)
        fill!(Ke, 0)
        for q in 1:getnquadpoints(cv)
            dΩ = getdetJdV(cv, q) * κ(spatial_coordinate(cv, q, getcoordinates(cell)))
            for i in 1:nbf, j in 1:nbf
                Ke[i, j] += (shape_gradient(cv, q, i) ⋅ shape_gradient(cv, q, j)) * dΩ
            end
        end
        assemble!(assembler, celldofs(cell), Ke)
    end
    return K
end

K = assemble_sparse_heat(dh, ip, qr, κ)
y_ref = K * Float64.(x_h)

rel_err = norm(y_h - y_ref) / norm(y_ref)
@printf "relative error vs assembled matrix (T = %s): %.3e\n" T rel_err
## Float32 accumulation over 27 dofs and atomics: expect ~sqrt(eps(T)) at worst
if use_atomics
    @test rel_err < (T === Float32 ? 5.0f-5 : 1.0e-13)
else
    @warn "noatomics: racy scatter, skipping the correctness check" rel_err
end

# ## Timing

function best_time(f!, n = 20)
    f!()
    return minimum(@elapsed(f!()) for _ in 1:n)
end

t_dev = best_time(() -> mul!(y_d, A, x_d))
t_csr = best_time(() -> mul!(y_ref, K, Float64.(x_h)))
@printf "matvec: device (thread-per-cell) %.3f ms | host cuSPARSE-analog (SparseMatrixCSC) %.3f ms\n" 1000t_dev 1000t_csr

# ## The workgroup-per-cell variant, verified and timed
#
# Same data, different kernel and launch geometry: `M³` threads per workgroup, one
# workgroup per cell (`ndrange = M³ * ncells` with workgroup size `M³`).

struct GPUHeatOperatorWGPC{TB, TD <: AbstractMatrix, TQ <: AbstractMatrix, SB, SD}
    backend::TB
    dofmap::TD
    Dq::TQ
    B::SB
    D::SD
    ndofs::Int
end

function LinearAlgebra.mul!(y::AbstractVector, A::GPUHeatOperatorWGPC, x::AbstractVector)
    M = size(A.B, 1)
    @assert M == size(A.B, 2) # the cooperative kernel assumes NQ == N
    fill!(y, 0)
    kernel! = heat_pa_kernel_wgpc!(A.backend, M^3)
    kernel!(y, x, A.dofmap, A.Dq, A.B, A.D; ndrange = M^3 * size(A.dofmap, 2))
    KernelAbstractions.synchronize(A.backend)
    return y
end

A_wg = GPUHeatOperatorWGPC(backend, A.dofmap, A.Dq, B, D, ndofs(dh))

mul!(y_d, A_wg, x_d)
rel_err_wg = norm(Array(y_d) - y_ref) / norm(y_ref)
@printf "workgroup-per-cell: relative error vs assembled matrix: %.3e\n" rel_err_wg
@test rel_err_wg < (T === Float32 ? 5.0f-5 : 1.0e-13)

t_wg = best_time(() -> mul!(y_d, A_wg, x_d))
@printf "matvec: device (workgroup-per-cell) %.3f ms | (thread-per-cell) %.3f ms\n" 1000t_wg 1000t_dev

# Note that on the *CPU* backend the workgroup-per-cell kernel is expected to be *slower*
# than thread-per-cell: the CPU emulates workgroups and barriers by splitting the kernel
# into one loop per phase, which costs overhead and buys nothing (a CPU thread has no
# register pressure problem at this size and nobody to cooperate with). The comparison
# only becomes meaningful on an actual GPU, where the localmem layout eliminates the
# per-thread register spill.

# ## Linear elasticity on the device
#
# The vector valued case, mirroring the elasticity section of the CPU how-to: gather and
# evaluate the reference gradient per displacement component (the dofmap from
# `lexicographic_dofmap(dh, ipv)` blocks the components, so component `c` of the local
# vector is `dofmap[l + (c - 1) * N³, e]`), assemble the 3×3 gradient tensor per
# quadrature point, apply the material pointwise, and integrate/scatter per component.
# The scratch is now ~9 gradient buffers plus the shared temporaries -- heavy register
# pressure for a thread-per-cell layout, which the workgroup-per-cell layout would fix.

@kernel function elast_pa_kernel!(
        y, @Const(x), @Const(dofmap), @Const(qp),
        B::SMatrix{NQ, N, T}, D::SMatrix{NQ, N, T},
    ) where {NQ, N, T}
    e = @index(Global, Linear)
    c = device_scratch(Val(NQ), Val(N), T)
    gx1 = MArray{Tuple{NQ, NQ, NQ}, T}(undef)
    gy1 = MArray{Tuple{NQ, NQ, NQ}, T}(undef)
    gz1 = MArray{Tuple{NQ, NQ, NQ}, T}(undef)
    gx2 = MArray{Tuple{NQ, NQ, NQ}, T}(undef)
    gy2 = MArray{Tuple{NQ, NQ, NQ}, T}(undef)
    gz2 = MArray{Tuple{NQ, NQ, NQ}, T}(undef)
    gx3 = MArray{Tuple{NQ, NQ, NQ}, T}(undef)
    gy3 = MArray{Tuple{NQ, NQ, NQ}, T}(undef)
    gz3 = MArray{Tuple{NQ, NQ, NQ}, T}(undef)
    n, nq = Val(N), Val(NQ)
    Bᵀ = transpose(B)
    Dᵀ = transpose(D)
    NNN = N * N * N
    ## Gather + evaluate, one component at a time (ue is reused)
    @inbounds for l in 1:NNN
        c.ue[l] = x[dofmap[l, e]]
    end
    device_evaluate_gradients!(gx1, gy1, gz1, c.ue, c, B, D, n, nq)
    @inbounds for l in 1:NNN
        c.ue[l] = x[dofmap[l + NNN, e]]
    end
    device_evaluate_gradients!(gx2, gy2, gz2, c.ue, c, B, D, n, nq)
    @inbounds for l in 1:NNN
        c.ue[l] = x[dofmap[l + 2 * NNN, e]]
    end
    device_evaluate_gradients!(gx3, gy3, gz3, c.ue, c, B, D, n, nq)
    ## Pointwise: ε = sym(ĝ ⋅ J⁻¹), σ = λ tr(ε) I + 2μ ε, ĥ = det(J) w σ ⋅ J⁻ᵀ
    @inbounds for q in 1:(NQ * NQ * NQ)
        d = qp[q, e]
        ĝ = Tensor{2, 3, T}(
            (
                gx1[q], gx2[q], gx3[q],
                gy1[q], gy2[q], gy3[q],
                gz1[q], gz2[q], gz3[q],
            )
        )
        ε = symmetric(ĝ ⋅ d.Jinv)
        σw = d.λw * tr(ε) * one(ε) + 2 * d.μw * ε
        h = σw ⋅ transpose(d.Jinv)
        gx1[q] = h[1, 1]; gx2[q] = h[2, 1]; gx3[q] = h[3, 1]
        gy1[q] = h[1, 2]; gy2[q] = h[2, 2]; gy3[q] = h[3, 2]
        gz1[q] = h[1, 3]; gz2[q] = h[2, 3]; gz3[q] = h[3, 3]
    end
    ## Integrate + scatter, one component at a time (ye is reused)
    device_integrate_gradients!(c.ye, gx1, gy1, gz1, c, Bᵀ, Dᵀ, n, nq)
    @inbounds for l in 1:NNN
        Atomix.@atomic y[dofmap[l, e]] += c.ye[l]
    end
    device_integrate_gradients!(c.ye, gx2, gy2, gz2, c, Bᵀ, Dᵀ, n, nq)
    @inbounds for l in 1:NNN
        Atomix.@atomic y[dofmap[l + NNN, e]] += c.ye[l]
    end
    device_integrate_gradients!(c.ye, gx3, gy3, gz3, c, Bᵀ, Dᵀ, n, nq)
    @inbounds for l in 1:NNN
        Atomix.@atomic y[dofmap[l + 2 * NNN, e]] += c.ye[l]
    end
end

struct GPUElasticityOperator{TB, TD <: AbstractMatrix, TQ <: AbstractMatrix, SB, SD}
    backend::TB
    dofmap::TD
    qp::TQ
    B::SB
    D::SD
    ndofs::Int
end

function LinearAlgebra.mul!(y::AbstractVector, A::GPUElasticityOperator, x::AbstractVector)
    fill!(y, 0)
    kernel! = elast_pa_kernel!(A.backend)
    kernel!(y, x, A.dofmap, A.qp, A.B, A.D; ndrange = size(A.dofmap, 2))
    KernelAbstractions.synchronize(A.backend)
    return y
end

# Host setup: same grid and rules; heterogeneous Lamé parameters folded into the per-point
# data as in the CPU how-to (J⁻¹ + premultiplied λ, μ: 11 floats per point).

ipv = ip^3
dh_e = close!(add!(DofHandler(grid), :u, ipv))
λ(x) = 2.0 + x[1]
μ(x) = 1.0 + 0.5 * sinpi(x[3])
dofmap_e = Ferrite.lexicographic_dofmap(dh_e, ipv)
qp_e = map(
    d -> (Jinv = convert(Tensor{2, 3, T}, d.Jinv), λw = T(d.λw), μw = T(d.μw)),
    Ferrite.quadrature_point_data(grid, qr) do x, J, w
        (Jinv = inv(J), λw = det(J) * w * λ(x), μw = det(J) * w * μ(x))
    end,
)

A_e = GPUElasticityOperator(backend, adapt(backend, dofmap_e), adapt(backend, qp_e), B, D, ndofs(dh_e))

xe_h = rand(T, ndofs(dh_e))
xe_d = adapt(backend, xe_h)
ye_d = adapt(backend, zeros(T, ndofs(dh_e)))
mul!(ye_d, A_e, xe_d)
ye_h = Array(ye_d);

# Verification against the assembled matrix (Float64, host). NOTE: this is the expensive
# part of the script -- ~300 MiB and most of the runtime goes into building this reference
# (the device operator data is ~12 MiB).

function assemble_sparse_elasticity(dh, ipv, qr, λ, μ)
    cv = CellValues(qr, ipv)
    K = allocate_matrix(dh)
    assembler = start_assemble(K)
    nbf = getnbasefunctions(cv)
    Ke = zeros(nbf, nbf)
    for cell in CellIterator(dh)
        reinit!(cv, cell)
        fill!(Ke, 0)
        for q in 1:getnquadpoints(cv)
            x_q = spatial_coordinate(cv, q, getcoordinates(cell))
            dΩ = getdetJdV(cv, q)
            for i in 1:nbf
                εᵢ = shape_symmetric_gradient(cv, q, i)
                for j in 1:nbf
                    εⱼ = shape_symmetric_gradient(cv, q, j)
                    Ke[i, j] += (λ(x_q) * tr(εᵢ) * tr(εⱼ) + 2 * μ(x_q) * (εᵢ ⊡ εⱼ)) * dΩ
                end
            end
        end
        assemble!(assembler, celldofs(cell), Ke)
    end
    return K
end

K_e = assemble_sparse_elasticity(dh_e, ipv, qr, λ, μ)
ye_ref = K_e * Float64.(xe_h)

rel_err_e = norm(ye_h - ye_ref) / norm(ye_ref)
@printf "elasticity: relative error vs assembled matrix (T = %s): %.3e\n" T rel_err_e
use_atomics && @test rel_err_e < (T === Float32 ? 5.0f-5 : 1.0e-13)

t_dev_e = best_time(() -> mul!(ye_d, A_e, xe_d))
t_csr_e = best_time(() -> mul!(ye_ref, K_e, Float64.(xe_h)))
@printf "elasticity matvec: device %.3f ms | host SparseMatrixCSC %.3f ms\n" 1000t_dev_e 1000t_csr_e

# ## Findings and next steps
#
# Measured (2026-09-16, p2 on the 16³ grid; heat: 35937 dofs, elasticity: 107811 dofs):
#
# | run | heat | elasticity |
# |---|---|---|
# | Metal, Apple M3, Float32, thread-per-cell | 0.36-0.53 ms | 0.76 ms |
# | CPU backend, 1 thread, Float64, same kernels | 0.70-0.80 ms | 2.8 ms |
# | host `SparseMatrixCSC` SpMV, 1 thread, Float64 | 0.89 ms | 7.3-7.6 ms |
#
# I.e. the *naive* device kernel beats the serial host SpMV by ~2x for heat and by ~10x
# for elasticity (note the Float32-vs-Float64 and 1-thread caveats), before any of the
# performance work listed below. Elasticity is the decisive case -- more arithmetic per
# byte and a ~40x smaller operator (~12 MiB of quadrature point data + dofmap against
# ~0.5 GB of assembled matrix) -- and the sum-factorized kernel wins there even serially
# on the CPU (2.8 ms vs 7.6 ms).
#
# What this script establishes:
#
# 1. The sum factorization kernels in Ferrite are **device-portable as-is** (after the
#    `Matrix -> AbstractMatrix` relaxation): the same `contract_*!` code runs in the CPU
#    evaluator and inside the KernelAbstractions kernel on `MArray` scratch.
# 2. The **data** side of the design ports with `adapt(backend, ...)` and nothing else:
#    dofmap and per-point data are isbits-element arrays by construction. To pass the
#    `ConstrainedDofMap`/`TensorProductEvaluator` structs (rather than raw arrays) to
#    device code, their hardcoded `Matrix`/`Vector`/`Array` field types must become type
#    parameters, with `Adapt.adapt_structure` for the data holder. For the evaluator this
#    is only worth doing for its role as a *host-side container* of `B`/`D` -- its scratch
#    arrays must not travel to the device (see 3).
# 3. The evaluator's *scratch model* is the real CPU/GPU divide: per-task heap buffers on
#    the CPU versus per-cell registers/local memory on the device. A device-side evaluator
#    (constructed inside the kernel, deal.II `Portable::FEEvaluation`-style) is the right
#    abstraction boundary for a future API.
#
# Performance caveats of the thread-per-cell layout (intentional simplifications):
#
# - ~300 floats of thread-private scratch at p = 2 causes register spilling; the standard
#   fix is one workgroup per cell with the contractions parallelized over the workgroup
#   threads and scratch in `@localmem` (this is what deal.II and MFEM do). The contraction
#   *loop structure* stays the same, so the `contract_*!` functions are the template for
#   those cooperative kernels.
# - The gather/scatter reads `dofmap[l, e]` with `l` fastest, so consecutive threads
#   (consecutive `e`) access memory with stride `nbf` -- transposing the dofmap gives
#   coalesced access.
# - The atomic scatter serializes on shared dofs; grid coloring (`create_coloring`, as in
#   the GPU assembly how-to) removes the atomics at the cost of one kernel launch per
#   color.
# - `Int` (64-bit) indices are wasteful on Metal in particular; the dofmap could be
#   `Int32`.
