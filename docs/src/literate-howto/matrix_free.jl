# # [Matrix-free operator evaluation](@id howto-matrix-free)
#
#-
#md # !!! tip
#md #     This example is also available as a Jupyter notebook:
#md #     [`matrix_free.ipynb`](@__NBVIEWER_ROOT_URL__/howto/matrix_free.ipynb).
#-
#
# ## Introduction
#
# Iterative solvers such as the conjugate gradient method only interact with the linear
# system through matrix-vector products `y = K * x`. This makes it possible to solve
# `K * u = f` without ever storing `K` -- as long as we can compute the *action* of `K` on a
# vector. Avoiding the assembled sparse matrix can save a substantial amount of memory,
# in particular for higher order elements where the number of non-zero entries per row
# grows quickly.
#
# There is a spectrum of choices for how much of the operator to precompute and store
# (using the terminology from [MFEM](https://mfem.org/howto/assembly_levels/)):
#
#  - **Full assembly**: store the global sparse matrix (what the tutorials do).
#  - **Element assembly**: store one dense element matrix `Ke` per cell and compute
#    `y = Σₑ Gᵀₑ (Ke * Gₑ x)` with `Gₑ` the gather/scatter for cell `e`.
#  - **Partial assembly**: store only a small tensor per quadrature point, and apply the
#    reference-element interpolation on the fly.
#  - **Fully matrix-free**: store nothing and recompute also the geometry mapping in every
#    product.
#
# This how-to implements **partial assembly** for the (generalized) Laplace operator
#
# ```math
# (K x)_i = \int_\Omega \nabla \delta u_i \cdot \kappa(\mathbf{x}) \nabla x_h \, d\Omega,
# ```
#
# i.e. the stiffness matrix of the heat equation with a spatially varying conductivity
# ``\kappa(\mathbf{x})``, discretized with quadratic hexahedra. The key observation is that
# the integrand at a quadrature point factors into
#
# ```math
# \hat{\nabla} \delta u_i \cdot \underbrace{\left[ \det(J_q) \, w_q \,
# J_q^{-1} \kappa(\mathbf{x}_q) J_q^{-T} \right]}_{D_q} \hat{\nabla} x_h,
# ```
#
# where ``J_q`` is the Jacobian of the geometry mapping and ``\hat{\nabla}`` denotes
# gradients with respect to the *reference* coordinates. Everything that varies from cell to
# cell -- geometry, coefficient, and quadrature weight -- is collected in the symmetric
# 3×3 tensor ``D_q``, which is precomputed and stored per quadrature point (6 floats).
# The reference-element gradients are the same for every cell and are applied on the fly
# in the matrix-vector product. Note that a spatially varying ``\kappa`` does not increase
# the storage compared to a constant one: it is folded into ``D_q`` either way.
#
# For quadratic hexahedra this stores 6 floats per quadrature point instead of
# ``27 \times 27`` matrix entries per cell (element assembly), or roughly 125 non-zero
# matrix entries per row (full assembly).
#
# ## Sum factorization
#
# Applying the reference gradients naively costs *more* arithmetic than multiplying with a
# stored element matrix. Partial assembly only pays off together with **sum factorization**,
# which exploits that the shape functions of `Lagrange{RefHexahedron, order}` are products
# of 1D shape functions,
#
# ```math
# N_i(\boldsymbol{\xi}) = N^\mathrm{1D}_a(\xi_1) N^\mathrm{1D}_b(\xi_2) N^\mathrm{1D}_c(\xi_3),
# ```
#
# and that the quadrature rule for `RefHexahedron` is a tensor product of 1D rules. The
# interpolation of values (and gradients) to the quadrature points then factors into three
# successive contractions with small 1D matrices -- for polynomial order ``p`` in dimension
# ``d`` this reduces the cost per cell from ``O(p^{2d})`` to ``O(d \, p^{d+1})``.
#
# Ferrite exposes this structure with two (internal) functions:
#
#  - [`Ferrite.tensor_product_interpolation(ip)`](@ref Ferrite.tensor_product_interpolation)
#    returns the 1D interpolation whose tensor product spans `ip`, and
#  - [`Ferrite.tensor_product_indices(ip)`](@ref Ferrite.tensor_product_indices) returns,
#    for every shape function `i` of `ip`, the tuple `(a, b, c)` of 1D shape function
#    indices such that `N[i](ξ) = N1D[a](ξ₁) * N1D[b](ξ₂) * N1D[c](ξ₃)`.
#
# !!! note "Why not `CellValues`?"
#     `CellValues` precomputes and stores the value and gradient of every shape function in
#     every quadrature point -- exactly the `nbasefunctions × nquadpoints` tables that sum
#     factorization avoids materializing. The evaluator below therefore does not use
#     `CellValues` in the matrix-vector product; it consumes the 1D building blocks
#     directly. (We do use standard assembly to *verify* the operator at the end.)
#
# !!! note "Boundary conditions"
#     For simplicity this how-to only considers the raw operator without constraints.
#     Dirichlet boundary conditions require special treatment for matrix-free operators
#     (e.g. applying the constraint condensation on the fly around the operator
#     application) and is left as an exercise for the reader.
#
# ## Prologue: a generic tensor-product evaluator
#
# Almost all of the machinery needed for sum-factorized operator evaluation is independent
# of the PDE being solved: it depends only on the interpolation and the quadrature rule.
# This prologue implements that machinery as a `TensorProductEvaluator`, closely modeled
# after deal.II's `FEEvaluation` class (see e.g.
# [deal.II step-37](https://dealii.org/current/doxygen/deal.II/step_37.html); MFEM's
# `QuadratureInterpolator` plays the same role). The problem-specific part of the how-to --
# what happens *at* each quadrature point -- reduces to a couple of lines sandwiched
# between calls to this evaluator.
#
# !!! note "A future Ferrite API?"
#     Everything in this section is problem-independent and could eventually be included in
#     Ferrite as a complement to `CellValues` for matrix-free methods. For now it lives in
#     this how-to while the design settles. Only *gradient* evaluation is implemented here
#     (sufficient for the Laplace operator); evaluation of *values* (needed for e.g. mass
#     matrices) follows the exact same pattern with contractions using only `B`, and an
#     implementation evaluating both can share intermediate contraction passes.

using Ferrite, LinearAlgebra, SparseArrays
using Test #src

# ### 1D contraction kernels
#
# The workhorses of sum factorization: contract a small 1D matrix `M` with one of the three
# dimensions of a rank-3 tensor. Note that the output dimension (the row index of `M`) can
# differ from the input dimension, so the same three functions implement both interpolation
# (`n1d → nq1d`, using the 1D matrices `B`/`D` defined below) and integration
# (`nq1d → n1d`, using their transposes). The two involved sizes are therefore passed as
# `Val`s: `P` is the length of the contracted dimension and `Q` the length of the
# corresponding output dimension. Since both are known at compile time the reduction loop
# is fully unrolled, with the sum accumulated in a register instead of read-modified-written
# through memory. Baking the (very short) loop lengths into the type like this mirrors what
# the established matrix-free implementations do -- deal.II and MFEM template their kernels
# on the polynomial degree for the same reason.

function contract_1!(out::Array{T, 3}, M::Matrix{T}, A::Array{T, 3}, ::Val{P}, ::Val{Q}) where {T, P, Q}
    ## out[q, j, k] = Σᵢ M[q, i] A[i, j, k]
    @inbounds for k in axes(A, 3), j in axes(A, 2), q in 1:Q
        s = zero(T)
        for i in 1:P
            s = muladd(M[q, i], A[i, j, k], s)
        end
        out[q, j, k] = s
    end
    return out
end

function contract_2!(out::Array{T, 3}, M::Matrix{T}, A::Array{T, 3}, ::Val{P}, ::Val{Q}) where {T, P, Q}
    ## out[i, q, k] = Σⱼ M[q, j] A[i, j, k]
    @inbounds for k in axes(A, 3), q in 1:Q, i in axes(A, 1)
        s = zero(T)
        for j in 1:P
            s = muladd(M[q, j], A[i, j, k], s)
        end
        out[i, q, k] = s
    end
    return out
end

function contract_3!(out::Array{T, 3}, M::Matrix{T}, A::Array{T, 3}, ::Val{P}, ::Val{Q}) where {T, P, Q}
    ## out[i, j, q] = Σₖ M[q, k] A[i, j, k]
    @inbounds for q in 1:Q, j in axes(A, 2), i in axes(A, 1)
        s = zero(T)
        for k in 1:P
            s = muladd(M[q, k], A[i, j, k], s)
        end
        out[i, j, q] = s
    end
    return out
end

# ### The evaluator
#
# The evaluator bundles the 1D operator matrices with the scratch buffers for one cell:
# `B` interpolates 1D nodal values to the 1D quadrature points and `D` evaluates the 1D
# derivative there (their transposes are stored explicitly since the integration step
# contracts with the transposed operators). `ue` holds the gathered local dof values (in
# lexicographic ordering), `ye` the local result, and `gx`/`gy`/`gz` the three components
# of the evaluated quantity in the quadrature points. As for the contraction kernels, the
# number of 1D quadrature points `NQ` and 1D basis functions `N` are lifted into the type.
#
# Since the evaluator owns scratch data it plays the same role as `CellValues` does for
# regular assembly: it is *not* thread-safe, and parallelizing the cell loop requires one
# evaluator per task (compare with the scratch data in the
# [multithreaded assembly how-to](@ref howto-threaded-assembly)).

struct TensorProductEvaluator{NQ, N, T}
    ## 1D operators
    B::Matrix{T}
    D::Matrix{T}
    Bᵀ::Matrix{T}
    Dᵀ::Matrix{T}
    ## Scratch buffers for one cell
    ue::Array{T, 3}                                  # (n, n, n) local values
    ye::Array{T, 3}                                  # (n, n, n) local result
    tmp::Array{T, 3}                                 # (n, n, n)
    t1::Array{T, 3}                                  # (nq, n, n)
    t2::Array{T, 3}                                  # (nq, n, n)
    s1::Array{T, 3}                                  # (nq, nq, n)
    s2::Array{T, 3}                                  # (nq, nq, n)
    s3::Array{T, 3}                                  # (nq, nq, n)
    gx::Array{T, 3}                                  # (nq, nq, nq)
    gy::Array{T, 3}                                  # (nq, nq, nq)
    gz::Array{T, 3}                                  # (nq, nq, nq)
end

function TensorProductEvaluator(ip::Lagrange{RefHexahedron}, qr1d::QuadratureRule{RefLine})
    ip1d = Ferrite.tensor_product_interpolation(ip)
    n = getnbasefunctions(ip1d)
    nq = getnquadpoints(qr1d)
    B = [Ferrite.reference_shape_value(ip1d, ξ, a) for ξ in Ferrite.getpoints(qr1d), a in 1:n]
    D = [Ferrite.reference_shape_gradient(ip1d, ξ, a)[1] for ξ in Ferrite.getpoints(qr1d), a in 1:n]
    return TensorProductEvaluator{nq, n, Float64}(
        B, D, collect(transpose(B)), collect(transpose(D)),
        zeros(n, n, n), zeros(n, n, n), zeros(n, n, n),
        zeros(nq, n, n), zeros(nq, n, n),
        zeros(nq, nq, n), zeros(nq, nq, n), zeros(nq, nq, n),
        zeros(nq, nq, nq), zeros(nq, nq, nq), zeros(nq, nq, nq),
    )
end

Ferrite.getnquadpoints(::TensorProductEvaluator{NQ}) where {NQ} = NQ^3
Ferrite.getnbasefunctions(::TensorProductEvaluator{<:Any, N}) where {N} = N^3

# The evaluator consumes and produces local dof values in *lexicographic* ordering (the
# ordering of the rank-3 scratch tensors), whereas Ferrite numbers the dofs by entity
# (vertices, then edges, faces, and the interior). The permutation between the two follows
# directly from `Ferrite.tensor_product_indices` (deal.II stores the equivalent permutation
# as `ShapeInfo::lexicographic_numbering`):

function lexicographic_numbering(ip::Lagrange{RefHexahedron})
    n = getnbasefunctions(Ferrite.tensor_product_interpolation(ip))
    return [LinearIndices((n, n, n))[abc...] for abc in Ferrite.tensor_product_indices(ip)]
end

# ### The evaluator interface
#
# Six functions make up the interface, corresponding to the five steps of a matrix-free
# operator application (`gather -> interpolate -> pointwise -> integrate -> scatter`),
# where the pointwise step is split into a getter and a setter (this is where the user's
# problem-specific code goes). The naming follows deal.II's `FEEvaluation`.
#
# First the gather and scatter steps, where `dofs` are the global dof indices of the cell
# *already permuted to lexicographic ordering* (see `partial_assembly` below, which stores
# the permuted dof map):

function read_dof_values!(ev::TensorProductEvaluator, x::AbstractVector, dofs::AbstractVector{Int})
    @inbounds for l in eachindex(ev.ue)
        ev.ue[l] = x[dofs[l]]
    end
    return ev
end

function distribute_local_to_global!(y::AbstractVector, ev::TensorProductEvaluator, dofs::AbstractVector{Int})
    @inbounds for l in eachindex(ev.ye)
        y[dofs[l]] += ev.ye[l]
    end
    return y
end

# The interpolation step computes the reference gradient `∇̂u` in all quadrature points,
# one component at a time. The `x`-component, for example, differentiates along the first
# dimension and interpolates along the other two: `gx = (B ⊗ B ⊗ D) ue`.

function evaluate_gradients!(ev::TensorProductEvaluator{NQ, N}) where {NQ, N}
    (; B, D, ue, t1, t2, s1, s2, s3, gx, gy, gz) = ev
    n, nq = Val(N), Val(NQ)
    contract_1!(t1, B, ue, n, nq)
    contract_1!(t2, D, ue, n, nq)
    contract_2!(s1, B, t1, n, nq)
    contract_2!(s2, D, t1, n, nq)
    contract_2!(s3, B, t2, n, nq)
    contract_3!(gx, B, s3, n, nq) # gx = (B ⊗ B ⊗ D) ue
    contract_3!(gy, B, s2, n, nq) # gy = (B ⊗ D ⊗ B) ue
    contract_3!(gz, D, s1, n, nq) # gz = (D ⊗ B ⊗ B) ue
    return ev
end

# The pointwise accessors, indexed by the (linear) quadrature point number:

@inline function get_gradient(ev::TensorProductEvaluator, q::Int)
    return @inbounds Vec(ev.gx[q], ev.gy[q], ev.gz[q])
end

@inline function submit_gradient!(ev::TensorProductEvaluator, h::Vec{3}, q::Int)
    @inbounds begin
        ev.gx[q] = h[1]
        ev.gy[q] = h[2]
        ev.gz[q] = h[3]
    end
    return ev
end

# Finally the integration step: the transpose of `evaluate_gradients!`, contracting the
# submitted quadrature point data with the transposed 1D operators and accumulating the
# three components into the local result `ye`.

function integrate_gradients!(ev::TensorProductEvaluator{NQ, N}) where {NQ, N}
    (; Bᵀ, Dᵀ, ye, tmp, t1, s1, gx, gy, gz) = ev
    n, nq = Val(N), Val(NQ)
    contract_3!(s1, Bᵀ, gx, nq, n)
    contract_2!(t1, Bᵀ, s1, nq, n)
    contract_1!(ye, Dᵀ, t1, nq, n)  # ye  = (Bᵀ ⊗ Bᵀ ⊗ Dᵀ) gx
    contract_3!(s1, Bᵀ, gy, nq, n)
    contract_2!(t1, Dᵀ, s1, nq, n)
    contract_1!(tmp, Bᵀ, t1, nq, n) # ye += (Bᵀ ⊗ Dᵀ ⊗ Bᵀ) gy
    ye .+= tmp
    contract_3!(s1, Dᵀ, gz, nq, n)
    contract_2!(t1, Bᵀ, s1, nq, n)
    contract_1!(tmp, Bᵀ, t1, nq, n) # ye += (Dᵀ ⊗ Bᵀ ⊗ Bᵀ) gz
    ye .+= tmp
    return ev
end

# This concludes the generic part -- note that the heat equation has not been mentioned
# once. Everything below this point is the problem-specific part.
#
# ## Commented program
#
# We start by setting up the problem: a quadratic Lagrange interpolation on a hexahedral
# grid, and a smoothly varying conductivity `κ`.

grid = generate_grid(Hexahedron, (16, 16, 16));

ip = Lagrange{RefHexahedron, 2}()

dh = DofHandler(grid)
add!(dh, :u, ip)
close!(dh);

κ(x::Vec{3}) = 2.0 + sinpi(x[1]) * cospi(2 * x[2]) * sinpi(x[3] / 2)

# The quadrature rule is where the tensor product structure of the *integration* comes
# from, so instead of relying on the memory layout of `QuadratureRule{RefHexahedron}(3)` we
# construct the 3D rule explicitly as the tensor product of a 1D rule with itself, with the
# first coordinate varying fastest to match the lexicographic ordering of the evaluator:

qr1d = QuadratureRule{RefLine}(3)
p1d = Ferrite.getpoints(qr1d)
w1d = Ferrite.getweights(qr1d)
qr = QuadratureRule{RefHexahedron}(
    vec([wx * wy * wz for wx in w1d, wy in w1d, wz in w1d]),
    vec([Vec(px[1], py[1], pz[1]) for px in p1d, py in p1d, pz in p1d]),
);

# With the interpolation and 1D quadrature rule we can construct the evaluator:

ev = TensorProductEvaluator(ip, qr1d)

# To see what the underlying trait provides, consider shape function 9 (the first edge dof,
# located at `ξ = (0, -1, -1)`): it is the product of 1D function 3 (the midpoint function)
# in `ξ₁` and 1D function 1 (the left vertex function) in `ξ₂` and `ξ₃`:

Ferrite.tensor_product_indices(ip)[9]

# ### Partial assembly
#
# The operator stores two things per cell: the lexicographically permuted global dof
# indices, and the tensor `D_q` for every quadrature point. The evaluator (with its
# scratch buffers) is shared between all cells.

struct PartialAssemblyOperator{E <: TensorProductEvaluator}
    ndofs::Int
    dofmap::Matrix{Int}                              # nbasefunctions × ncells
    Dq::Matrix{SymmetricTensor{2, 3, Float64, 6}}    # nquadpoints × ncells
    ev::E
end

Base.size(A::PartialAssemblyOperator) = (A.ndofs, A.ndofs)
Base.size(A::PartialAssemblyOperator, d::Int) = size(A)[d]
Base.eltype(::PartialAssemblyOperator) = Float64

# The setup loop computes `D_q = det(J) w κ(x) J⁻¹ J⁻ᵀ` for every quadrature point of every
# cell. This is the "partial" assembly: it visits every cell like regular assembly, but the
# result is 6 floats per quadrature point instead of an element matrix. Note how the dof
# permutation is folded into the stored `dofmap` such that the matrix-vector product can
# gather straight into lexicographic ordering.
#
# The geometry mapping is evaluated with the geometric (trilinear) interpolation of the
# grid: `J = Σₐ xₐ ⊗ ∇̂Nₐ(ξ_q)`.

function partial_assembly(
        dh::DofHandler, ip, qr::QuadratureRule, κ::Function, ev::TensorProductEvaluator,
    )
    grid = dh.grid
    ncells = getncells(grid)
    nqp = getnquadpoints(qr)
    @assert nqp == getnquadpoints(ev) && getnbasefunctions(ip) == getnbasefunctions(ev)
    w = Ferrite.getweights(qr)
    lex = lexicographic_numbering(ip)
    ## Evaluate the geometric interpolation in the quadrature points
    ip_geo = geometric_interpolation(getcelltype(grid))
    ngeo = getnbasefunctions(ip_geo)
    N_geo = [Ferrite.reference_shape_value(ip_geo, ξ, a) for a in 1:ngeo, ξ in Ferrite.getpoints(qr)]
    dNdξ_geo = [Ferrite.reference_shape_gradient(ip_geo, ξ, a) for a in 1:ngeo, ξ in Ferrite.getpoints(qr)]
    ## Allocate the per-cell data
    dofmap = Matrix{Int}(undef, getnbasefunctions(ip), ncells)
    Dq = Matrix{SymmetricTensor{2, 3, Float64, 6}}(undef, nqp, ncells)
    for cell in CellIterator(dh)
        e = cellid(cell)
        x = getcoordinates(cell)
        for (i, dof) in pairs(celldofs(cell))
            dofmap[lex[i], e] = dof
        end
        for q in 1:nqp
            J = zero(Tensor{2, 3})
            x_q = zero(Vec{3})
            for a in 1:ngeo
                J += x[a] ⊗ dNdξ_geo[a, q]
                x_q += N_geo[a, q] * x[a]
            end
            Jinv = inv(J)
            Dq[q, e] = det(J) * w[q] * κ(x_q) * dott(Jinv)
        end
    end
    return PartialAssemblyOperator(ndofs(dh), dofmap, Dq, ev)
end

A = partial_assembly(dh, ip, qr, κ, ev);

# ### The matrix-vector product
#
# With the evaluator doing the heavy lifting, the operator application reduces to the
# problem-specific pointwise operation `h_q = D_q ⋅ ∇̂u_q` sandwiched between the generic
# evaluator calls. By overloading `LinearAlgebra.mul!` the operator can be dropped into any
# iterative solver that accepts a general linear operator (e.g. the packages Krylov.jl,
# IterativeSolvers.jl, or KrylovKit.jl).

function LinearAlgebra.mul!(y::AbstractVector, A::PartialAssemblyOperator, x::AbstractVector)
    (; dofmap, Dq, ev) = A
    fill!(y, 0)
    for e in axes(dofmap, 2)
        dofs = view(dofmap, :, e)
        read_dof_values!(ev, x, dofs)
        evaluate_gradients!(ev)
        @inbounds for q in 1:getnquadpoints(ev)
            submit_gradient!(ev, Dq[q, e] ⋅ get_gradient(ev, q), q)
        end
        integrate_gradients!(ev)
        distribute_local_to_global!(y, ev, dofs)
    end
    return y
end

Base.:*(A::PartialAssemblyOperator, x::AbstractVector) = mul!(similar(x, size(A, 1)), A, x)

# For comparison: a mass matrix would store the scalar `det(J) w ρ(x)` per quadrature
# point and its product loop would be `submit_value!(ev, m_q * get_value(ev, q), q)`
# between `evaluate_values!` and `integrate_values!` -- the cell loop, gather/scatter, and
# contraction machinery are untouched. This separation is what makes the evaluator a
# candidate for a library abstraction.
#
# Note that the loop over the cells can be parallelized without further ado on the CPU
# (given one evaluator per task) *except* for the scatter step, which requires the same
# treatment as parallel assembly: grid coloring or atomic additions, see the
# [multithreaded assembly how-to](@ref howto-threaded-assembly).
#
# ## Verification
#
# To verify the operator we assemble the same bilinear form into a sparse matrix with
# standard assembly (compare with the [heat equation tutorial](@ref tutorial-heat-equation))
# and compare the matrix-vector products.

function assemble_sparse(dh::DofHandler, ip, qr::QuadratureRule, κ::Function)
    cv = CellValues(qr, ip)
    K = allocate_matrix(dh)
    assembler = start_assemble(K)
    nbf = getnbasefunctions(cv)
    Ke = zeros(nbf, nbf)
    for cell in CellIterator(dh)
        reinit!(cv, cell)
        fill!(Ke, 0)
        for q in 1:getnquadpoints(cv)
            x_q = spatial_coordinate(cv, q, getcoordinates(cell))
            dΩ = getdetJdV(cv, q) * κ(x_q)
            for i in 1:nbf
                ∇Nᵢ = shape_gradient(cv, q, i)
                for j in 1:nbf
                    Ke[i, j] += (∇Nᵢ ⋅ shape_gradient(cv, q, j)) * dΩ
                end
            end
        end
        assemble!(assembler, celldofs(cell), Ke)
    end
    return K
end

K = assemble_sparse(dh, ip, qr, κ)

x = rand(ndofs(dh))
y_pa = A * x
y_csr = K * x
y_pa ≈ y_csr
@test y_pa ≈ y_csr #src

# ## Storage and runtime comparison
#
# The point of partial assembly is the memory footprint. For this problem (4096 cells with
# 27 quadrature points and 27 basis functions each, 35937 dofs):

storage_pa = Base.summarysize(A.Dq) + Base.summarysize(A.dofmap)
storage_csr = Base.summarysize(K.nzval) + Base.summarysize(K.rowval) + Base.summarysize(K.colptr)
(pa = Base.format_bytes(storage_pa), csr = Base.format_bytes(storage_csr))

# Note that the `dofmap` (the gather indices) dominates the partial assembly storage here;
# the `D_q` data itself is only `27 * 6 * 8 = 1296` bytes per cell. The same dof
# information is also needed by any element-assembly or matrix-free implementation, and on
# the sparse matrix side the analogous index data (`rowval`/`colptr`) is included in the
# count above.
#
# Finally we compare the runtime of the two matrix-vector products:

function best_time(f!, n = 20)
    f!() # warmup (compilation)
    return minimum(@elapsed(f!()) for _ in 1:n)
end

t_pa = best_time(() -> mul!(y_pa, A, x))
t_csr = best_time(() -> mul!(y_csr, K, x))
(pa = t_pa, csr = t_csr, ratio = t_pa / t_csr)

# The serial reference implementation above is expected to land within a small factor
# (roughly 2x) of the sparse matrix-vector product: at `p = 2` sum factorization performs a
# comparable number of floating point operations, but the sparse product is a tight,
# bandwidth-bound loop that is hard to beat in a serial apples-to-apples comparison. The
# trade-off tilts towards partial assembly with each of the following, compounding, factors:
#
#  - **Memory**: already at `p = 2` the operator data is ~5x smaller, and the gap grows
#    with the polynomial order (per cell, the stored data grows as `O(p³)` compared to
#    `O(p⁶)` for the element matrices and the sparse matrix).
#  - **Parallelism**: the sparse product is bandwidth bound and stops scaling once a few
#    cores saturate the memory bus, whereas the cellwise operator application is compute
#    bound and scales like assembly does (see the
#    [multithreaded assembly how-to](@ref howto-threaded-assembly)).
#  - **Polynomial order**: the arithmetic advantage of sum factorization grows with `p` as
#    well -- `O(p⁴)` against `O(p⁶)` per cell in 3D.
#
# This structure -- gather, small dense contractions, pointwise operation, transposed
# contractions, scatter -- is also exactly the shape of computation that maps well onto
# GPUs, where one block of threads processes one (or a few) cells and the 1D matrices live
# in shared memory. See the [GPU assembly how-to](gpu_assembly.md) for the Ferrite GPU
# infrastructure.

#md # ## [Plain program](@id matrix_free-plain-program)
#md #
#md # Here follows a version of the program without any comments.
#md # The file is also available here: [`matrix_free.jl`](matrix_free.jl).
#md #
#md # ```julia
#md # @__CODE__
#md # ```
