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
# This how-to implements operators at the **partial assembly** level, first for the heat
# equation and then for linear elasticity. For the heat equation with a spatially varying
# conductivity ``\kappa(\mathbf{x})``, discretized with quadratic hexahedra, the key
# observation is that the integrand at a quadrature point factors into
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
# The reference-element shape functions are the same for every cell and are applied on the
# fly in the matrix-vector product. Note that a spatially varying ``\kappa`` does not
# increase the storage compared to a constant one: it is folded into ``D_q`` either way.
#
# For quadratic hexahedra this stores 6 floats per quadrature point instead of
# ``27 \times 27`` matrix entries per cell (element assembly), or roughly 125 non-zero
# matrix entries per row (full assembly).
#
# ## Sum factorization and the evaluator
#
# Applying the reference shape functions naively costs *more* arithmetic than multiplying
# with a stored element matrix. Partial assembly only pays off together with
# **sum factorization**, which exploits that the shape functions of
# `Lagrange{RefHexahedron, order}` are products of 1D shape functions,
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
# Ferrite ships (experimental, internal) infrastructure for this, see the
# [devdocs on matrix-free evaluation](../devdocs/matrix_free.md) for the complete API:
#
#  - [`Ferrite.tensor_product_interpolation`](@ref Ferrite.tensor_product_interpolation)
#    and [`Ferrite.tensor_product_indices`](@ref Ferrite.tensor_product_indices) expose the
#    tensor product structure of the interpolations.
#  - [`Ferrite.TensorProductEvaluator`](@ref Ferrite.TensorProductEvaluator) implements
#    sum-factorized evaluation and integration for scalar and vector fields. Its interface
#    is modeled after deal.II's `FEEvaluation` class (see e.g.
#    [deal.II step-37](https://dealii.org/current/doxygen/deal.II/step_37.html)).
#  - [`Ferrite.lexicographic_dofmap`](@ref Ferrite.lexicographic_dofmap) collects the cell
#    dof indices permuted to the lexicographic (tensor product) ordering that the evaluator
#    uses internally.
#  - [`Ferrite.quadrature_point_data`](@ref Ferrite.quadrature_point_data) precomputes user
#    data for every quadrature point of every cell -- this is the "partial assembly" step.
#
# With this, a matrix-free operator application follows a five step structure per cell,
# where only step 3 depends on the weak form:
#
# 1. `read_dof_values!`: gather the local dof values,
# 2. `evaluate_gradients!`: compute reference gradients in all quadrature points,
# 3. `submit_gradient!(ev, f(get_gradient(ev, q)), q)` for each quadrature point, where `f`
#    contracts the gradient with the stored quadrature point data,
# 4. `integrate_gradients!`: multiply with the test function gradients and integrate,
# 5. `distribute_local_to_global!`: scatter-add the local result.
#
# !!! note "Why not `CellValues`?"
#     `CellValues` precomputes and stores the value and gradient of every shape function in
#     every quadrature point -- exactly the `nbasefunctions × nquadpoints` tables that sum
#     factorization avoids materializing. This is why the evaluator exposes gradients only
#     through the quadrature point accessors and no `shape_gradient(cv, q, i)` style
#     interface. (We do use `CellValues` and standard assembly to *verify* the operators
#     below.)
#
# !!! note "Boundary conditions"
#     For simplicity this how-to only considers the raw operators without constraints.
#     Dirichlet boundary conditions require special treatment for matrix-free operators
#     (e.g. applying the constraint condensation on the fly around the operator
#     application) and is left as an exercise for the reader.
#
# ## A generic operator
#
# Both operators in this how-to -- and cell-integral bilinear forms in general -- share
# everything except (i) the data stored per quadrature point and (ii) the pointwise
# operation contracting that data with the evaluated gradient. We therefore define one
# operator type, parameterized by the quadrature point data and the pointwise function:

using Ferrite, LinearAlgebra, SparseArrays
using Test #src

struct MatrixFreeOperator{D, F <: Function, E <: Ferrite.TensorProductEvaluator}
    ndofs::Int
    dofmap::Matrix{Int}   # nbasefunctions × ncells, lexicographically permuted
    qpdata::Matrix{D}     # nquadpoints × ncells
    pointwise::F          # (qpdata, ĝ) -> ĥ
    ev::E
end

Base.size(A::MatrixFreeOperator) = (A.ndofs, A.ndofs)
Base.size(A::MatrixFreeOperator, d::Int) = size(A)[d]
Base.eltype(::MatrixFreeOperator) = Float64

# The matrix-vector product is the five step structure from above. By overloading
# `LinearAlgebra.mul!` the operator can be dropped into any iterative solver that accepts a
# general linear operator (e.g. the packages Krylov.jl, IterativeSolvers.jl, or
# KrylovKit.jl).

function LinearAlgebra.mul!(y::AbstractVector, A::MatrixFreeOperator, x::AbstractVector)
    (; dofmap, qpdata, pointwise, ev) = A
    fill!(y, 0)
    for e in axes(dofmap, 2)
        dofs = view(dofmap, :, e)
        Ferrite.read_dof_values!(ev, x, dofs)
        Ferrite.evaluate_gradients!(ev)
        @inbounds for q in 1:Ferrite.getnquadpoints(ev)
            ĝ = Ferrite.get_gradient(ev, q)
            Ferrite.submit_gradient!(ev, pointwise(qpdata[q, e], ĝ), q)
        end
        Ferrite.integrate_gradients!(ev)
        Ferrite.distribute_local_to_global!(y, ev, dofs)
    end
    return y
end

Base.:*(A::MatrixFreeOperator, x::AbstractVector) = mul!(similar(x, size(A, 1)), A, x)

# Note that the loop over the cells can be parallelized without further ado on the CPU
# (given one evaluator per task) *except* for the scatter step, which requires the same
# treatment as parallel assembly: grid coloring or atomic additions, see the
# [multithreaded assembly how-to](@ref howto-threaded-assembly).
#
# ## The heat equation
#
# A quadratic Lagrange interpolation on a hexahedral grid, and a smoothly varying
# conductivity `κ`:

grid = generate_grid(Hexahedron, (16, 16, 16));

ip = Lagrange{RefHexahedron, 2}()

dh = DofHandler(grid)
add!(dh, :u, ip)
close!(dh);

κ(x::Vec{3}) = 2.0 + sinpi(x[1]) * cospi(2 * x[2]) * sinpi(x[3] / 2)

# The quadrature rule is where the tensor product structure of the *integration* comes
# from, so we construct the 3D rule explicitly as the tensor product of a 1D rule with
# itself, with the first coordinate varying fastest to match the lexicographic ordering of
# the evaluator:

qr1d = QuadratureRule{RefLine}(3)
p1d = Ferrite.getpoints(qr1d)
w1d = Ferrite.getweights(qr1d)
qr = QuadratureRule{RefHexahedron}(
    vec([wx * wy * wz for wx in w1d, wy in w1d, wz in w1d]),
    vec([Vec(px[1], py[1], pz[1]) for px in p1d, py in p1d, pz in p1d]),
);

# Setting up the operator is now three lines: the evaluator, the permuted dof map, and the
# partial assembly loop computing `D_q = det(J) w κ(x) J⁻¹ J⁻ᵀ` -- the latter visits every
# cell like regular assembly, but stores 6 floats per quadrature point instead of an
# element matrix. The pointwise operation of the heat operator is simply `D_q ⋅ ∇̂u`.

ev = Ferrite.TensorProductEvaluator(ip, qr1d)
dofmap = Ferrite.lexicographic_dofmap(dh, ip)
Dq = Ferrite.quadrature_point_data(grid, qr) do x, J, w
    Jinv = inv(J)
    return det(J) * w * κ(x) * dott(Jinv)
end

A = MatrixFreeOperator(ndofs(dh), dofmap, Dq, (D, ĝ) -> D ⋅ ĝ, ev);

# ### Verification
#
# To verify the operator we assemble the same bilinear form into a sparse matrix with
# standard assembly (compare with the [heat equation tutorial](@ref tutorial-heat-equation))
# and compare the matrix-vector products for a random input vector.

function assemble_sparse_heat(dh::DofHandler, ip, qr::QuadratureRule, κ::Function)
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

K = assemble_sparse_heat(dh, ip, qr, κ)

x = rand(ndofs(dh))
y_pa = A * x
y_csr = K * x
y_pa ≈ y_csr
@test y_pa ≈ y_csr #src

# ### Storage and runtime
#
# The point of partial assembly is the memory footprint. For this problem (4096 cells with
# 27 quadrature points and 27 basis functions each, 35937 dofs):

storage(A::MatrixFreeOperator) = Base.summarysize(A.qpdata) + Base.summarysize(A.dofmap)
storage(K::SparseMatrixCSC) = Base.summarysize(K.nzval) + Base.summarysize(K.rowval) + Base.summarysize(K.colptr)
(pa = Base.format_bytes(storage(A)), csr = Base.format_bytes(storage(K)))

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

# The serial reference implementation is expected to land within a small factor (roughly
# 2x) of the sparse matrix-vector product: at `p = 2` sum factorization performs a
# comparable number of floating point operations, but the sparse product is a tight,
# bandwidth-bound loop that is hard to beat in a serial apples-to-apples comparison. The
# trade-off tilts towards partial assembly with each of the following, compounding, factors:
#
#  - **Memory**: already at `p = 2` the operator data is ~5x smaller, and the gap grows
#    with the polynomial order (per cell, the stored data grows as `O(p³)` compared to
#    `O(p⁶)` for the element matrices and the sparse matrix).
#  - **Parallelism**: the sparse product is bandwidth bound and stops scaling once a few
#    cores saturate the memory bus, whereas the cellwise operator application is compute
#    bound and scales like assembly does.
#  - **Polynomial order**: the arithmetic advantage of sum factorization grows with `p` as
#    well -- `O(p⁴)` against `O(p⁶)` per cell in 3D.
#
# ## Linear elasticity
#
# For linear elasticity the displacement field is vector valued, `ip^3`, and the evaluated
# gradient in a quadrature point is a second order tensor (`ĝ[i, j] = ∂u_i/∂ξ_j`). Nothing
# changes in the operator structure -- the evaluator applies the same 1D contractions once
# per component -- so the `MatrixFreeOperator` defined above is reused as is, with new
# quadrature point data and a new pointwise operation.
#
# For the per-point data we choose the *factored* representation: the inverse Jacobian
# `J⁻¹` together with the (premultiplied) Lamé parameters, 11 floats per point. The
# pointwise operation then explicitly maps to the spatial gradient, evaluates the stress,
# and maps back:
#
# ```math
# \varepsilon = \mathrm{sym}(\hat{g} \cdot J^{-1}), \quad
# \sigma = \lambda \, \mathrm{tr}(\varepsilon) I + 2 \mu \varepsilon, \quad
# \hat{h} = \det(J) \, w \, \sigma \cdot J^{-T}.
# ```
#
# Alternatively one can fold everything into a single pulled-back fourth order tensor per
# point (45 independent components with major symmetry), trading storage for fewer
# operations in the product -- for heterogeneous *anisotropic* elasticity that is the
# natural representation since the pointwise stiffness carries 21 independent components
# anyway. That the choice of representation is up to the user, and invisible to the
# evaluator, is the reason Ferrite does not prescribe a data structure for the quadrature
# point data.
#
# We use a smaller grid here, for the simple reason that the *assembled reference matrix*
# we verify against below is getting expensive: on the 16³ grid it would occupy about
# 300 MiB, whereas the matrix-free elasticity operator needs about 12 MiB. This is the
# storage argument of partial assembly making itself felt already at `p = 2` for vector
# valued problems.

grid_e = generate_grid(Hexahedron, (8, 8, 8))

ipv = ip^3

dh_e = DofHandler(grid_e)
add!(dh_e, :u, ipv)
close!(dh_e);

λ(x::Vec{3}) = 2.0 + x[1]
μ(x::Vec{3}) = 1.0 + 0.5 * sinpi(x[3])

ev_e = Ferrite.TensorProductEvaluator(ipv, qr1d)
dofmap_e = Ferrite.lexicographic_dofmap(dh_e, ipv)
data_e = Ferrite.quadrature_point_data(grid_e, qr) do x, J, w
    return (Jinv = inv(J), λw = det(J) * w * λ(x), μw = det(J) * w * μ(x))
end

function elasticity_pointwise(d, ĝ)
    ε = symmetric(ĝ ⋅ d.Jinv)
    σw = d.λw * tr(ε) * one(ε) + 2 * d.μw * ε
    return σw ⋅ d.Jinv'
end

A_e = MatrixFreeOperator(ndofs(dh_e), dofmap_e, data_e, elasticity_pointwise, ev_e);

# Verify against standard assembly of the elasticity stiffness matrix (compare with the
# [linear elasticity tutorial](@ref tutorial-linear-elasticity)):

function assemble_sparse_elasticity(dh::DofHandler, ipv, qr::QuadratureRule, λ::Function, μ::Function)
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

x_e = rand(ndofs(dh_e))
y_pa_e = A_e * x_e
y_csr_e = K_e * x_e
y_pa_e ≈ y_csr_e
@test y_pa_e ≈ y_csr_e #src

# The storage advantage is considerably larger than for the heat equation -- with 81
# basis functions per cell the element matrices and the sparse matrix grow with the square,
# while the quadrature point data does not grow at all (the 11 floats per point would serve
# any polynomial order):

(pa = Base.format_bytes(storage(A_e)), csr = Base.format_bytes(storage(K_e)))

#-

t_pa_e = best_time(() -> mul!(y_pa_e, A_e, x_e))
t_csr_e = best_time(() -> mul!(y_csr_e, K_e, x_e))
(pa = t_pa_e, csr = t_csr_e, ratio = t_pa_e / t_csr_e)

# In contrast to the heat equation, the matrix-free elasticity operator reaches parity with
# the sparse matrix-vector product already in this serial comparison: with three coupled
# components the sparse matrix carries ~9x the non-zeros per node pair, while the
# sum-factorized operator only repeats the (cheap) 1D contractions three times and pays for
# the extra coupling in the pointwise operation -- a few flops per quadrature point on data
# that is already in cache.
#
# ## Generalizing to other weak forms
#
# The two operators above suggest how this factors for cell-integral weak forms in
# general: a problem is characterized by which quantities are evaluated per field, the
# pointwise operation, and the data stored per quadrature point -- everything else
# (contractions, dof permutations, gather/scatter, the geometry loop) is shared
# infrastructure. Some directions not covered by this how-to:
#
#  - **Mass and advection terms** need quadrature point *values* in addition to (or instead
#    of) gradients. These follow the same pattern with contractions using only the 1D
#    value matrix, and evaluating values and gradients together can share intermediate
#    contraction passes. (Value evaluation is not yet implemented by the evaluator.)
#  - **Mixed problems** (e.g. Stokes) use one evaluator per field, sharing the quadrature
#    rule, with the pointwise operation coupling the fields -- analogous to using multiple
#    `CellValues` in regular assembly.
#  - **Nonlinear problems** fit naturally: the Krylov solver inside each Newton step
#    applies the *linearized* operator, so the quadrature point data (e.g. the consistent
#    tangent, for plasticity computed from internal variables that live per quadrature
#    point anyway) is refreshed once per Newton step and reused across all solver
#    iterations -- a much cheaper "assembly" than rebuilding a matrix. The residual can be
#    computed with the same evaluate/pointwise/integrate structure, with the constitutive
#    law replacing the linear pointwise contraction.
#  - **Simplices** have no tensor product structure with nodal Lagrange bases, so there is
#    no sum factorization to exploit -- this is why high-order matrix-free codes are
#    hypercube-centric. The evaluator *interface* still makes sense backed by dense
#    (element-assembly level) kernels as a fallback.
#  - **H(div)/H(curl) interpolations** on hexahedra (Raviart-Thomas, Nédélec) do have
#    tensor product structure, but an *anisotropic* one (different 1D polynomial degrees
#    per direction and component) and Piola mappings instead of the gradient pullback --
#    a genuine extension of the machinery.
#  - **Face integrals** (DG, Neumann terms) require a facet analogue of the evaluator with
#    2D tensor product contractions on the facets (deal.II's `FEFaceEvaluation`).
#
# Finally, this cellwise structure -- gather, small dense contractions, pointwise
# operation, transposed contractions, scatter -- is exactly the shape of computation that
# maps well onto GPUs, where one block of threads processes one (or a few) cells and the 1D
# matrices live in shared memory. See the [GPU assembly how-to](gpu_assembly.md) for the
# Ferrite GPU infrastructure.

#md # ## [Plain program](@id matrix_free-plain-program)
#md #
#md # Here follows a version of the program without any comments.
#md # The file is also available here: [`matrix_free.jl`](matrix_free.jl).
#md #
#md # ```julia
#md # @__CODE__
#md # ```
