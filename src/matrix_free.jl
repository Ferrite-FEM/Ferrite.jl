# Experimental infrastructure for matrix-free operator evaluation with sum factorization.
#
# Nothing in this file is specific to a PDE: the evaluator interpolates fields (and their
# gradients) between the cell dofs and the quadrature points, and what happens *at* the
# quadrature points is up to the user. See the "Matrix-free operator evaluation" how-to in
# the documentation for usage, background, and references. The API is modeled after
# deal.II's FEEvaluation class and is currently limited to (vectorized) Lagrange
# interpolations on hexahedra with tensor product quadrature rules.
#
# NOTE: This API is experimental and expected to change between releases.

#####################
# Sum factorization #
#####################

# Contraction of the 1D operator matrix `M` with one of the three dimensions of the rank-3
# tensor `A`. The output dimension (the row index of `M`) can differ from the input
# dimension, so the same functions implement both interpolation (`n1d -> nq1d`) and, with
# the transposed operator, integration (`nq1d -> n1d`). The two sizes are passed as `Val`s:
# `P` is the length of the contracted dimension and `Q` the length of the corresponding
# output dimension. Both are known at compile time such that the reduction loop is fully
# unrolled, with the sum accumulated in a register.

@inline function contract_1!(out::AbstractArray{T, 3}, M::AbstractMatrix{T}, A::AbstractArray{T, 3}, ::Val{P}, ::Val{Q}) where {T, P, Q}
    # out[q, j, k] = Σᵢ M[q, i] A[i, j, k]
    @inbounds for k in axes(A, 3), j in axes(A, 2), q in 1:Q
        s = zero(T)
        for i in 1:P
            s = muladd(M[q, i], A[i, j, k], s)
        end
        out[q, j, k] = s
    end
    return out
end

@inline function contract_2!(out::AbstractArray{T, 3}, M::AbstractMatrix{T}, A::AbstractArray{T, 3}, ::Val{P}, ::Val{Q}) where {T, P, Q}
    # out[i, q, k] = Σⱼ M[q, j] A[i, j, k]
    @inbounds for k in axes(A, 3), q in 1:Q, i in axes(A, 1)
        s = zero(T)
        for j in 1:P
            s = muladd(M[q, j], A[i, j, k], s)
        end
        out[i, q, k] = s
    end
    return out
end

@inline function contract_3!(out::AbstractArray{T, 3}, M::AbstractMatrix{T}, A::AbstractArray{T, 3}, ::Val{P}, ::Val{Q}) where {T, P, Q}
    # out[i, j, q] = Σₖ M[q, k] A[i, j, k]
    @inbounds for q in 1:Q, j in axes(A, 2), i in axes(A, 1)
        s = zero(T)
        for k in 1:P
            s = muladd(M[q, k], A[i, j, k], s)
        end
        out[i, j, q] = s
    end
    return out
end

###########################
# TensorProductEvaluator  #
###########################

"""
    TensorProductEvaluator(ip, qr1d::QuadratureRule{RefLine})

Evaluator for sum-factorized evaluation and integration of a finite element field in the
tensor product quadrature points of a cell. `ip` is the interpolation of the field --
either a scalar `Lagrange{RefHexahedron, order}` or its vectorized counterpart
(`ip_scalar ^ 3`) -- and `qr1d` the 1D quadrature rule whose 3-fold tensor product (with
the first coordinate varying fastest) defines the quadrature points.

The evaluator bundles the 1D shape value/derivative matrices with the scratch buffers for
one cell and is used as follows for a (linear) operator application:

```julia
read_dof_values!(ev, x, dofs)      # gather (dofs in lexicographic ordering)
evaluate_gradients!(ev)            # reference gradients in all quadrature points
for q in 1:getnquadpoints(ev)
    ĝ = get_gradient(ev, q)
    ĥ = ...                        # problem specific pointwise operation
    submit_gradient!(ev, ĥ, q)
end
integrate_gradients!(ev)           # multiply with test function gradients and integrate
distribute_local_to_global!(y, ev, dofs) # scatter
```

Since the evaluator owns scratch data it is not thread-safe: parallelizing the cell loop
requires one evaluator per task, like `CellValues` for regular assembly.

The number of 1D quadrature points `NQ` and 1D basis functions `N`, as well as the number
of field components `NC`, are compile time constants (type parameters), which allows the
short sum factorization loops to be unrolled completely.

!!! warning "Experimental API"
    This type and its associated functions are experimental and may change between
    releases without further notice.
"""
struct TensorProductEvaluator{NQ, N, NC, T}
    # 1D operators: shape values and shape derivatives in the 1D quadrature points, and
    # their transposes.
    B::Matrix{T}                                     # (nq, n)
    D::Matrix{T}                                     # (nq, n)
    Bᵀ::Matrix{T}                                    # (n, nq)
    Dᵀ::Matrix{T}                                    # (n, nq)
    # Scratch buffers for one cell; the trailing dimension is the field component.
    ue::Array{T, 4}                                  # (n, n, n, nc) local dof values
    ye::Array{T, 4}                                  # (n, n, n, nc) local result
    tmp::Array{T, 3}                                 # (n, n, n)
    t1::Array{T, 3}                                  # (nq, n, n)
    t2::Array{T, 3}                                  # (nq, n, n)
    s1::Array{T, 3}                                  # (nq, nq, n)
    s2::Array{T, 3}                                  # (nq, nq, n)
    s3::Array{T, 3}                                  # (nq, nq, n)
    gx::Array{T, 4}                                  # (nq, nq, nq, nc)
    gy::Array{T, 4}                                  # (nq, nq, nq, nc)
    gz::Array{T, 4}                                  # (nq, nq, nq, nc)
end

function TensorProductEvaluator(ip::Lagrange{RefHexahedron}, qr1d::QuadratureRule{RefLine}; T::Type = Float64)
    return _tensor_product_evaluator(ip, qr1d, 1, T)
end
function TensorProductEvaluator(
        ipv::VectorizedInterpolation{3, RefHexahedron, <:Any, <:Lagrange}, qr1d::QuadratureRule{RefLine};
        T::Type = Float64,
    )
    return _tensor_product_evaluator(ipv.ip, qr1d, 3, T)
end

function _tensor_product_evaluator(
        ip::Lagrange{RefHexahedron}, qr1d::QuadratureRule{RefLine}, nc::Int, ::Type{T}
    ) where {T}
    ip1d = tensor_product_interpolation(ip)
    n = getnbasefunctions(ip1d)
    nq = getnquadpoints(qr1d)
    B = T[reference_shape_value(ip1d, ξ, a) for ξ in getpoints(qr1d), a in 1:n]
    D = T[reference_shape_gradient(ip1d, ξ, a)[1] for ξ in getpoints(qr1d), a in 1:n]
    return TensorProductEvaluator{nq, n, nc, T}(
        B, D, collect(transpose(B)), collect(transpose(D)),
        zeros(T, n, n, n, nc), zeros(T, n, n, n, nc), zeros(T, n, n, n),
        zeros(T, nq, n, n), zeros(T, nq, n, n),
        zeros(T, nq, nq, n), zeros(T, nq, nq, n), zeros(T, nq, nq, n),
        zeros(T, nq, nq, nq, nc), zeros(T, nq, nq, nq, nc), zeros(T, nq, nq, nq, nc),
    )
end

getnquadpoints(::TensorProductEvaluator{NQ}) where {NQ} = NQ^3
getnbasefunctions(::TensorProductEvaluator{<:Any, N, NC}) where {N, NC} = N^3 * NC

##################################
# Lexicographic (re)numbering    #
##################################

"""
    lexicographic_numbering(ip)

Return the permutation from Ferrite's local dof numbering (by entity: vertices, edges,
faces, interior; for vectorized interpolations with components interleaved) to the
lexicographic (tensor product) numbering used by the [`TensorProductEvaluator`](@ref
Ferrite.TensorProductEvaluator): local dof `i` of `ip` corresponds to linear index
`lexicographic_numbering(ip)[i]` of the evaluator's local buffers. Derived from
[`tensor_product_indices`](@ref Ferrite.tensor_product_indices). deal.II stores the
equivalent permutation as `ShapeInfo::lexicographic_numbering`.
"""
function lexicographic_numbering(ip::Lagrange{RefHexahedron})
    n = getnbasefunctions(tensor_product_interpolation(ip))
    return [LinearIndices((n, n, n))[abc...] for abc in tensor_product_indices(ip)]
end
function lexicographic_numbering(ipv::VectorizedInterpolation{3, RefHexahedron, <:Any, <:Lagrange})
    lex = lexicographic_numbering(ipv.ip)
    n = length(lex)
    out = Vector{Int}(undef, 3 * n)
    for i in 1:n, c in 1:3
        # Ferrite interleaves the components node by node; the evaluator blocks them.
        out[3 * (i - 1) + c] = lex[i] + (c - 1) * n
    end
    return out
end

"""
    lexicographic_dofmap(dh::DofHandler, ip)

Return a `getnbasefunctions(ip) × getncells(grid)` matrix such that column `e` contains
the global dof indices of cell `e` permuted to the lexicographic ordering consumed by the
[`TensorProductEvaluator`](@ref Ferrite.TensorProductEvaluator) (see
[`lexicographic_numbering`](@ref Ferrite.lexicographic_numbering)). The dof handler must
have a single field, discretized with `ip`.
"""
function lexicographic_dofmap(dh::DofHandler, ip::Interpolation)
    lex = lexicographic_numbering(ip)
    grid = get_grid(dh)
    dofmap = Matrix{Int}(undef, length(lex), getncells(grid))
    for cell in CellIterator(dh)
        dofs = celldofs(cell)
        length(dofs) == length(lex) || error("the dof handler doesn't match the interpolation")
        for (i, dof) in pairs(dofs)
            dofmap[lex[i], cellid(cell)] = dof
        end
    end
    return dofmap
end

"""
    ConstrainedDofMap

Constraint-aware version of the dof map returned by [`lexicographic_dofmap`](@ref
Ferrite.lexicographic_dofmap): positive entries are regular global dof indices, and a
negative entry `-k` marks a constrained dof, where `k` indexes into a compressed table of
`(master dof, coefficient)` pairs describing the (homogeneous part of the) constraint. A
plain Dirichlet dof has an empty master list. Constructed with
`lexicographic_dofmap(dh, ip, ch)`.
"""
struct ConstrainedDofMap{Tv}
    dofmap::Matrix{Int}       # > 0: global dof index, -k: constraint number k
    offsets::Vector{Int}      # masters/coefficients of constraint k: offsets[k]:(offsets[k + 1] - 1)
    masters::Vector{Int}
    coefficients::Vector{Tv}
end

"""
    lexicographic_dofmap(dh::DofHandler, ip, ch::ConstraintHandler)

Constraint-aware version returning a [`ConstrainedDofMap`](@ref Ferrite.ConstrainedDofMap):
[`read_dof_values!`](@ref Ferrite.read_dof_values!) and
[`distribute_local_to_global!`](@ref Ferrite.distribute_local_to_global!) then apply the
*homogeneous* part of the constraints in `ch` during the gather and (transposed) during the
scatter, i.e. the resulting operator is `Cᵀ A C` in terms of the constraint matrix `C`
(see [`create_constraint_matrix`](@ref)). Constraint inhomogeneities are not applied --
handle them by lifting, see the matrix-free how-to in the documentation.
"""
function lexicographic_dofmap(dh::DofHandler, ip::Interpolation, ch::ConstraintHandler)
    @assert isclosed(ch)
    dofmap = lexicographic_dofmap(dh, ip)
    np = length(ch.prescribed_dofs)
    offsets = Vector{Int}(undef, np + 1)
    offsets[1] = 1
    masters = Int[]
    coefficients = eltype(ch.inhomogeneities)[]
    for k in 1:np
        dofcoef = ch.dofcoefficients[k]
        if dofcoef !== nothing
            for (d, v) in dofcoef
                # Prescribed masters are part of the effective inhomogeneity (see
                # `create_constraint_matrix`), not of the homogeneous application.
                ch.isconstrained[d] && continue
                push!(masters, d)
                push!(coefficients, v)
            end
        end
        offsets[k + 1] = length(masters) + 1
    end
    for l in eachindex(dofmap)
        k = get(ch.dofmapping, dofmap[l], 0)
        k != 0 && (dofmap[l] = -k)
    end
    return ConstrainedDofMap(dofmap, offsets, masters, coefficients)
end

###########################
# Gather / scatter        #
###########################

"""
    read_dof_values!(ev::TensorProductEvaluator, x::AbstractVector, dofs::AbstractVector{Int})
    read_dof_values!(ev::TensorProductEvaluator, x::AbstractVector, dofmap, e::Int)

Gather the local dof values `x[dofs]` of cell `e` into the evaluator, where `dofs` are
permuted to lexicographic ordering (column `e` of the `dofmap` from
[`lexicographic_dofmap`](@ref Ferrite.lexicographic_dofmap)).

When `dofmap` is a [`ConstrainedDofMap`](@ref Ferrite.ConstrainedDofMap) the homogeneous
part of the constraints is applied during the gather: a constrained dof takes the value
`Σₘ aₘ x[m]` of its masters (zero for a plain Dirichlet dof) and the entry `x[d]` of a
constrained dof `d` is never read.
"""
function read_dof_values!(ev::TensorProductEvaluator, x::AbstractVector, dofs::AbstractVector{Int})
    @inbounds for l in eachindex(ev.ue)
        ev.ue[l] = x[dofs[l]]
    end
    return ev
end
function read_dof_values!(ev::TensorProductEvaluator, x::AbstractVector, dofmap::Matrix{Int}, e::Int)
    return read_dof_values!(ev, x, view(dofmap, :, e))
end
function read_dof_values!(ev::TensorProductEvaluator, x::AbstractVector, m::ConstrainedDofMap, e::Int)
    (; dofmap, offsets, masters, coefficients) = m
    ue = ev.ue
    @inbounds for l in eachindex(ue)
        d = dofmap[l, e]
        if d > 0
            ue[l] = x[d]
        else
            v = zero(eltype(ue))
            for j in offsets[-d]:(offsets[-d + 1] - 1)
                v += coefficients[j] * x[masters[j]]
            end
            ue[l] = v
        end
    end
    return ev
end

"""
    distribute_local_to_global!(y::AbstractVector, ev::TensorProductEvaluator, dofs::AbstractVector{Int})
    distribute_local_to_global!(y::AbstractVector, ev::TensorProductEvaluator, dofmap, e::Int)

Scatter-add the local result of [`integrate_gradients!`](@ref Ferrite.integrate_gradients!)
into `y[dofs]` for cell `e`, where `dofs` are permuted to lexicographic ordering (column
`e` of the `dofmap` from [`lexicographic_dofmap`](@ref Ferrite.lexicographic_dofmap)).

When `dofmap` is a [`ConstrainedDofMap`](@ref Ferrite.ConstrainedDofMap) the transposed
homogeneous constraints are applied during the scatter: the contribution of a constrained
dof is distributed to its masters (scaled by the constraint coefficients) and `y[d]` of a
constrained dof `d` is never written -- combined with the constrained gather this realizes
the operator `Cᵀ A C`, which has zero rows and columns for all constrained dofs.
"""
function distribute_local_to_global!(y::AbstractVector, ev::TensorProductEvaluator, dofs::AbstractVector{Int})
    @inbounds for l in eachindex(ev.ye)
        y[dofs[l]] += ev.ye[l]
    end
    return y
end
function distribute_local_to_global!(y::AbstractVector, ev::TensorProductEvaluator, dofmap::Matrix{Int}, e::Int)
    return distribute_local_to_global!(y, ev, view(dofmap, :, e))
end
function distribute_local_to_global!(y::AbstractVector, ev::TensorProductEvaluator, m::ConstrainedDofMap, e::Int)
    (; dofmap, offsets, masters, coefficients) = m
    ye = ev.ye
    @inbounds for l in eachindex(ye)
        d = dofmap[l, e]
        if d > 0
            y[d] += ye[l]
        else
            for j in offsets[-d]:(offsets[-d + 1] - 1)
                y[masters[j]] += coefficients[j] * ye[l]
            end
        end
    end
    return y
end

###########################
# Evaluate / integrate    #
###########################

"""
    evaluate_gradients!(ev::TensorProductEvaluator)

Compute the gradient with respect to the *reference* coordinates of the field defined by
the gathered dof values (see [`read_dof_values!`](@ref Ferrite.read_dof_values!)) in all
quadrature points. Access the result with [`get_gradient`](@ref Ferrite.get_gradient).

The evaluation is sum-factorized: for each field component and gradient direction the
interpolation factors into three successive 1D contractions, e.g.
`gx = (B ⊗ B ⊗ D) ue` for the first reference coordinate direction.
"""
function evaluate_gradients!(ev::TensorProductEvaluator{NQ, N, NC}) where {NQ, N, NC}
    (; B, D, ue, t1, t2, s1, s2, s3, gx, gy, gz) = ev
    n, nq = Val(N), Val(NQ)
    for c in 1:NC
        uc = @view ue[:, :, :, c]
        contract_1!(t1, B, uc, n, nq)
        contract_1!(t2, D, uc, n, nq)
        contract_2!(s1, B, t1, n, nq)
        contract_2!(s2, D, t1, n, nq)
        contract_2!(s3, B, t2, n, nq)
        contract_3!(@view(gx[:, :, :, c]), B, s3, n, nq) # gx = (B ⊗ B ⊗ D) ue
        contract_3!(@view(gy[:, :, :, c]), B, s2, n, nq) # gy = (B ⊗ D ⊗ B) ue
        contract_3!(@view(gz[:, :, :, c]), D, s1, n, nq) # gz = (D ⊗ B ⊗ B) ue
    end
    return ev
end

"""
    integrate_gradients!(ev::TensorProductEvaluator)

Contract the quadrature point data set by [`submit_gradient!`](@ref
Ferrite.submit_gradient!) with the gradients of the test functions and integrate, i.e.
compute `yₐ = Σ_q ∇̂Nₐ(ξ_q) ⋅ ĥ_q` for all local (lexicographic) dofs `a`. This is the
transpose of [`evaluate_gradients!`](@ref Ferrite.evaluate_gradients!). Scatter the result
with [`distribute_local_to_global!`](@ref Ferrite.distribute_local_to_global!).
"""
function integrate_gradients!(ev::TensorProductEvaluator{NQ, N, NC}) where {NQ, N, NC}
    (; Bᵀ, Dᵀ, ye, tmp, t1, s1, gx, gy, gz) = ev
    n, nq = Val(N), Val(NQ)
    for c in 1:NC
        yc = @view ye[:, :, :, c]
        contract_3!(s1, Bᵀ, @view(gx[:, :, :, c]), nq, n)
        contract_2!(t1, Bᵀ, s1, nq, n)
        contract_1!(yc, Dᵀ, t1, nq, n)  # ye  = (Bᵀ ⊗ Bᵀ ⊗ Dᵀ) gx
        contract_3!(s1, Bᵀ, @view(gy[:, :, :, c]), nq, n)
        contract_2!(t1, Dᵀ, s1, nq, n)
        contract_1!(tmp, Bᵀ, t1, nq, n) # ye += (Bᵀ ⊗ Dᵀ ⊗ Bᵀ) gy
        yc .+= tmp
        contract_3!(s1, Dᵀ, @view(gz[:, :, :, c]), nq, n)
        contract_2!(t1, Bᵀ, s1, nq, n)
        contract_1!(tmp, Bᵀ, t1, nq, n) # ye += (Dᵀ ⊗ Bᵀ ⊗ Bᵀ) gz
        yc .+= tmp
    end
    return ev
end

###########################
# Quadrature point access #
###########################

"""
    get_gradient(ev::TensorProductEvaluator, q::Int)

Return the reference gradient computed by [`evaluate_gradients!`](@ref
Ferrite.evaluate_gradients!) in quadrature point `q`: a `Vec{3}` for scalar fields and a
`Tensor{2, 3}` (with `ĝ[i, j] = ∂u_i / ∂ξ_j`) for vector fields.
"""
@inline function get_gradient(ev::TensorProductEvaluator{NQ, <:Any, 1, T}, q::Int) where {NQ, T}
    return @inbounds Vec{3, T}((ev.gx[q], ev.gy[q], ev.gz[q]))
end
@inline function get_gradient(ev::TensorProductEvaluator{NQ, <:Any, 3, T}, q::Int) where {NQ, T}
    Q = NQ^3
    return @inbounds Tensor{2, 3, T}(
        (
            ev.gx[q], ev.gx[q + Q], ev.gx[q + 2Q],
            ev.gy[q], ev.gy[q + Q], ev.gy[q + 2Q],
            ev.gz[q], ev.gz[q + Q], ev.gz[q + 2Q],
        )
    )
end

"""
    submit_gradient!(ev::TensorProductEvaluator, ĥ, q::Int)

Set the quantity to be contracted with the reference gradient of the test functions by
[`integrate_gradients!`](@ref Ferrite.integrate_gradients!) in quadrature point `q`:
a `Vec{3}` for scalar fields and a (symmetric or general) second order tensor for vector
fields. Note that this overwrites the value evaluated by [`evaluate_gradients!`](@ref
Ferrite.evaluate_gradients!) in the same point.
"""
@inline function submit_gradient!(ev::TensorProductEvaluator{<:Any, <:Any, 1}, ĥ::Vec{3}, q::Int)
    @inbounds begin
        ev.gx[q] = ĥ[1]
        ev.gy[q] = ĥ[2]
        ev.gz[q] = ĥ[3]
    end
    return ev
end
@inline function submit_gradient!(ev::TensorProductEvaluator{NQ, <:Any, 3}, ĥ::SecondOrderTensor{3}, q::Int) where {NQ}
    Q = NQ^3
    @inbounds begin
        ev.gx[q] = ĥ[1, 1]; ev.gx[q + Q] = ĥ[2, 1]; ev.gx[q + 2Q] = ĥ[3, 1]
        ev.gy[q] = ĥ[1, 2]; ev.gy[q + Q] = ĥ[2, 2]; ev.gy[q + 2Q] = ĥ[3, 2]
        ev.gz[q] = ĥ[1, 3]; ev.gz[q + Q] = ĥ[2, 3]; ev.gz[q + 2Q] = ĥ[3, 3]
    end
    return ev
end

###########################
# Quadrature point data   #
###########################

"""
    quadrature_point_data(f::Function, grid::AbstractGrid, qr::QuadratureRule)

Precompute user data for every quadrature point of every cell ("partial assembly"): return
a `getnquadpoints(qr) × getncells(grid)` matrix with entries `f(x_q, J_q, w_q)`, where
`x_q` is the spatial coordinate of the quadrature point, `J_q = ∂x/∂ξ` the Jacobian of the
geometry mapping, and `w_q` the quadrature weight. For example, for the Laplace operator
with a variable coefficient `κ`, the natural per-point data is
```julia
Dq = quadrature_point_data(grid, qr) do x, J, w
    Jinv = inv(J)
    return det(J) * w * κ(x) * dott(Jinv)
end
```
"""
function quadrature_point_data(f::F, grid::AbstractGrid{3}, qr::QuadratureRule) where {F <: Function}
    ip_geo = geometric_interpolation(getcelltype(grid))
    ngeo = getnbasefunctions(ip_geo)
    w = getweights(qr)
    nqp = getnquadpoints(qr)
    N_geo = [reference_shape_value(ip_geo, ξ, a) for a in 1:ngeo, ξ in getpoints(qr)]
    dNdξ_geo = [reference_shape_gradient(ip_geo, ξ, a) for a in 1:ngeo, ξ in getpoints(qr)]
    columns = map(1:getncells(grid)) do e
        x = getcoordinates(grid, e)
        return map(1:nqp) do q
            J = zero(Tensor{2, 3})
            x_q = zero(Vec{3})
            for a in 1:ngeo
                J += x[a] ⊗ dNdξ_geo[a, q]
                x_q += N_geo[a, q] * x[a]
            end
            return f(x_q, J, w[q])
        end
    end
    return reduce(hcat, columns)
end
