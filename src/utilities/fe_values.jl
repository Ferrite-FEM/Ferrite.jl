immutable FEValues{dim, T <: Real, FS <: FunctionSpace}
    N::Vector{Vector{T}}
    dNdx::Vector{Vector{Vec{dim, T}}}
    dNdξ::Vector{Vector{Vec{dim, T}}}
    detJdV::Vector{T}
    quad_rule::QuadratureRule{dim, T}
    function_space::FS
end

"""
Initializes an `FEValues` object from a function space and a quadrature rule.
"""
function FEValues{dim, T, FS <: FunctionSpace}(::Type{T}, quad_rule::QuadratureRule{dim}, func_space::FS)
        n_basefuncs = n_basefunctions(func_space)

        n_qpoints = length(points(quad_rule))

        N = [zeros(T, n_basefuncs) for i in 1:n_qpoints]
        dNdx = [[zero(Vec{dim, T}) for i in 1:n_basefuncs] for j in 1:n_qpoints]
        dNdξ = [[zero(Vec{dim, T}) for i in 1:n_basefuncs] for j in 1:n_qpoints]

        for (i, (ξ, w)) in enumerate(zip(quad_rule.points, quad_rule.weights))
            value!(func_space, N[i], ξ)
            derivative!(func_space, dNdξ[i], ξ)
        end

        FEValues(N, dNdx, dNdξ, zeros(T, n_qpoints), quad_rule, func_space)
end
function FEValues{dim, FS <: FunctionSpace}(quad_rule::QuadratureRule{dim}, func_space::FS)
    FEValues(Float64, quad_rule, func_space)
end

"""
Updates the `FEValues` object for the current element with coordinate matrix `x`.
"""
function reinit!{dim, T}(fe_v::FEValues{dim}, x::Vector{Vec{dim, T}})
    for i in 1:length(points(fe_v.quad_rule))
        ξ = points(fe_v.quad_rule)[i]
        w = weights(fe_v.quad_rule)[i]
        n_basefuncs = n_basefunctions(get_functionspace(fe_v))
        fev_J = zero(Tensor{2, dim})
        for j in 1:n_basefuncs
            fev_J += fe_v.dNdξ[i][j] ⊗ x[j]
        end
        Jinv = inv(fev_J)
        for j in 1:n_basefuncs
            fe_v.dNdx[i][j] = Jinv ⋅ fe_v.dNdξ[i][j]
        end
        fe_v.detJdV[i] = det(fev_J) * w
    end
end

"""
Returns the quadrature rule.
"""
get_quadrule(fe_v::FEValues) = fe_v.quad_rule

"""
Returns the function space.
"""
get_functionspace(fe_v::FEValues) = fe_v.function_space

"""
Gets the product between the determinant of the Jacobian and the quadrature point weight for a given quadrature point.
"""

@inline detJdV(fe_v::FEValues, q_point::Int) = fe_v.detJdV[q_point]

"""
    shape_value(fe_v, q_point::Int) -> value

Gets the value of the shape function for a given quadrature point
"""
@inline shape_value(fe_v::FEValues, q_point::Int) = fe_v.N[q_point]

"""
    shape_value(fe_v, q_point::Int, base_func::Int) -> value

Gets the value of the shape function at a given quadrature point and given base function
"""
@inline shape_value(fe_v::FEValues, q_point::Int, base_func::Int) = fe_v.N[q_point][base_func]

"""
    shape_gradient(fe_v, q_point::Int) -> gradients::Vector{Tensor{2}}

Get the gradients of the shape functions for a given quadrature point
"""
@inline shape_gradient(fe_v::FEValues, q_point::Int) = fe_v.dNdx[q_point]

"""
    shape_gradient(fe_v, q_point::Int, base_func::Int) -> gradient::Tensor{2}

Get the gradient of the shape functions for a given quadrature point and base function
"""
@inline shape_gradient(fe_v::FEValues, q_point::Int, base_func::Int) = fe_v.dNdx[q_point][base_func]


const shape_derivative = shape_gradient

"""
    function_scalar_value(fe_v, q_point::Int, u::Vector) -> value

Computes the value in a quadrature point for a scalar valued function
"""
@inline function function_scalar_value{dim, T}(fe_v::FEValues{dim}, q_point::Int, u::Vector{T})
    n_base_funcs = n_basefunctions(get_functionspace(fe_v))
    @assert length(u) == n_base_funcs
    N = shape_value(fe_v, q_point)
    s = zero(T)
    @inbounds for i in 1:n_base_funcs
        s += N[i] * u[i]
    end
    return s
end

"""
    function_vector_value(fe_v, q_point::Int, u::Vector{Tensor{1}}) -> value::Tensor{1}

Computes the value in a quadrature point for a vector valued function.
"""
@inline function function_vector_value{dim, T}(fe_v::FEValues{dim}, q_point::Int, u::Vector{Vec{dim, T}})
    n_base_funcs = n_basefunctions(get_functionspace(fe_v))
    @assert length(u) == n_base_funcs
    vec = zero(Vec{dim, T})
    N = shape_value(fe_v, q_point)
    @inbounds for i in 1:n_base_funcs
        vec += N[i] * u[i]
    end
    return vec
end

"""
    function_scalar_gradient(fe_v, q_point::Int, u::Vector) -> grad::Tensor{1}

Computes the gradient in a quadrature point for a scalar valued function.
"""
@inline function function_scalar_gradient{dim, T}(fe_v::FEValues{dim}, q_point::Int, u::Vector{T})
    n_base_funcs = n_basefunctions(get_functionspace(fe_v))
    @assert length(u) == n_base_funcs
    dN = shape_gradient(fe_v, q_point)
    grad = zero(Vec{dim, T})
    @inbounds for i in 1:n_base_funcs
        grad += dN[i] * u[i]
    end
    return grad
end

"""
    function_vector_gradient(fe_v, q_point::Int, u::Vector{Tensor{1}}) -> grad::Tensor{2}

Computes the gradient (jacobian) in a quadrature point for a vector valued function.
"""
@inline function function_vector_gradient{dim, T}(fe_v::FEValues{dim}, q_point::Int, u::Vector{Vec{dim, T}})
    n_base_funcs = n_basefunctions(get_functionspace(fe_v))
    @assert length(u) == n_base_funcs
    dN = shape_gradient(fe_v, q_point)
    grad = zero(Tensor{2, dim, T})
    for i in 1:n_base_funcs
        grad += u[i] ⊗ dN[i]
    end
    return grad
end

"""
    function_vector_symmetric_gradient(fe_v, q_point::Int, u::Vector{Tensor{1}}) -> sym_grad::SymmetricTensor{2}

Computes the symmetric gradient (jacobian) in a quadrature point for a vector valued function.
Result is stored in `grad`.
"""
@inline function function_vector_symmetric_gradient{dim, T}(fe_v::FEValues{dim}, q_point::Int, u::Vector{Vec{dim, T}})
    grad = function_vector_gradient(fe_v, q_point, u)
    return convert(SymmetricTensor{2, dim, T}, grad)
end

"""
    function_vector_divergence(fe_v, q_point::Int, u::Vector{Tensor{1}}) -> divergence

Computes the divergence in a quadrature point for a vector valued function.
"""
@inline function function_vector_divergence{dim, T}(fe_v::FEValues{dim}, q_point::Int, u::Vector{Vec{dim, T}})
    n_base_funcs = n_basefunctions(get_functionspace(fe_v))
    @assert length(u) == n_base_funcs
    dN = shape_gradient(fe_v, q_point)
    diverg = zero(T)
    @inbounds for i in 1:n_base_funcs
        diverg += dN[i] ⋅ u[i]
    end
    return diverg
end
