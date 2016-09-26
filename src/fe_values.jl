"""
An `FEValues` object facilitates the process of evaluating values shape functions, gradients of shape functions,
values of nodal functions, gradients and divergences of nodal functions etc.

**Constructor**

    FEValues([::Type{T}], quad_rule::QuadratureRule, function_space::FunctionSpace, [geometric_space::FunctionSpace])


**Arguments**

* `T` an optional argument to determine the type the internal data is stored as.
* `quad_rule` an instance of a [`QuadratureRule`](@ref)
* `function_space` an instance of a [`FunctionSpace`](@ref) used to interpolate the approximated function
* `geometric_space` an optional instance of a [`FunctionSpace`](@ref) which is used to interpolate the geometry 

** Common methods**

* [`get_quadrule`](@ref)
* [`get_functionspace`](@ref)
* [`get_geometricspace`](@ref)
* [`detJdV`](@ref)

* [`shape_value`](@ref)
* [`shape_gradient`](@ref)
* [`shape_divergence`](@ref)
* [`shape_derivative`](@ref)

* [`function_scalar_value`](@ref)
* [`function_vector_value`](@ref)
* [`function_scalar_gradient`](@ref)
* [`function_vector_divergence`](@ref)
* [`function_vector_gradient`](@ref)
* [`function_vector_symmetric_gradient`](@ref)
"""
immutable FEValues{dim, T <: Real, FS <: FunctionSpace, GS <: FunctionSpace}
    N::Vector{Vector{T}}
    dNdx::Vector{Vector{Vec{dim, T}}}
    dNdξ::Vector{Vector{Vec{dim, T}}}
    detJdV::Vector{T}
    quad_rule::QuadratureRule{dim, T}
    function_space::FS
    dMdξ::Vector{Vector{Vec{dim, T}}}
    geometric_space::GS
end

FEValues{dim, FS <: FunctionSpace, GS <: FunctionSpace}(quad_rule::QuadratureRule{dim}, func_space::FS, geom_space::GS=func_space) = FEValues(Float64, quad_rule, func_space, geom_space)

function FEValues{dim, T, FS <: FunctionSpace, GS <: FunctionSpace}(::Type{T}, quad_rule::QuadratureRule{dim}, func_space::FS, geom_space::GS=func_space)
    n_qpoints = length(points(quad_rule))

    # Function interpolation
    n_func_basefuncs = n_basefunctions(func_space)

    N = [zeros(T, n_func_basefuncs) for i in 1:n_qpoints]
    dNdx = [[zero(Vec{dim, T}) for i in 1:n_func_basefuncs] for j in 1:n_qpoints]
    dNdξ = [[zero(Vec{dim, T}) for i in 1:n_func_basefuncs] for j in 1:n_qpoints]

    # Geometry interpolation
    n_geom_basefuncs = n_basefunctions(geom_space)
    dMdξ = [[zero(Vec{dim, T}) for i in 1:n_geom_basefuncs] for j in 1:n_qpoints]

    for (i, (ξ, w)) in enumerate(zip(quad_rule.points, quad_rule.weights))
        value!(func_space, N[i], ξ)
        derivative!(func_space, dNdξ[i], ξ)
        derivative!(geom_space, dMdξ[i], ξ)
    end

    FEValues(N, dNdx, dNdξ, zeros(T, n_qpoints), quad_rule, func_space, dMdξ, geom_space)
end



"""
Updates the `FEValues` object for an element.

    reinit!{dim, T}(fe_v::FEValues{dim}, x::Vector{Vec{dim, T}})

** Arguments **

* `fe_values`: the `FEValues` object
* `x`: A `Vector` of `Vec`, one for each nodal position in the element.

** Result **

* nothing


**Details**


"""
function reinit!{dim, T}(fe_v::FEValues{dim}, x::Vector{Vec{dim, T}})
    n_geom_basefuncs = n_basefunctions(get_geometricspace(fe_v))
    n_func_basefuncs = n_basefunctions(get_functionspace(fe_v))
    @assert length(x) == n_geom_basefuncs

    for i in 1:length(points(fe_v.quad_rule))
        w = weights(fe_v.quad_rule)[i]
        fev_J = zero(Tensor{2, dim})
        for j in 1:n_geom_basefuncs
            fev_J += fe_v.dMdξ[i][j] ⊗ x[j]
        end
        Jinv = inv(fev_J)
        for j in 1:n_func_basefuncs
            fe_v.dNdx[i][j] = Jinv ⋅ fe_v.dNdξ[i][j]
        end
        fe_v.detJdV[i] = det(fev_J) * w
    end
end

"""
The quadrature rule for the `FEValues` type.

    get_quadrule(fe_v::FEValues)

** Arguments **

* `fe_values`: the `FEValues` object

** Results **

* `::QuadratureRule`: the quadrature rule.

"""
get_quadrule(fe_v::FEValues) = fe_v.quad_rule

"""
The function space for the `FEValues` type.

**Arguments**

* `fe_values`: the `FEValues` object

**Results**

* `::FunctionSpace`: the function space

"""
get_functionspace(fe_v::FEValues) = fe_v.function_space

"""
The function space used for geometric interpolation for the `FEValues` type.

**Arguments**

* `fe_values`: the `FEValues` object

**Results**

* `::FunctionSpace`: the geometric interpolation function space

"""
get_geometricspace(fe_v::FEValues) = fe_v.geometric_space


"""
The product between the determinant of the Jacobian and the quadrature point weight for a given quadrature point: ``\\det(J(\\mathbf{x})) w_q``

    detJdV(fe_v::FEValues, quadrature_point::Int)

** Arguments:**

* `fe_values`: the `FEValues` object
* `quadrature_point` The quadrature point number

**Results:**

* `::Number`

**Details:**

This value is typically used when one integrates a function on a finite element as

``\\int\\limits_\\Omega f(\\mathbf{x}) d \\Omega \\approx \\sum\\limits_{q = 1}^{n_q} f(\\mathbf{x}_q) \\det(J(\\mathbf{x})) w_q``

"""
@inline detJdV(fe_v::FEValues, q_point::Int) = fe_v.detJdV[q_point]



"""
Computes the value of the shape function
    shape_value(fe_v::FEValues, quadrature_point::Int, [base_function::Int])

Gets the values of the shape function for a given quadrature point and base_func

"""
@inline shape_value(fe_v::FEValues, q_point::Int) = fe_v.N[q_point]
@inline shape_value(fe_v::FEValues, q_point::Int, base_func::Int) = fe_v.N[q_point][base_func]

"""
Get the gradients of the shape functions for a given quadrature point
"""
@inline shape_gradient(fe_v::FEValues, q_point::Int) = fe_v.dNdx[q_point]

"""
Get the gradient of the shape functions for a given quadrature point and base function
"""
@inline shape_gradient(fe_v::FEValues, q_point::Int, base_func::Int) = fe_v.dNdx[q_point][base_func]

"""
Get the divergence of the shape functions for a given quadrature point and base function
"""
@inline shape_divergence(fe_v::FEValues, q_point::Int, base_func::Int) = sum(fe_v.dNdx[q_point][base_func])


const shape_derivative = shape_gradient

"""
Computes the value in a quadrature point for a scalar valued function

    function_scalar_value{dim, T}(fe_v::FEValues{dim}, q_point::Int, u::Vector{T})

**Arguments:**

* `fe_v`: the `FEValues` object
* `q_point`: the quadrature point number
* `u`: the value of the function in the nodes

**Results:**

* `::Number`: the value of the function

**Details:**

The value of a scalar valued function is computed as ``T(\\mathbf{x}) = \\sum\\limits_{i = 1}^n N_i (\\mathbf{x}) T_i``
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
    function_vector_value{dim, T}(fe_v::FEValues{dim}, q_point::Int, u::Vector{Vec{dim, T}})

Computes the value in a quadrature point for a vector valued function.

**Arguments:**

* `fe_v`: the `FEValues` object
* `q_point`: the quadrature point number
* `u`: the value of the function in the nodes

**Results:**

* `::Vec{dim, T}`: the value of the function

**Details:**

The value of a vector valued function is computed as ``\\mathbf{u}(\\mathbf{x}) = \\sum\\limits_{i = 1}^n N_i (\\mathbf{x}) \\mathbf{u}_i``
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
    function_scalar_gradien{dim, T}t(fe_v::FEValues{dim}, q_point::Int, u::Vector{T}) -> grad::Tensor{1}

Computes the gradient for a scalar valued function in a quadrature point .

**Arguments:**

* `fe_v`: the `FEValues` object
* `q_point`: the quadrature point number
* `u`: the value of the function in the nodes

**Results:**

* `::Vec{dim, T}`: the gradient

**Details:**

The gradient of a scalar function is computed as ``\\mathbf{\\nabla} T(\\mathbf{x}) = \\sum\\limits_{i = 1}^n \\mathbf{\\nabla} N_i (\\mathbf{x}) T_i``
where ``T_i`` are the nodal values of the function.
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
Computes the gradient for a vector valued function in a quadrature point.

    function_vector_gradient{dim, T}(fe_v::FEValues{dim}, q_point::Int, u::Vector{Vec{dim, T}})

**Arguments:**

* `fe_v`: the `FEValues` object
* `q_point`: the quadrature point number
* `u`: the value of the function in the nodes

**Results:**

* `::Tensor{2, dim, T}`: the gradient

**Details:**

The gradient of a scalar function is computed as ``\\mathbf{\\nabla} \\mathbf{u}(\\mathbf{x}) = \\sum\\limits_{i = 1}^n \\mathbf{\\nabla} N_i (\\mathbf{x}) \\otimes \\mathbf{u}_i``
where ``\\mathbf{u}_i`` are the nodal values of the function.
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
Computes the gradient for a vector valued function in a quadrature point.

    function_vector_symmetric_gradient{dim, T}(fe_v::FEValues{dim}, q_point::Int, u::Vector{Vec{dim, T}})

**Arguments:**

* `fe_v`: the `FEValues` object
* `q_point`: the quadrature point number
* `u`: the value of the function in the nodes

**Results:**

* `::SymmetricTensor{2, dim, T}`: the symmetric gradient

**Details:**

The symmetric gradient of a scalar function is computed as

``\\left[ \\mathbf{\\nabla}  \\mathbf{u}(\\mathbf{x_q}) \\right]^\\text{sym} =  \\sum\\limits_{i = 1}^n  \\frac{1}{2} \\left[ \\mathbf{\\nabla} N_i (\\mathbf{x}_q) \\otimes \\mathbf{u}_i + \\mathbf{u}_i  \\otimes  \\mathbf{\\nabla} N_i (\\mathbf{x}_q) \\right]``

where ``\\mathbf{u}_i`` are the nodal values of the function.
"""
@inline function function_vector_symmetric_gradient{dim, T}(fe_v::FEValues{dim}, q_point::Int, u::Vector{Vec{dim, T}})
    grad = function_vector_gradient(fe_v, q_point, u)
    return symmetric(grad)
end

"""
Computes the divergence in a quadrature point for a vector valued function.

    function_vector_divergence{dim, T}(fe_v::FEValues{dim}, q_point::Int, u::Vector{Vec{dim, T}})

**Arguments:**

* `fe_v`: the `FEValues` object
* `q_point`: the quadrature point number
* `u`: the value of the function in the nodes

**Results:**

* `::Number`: the divergence of the function

**Details:**

The divergence of a vector valued functions in the quadrature point ``\\mathbf{x}_q)`` is computed as

``\\mathbf{\\nabla} \\cdot \\mathbf{u}(\\mathbf{x_q}) = \\sum\\limits_{i = 1}^n \\mathbf{\\nabla} N_i (\\mathbf{x_q}) \\cdot \\mathbf{u}_i``


where ``\\mathbf{u}_i`` are the nodal values of the function.

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
