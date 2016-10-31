# Common methods for both FECellValues and FEBoundaryValues

"""
The quadrature rule for the `AbstractFEValues` type.

    getquadrule(fe_v::AbstractFEValues)

** Arguments **

* `fe_v`: the `AbstractFEValues` object

** Results **

* `::QuadratureRule`: the quadrature rule.

"""
getquadrule(fe_cv::FECellValues) = fe_cv.quad_rule
getquadrule(fe_cv::FECellVectorValues) = fe_cv.quad_rule
getquadrule(fe_bv::FEBoundaryValues) = fe_bv.quad_rule[fe_bv.current_boundary[]]

"""
The function space for the `AbstractFEValues` type.

    getfunctionspace(fe_v::AbstractFEValues)

**Arguments**

* `fe_v`: the `AbstractFEValues` object

**Results**

* `::FunctionSpace`: the function space

"""
getfunctionspace(fe_v::AbstractFEValues) = fe_v.function_space

"""
The function space used for geometric interpolation for the `AbstractFEValues` type.

    getgeometricspace(fe_v::AbstractFEValues)

**Arguments**

* `fe_v`: the `AbstractFEValues` object

**Results**

* `::FunctionSpace`: the geometric interpolation function space

"""
getgeometricspace(fe_v::AbstractFEValues) = fe_v.geometric_space

"""
The product between the determinant of the Jacobian and the quadrature point weight for a given quadrature point: ``\\det(J(\\mathbf{x})) w_q``

    getdetJdV(fe_v::AbstractFEValues, quadrature_point::Int)

** Arguments:**

* `fe_v`: the `AbstractFEValues` object
* `quadrature_point` The quadrature point number

**Results:**

* `::Number`

**Details:**

This value is typically used when one integrates a function on a finite element cell or boundary as

``\\int\\limits_\\Omega f(\\mathbf{x}) d \\Omega \\approx \\sum\\limits_{q = 1}^{n_q} f(\\mathbf{x}_q) \\det(J(\\mathbf{x})) w_q``
``\\int\\limits_\\Gamma f(\\mathbf{x}) d \\Gamma \\approx \\sum\\limits_{q = 1}^{n_q} f(\\mathbf{x}_q) \\det(J(\\mathbf{x})) w_q``

"""
@inline getdetJdV(fe_cv::AbstractFECellValues, q_point::Int) = fe_cv.detJdV[q_point]
@inline getdetJdV(fe_bv::FEBoundaryValues, q_point::Int) = fe_bv.detJdV[q_point, fe_bv.current_boundary[]]

"""
Computes the value of the shape function

    shape_value(fe_v::AbstractFEValues, quadrature_point::Int, base_function::Int)

Gets the values of the shape function for a given quadrature point and base_func

"""
@inline shape_value(fe_cv::AbstractFECellValues, q_point::Int, base_func::Int) = fe_cv.N[base_func, q_point]
@inline shape_value(fe_bv::FEBoundaryValues, q_point::Int, base_func::Int) = fe_bv.N[base_func, q_point, fe_bv.current_boundary[]]

@inline geometric_value(fe_cv::AbstractFECellValues, q_point::Int, base_func::Int) = fe_cv.M[base_func, q_point]
@inline geometric_value(fe_bv::FEBoundaryValues, q_point::Int, base_func::Int) = fe_bv.M[base_func, q_point, fe_bv.current_boundary[]]



"""
Get the gradient of the shape functions for a given quadrature point and base function
"""
@inline shape_gradient(fe_cv::AbstractFECellValues, q_point::Int, base_func::Int) = fe_cv.dNdx[base_func, q_point]
@inline shape_gradient(fe_bv::FEBoundaryValues, q_point::Int, base_func::Int) = fe_bv.dNdx[base_func, q_point, fe_bv.current_boundary[]]
@inline shape_symmetric_gradient(fe_cv::FECellVectorValues, q_point::Int, base_func::Int) = symmetric(shape_gradient(fe_cv, q_point, base_func))
const shape_derivative = shape_gradient

"""
Get the divergence of the shape functions for a given quadrature point and base function
"""
@inline shape_divergence(fe_cv::AbstractFECellValues, q_point::Int, base_func::Int) = sum(fe_cv.dNdx[base_func, q_point])
@inline shape_divergence(fe_bv::FEBoundaryValues, q_point::Int, base_func::Int) = sum(fe_bv.dNdx[base_func, q_point, fe_bv.current_boundary[]])


"""
Computes the value in a quadrature point for a scalar or vector valued function

    function_value{dim, T}(fe_v::AbstractFEValues{dim}, q_point::Int, u::Vector{T})
    function_value{dim, T}(fe_v::AbstractFEValues{dim}, q_point::Int, u::Vector{Vec{dim, T}})

**Arguments:**

* `fe_v`: the `AbstractFEValues` object
* `q_point`: the quadrature point number
* `u`: the value of the function in the nodes

**Results:**

* `::Number`: the value of a scalar valued function
* `::Vec{dim, T}` the value of a vector valued function

**Details:**

The value of a scalar valued function is computed as ``u(\\mathbf{x}) = \\sum\\limits_{i = 1}^n N_i (\\mathbf{x}) u_i``
where ``u_i`` are the value of ``u`` in the nodes. For a vector valued function the value is calculated as
``\\mathbf{u}(\\mathbf{x}) = \\sum\\limits_{i = 1}^n N_i (\\mathbf{x}) \\mathbf{u}_i`` where ``\\mathbf{u}_i`` are the
nodal values of ``\\mathbf{u}``.
"""
function function_value{dim, T}(fe_v::AbstractFEValues{dim}, q_point::Int, u::Vector{T})
    n_base_funcs = getnbasefunctions(getfunctionspace(fe_v))
    @assert length(u) == n_base_funcs
    s = zero(T)
    @inbounds for i in 1:n_base_funcs
        s += shape_value(fe_v, q_point, i) * u[i]
    end
    return s
end

function function_value{dim, T}(fe_v::AbstractFEValues{dim}, q_point::Int, u::Vector{Vec{dim, T}})
    n_base_funcs = getnbasefunctions(getfunctionspace(fe_v))
    @assert length(u) == n_base_funcs
    vec = zero(Vec{dim, T})
    @inbounds for i in 1:n_base_funcs
        vec += shape_value(fe_v, q_point, i) * u[i]
    end
    return vec
end

function function_value{dim, T}(fe_v::FECellVectorValues{dim}, q_point::Int, u::Vector{Vec{dim, T}})
    n_base_funcs = getnbasefunctions(getfunctionspace(fe_v))
    @assert length(u) == n_base_funcs
    vec = zero(Vec{dim, T})
    basefunc = 1
    @inbounds for i in 1:n_base_funcs
        for j in 1:dim
            vec += shape_value(fe_v, q_point, basefunc) * u[i][j]
            basefunc += 1
        end
    end
    return vec
end

"""
Computes the gradient in a quadrature point for a scalar or vector valued function

    function_scalar_gradient{dim, T}(fe_v::AbstractFEValues{dim}, q_point::Int, u::Vector{T})
    function_vector_gradient{dim, T}(fe_v::AbstractFEValues{dim}, q_point::Int, u::Vector{Vec{dim, T}})

**Arguments:**

* `fe_v`: the `AbstractFEValues` object
* `q_point`: the quadrature point number
* `u`: the value of the function in the nodes

**Results:**

* `::Vec{dim, T}`: the gradient of a scalar valued function
* `::Tensor{2, dim, T}`: the gradient of a vector valued function

**Details:**

The gradient of a scalar function is computed as
``\\mathbf{\\nabla} u(\\mathbf{x}) = \\sum\\limits_{i = 1}^n \\mathbf{\\nabla} N_i (\\mathbf{x}) u_i``
where ``u_i`` are the nodal values of the function.
For a vector valued function the gradient is computed as
``\\mathbf{\\nabla} \\mathbf{u}(\\mathbf{x}) = \\sum\\limits_{i = 1}^n \\mathbf{\\nabla} N_i (\\mathbf{x}) \\otimes \\mathbf{u}_i``
where ``\\mathbf{u}_i`` are the nodal values of ``\\mathbf{u}``.
"""
function function_gradient{dim, T}(fe_v::AbstractFEValues{dim}, q_point::Int, u::Vector{T})
    n_base_funcs = getnbasefunctions(getfunctionspace(fe_v))
    @assert length(u) == n_base_funcs
    grad = zero(Vec{dim, T})
    @inbounds for i in 1:n_base_funcs
        grad += shape_gradient(fe_v, q_point, i) * u[i]
    end
    return grad
end

function function_gradient{dim, T}(fe_v::AbstractFEValues{dim}, q_point::Int, u::Vector{Vec{dim, T}})
    n_base_funcs = getnbasefunctions(getfunctionspace(fe_v))
    @assert length(u) == n_base_funcs
    grad = zero(Tensor{2, dim, T})
    @inbounds for i in 1:n_base_funcs
        grad += u[i] ⊗ shape_gradient(fe_v, q_point, i)
    end
    return grad
end

function function_gradient{dim, T}(fe_v::FECellVectorValues{dim}, q_point::Int, u::Vector{Vec{dim, T}})
    n_base_funcs = getnbasefunctions(getfunctionspace(fe_v))
    @assert length(u) == n_base_funcs
    grad = zero(Tensor{2, dim, T})
    basefunc_count = 1
    @inbounds for i in 1:n_base_funcs
        for j in 1:dim
            grad += u[i][j] * shape_gradient(fe_v, q_point, basefunc_count)
            basefunc_count += 1
        end
    end
    return grad
end


const function_derivative = function_gradient

"""
Computes the symmetric gradient for a vector valued function in a quadrature point.

    function_symmetric_gradient{dim, T}(fe_v::AbstractFEValues{dim}, q_point::Int, u::Vector{Vec{dim, T}})

**Arguments:**

* `fe_v`: the `AbstractFEValues` object
* `q_point`: the quadrature point number
* `u`: the value of the function in the nodes

**Results:**

* `::SymmetricTensor{2, dim, T}`: the symmetric gradient

**Details:**

The symmetric gradient of a scalar function is computed as

``\\left[ \\mathbf{\\nabla}  \\mathbf{u}(\\mathbf{x_q}) \\right]^\\text{sym} =  \\sum\\limits_{i = 1}^n  \\frac{1}{2} \\left[ \\mathbf{\\nabla} N_i (\\mathbf{x}_q) \\otimes \\mathbf{u}_i + \\mathbf{u}_i  \\otimes  \\mathbf{\\nabla} N_i (\\mathbf{x}_q) \\right]``

where ``\\mathbf{u}_i`` are the nodal values of the function.
"""
function function_symmetric_gradient{dim, T}(fe_v::AbstractFEValues{dim}, q_point::Int, u::Vector{Vec{dim, T}})
    grad = function_gradient(fe_v, q_point, u)
    return symmetric(grad)
end

"""
Computes the divergence in a quadrature point for a vector valued function.

    function_divergence{dim, T}(fe_v::AbstractFEValues{dim}, q_point::Int, u::Vector{Vec{dim, T}})

**Arguments:**

* `fe_v`: the `AbstractFEValues` object
* `q_point`: the quadrature point number
* `u`: the value of the function in the nodes

**Results:**

* `::Number`: the divergence of the function

**Details:**

The divergence of a vector valued functions in the quadrature point ``\\mathbf{x}_q)`` is computed as

``\\mathbf{\\nabla} \\cdot \\mathbf{u}(\\mathbf{x_q}) = \\sum\\limits_{i = 1}^n \\mathbf{\\nabla} N_i (\\mathbf{x_q}) \\cdot \\mathbf{u}_i``


where ``\\mathbf{u}_i`` are the nodal values of the function.
"""
function function_divergence{dim, T}(fe_v::AbstractFEValues{dim}, q_point::Int, u::Vector{Vec{dim, T}})
    n_base_funcs = getnbasefunctions(getfunctionspace(fe_v))
    @assert length(u) == n_base_funcs
    diverg = zero(T)
    @inbounds for i in 1:n_base_funcs
        diverg += shape_gradient(fe_v, q_point, i) ⋅ u[i]
    end
    return diverg
end

function function_divergence{dim, T}(fe_v::FECellVectorValues{dim}, q_point::Int, u::Vector{Vec{dim, T}})
    n_base_funcs = getnbasefunctions(getfunctionspace(fe_v))
    @assert length(u) == n_base_funcs
    diverg = zero(T)
    basefunc_count = 1
    @inbounds for i in 1:n_base_funcs
        for j in 1:dim
            grad = shape_gradient(fe_v, q_point, basefunc_count)
            basefunc_count += 1
            for k in 1:dim
                diverg += grad[k, k] * u[i][j]
            end
        end
    end
    return diverg
end


"""
    spatial_coordinate{dim, T}(fe_v::AbstractFEValues{dim}, q_point::Int, x::Vector{Vec{dim, T}})

Computes the spatial coordinate in a quadrature point.

**Arguments:**

* `fe_v`: the `AbstractFEValues` object
* `q_point`: the quadrature point number
* `x`: the nodal coordinates of the cell

**Results:**

* `::Vec{dim, T}`: the spatial coordinate

**Details:**

The coordinate is computed, using the geometric interpolation space, as ``\\mathbf{x} = \\sum\\limits_{i = 1}^n M_i (\\mathbf{x}) \\mathbf{\\hat{x}}_i``
"""
function spatial_coordinate{dim, T}(fe_v::AbstractFEValues{dim}, q_point::Int, x::Vector{Vec{dim, T}})
    n_base_funcs = getnbasefunctions(getgeometricspace(fe_v))
    @assert length(x) == n_base_funcs
    vec = zero(Vec{dim, T})
    @inbounds for i in 1:n_base_funcs
        vec += geometric_value(fe_v, q_point, i) * x[i]
    end
    return vec
end
