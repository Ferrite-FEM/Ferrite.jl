immutable FEFaceValues{dim, T <: Real, FS <: FunctionSpace, GS <: FunctionSpace}
    N::Vector{Vector{Vector{T}}}
    dNdx::Vector{Vector{Vector{Vec{dim, T}}}}
    dNdξ::Vector{Vector{Vector{Vec{dim, T}}}}
    detJdS::Vector{Vector{T}}
    quad_rule::Vector{QuadratureRule{dim, T}}
    function_space::FS
    dMdξ::Vector{Vector{Vector{Vec{dim, T}}}}
    geometric_space::GS
    current_boundary::Ref{Int}
end

FEFaceValues{dim_qr, FS <: FunctionSpace, GS <: FunctionSpace}(quad_rule::QuadratureRule{dim_qr}, func_space::FS, geom_space::GS=func_space) = FEFaceValues(Float64, quad_rule, func_space, geom_space)

function FEFaceValues{dim_qr, T, FS <: FunctionSpace, GS <: FunctionSpace}(::Type{T}, quad_rule::QuadratureRule{dim_qr}, func_space::FS, geom_space::GS=func_space)
    @assert n_dim(func_space) == n_dim(geom_space)
    @assert ref_shape(func_space) == ref_shape(geom_space)
    n_qpoints = length(points(quad_rule))
    dim = dim_qr + 1

    # Function interpolation
    n_func_basefuncs = n_basefunctions(func_space)
    boundary_quad_rule = boundary_information(func_space,quad_rule)
    n_bounds = length(boundary_quad_rule)

    N =    [[zeros(T, n_func_basefuncs) for i in 1:n_qpoints]                      for k in 1:n_bounds]
    dNdx = [[[zero(Vec{dim, T}) for i in 1:n_func_basefuncs] for j in 1:n_qpoints] for k in 1:n_bounds]
    dNdξ = [[[zero(Vec{dim, T}) for i in 1:n_func_basefuncs] for j in 1:n_qpoints] for k in 1:n_bounds]

    # Geometry interpolation
    n_geom_basefuncs = n_basefunctions(geom_space)
    dMdξ = [[[zero(Vec{dim, T}) for i in 1:n_geom_basefuncs] for j in 1:n_qpoints] for k in 1:n_bounds]
    for k in 1:n_bounds, (i, ξ) in enumerate(boundary_quad_rule[k].points)
        value!(func_space, N[k][i], ξ)
        derivative!(func_space, dNdξ[k][i], ξ)
        derivative!(geom_space, dMdξ[k][i], ξ)
    end

    detJdS = [zeros(T, n_qpoints) for i in 1:n_bounds]

    FEFaceValues(N, dNdx, dNdξ, detJdS, boundary_quad_rule, func_space, dMdξ, geom_space, Ref(0))
end

function reinit!{dim, T}(fe_fv::FEFaceValues{dim}, x::Vector{Vec{dim, T}}, boundary::Int)
    n_geom_basefuncs = n_basefunctions(get_geometricspace(fe_fv))
    n_func_basefuncs = n_basefunctions(get_functionspace(fe_fv))
    @assert length(x) == n_geom_basefuncs

    fe_fv.current_boundary[] = boundary
    cb = current_boundary(fe_fv)

    for i in 1:length(points(fe_fv.quad_rule[cb]))
        w = weights(fe_fv.quad_rule[cb])[i]
        fefv_J = zero(Tensor{2, dim})
        for j in 1:n_geom_basefuncs
            fefv_J += fe_fv.dMdξ[cb][i][j] ⊗ x[j]
        end
        Jinv = inv(fefv_J)
        for j in 1:n_func_basefuncs
            fe_fv.dNdx[cb][i][j] = Jinv ⋅ fe_fv.dNdξ[cb][i][j]
        end
        detJ = sqrt(fefv_J[1,1]^2 + fefv_J[2,1]^2) # det(fefv_J)
        detJ <= 0.0 && throw(ArgumentError("detJ is not positive: detJ = $(detJ)"))
        fe_fv.detJdS[cb][i] = detJ * w
    end
end

"""
The current active boundary `FEFaceValues` type.

    current_boundary(fe_fv::FEFaceValues)

** Arguments **

* `fe_face_values`: the `FEFaceValues` object

** Results **

* `::Int`: the current active boundary (from last `reinit!`).

"""
current_boundary(fe_fv::FEFaceValues) = fe_fv.current_boundary[]

"""
The quadrature rule for the `FEFaceValues` type.

    get_quadrule(fe_fv::FEFaceValues)

** Arguments **

* `fe_face_values`: the `FEFaceValues` object

** Results **

* `::QuadratureRule`: the quadrature rule for the current boundary.

"""
get_quadrule(fe_fv::FEFaceValues) = fe_fv.quad_rule[current_boundary(fe_fv)]

"""
The function space for the `FEFaceValues` type.

**Arguments**

* `fe_face_values`: the `FEFaceValues` object

**Results**

* `::FunctionSpace`: the function space

"""
get_functionspace(fe_fv::FEFaceValues) = fe_fv.function_space

"""
The function space used for geometric interpolation for the `FEFaceValues` type.

**Arguments**

* `fe_face_values`: the `FEFaceValues` object

**Results**

* `::FunctionSpace`: the geometric interpolation function space

"""
get_geometricspace(fe_fv::FEFaceValues) = fe_fv.geometric_space


"""
The product between the determinant of the Jacobian and the quadrature point weight for a given quadrature point: ``\\det(J(\\mathbf{x})) w_q``

    detJdS(fe_fv::FEFaceValues, quadrature_point::Int)

** Arguments:**

* `fe_face_values`: the `FEFaceValues` object
* `quadrature_point` The quadrature point number

**Results:**

* `::Number`

**Details:**

This value is typically used when one integrates a function on a finite element boundary as

``\\int\\limits_\\Gamma f(\\mathbf{x}) d \\Gamma \\approx \\sum\\limits_{q = 1}^{n_q} f(\\mathbf{x}_q) \\det(J(\\mathbf{x})) w_q``

"""
@inline detJdS(fe_fv::FEFaceValues, q_point::Int) = fe_fv.detJdS[fe_fv.current_boundary[]][q_point]



"""
Computes the value of the shape function at a given quadrature point on the current boundary
    shape_value(fe_fv::FEValues, quadrature_point::Int, [base_function::Int])

Gets the values of the shape function for a given quadrature point and base_func on the current boundary

"""
@inline shape_value(fe_fv::FEFaceValues, q_point::Int) = fe_fv.N[fe_fv.current_boundary[]][q_point]
@inline shape_value(fe_fv::FEFaceValues, q_point::Int, base_func::Int) = fe_fv.N[fe_fv.current_boundary[]][q_point][base_func]

"""
Get the gradients of the shape functions for a given quadrature point on the current boundary
"""
@inline shape_gradient(fe_fv::FEFaceValues, q_point::Int) = fe_fv.dNdx[fe_fv.current_boundary[]][q_point]

"""
Get the gradient of the shape functions for a given quadrature point and base function on the current boundary
"""
@inline shape_gradient(fe_fv::FEFaceValues, q_point::Int, base_func::Int) = fe_fv.dNdx[fe_fv.current_boundary[]][q_point][base_func]

"""
Get the divergence of the shape functions for a given quadrature point and base function on the current boundary
"""
@inline shape_divergence(fe_fv::FEFaceValues, q_point::Int, base_func::Int) = sum(fe_fv.dNdx[fe_fv.current_boundary[]][q_point][base_func])


"""
Computes the value in a quadrature point for a scalar valued function on the boundary

    function_scalar_value{dim, T}(fe_fv::FEFaceValues{dim}, q_point::Int, u::Vector{T})

**Arguments:**

* `fe_fv`: the `FEFaceValues` object
* `q_point`: the quadrature point number
* `u`: the value of the function in the nodes

**Results:**

* `::Number`: the value of the function

**Details:**

The value of a scalar valued function is computed as ``T(\\mathbf{x}) = \\sum\\limits_{i = 1}^n N_i (\\mathbf{x}) T_i``
"""
@inline function function_scalar_value{dim, T}(fe_fv::FEFaceValues{dim}, q_point::Int, u::Vector{T})
    n_base_funcs = n_basefunctions(get_functionspace(fe_fv))
    @assert length(u) == n_base_funcs
    N = shape_value(fe_fv, q_point)
    s = zero(T)
    @inbounds for i in 1:n_base_funcs
        s += N[i] * u[i]
    end
    return s
end

"""
    function_vector_value{dim, T}(fe_v::FEFaceValues{dim}, q_point::Int, u::Vector{Vec{dim, T}})

Computes the value in a quadrature point for a vector valued function.

**Arguments:**

* `fe_fv`: the `FEFaceValues` object
* `q_point`: the quadrature point number
* `u`: the value of the function in the nodes

**Results:**

* `::Vec{dim, T}`: the value of the function

**Details:**

The value of a vector valued function is computed as ``\\mathbf{u}(\\mathbf{x}) = \\sum\\limits_{i = 1}^n N_i (\\mathbf{x}) \\mathbf{u}_i``
"""
@inline function function_vector_value{dim, T}(fe_fv::FEFaceValues{dim}, q_point::Int, u::Vector{Vec{dim, T}})
    n_base_funcs = n_basefunctions(get_functionspace(fe_fv))
    @assert length(u) == n_base_funcs
    vec = zero(Vec{dim, T})
    N = shape_value(fe_fv, q_point)
    @inbounds for i in 1:n_base_funcs
        vec += N[i] * u[i]
    end
    return vec
end

"""
    function_scalar_gradient{dim, T}(fe_fv::FEFaceValues{dim}, q_point::Int, u::Vector{T}) -> grad::Tensor{1}

Computes the gradient for a scalar valued function in a quadrature point.

**Arguments:**

* `fe_fv`: the `FEFaceValues` object
* `q_point`: the quadrature point number
* `u`: the value of the function in the nodes

**Results:**

* `::Vec{dim, T}`: the gradient

**Details:**

The gradient of a scalar function is computed as ``\\mathbf{\\nabla} T(\\mathbf{x}) = \\sum\\limits_{i = 1}^n \\mathbf{\\nabla} N_i (\\mathbf{x}) T_i``
where ``T_i`` are the nodal values of the function.
"""
@inline function function_scalar_gradient{dim, T}(fe_fv::FEFaceValues{dim}, q_point::Int, u::Vector{T})
    n_base_funcs = n_basefunctions(get_functionspace(fe_fv))
    @assert length(u) == n_base_funcs
    dN = shape_gradient(fe_fv, q_point)
    grad = zero(Vec{dim, T})
    @inbounds for i in 1:n_base_funcs
        grad += dN[i] * u[i]
    end
    return grad
end


"""
Computes the gradient for a vector valued function in a quadrature point.

    function_vector_gradient{dim, T}(fe_v::FEFaceValues{dim}, q_point::Int, u::Vector{Vec{dim, T}})

**Arguments:**

* `fe_fv`: the `FEFaceValues` object
* `q_point`: the quadrature point number
* `u`: the value of the function in the nodes

**Results:**

* `::Tensor{2, dim, T}`: the gradient

**Details:**

The gradient of a scalar function is computed as ``\\mathbf{\\nabla} \\mathbf{u}(\\mathbf{x}) = \\sum\\limits_{i = 1}^n \\mathbf{\\nabla} N_i (\\mathbf{x}) \\otimes \\mathbf{u}_i``
where ``\\mathbf{u}_i`` are the nodal values of the function.
"""
@inline function function_vector_gradient{dim, T}(fe_fv::FEFaceValues{dim}, q_point::Int, u::Vector{Vec{dim, T}})
    n_base_funcs = n_basefunctions(get_functionspace(fe_v))
    @assert length(u) == n_base_funcs
    dN = shape_gradient(fe_fv, q_point)
    grad = zero(Tensor{2, dim, T})
    for i in 1:n_base_funcs
        grad += u[i] ⊗ dN[i]
    end
    return grad
end

"""
Computes the gradient for a vector valued function in a quadrature point.

    function_vector_symmetric_gradient{dim, T}(fe_fv::FEFaceValues{dim}, q_point::Int, u::Vector{Vec{dim, T}})

**Arguments:**

* `fe_fv`: the `FEFaceValues` object
* `q_point`: the quadrature point number
* `u`: the value of the function in the nodes

**Results:**

* `::SymmetricTensor{2, dim, T}`: the symmetric gradient

**Details:**

The symmetric gradient of a scalar function is computed as

``\\left[ \\mathbf{\\nabla}  \\mathbf{u}(\\mathbf{x_q}) \\right]^\\text{sym} =  \\sum\\limits_{i = 1}^n  \\frac{1}{2} \\left[ \\mathbf{\\nabla} N_i (\\mathbf{x}_q) \\otimes \\mathbf{u}_i + \\mathbf{u}_i  \\otimes  \\mathbf{\\nabla} N_i (\\mathbf{x}_q) \\right]``

where ``\\mathbf{u}_i`` are the nodal values of the function.
"""
@inline function function_vector_symmetric_gradient{dim, T}(fe_fv::FEFaceValues{dim}, q_point::Int, u::Vector{Vec{dim, T}})
    grad = function_vector_gradient(fe_fv, q_point, u)
    return symmetric(grad)
end

"""
Computes the divergence in a quadrature point for a vector valued function.

    function_vector_divergence{dim, T}(fe_fv::FEFaceValues{dim}, q_point::Int, u::Vector{Vec{dim, T}})

**Arguments:**

* `fe_fv`: the `FEFaceValues` object
* `q_point`: the quadrature point number
* `u`: the value of the function in the nodes

**Results:**

* `::Number`: the divergence of the function

**Details:**

The divergence of a vector valued functions in the quadrature point ``\\mathbf{x}_q)`` is computed as

``\\mathbf{\\nabla} \\cdot \\mathbf{u}(\\mathbf{x_q}) = \\sum\\limits_{i = 1}^n \\mathbf{\\nabla} N_i (\\mathbf{x_q}) \\cdot \\mathbf{u}_i``


where ``\\mathbf{u}_i`` are the nodal values of the function.

"""
@inline function function_vector_divergence{dim, T}(fe_fv::FEFaceValues{dim}, q_point::Int, u::Vector{Vec{dim, T}})
    n_base_funcs = n_basefunctions(get_functionspace(fe_fv))
    @assert length(u) == n_base_funcs
    dN = shape_gradient(fe_fv, q_point)
    diverg = zero(T)
    @inbounds for i in 1:n_base_funcs
        diverg += dN[i] ⋅ u[i]
    end
    return diverg
end
