type FEValues{T <: Real, QR <: QuadratureRule, FS <: FunctionSpace}
    J::Matrix{T}
    Jinv::Matrix{T}
    N::Vector{Vector{T}}
    dNdx::Vector{Matrix{T}}
    dNdξ::Vector{Matrix{T}}
    detJdV::Vector{T}
    quad_rule::QR
    function_space::FS
end

"""
Initializes an FEValues object from a function space and a quadrature rule.
"""
function FEValues{QR <: QuadratureRule, FS <: FunctionSpace}(T::Type, quad_rule::QR, func_space::FS)
        n_basefuncs = n_basefunctions(func_space)
        dim = n_dim(func_space)

        n_qpoints = length(points(quad_rule))

        N = [zeros(T, n_basefuncs) for i in 1:n_qpoints]
        dNdx = [zeros(T, dim, n_basefuncs) for i in 1:n_qpoints]
        dNdξ = [zeros(T, dim, n_basefuncs) for i in 1:n_qpoints]
        J = zeros(T, dim, dim)
        Jinv = similar(J)

        for (i, (ξ, w)) in enumerate(zip(quad_rule.points, quad_rule.weights))
            value!(func_space, N[i], ξ)
            derivative!(func_space, dNdξ[i], ξ)
        end

        FEValues{T, QR, FS}(J, Jinv, N, dNdx, dNdξ, zeros(T, n_qpoints), quad_rule, func_space)
end

function reinit!(fe_v::FEValues, x::Matrix)
    for (i, (ξ, w)) in enumerate(zip(fe_v.quad_rule.points, fe_v.quad_rule.weights))
        @into! fe_v.J = fe_v.dNdξ[i] * x'
        inv_spec!(fe_v.Jinv, fe_v.J)
        @into! fe_v.dNdx[i] = fe_v.Jinv * fe_v.dNdξ[i]
        fe_v.detJdV[i] = det_spec(fe_v.J) * w
    end
end

get_quadrule(fe_v::FEValues) = fe_v.quad_rule
get_functionspace(fe_v::FEValues) = fe_v.function_space

"""
Gets the product between the determinant of the Jacobian and the quadrature point weight for a given quadrature point.
"""

@inline detJdV(fe_v::FEValues, q_point::Int) = fe_v.detJdV[q_point]

"""
Gets the value of the shape function for a given quadrature point
"""
@inline shape_value(fe_v::FEValues, q_point::Int) = fe_v.N[q_point]

"""
Gets the value of the shape function at a given quadrature point and given base function
"""
@inline shape_value(fe_v::FEValues, q_point::Int, base_func::Int) = fe_v.N[q_point][base_func]

"""
Get the derivatives of the shape functions for a given quadrature point
"""
@inline shape_derivative(fe_v::FEValues, q_point::Int) = fe_v.dNdx[q_point]

"""
Get the derivatives of the shape functions for a given quadrature point and base function
"""
@inline shape_derivative(fe_v::FEValues, q_point::Int, base_func::Int) = fe_v.dNdx[q_point][:, base_func]

"""
Get the derivatives of the shape functions for a given quadrature point, base_function and component
"""
@inline shape_derivative(fe_v::FEValues, q_point::Int, base_func::Int, component::Int) = fe_v.dNdx[q_point][component, base_func]


