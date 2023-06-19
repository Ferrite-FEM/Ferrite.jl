# Defines InterfaceValues and common methods
"""
    InterfaceValues(quad_rule::FaceQuadratureRule, func_interpol::Interpolation, [geom_interpol::Interpolation])

An `InterfaceValues` object facilitates the process of evaluating values, averages, jumps and gradients of shape functions
and nodal functions on the interfaces of finite elements.

**Arguments:**

* `quad_rule`: an instance of a [`FaceQuadratureRule`](@ref)
* `func_interpol`: an instance of an [`Interpolation`](@ref) used to interpolate the approximated function
* `geom_interpol`: an optional instance of an [`Interpolation`](@ref) which is used to interpolate the geometry.
  By default linear Lagrange interpolation is used.
 
**associated methods:**

* [`shape_value_average`](@ref)
* [`shape_value_jump`](@ref)
* [`shape_gradient_average`](@ref)
* [`shape_gradient_jump`](@ref)

**Common methods:**

* [`reinit!`](@ref)
* [`getnquadpoints`](@ref)
* [`getdetJdV`](@ref)

* [`shape_value`](@ref)
* [`shape_gradient`](@ref)
* [`shape_divergence`](@ref)
* [`shape_curl`](@ref)

* [`function_value`](@ref)
* [`function_gradient`](@ref)
* [`function_symmetric_gradient`](@ref)
* [`function_divergence`](@ref)
* [`function_curl`](@ref)
* [`spatial_coordinate`](@ref)
"""
InterfaceValues

struct InterfaceValues{FV<:FaceValues} <: AbstractValues
    face_values::FV
    face_values_neighbor::FV
end
function InterfaceValues(quad_rule::FaceQuadratureRule, func_interpol::Interpolation,
    geom_interpol::Interpolation = func_interpol)
    #@assert isDiscontinuous(func_interpol) "`InterfaceValues` is designed for discontinuous interpolations. a continuous interpolation is passed" TODO: add this when sparsity_pattern is merged
    face_values = FaceValues(quad_rule, func_interpol, geom_interpol)
    face_values_neighbor = copy(face_values)
    return InterfaceValues{FaceValues}(face_values,face_values_neighbor)
end
function reinit!(iv::InterfaceValues, coords::AbstractVector{Vec{dim,T}}, f::Int, ncoords::AbstractVector{Vec{dim,T}}, nf::Int) where {dim,T}
    reinit!(iv.face_values,coords,f)
    reinit!(iv.face_values_neighbor,ncoords,nf)
    @assert getnquadpoints(iv.face_values) == getnquadpoints(iv.face_values_neighbor)
end
# Maybe move this to common_values.jl?

"""
    shape_value_average(iv::InterfaceValues, qp::Int, base_function::Int)

Compute the average of the shape function value at the quadrature point from both sides of the interface.
"""
shape_value_average

"""
    shape_value_jump(iv::InterfaceValues, qp::Int, base_function::Int)

Compute the jump of the shape function value at the quadrature point over the interface.
The jump of scalar shape values is a vector.
"""
shape_value_jump

"""
    shape_gradient_average(iv::InterfaceValues, qp::Int, base_function::Int)

Compute the average of the shape function gradient at the quadrature point from both sides of the interface.
"""
shape_gradient_average

"""
    shape_gradient_jump(iv::InterfaceValues, qp::Int, base_function::Int)

Compute the jump of the shape function gradient at the quadrature point over the interface.
The jump of the gradient vector is a scalar.
"""
shape_gradient_jump

for (func,                      f_nbf,                  f_,                 multiplier,             operator) in (
    (:shape_value,              :getnbasefunctions,     :shape_value,       :(1),                   :*),
    (:shape_value_average,      :getnbasefunctions,     :shape_value,       :(0.5),                 :*),
    (:shape_value_jump,         :getnbasefunctions,     :shape_value,       :(getnormal(fv, qp)),   :*),
    (:shape_gradient,           :getnbasefunctions,     :shape_gradient,    :(1),                   :*),
    (:shape_gradient_average,   :getnbasefunctions,     :shape_gradient,    :(0.5),                 :*),
    (:shape_gradient_jump,      :getnbasefunctions,     :shape_gradient,    :(getnormal(fv, qp)),   :⋅),
    (:geometric_value,          :getngeobasefunctions,  :geometric_value,   :(1),                   :*),
)
    @eval begin
        function $(func)(iv::InterfaceValues, qp::Int, i::Int)
            nbf = $(f_nbf)(iv)
            if i <= nbf/2
                fv = iv.face_values
                return $(operator)($(multiplier), $(f_)(fv, qp, i))
            elseif i <= nbf
                fv = iv.face_values_neighbor
                return $(operator)($(multiplier), $(f_)(fv, qp, i - nbf/2))
            end
            error("Invalid base function $i. Interface has only $(nbf) base functions")
        end
    end
end
