# Defines InterfaceValues and common methods
"""
    InterfaceValues(quad_rule::FaceQuadratureRule, func_interpol::Interpolation, [geom_interpol::Interpolation])

An `InterfaceValues` object facilitates the process of evaluating values of shape functions, gradients of shape functions,
values of nodal functions, gradients and divergences of nodal functions etc. on the interfaces of finite elements.

**Arguments:**

* `quad_rule`: an instance of a [`FaceQuadratureRule`](@ref)
* `func_interpol`: an instance of an [`Interpolation`](@ref) used to interpolate the approximated function
* `geom_interpol`: an optional instance of an [`Interpolation`](@ref) which is used to interpolate the geometry.
  By default linear Lagrange interpolation is used.

**Common methods:**

* [`reinit!`](@ref)
* [`getnquadpoints`](@ref)
* [`getdetJdV`](@ref)

* [`shape_value`](@ref)
* [`shape_gradient`](@ref)
* [`shape_divergence`](@ref)
* [`shape_curl`](@ref)

"""
InterfaceValues

struct InterfaceValues{FV<:FaceValues} <: AbstractValues
    face_values::FV
    face_values_neighbor::FV
end
function InterfaceValues(quad_rule::FaceQuadratureRule, func_interpol::Interpolation,
    geom_interpol::Interpolation = func_interpol)
    face_values = FaceValues(quad_rule, func_interpol, geom_interpol)
    face_values_neighbor = copy(face_values)
    return InterfaceValues{FaceScalarValues}(face_values,face_values_neighbor)
end
function reinit!(iv::InterfaceValues, coords::AbstractVector{Vec{dim,T}}, f::Int, ncoords::AbstractVector{Vec{dim,T}}, nf::Int) where {dim,T}
    reinit!(iv.face_values,coords,f)
    reinit!(iv.face_values_neighbor,ncoords,nf)
    @assert getnquadpoints(iv.face_values) == getnquadpoints(iv.face_values_neighbor)
end

for (func,                      f_,                 multiplier,             operator) in (
    (:shape_value,              :shape_value,       :(1),                   :*),
    (:shape_value_average,      :shape_value,       :(0.5),                 :*),
    (:shape_value_jump,         :shape_value,       :(getnormal(fv, qp)),   :*),
    (:shape_gradient,           :shape_gradient,    :(1),                   :*),
    (:shape_gradient_average,   :shape_gradient,    :(0.5),                 :*),
    (:shape_gradient_jump,      :shape_gradient,    :(getnormal(fv, qp)),   :⋅),
)
    @eval begin
        function $(func)(iv::InterfaceValues, qp::Int, i::Int)
            nbf = getnbasefunctions(iv.face_values)
            if i <= nqp
                fv = iv.face_values
                return operator(multiplier, f_(fv, qp, i))
            else 
                fv = iv.face_values_neighbor
                return operator(multiplier, f_(fv, qp, i - nbf))
            end
            error("Invalid base function $i. Interface has only $(2*nbf) base functions")
        end
    end
end
