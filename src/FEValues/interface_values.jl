# Defines InterfaceScalarValues and InterfaceVectorValues and common methods
# InterfaceValues
struct InterfaceValues{FV<:FaceValues}
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
