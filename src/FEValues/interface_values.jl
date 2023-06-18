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
shape_value(iv::InterfaceValues, qp::Int, base_func::Int) = f_value_(iv, qp, base_func, shape_value) 
shape_value_jump(iv::InterfaceValues, qp::Int, base_func::Int) = f_jump_(iv, qp, base_func, shape_value) 
shape_value_average(iv::InterfaceValues, qp::Int, base_func::Int) = f_average(iv, qp, base_func, shape_value) 

shape_gradient(iv::InterfaceValues, qp::Int, base_func::Int) = f_value_(iv, qp, base_func, shape_gradient) 
shape_gradient_jump(iv::InterfaceValues, qp::Int, base_func::Int) = f_jump_(iv, qp, base_func, shape_gradient) 
shape_gradient_average(iv::InterfaceValues, qp::Int, base_func::Int) = f_average(iv, qp, base_func, shape_gradient) 

function f_value_(iv::InterfaceValues, qp::Int, i::Int, f_::Function)
    nbf = getnbasefunctions(iv.face_values)
    if i <= nqp
        return f_(iv.face_values, qp, i)
    else 
        return f_(iv.face_values_neighbor, qp, i - nbf)
    end
    error("Invalid base function $i. Interface has only $(2*nbf) base functions")
end
function f_jump_(iv::InterfaceValues, qp::Int, i::Int, f_::Function)
    nbf = getnbasefunctions(iv.face_values)
    if i <= nqp
        jump_mag = f_(iv.face_values, qp, i)
        return jump_mag isa Number ? jump_mag * getnormal(iv.face_values, qp) :  jump_mag ⋅ getnormal(iv.face_values, qp)
    else 
        jump_mag = f_(iv.face_values_neighbor, qp, i - nbf)
        return jump_mag isa Number ? jump_mag * getnormal(iv.face_values_neighbor, qp) :  jump_mag ⋅ getnormal(iv.face_values_neighbor, qp)
    end
    error("Invalid base function $i. Interface has only $(2*nbf) base functions")
end
function f_average_(iv::InterfaceValues, qp::Int, base_func::Int, f_::Function)
    nbf = getnbasefunctions(iv.face_values)
    if i <= nqp
        return 0.5 * f_(iv.face_values, qp, i)
    else 
        return 0.5 * f_(iv.face_values_neighbor, qp, i - nbf)
    end
    error("Invalid base function $i. Interface has only $(2*nbf) base functions")
end