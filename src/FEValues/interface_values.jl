# Defines InterfaceValues and common methods
"""
    InterfaceValues(grid::AbstractGrid, quad_rule::FaceQuadratureRule, func_interpol_a::Interpolation, [geom_interpol_a::Interpolation], [func_interpol_b::Interpolation], [geom_interpol_b::Interpolation])

An `InterfaceValues` object facilitates the process of evaluating values, averages, jumps and gradients of shape functions
and function on the interfaces of finite elements.

**Arguments:**

* `quad_rule_a`: an instance of a [`FaceQuadratureRule`](@ref) for element A.
* `quad_rule_b`: an instance of a [`FaceQuadratureRule`](@ref) for element B.
* `func_interpol_a`: an instance of an [`Interpolation`](@ref) used to interpolate the approximated function for element A.
* `func_interpol_b`: an instance of an [`Interpolation`](@ref) used to interpolate the approximated function for element B.
  It defaults to `func_interpol_a`.
* `geom_interpol_a`: an optional instance of an [`Interpolation`](@ref) which is used to interpolate the geometry for element A.
  It defaults to `func_interpol_a`.
* `geom_interpol_b`: an optional instance of an [`Interpolation`](@ref) which is used to interpolate the geometry for element B.
  It defaults to `func_interpol_b`.
 
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

struct InterfaceValues{FVA, FVB} <: AbstractValues
    face_values_a::FVA
    face_values_b::FVB
end
function InterfaceValues(quad_rule_a::FaceQuadratureRule, func_interpol_a::Interpolation,
    geom_interpol_a::Interpolation = func_interpol_a; quad_rule_b::FaceQuadratureRule = deepcopy(quad_rule_a),
    func_interpol_b::Interpolation = func_interpol_a, geom_interpol_b::Interpolation = func_interpol_b)
    face_values_a = FaceValues(quad_rule_a, func_interpol_a, geom_interpol_a)
    face_values_b = FaceValues(quad_rule_b, func_interpol_b, geom_interpol_b)
    return InterfaceValues{typeof(face_values_a), typeof(face_values_b)}(face_values_a, face_values_b)
end

getnbasefunctions(iv::InterfaceValues) = getnbasefunctions(iv.face_values_a) + getnbasefunctions(iv.face_values_b)
getngeobasefunctions(iv::InterfaceValues) = getngeobasefunctions(iv.face_values_a) + getngeobasefunctions(iv.face_values_b)

"""
    getnquadpoints(iv::InterfaceValues)

Return the number of quadrature points in `iv`s element A's [`FaceValues`](@ref)' quadrature for the current
(most recently [`reinit!`](@ref)ed) interface.
"""
getnquadpoints(iv::InterfaceValues) = getnquadpoints(iv.face_values_a.qr, iv.face_values_a.current_face[])

@propagate_inbounds getdetJdV(iv::InterfaceValues, q_point::Int) = getdetJdV(iv.face_values_a, q_point)

"""
    reinit!(iv::InterfaceValues, face_a::Int, face_b::Int, cell_a_coords::AbstractVector{Vec{dim, T}}, cell_b_coords::AbstractVector{Vec{dim, T}}, cell_a::AbstractCell, cell_b::AbstractCell)

Update the [`FaceValues`](@ref) in the interface (A and B) using their corresponding cell coordinates and [`FaceIndex`](@ref). This involved recalculating the transformation matrix [`transform_interface_point`](@ref)
and mutating element B's quadrature points and its [`FaceValues`](@ref) `M, N, dMdξ, dNdξ`.
"""
function reinit!(iv::InterfaceValues, face_a::Int, face_b::Int, cell_a_coords::AbstractVector{Vec{dim, T}}, cell_b_coords::AbstractVector{Vec{dim, T}}, cell_a::AbstractCell, cell_b::AbstractCell) where {dim, T}
    reinit!(iv.face_values_a, cell_a_coords, face_a)
    dim == 1 && return reinit!(iv.face_values_b, cell_b_coords, face_b) 
    iv.face_values_b.current_face[] = face_b
    interface_transformation = InterfaceTransformation(cell_a, cell_b, face_a, face_b)
    quad_points_a = getpoints(iv.face_values_a.qr, face_a)
    quad_points_b = getpoints(iv.face_values_b.qr, face_b)
    transform_interface_points!(quad_points_b, quad_points_a, interface_transformation)
    @boundscheck checkface(iv.face_values_b, face_b)
    # This is the bottleneck, cache it?
    for idx in eachindex(IndexCartesian(), @view iv.face_values_b.N[:, :, face_b])
        @boundscheck idx[2] > length(quad_points_b) && continue
        iv.face_values_b.dNdξ[idx, face_b], iv.face_values_b.N[idx, face_b] = shape_gradient_and_value(iv.face_values_b.func_interp, quad_points_b[idx[2]], idx[1])
    end
    for idx in eachindex(IndexCartesian(), @view iv.face_values_b.M[:, :, face_b])
        @boundscheck idx[2] > length(quad_points_b) && continue
        iv.face_values_b.dMdξ[idx, face_b], iv.face_values_b.M[idx, face_b] = shape_gradient_and_value(iv.face_values_b.geo_interp, quad_points_b[idx[2]], idx[1])
    end  
    reinit!(iv.face_values_b, cell_b_coords, face_b)
end

"""
    getnormal(iv::InterfaceValues, qp::Int, here::Bool = true)

Return the normal at the quadrature point `qp` on the interface. 

For `InterfaceValues`, `use_elemet_a` determines which element to use for calculating divergence of the function.
`true` uses the element A's face nomal vector, while `false` uses element B's, which is the default.
"""
getnormal(iv::InterfaceValues, qp::Int, here::Bool = false) = here ? iv.face_values_a.normals[qp] : iv.face_values_b.normals[qp]

"""
    function_value(iv::InterfaceValues, q_point::Int, u::AbstractVector; here::Bool = true)

Compute the value of the function in a quadrature point. `u` is a vector with values
for the degrees of freedom. For a scalar valued function, `u` contains scalars.
For a vector valued function, `u` can be a vector of scalars (for use of `VectorValues`)
or `u` can be a vector of `Vec`s (for use with ScalarValues).

`here` determines which element to use for calculating function value.
`true` uses the element A's nodal values, which is the default, while `false` uses element B's.

The value of a scalar valued function is computed as ``u(\\mathbf{x}) = \\sum\\limits_{i = 1}^n N_i (\\mathbf{x}) u_i``
where ``u_i`` are the value of ``u`` in the nodes. For a vector valued function the value is calculated as
``\\mathbf{u}(\\mathbf{x}) = \\sum\\limits_{i = 1}^n N_i (\\mathbf{x}) \\mathbf{u}_i`` where ``\\mathbf{u}_i`` are the
nodal values of ``\\mathbf{u}``.
"""
function function_value(iv::InterfaceValues, q_point::Int, u::AbstractVector, dof_range = eachindex(u); here::Bool = true)
    fv = here ? iv.face_values_a : iv.face_values_b
    function_value(fv, q_point, u, dof_range)
end

shape_value_type(::InterfaceValues{<:FaceValues{<:Any, N_t}}) where N_t = N_t

"""
    function_gradient(iv::InterfaceValues, q_point::Int, u::AbstractVector; here::Bool = true)

Compute the gradient of the function in a quadrature point. `u` is a vector with values
for the degrees of freedom. For a scalar valued function, `u` contains scalars.
For a vector valued function, `u` can be a vector of scalars (for use of `VectorValues`)
or `u` can be a vector of `Vec`s (for use with ScalarValues).

`here` determines which element to use for calculating function gradient. 
`true` uses the element A's nodal values for calculating the gradient, which is the default, while `false` uses element B's.

The gradient of a scalar function or a vector valued function with use of `VectorValues` is computed as
``\\mathbf{\\nabla} u(\\mathbf{x}) = \\sum\\limits_{i = 1}^n \\mathbf{\\nabla} N_i (\\mathbf{x}) u_i`` or
``\\mathbf{\\nabla} u(\\mathbf{x}) = \\sum\\limits_{i = 1}^n \\mathbf{\\nabla} \\mathbf{N}_i (\\mathbf{x}) u_i`` respectively,
where ``u_i`` are the nodal values of the function.
For a vector valued function with use of `ScalarValues` the gradient is computed as
``\\mathbf{\\nabla} \\mathbf{u}(\\mathbf{x}) = \\sum\\limits_{i = 1}^n \\mathbf{u}_i \\otimes \\mathbf{\\nabla} N_i (\\mathbf{x})``
where ``\\mathbf{u}_i`` are the nodal values of ``\\mathbf{u}``.
"""
function function_gradient(iv::InterfaceValues, q_point::Int, u::AbstractVector, dof_range = eachindex(u); here::Bool = true)
    fv = here ? iv.face_values_a : iv.face_values_b
    function_gradient(fv, q_point, u, dof_range)
end

# TODO: Deprecate this, nobody is using this in practice...
function function_gradient(iv::InterfaceValues, q_point::Int, u::AbstractVector{<:Vec}; here::Bool = true)
    fv = here ? iv.face_values_a : iv.face_values_b
    function_gradient(fv, q_point, u)
end

"""
    function_symmetric_gradient(iv::InterfaceValues, q_point::Int, u::AbstractVector; here::Bool = true)

Compute the symmetric gradient of the function, see [`function_gradient`](@ref).
Return a `SymmetricTensor`.

For `InterfaceValues`, `here` determines which element to use for calculating function gradient.
`true` uses the element A's nodal values for calculating the gradient, which is the default, while `false` uses element B's.

The symmetric gradient of a scalar function is computed as
``\\left[ \\mathbf{\\nabla}  \\mathbf{u}(\\mathbf{x_q}) \\right]^\\text{sym} =  \\sum\\limits_{i = 1}^n  \\frac{1}{2} \\left[ \\mathbf{\\nabla} N_i (\\mathbf{x}_q) \\otimes \\mathbf{u}_i + \\mathbf{u}_i  \\otimes  \\mathbf{\\nabla} N_i (\\mathbf{x}_q) \\right]``
where ``\\mathbf{u}_i`` are the nodal values of the function.
"""
function function_symmetric_gradient(iv::InterfaceValues, q_point::Int, u::AbstractVector, dof_range = eachindex(u); here::Bool = true)
    fv = here ? iv.face_values_a : iv.face_values_b
    function_symmetric_gradient(fv, q_point, u, dof_range)
end

# TODO: Deprecate this, nobody is using this in practice...
function function_symmetric_gradient(iv::InterfaceValues, q_point::Int, u::AbstractVector{<:Vec}; here::Bool = true)
    fv = here ? iv.face_values_a : iv.face_values_b
    function_symmetric_gradient(fv, q_point, u)
end

"""
    function_divergence(iv::InterfaceValues, q_point::Int, u::AbstractVector; here::Bool = true)

Compute the divergence of the vector valued function in a quadrature point.

`here` determines which element to use for calculating divergence of the function.
`true` uses the element A's nodal values for calculating the divergence from gradient, which is the default, while `false` uses element B's.

The divergence of a vector valued functions in the quadrature point ``\\mathbf{x}_q)`` is computed as
``\\mathbf{\\nabla} \\cdot \\mathbf{u}(\\mathbf{x_q}) = \\sum\\limits_{i = 1}^n \\mathbf{\\nabla} N_i (\\mathbf{x_q}) \\cdot \\mathbf{u}_i``
where ``\\mathbf{u}_i`` are the nodal values of the function.
"""
function_divergence(iv::InterfaceValues, q_point::Int, u::AbstractVector, dof_range = eachindex(u); here::Bool = true) =
    divergence_from_gradient(function_gradient(iv, q_point, u, dof_range; here = here))

# TODO: Deprecate this, nobody is using this in practice...
function function_divergence(iv::InterfaceValues, q_point::Int, u::AbstractVector{<:Vec}; here::Bool = true)
    fv = here ? iv.face_values_a : iv.face_values_b
    function_divergence(fv, q_point, u)
end

"""
    function_curl(iv::InterfaceValues, q_point::Int, u::AbstractVector; here::Bool = true)

Compute the curl of the vector valued function in a quadrature point.

`here` determines which element to use for calculating curl of the function.
`true` uses the element A's nodal values for calculating the curl from gradient, which is the default, while `false` uses element B's.

The curl of a vector valued functions in the quadrature point ``\\mathbf{x}_q)`` is computed as
``\\mathbf{\\nabla} \\times \\mathbf{u}(\\mathbf{x_q}) = \\sum\\limits_{i = 1}^n \\mathbf{\\nabla} N_i \\times (\\mathbf{x_q}) \\cdot \\mathbf{u}_i``
where ``\\mathbf{u}_i`` are the nodal values of the function.
"""
function_curl(iv::InterfaceValues, q_point::Int, u::AbstractVector, dof_range = eachindex(u); here::Bool = true) =
    curl_from_gradient(function_gradient(iv, q_point, u, dof_range; here))

# TODO: Deprecate this, nobody is using this in practice...
function_curl(iv::InterfaceValues, q_point::Int, u::AbstractVector{<:Vec}; here::Bool = true) =
    curl_from_gradient(function_gradient(iv, q_point, u; here))

"""
    shape_value_average(iv::InterfaceValues, qp::Int, base_function::Int)

Compute the average of the shape function value at the quadrature point on interface.
"""
shape_value_average

"""
    shape_value_jump(iv::InterfaceValues, qp::Int, base_function::Int)

Compute the jump of the shape function value at the quadrature point over the interface.

`normal_dotted::Bool` determines whether to use the definition ``\\llbracket v \\rrbracket=v^- -v^+`` if it's `false`, or
 the definition  ``\\llbracket v \\rrbracket=v^- ⋅ \\vec{n}^- + v^+ ⋅ \\vec{n}^+`` if it's `true`, which is the default.

!!! note
    If `normal_dotted == true` then the jump of scalar shape values is a vector.
"""
shape_value_jump

"""
    shape_gradient_average(iv::InterfaceValues, qp::Int, base_function::Int)

Compute the average of the shape function gradient at the quadrature point on the interface.
"""
shape_gradient_average

"""
    shape_gradient_jump(iv::InterfaceValues, qp::Int, base_function::Int)

Compute the jump of the shape function gradient at the quadrature point over the interface.

This function uses the definition ``\\llbracket \\vec{v} \\rrbracket=\\vec{v}^- -\\vec{v}^+``. to obtain the form 
``\\llbracket \\vec{v} \\rrbracket=\\vec{v}^- ⋅ \\vec{n}^- + \\vec{v}^+ ⋅ \\vec{n}^+``one can simple multiply by the normal of face A (which is the default normal for [`getnormal`](@ref) with [`InterfaceValues`](@ref)).
"""
shape_gradient_jump

for (func,                      f_,                 multiplier, ) in (
    (:shape_value,              :shape_value,       :(1),       ),
    (:shape_value_average,      :shape_value,       :(0.5),     ),
    (:shape_gradient,           :shape_gradient,    :(1),       ),
    (:shape_gradient_average,   :shape_gradient,    :(0.5),     ),
)
    @eval begin
        function $(func)(iv::InterfaceValues, qp::Int, i::Int)
            nbf = getnbasefunctions(iv)
            nbf_a = getnbasefunctions(iv.face_values_a)
            if i <= nbf_a
                fv = iv.face_values_a
                f_value = $(f_)(fv, qp, i)
                return $(multiplier) * f_value
            elseif i <= nbf
                fv = iv.face_values_b
                f_value = $(f_)(fv, qp, i - nbf_a)
                return $(multiplier) * f_value
            end
            error("Invalid base function $i. Interface has only $(nbf) base functions")
        end
    end
end

for (func,                      f_,                 ) in (
    (:shape_value_jump,         :shape_value,       ),
    (:shape_gradient_jump,      :shape_gradient,    ),
)
    @eval begin
        function $(func)(iv::InterfaceValues, qp::Int, i::Int)
            f_value = $(f_)(iv, qp, i)
            nbf_a = getnbasefunctions(iv.face_values_a)
            return i <= nbf_a ? -f_value : f_value
        end
    end
end

"""
    function_value_average(iv::InterfaceValues, qp::Int, u_a::AbstractVector, u_b::AbstractVector, dof_range_a = eachindex(u_a), dof_range_b = eachindex(u_b))

Compute the average of the function value at the quadrature point on interface.
"""
function_value_average

"""
    function_value_jump(iv::InterfaceValues, qp::Int, u_a::AbstractVector, u_b::AbstractVector, dof_range_a = eachindex(u_a), dof_range_b = eachindex(u_b))

Compute the jump of the function value at the quadrature point over the interface.

This function uses the definition ``\\llbracket \\vec{v} \\rrbracket=\\vec{v}^- -\\vec{v}^+``. to obtain the form 
``\\llbracket \\vec{v} \\rrbracket=\\vec{v}^- ⋅ \\vec{n}^- + \\vec{v}^+ ⋅ \\vec{n}^+``one can simple multiply by the normal of face A (which is the default normal for [`getnormal`](@ref) with [`InterfaceValues`](@ref)).
"""
function_value_jump

"""
    function_gradient_average(iv::InterfaceValues, qp::Int, u_a::AbstractVector, u_b::AbstractVector, dof_range_a = eachindex(u_a), dof_range_b = eachindex(u_b))

Compute the average of the function gradient at the quadrature point on the interface.
"""
function_gradient_average

"""
    function_gradient_jump(iv::InterfaceValues, qp::Int, u_a::AbstractVector, u_b::AbstractVector, dof_range_a = eachindex(u_a), dof_range_b = eachindex(u_b))

Compute the jump of the function gradient at the quadrature point over the interface.

This function uses the definition ``\\llbracket \\vec{v} \\rrbracket=\\vec{v}^- -\\vec{v}^+``. to obtain the form 
``\\llbracket \\vec{v} \\rrbracket=\\vec{v}^- ⋅ \\vec{n}^- + \\vec{v}^+ ⋅ \\vec{n}^+``one can simple multiply by the normal of face A (which is the default normal for [`getnormal`](@ref) with [`InterfaceValues`](@ref)).
"""
function_gradient_jump

for (func,                          f_,                 ) in (
    (:function_value_average,       :function_value,    ),
    (:function_gradient_average,    :function_gradient, ),
)
    @eval begin
        function $(func)(iv::InterfaceValues, qp::Int, u_a::AbstractVector, u_b::AbstractVector, dof_range_a = eachindex(u_a), dof_range_b = eachindex(u_b))
            f_value_here = $(f_)(iv, qp, u_a, dof_range_a, here = true)
            f_value_there = $(f_)(iv, qp, u_b, dof_range_b, here = false)
            fv = iv.face_values_a
            result = 0.5 * f_value_here 
            fv = iv.face_values_b
            result += 0.5 * f_value_there
            return result
        end
        # TODO: Deprecate this, nobody is using this in practice...
        function $(func)(iv::InterfaceValues, qp::Int, u_a::AbstractVector{<:Vec}, u_b::AbstractVector{<:Vec})
            f_value_here = $(f_)(iv, qp, u_a, here = true)
            f_value_there = $(f_)(iv, qp, u_b, here = false)
            fv = iv.face_values_a
            result = 0.5 * f_value_here
            fv = iv.face_values_b
            result += 0.5 * f_value_there
            return result
        end
    end
end

for (func,                          f_,                 ) in (
    (:function_value_jump,          :function_value,    ),
    (:function_gradient_jump,       :function_gradient, ),
)
    @eval begin
        function $(func)(iv::InterfaceValues, qp::Int, u_a::AbstractVector, u_b::AbstractVector, dof_range_a = eachindex(u_a), dof_range_b = eachindex(u_b))
            f_value_here = $(f_)(iv, qp, u_a, dof_range_a, here = true)
            f_value_there = $(f_)(iv, qp, u_b, dof_range_b, here = false)
            return f_value_there - f_value_here 
        end
        # TODO: Deprecate this, nobody is using this in practice...
        function $(func)(iv::InterfaceValues, qp::Int, u_a::AbstractVector{<:Vec}, u_b::AbstractVector{<:Vec})
            f_value_here = $(f_)(iv, qp, u_a, here = true)
            f_value_there = $(f_)(iv, qp, u_b, here = false)
            return f_value_there - f_value_here 
        end
    end
end

spatial_coordinate(iv::InterfaceValues, q_point::Int, x::AbstractVector{Vec{dim,T}}) where {dim,T} =
    spatial_coordinate(iv.face_values_a, q_point, x)

"""
    get_transformation_matrix(interface_transformation::InterfaceTransformation)

Returns the transformation matrix corresponding to the interface information stored in `InterfaceTransformation`.
"""
get_transformation_matrix

function get_transformation_matrix(interface_transformation::InterfaceTransformation{4})
    flipped = interface_transformation.flipped
    shift_index = interface_transformation.shift_index
    lowest_node_shift_index = interface_transformation.lowest_node_shift_index
    θ = shift_index/2
    θpre = lowest_node_shift_index/2
    flipping = SMatrix{3,3}(0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    M = flipped ? rotation_matrix_pi(-θpre) * flipping * rotation_matrix_pi(θ + θpre) :  rotation_matrix_pi(θ) 
    return M
end

function get_transformation_matrix(interface_transformation::InterfaceTransformation{3})
    flipped = interface_transformation.flipped
    shift_index = interface_transformation.shift_index
    lowest_node_shift_index = interface_transformation.lowest_node_shift_index
    θ = 2/3 * shift_index
    θpre = 2/3 * lowest_node_shift_index
    
    flipping = SMatrix{3,3}(1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0)

    translate_1 = SMatrix{3,3}(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, -sinpi(2/3)/3, -0.5, 1.0) 
    stretch_1 = SMatrix{3,3}(sinpi(2/3), 0.5, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0) 

    translate_2 = SMatrix{3,3}(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, sinpi(2/3)/3, 0.5, 1.0) 
    stretch_2 = SMatrix{3,3}(1/sinpi(2/3), -1/2/sinpi(2/3), 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0) 

    M = flipped ? stretch_2 * translate_2 * rotation_matrix_pi(-θpre) * flipping * rotation_matrix_pi(θ + θpre) * translate_1 * stretch_1 : stretch_2 * translate_2 * rotation_matrix_pi(θ) * translate_1 * stretch_1
    return M
end

@doc raw"""
    transform_interface_points!(dst::Vector{Vec{3, Float64}}, points::Vector{Vec{3, Float64}}, interface_transformation::InterfaceTransformation)

Transform the points from face A to face B using the orientation information of the interface and store it in the vecotr dst.
For 3D, the faces are transformed to regular polygons such that the rotation angle is the shift in reference node index × 2π ÷ number of edges in face.
If the face is flipped then the flipping is about the axis that perserves the position of the first node (which is the reference node after being rotated to be in the first position,
it's rotated back in the opposite direction after flipping).
Take for example the interface
```
        2           3
        | \         | \
        |  \        |  \
y       | A \       | B \ 
↑       |    \      |    \
→  x    1-----3     1-----2  
```
Transforming A to a equilateral triangle and translating it such that {0,0} is equidistant to all nodes
```
        3
        +
       / \         
      /   \
     /  x  \                          
    /   ↑   \                        
   /  ←      \                          
  /  y        \                            
2+-------------+1                     
```
Rotating it -270° (or 120°) such that the reference node (the node with smallest index) is at index 1
```
        1
        +
       / \         
      /   \
     /  x  \                          
    /   ↑   \                        
   /  ←      \                          
  /  y        \                            
3+-------------+2                     
```
Flipping about the x axis (such that the position of the reference node doesn't change) and rotating 270° (or -120°)
```
        2
        +
       / \         
      /   \
     /  x  \                          
    /   ↑   \                        
   /  ←      \                          
  /  y        \                            
3+-------------+1                     
```
Transforming back to triangle B
```
       3           
       | \
       |  \
y      |   \ 
↑      |    \
→ x    1-----2  
```
"""
transform_interface_points!

function transform_interface_points!(dst::Vector{Vec{3, Float64}}, points::Vector{Vec{3, Float64}}, interface_transformation::InterfaceTransformation)
    cell_a_refshape = interface_transformation.cell_a_refshape
    cell_b_refshape = interface_transformation.cell_b_refshape
    face_a = interface_transformation.face_a_index
    face_b = interface_transformation.face_b_index
    
    M = get_transformation_matrix(interface_transformation)
    for (idx, point) in pairs(points)
        point = element_to_face_transformation(point, cell_a_refshape, face_a)
        result = M * Vec(point[1],point[2], 1.0)
        dst[idx] = face_to_element_transformation(Vec(result[1],result[2]), cell_b_refshape, face_b)
    end
    return nothing
end

function transform_interface_points!(dst::Vector{Vec{2, Float64}}, points::Vector{Vec{2, Float64}}, interface_transformation::InterfaceTransformation{2})
    flipped = interface_transformation.flipped
    cell_a_refshape = interface_transformation.cell_a_refshape
    cell_b_refshape = interface_transformation.cell_b_refshape
    face_a = interface_transformation.face_a_index
    face_b = interface_transformation.face_b_index
    
    for (idx, point) in pairs(points)
        point = element_to_face_transformation(point, cell_a_refshape, face_a)
        flipped && (point *= -1) 
        dst[idx] = face_to_element_transformation(point, cell_b_refshape, face_b)
    end
    return nothing
end
