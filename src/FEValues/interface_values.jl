# Defines InterfaceValues and common methods
"""
    InterfaceValues(quad_rule::FaceQuadratureRule, ip_here::Interpolation, [geo_ip_here::Interpolation]; kwargs...)

An `InterfaceValues` object facilitates the process of evaluating values, averages, jumps
and gradients of shape functions and function on the interfaces between elements.

The first element of the interface is denoted "here" and the second element "there".

**Arguments:**
* `quad_rule`: an instance of a [`FaceQuadratureRule`](@ref) used for the "here" element.
  The quadrature points are translated to the "there" element during `reinit!`.
* `ip_here`: an instance of an [`Interpolation`](@ref) used to interpolate the approximated
  function on the "here" element.
* `geo_ip_here`: an optional instance of an [`Interpolation`](@ref) used to interpolate the
  geometry for the "here" element. Defaults to linear Lagrange interpolation.

**Keyword arguments:**
* `ip_there`: an optional instance of an [`Interpolation`](@ref) used to interpolate the
  approximated function on the "there" element. Defaults to `ip_here`.
* `geo_ip_there`: an optional instance of an [`Interpolation`](@ref) used to interpolate the
  geometry for the "there" element. Defaults to linear Lagrange interpolation.

**Associated methods:**
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
    here::FVA
    there::FVB
end
function InterfaceValues(quad_rule::FaceQuadratureRule, ip_here::Interpolation,
        geo_ip_here::Interpolation = default_geometric_interpolation(ip_here);
        ip_there::Interpolation = ip_here, geo_ip_there::Interpolation = default_geometric_interpolation(ip_there))
    here = FaceValues(quad_rule, ip_here, geo_ip_here)
    # TODO: Replace deepcopy
    there = FaceValues(deepcopy(quad_rule), ip_there, geo_ip_there)
    return InterfaceValues{typeof(here), typeof(there)}(here, there)
end

function getnbasefunctions(iv::InterfaceValues)
    return getnbasefunctions(iv.here) + getnbasefunctions(iv.there)
end
# function getngeobasefunctions(iv::InterfaceValues)
#     return getngeobasefunctions(iv.here) + getngeobasefunctions(iv.there)
# end

"""
    getnquadpoints(iv::InterfaceValues)

Return the number of quadrature points in `iv`s element A's [`FaceValues`](@ref)' quadrature for the current
(most recently [`reinit!`](@ref)ed) interface.
"""
getnquadpoints(iv::InterfaceValues) = getnquadpoints(iv.here.qr, iv.here.current_face[])

@propagate_inbounds function getdetJdV(iv::InterfaceValues, q_point::Int)
    return getdetJdV(iv.here, q_point)
end

"""
    reinit!(iv::InterfaceValues, face_here::Int, face_there::Int, coords_here::AbstractVector{Vec{dim, T}}, coords_there::AbstractVector{Vec{dim, T}}, cell_here::AbstractCell, cell_there::AbstractCell)

Update the [`FaceValues`](@ref) in the interface (A and B) using their corresponding cell coordinates and [`FaceIndex`](@ref). This involved recalculating the transformation matrix [`transform_interface_point`](@ref)
and mutating element B's quadrature points and its [`FaceValues`](@ref) `M, N, dMdξ, dNdξ`.
"""
function reinit!(iv::InterfaceValues, face_here::Int, face_there::Int, coords_here::AbstractVector{Vec{dim, T}}, coords_there::AbstractVector{Vec{dim, T}}, cell_here::AbstractCell, cell_there::AbstractCell) where {dim, T}

    # reinit! the here side as normal
    reinit!(iv.here, coords_here, face_here)
    dim == 1 && return reinit!(iv.there, coords_there, face_there)
    # Transform the quadrature points from the here side to the there side
    iv.there.current_face[] = face_there
    interface_transformation = InterfaceTransformation(cell_here, cell_there, face_here, face_there)
    quad_points_a = getpoints(iv.here.qr, face_here)
    quad_points_b = getpoints(iv.there.qr, face_there)
    transform_interface_points!(quad_points_b, quad_points_a, interface_transformation)
    @boundscheck checkface(iv.there, face_there)
    # TODO: This is the bottleneck, cache it?
    @assert length(quad_points_a) <= length(quad_points_b)
    # Re-evalutate shape functions in the transformed quadrature points
    for qp in 1:length(quad_points_a), i in 1:getnbasefunctions(iv.there)
        iv.there.dNdξ[i, qp, face_there], iv.there.N[i, qp, face_there] = shape_gradient_and_value(iv.there.func_interp, quad_points_b[qp], i)
    end
    for qp in 1:length(quad_points_a), i in 1:getngeobasefunctions(iv.there)
        iv.there.dMdξ[i, qp, face_there], iv.there.M[i, qp, face_there] = shape_gradient_and_value(iv.there.geo_interp, quad_points_b[qp], i)
    end
    # reinit! the "there" side
    reinit!(iv.there, coords_there, face_there)
    return iv
end

"""
    getnormal(iv::InterfaceValues, qp::Int; here::Bool = true)

Return the normal vector in the quadrature point `qp` on the interface. If `here = true`
(default) the normal to the "here" element is returned, otherwise the normal to the "there"
element.
"""
function getnormal(iv::InterfaceValues, qp::Int; here::Bool=true)
    return here ? iv.here.normals[qp] : iv.there.normals[qp]
end

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
    fv = here ? iv.here : iv.there
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
    fv = here ? iv.here : iv.there
    function_gradient(fv, q_point, u, dof_range)
end

# TODO: Deprecate this, nobody is using this in practice...
function function_gradient(iv::InterfaceValues, q_point::Int, u::AbstractVector{<:Vec}; here::Bool = true)
    fv = here ? iv.here : iv.there
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
    fv = here ? iv.here : iv.there
    function_symmetric_gradient(fv, q_point, u, dof_range)
end

# TODO: Deprecate this, nobody is using this in practice...
function function_symmetric_gradient(iv::InterfaceValues, q_point::Int, u::AbstractVector{<:Vec}; here::Bool = true)
    fv = here ? iv.here : iv.there
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
    fv = here ? iv.here : iv.there
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
            nbf_a = getnbasefunctions(iv.here)
            if i <= nbf_a
                fv = iv.here
                f_value = $(f_)(fv, qp, i)
                return $(multiplier) * f_value
            elseif i <= nbf
                fv = iv.there
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
            nbf_a = getnbasefunctions(iv.here)
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
            fv = iv.here
            result = 0.5 * f_value_here
            fv = iv.there
            result += 0.5 * f_value_there
            return result
        end
        # TODO: Deprecate this, nobody is using this in practice...
        function $(func)(iv::InterfaceValues, qp::Int, u_a::AbstractVector{<:Vec}, u_b::AbstractVector{<:Vec})
            f_value_here = $(f_)(iv, qp, u_a, here = true)
            f_value_there = $(f_)(iv, qp, u_b, here = false)
            fv = iv.here
            result = 0.5 * f_value_here
            fv = iv.there
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
    spatial_coordinate(iv.here, q_point, x)

"""
    get_transformation_matrix(interface_transformation::InterfaceTransformation)

Returns the transformation matrix corresponding to the interface information stored in `InterfaceTransformation`.
"""
get_transformation_matrix

function get_transformation_matrix(interface_transformation::InterfaceTransformation{RefShapeA}) where RefShapeA
    flipped = interface_transformation.flipped
    shift_index = interface_transformation.shift_index
    lowest_node_shift_index = interface_transformation.lowest_node_shift_index
    face_a = interface_transformation.face_a
    nfacenodes = length(reference_faces(RefShapeA)[face_a])
    θ = 2*shift_index/nfacenodes
    θpre = 2*lowest_node_shift_index/nfacenodes
    if nfacenodes == 3 # Triangle
        flipping = SMatrix{3,3}(1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0)

        translate_1 = SMatrix{3,3}(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, -sinpi(2/3)/3, -0.5, 1.0)
        stretch_1 = SMatrix{3,3}(sinpi(2/3), 0.5, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)

        translate_2 = SMatrix{3,3}(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, sinpi(2/3)/3, 0.5, 1.0)
        stretch_2 = SMatrix{3,3}(1/sinpi(2/3), -1/2/sinpi(2/3), 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)

        return flipped ? stretch_2 * translate_2 * rotation_matrix_pi(-θpre) * flipping * rotation_matrix_pi(θ + θpre) * translate_1 * stretch_1 : stretch_2 * translate_2 * rotation_matrix_pi(θ) * translate_1 * stretch_1
    elseif nfacenodes == 4 # Quadrilateral
        flipping = SMatrix{3,3}(0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0)
        return flipped ? rotation_matrix_pi(-θpre) * flipping * rotation_matrix_pi(θ + θpre) :  rotation_matrix_pi(θ)
    end

    error("transformation is not implemented")
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

function transform_interface_points!(dst::Vector{Vec{3, Float64}}, points::Vector{Vec{3, Float64}}, interface_transformation::InterfaceTransformation{RefShapeA, RefShapeB}) where {RefShapeA, RefShapeB}
    face_a = interface_transformation.face_a
    face_b = interface_transformation.face_b

    M = get_transformation_matrix(interface_transformation)
    for (idx, point) in pairs(points)
        face_point = element_to_face_transformation(point, RefShapeA, face_a)
        result = M * Vec(face_point[1],face_point[2], 1.0)
        dst[idx] = face_to_element_transformation(Vec(result[1],result[2]), RefShapeB, face_b)
    end
    return nothing
end

function transform_interface_points!(dst::Vector{Vec{2, Float64}}, points::Vector{Vec{2, Float64}}, interface_transformation::InterfaceTransformation{RefShapeA, RefShapeB}) where {RefShapeA, RefShapeB}
    face_a = interface_transformation.face_a
    face_b = interface_transformation.face_b
    flipped = interface_transformation.flipped

    for (idx, point) in pairs(points)
        face_point = element_to_face_transformation(point, RefShapeA, face_a)
        flipped && (face_point *= -1)
        dst[idx] = face_to_element_transformation(face_point, RefShapeB, face_b)
    end
    return nothing
end

function Base.show(io::IO, m::MIME"text/plain", iv::InterfaceValues)
    println(io, "InterfaceValues with")
    print(io, "{Here} ")
    show(io,m,iv.here)
    println(io)
    print(io, "{There} ")
    show(io,m,iv.there)
end
