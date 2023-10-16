# Defines InterfaceValues and common methods
"""
    InterfaceValues(facevalues_here, [facevalues_there])
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

# TODO: Should the kwargs be removed? If you need different things on the two
# sides, use two FaceValues?
function InterfaceValues(quad_rule_here::FaceQuadratureRule, ip_here::Interpolation,
        geo_ip_here::Interpolation = default_geometric_interpolation(ip_here);
        quad_rule_there::FaceQuadratureRule = deepcopy(quad_rule_here),
        ip_there::Interpolation = ip_here, geo_ip_there::Interpolation = default_geometric_interpolation(ip_there))
    quad_rule_here == quad_rule_there && getrefshape(geo_ip_here) != getrefshape(geo_ip_there) &&
        throw(ArgumentError("Constructing InterfaceValues with a single FaceQuadratureRule isn't valid for mixed grids, please consider passing quad_rule_there"))
    here = FaceValues(quad_rule_here, ip_here, geo_ip_here)
    there = FaceValues(quad_rule_there, ip_there, geo_ip_there)
    return InterfaceValues{typeof(here), typeof(there)}(here, there)
end

InterfaceValues(facevalues_here::FVA, facevalues_there::FVB = deepcopy(facevalues_here)) where {FVA <: FaceValues, FVB <: FaceValues} =
    InterfaceValues{FVA,FVB}(facevalues_here, facevalues_there)

function getnbasefunctions(iv::InterfaceValues)
    return getnbasefunctions(iv.here) + getnbasefunctions(iv.there)
end
# function getngeobasefunctions(iv::InterfaceValues)
#     return getngeobasefunctions(iv.here) + getngeobasefunctions(iv.there)
# end

"""
    getnquadpoints(iv::InterfaceValues)

Return the number of quadrature points on the interface for the current (most recently
[`reinit!`](@ref)ed) interface.
"""
getnquadpoints(iv::InterfaceValues) = getnquadpoints(iv.here)

@propagate_inbounds function getdetJdV(iv::InterfaceValues, q_point::Int)
    return getdetJdV(iv.here, q_point)
end

"""
    reinit!(
        iv::InterfaceValues,
        cell_here::AbstractCell, coords_here::AbstractVector{Vec{dim, T}}, face_here::Int,
        cell_there::AbstractCell, coords_there::AbstractVector{Vec{dim, T}}, face_there::Int
    )

Update the [`InterfaceValues`](@ref) for the interface between `cell_here` (with cell
coordinates `coords_here`) and `cell_there` (with cell coordinates `coords_there`).
`face_here` and `face_there` are the (local) face numbers for the respective cell.
"""
function reinit!(
        iv::InterfaceValues,
        cell_here::AbstractCell, coords_here::AbstractVector{Vec{dim, T}}, face_here::Int,
        cell_there::AbstractCell, coords_there::AbstractVector{Vec{dim, T}}, face_there::Int
    ) where {dim, T}

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
    getnormal(iv::InterfaceValues, qp::Int; here::Bool=true)

Return the normal vector in the quadrature point `qp` on the interface. If `here = true`
(default) the outward normal to the "here" element is returned, otherwise the outward normal
to the "there" element.
"""
function getnormal(iv::InterfaceValues, qp::Int; here::Bool=true)
    # TODO: Does it make sense to allow the kwarg here? You can juse negate the vector
    #       yourself since getnormal(iv, qp; here=false) == -getnormal(iv, qp; here=true).
    return getnormal(here ? iv.here : iv.there, qp)
end

"""
    function_value(iv::InterfaceValues, q_point::Int, u; here::Bool)
    function_value(iv::InterfaceValues, q_point::Int, u, dof_range_here, dof_range_there; here::Bool)

Compute the value of the function in quadrature point `q_point` on the "here" (`here=true`)
or "there" (`here=false`) side of the interface. `u_here` and `u_there` are the values of
the degrees of freedom for the respeciv element.

`u` is a vector of scalar values for the degrees of freedom.
This function can be used with a single `u` vector containing the dofs of both elements of the interface or
two vectors (`u_here` and `u_there`) which contain the dofs of each cell of the interface respectively.

`here` determines which element to use for calculating function value.
`true` uses the value on the first element's side of the interface, while `false` uses the value on the second element's side.

The value of a scalar valued function is computed as ``u(\\mathbf{x}) = \\sum\\limits_{i = 1}^n N_i (\\mathbf{x}) u_i``
where ``u_i`` are the value of ``u`` in the nodes. For a vector valued function the value is calculated as
``\\mathbf{u}(\\mathbf{x}) = \\sum\\limits_{i = 1}^n N_i (\\mathbf{x}) \\mathbf{u}_i`` where ``\\mathbf{u}_i`` are the
nodal values of ``\\mathbf{u}``.
"""
function_value(::InterfaceValues, ::Int, args...; kwargs...)


"""
    function_gradient(iv::InterfaceValues, q_point::Int, u; here::Bool)
    function_gradient(iv::InterfaceValues, q_point::Int, u, dof_range_here, dof_range_there; here::Bool)

Compute the gradient of the function in a quadrature point. `u` is a vector of scalar values for the degrees of freedom.
This function can be used with a single `u` vector containing the dofs of both elements of the interface or
two vectors (`u_here` and `u_there`) which contain the dofs of each cell of the interface respectively.

`here` determines which element to use for calculating function value.
`true` uses the value on the first element's side of the interface, while `false` uses the value on the second element's side.

The gradient of a scalar function or a vector valued function with use of `VectorValues` is computed as
``\\mathbf{\\nabla} u(\\mathbf{x}) = \\sum\\limits_{i = 1}^n \\mathbf{\\nabla} N_i (\\mathbf{x}) u_i`` or
``\\mathbf{\\nabla} u(\\mathbf{x}) = \\sum\\limits_{i = 1}^n \\mathbf{\\nabla} \\mathbf{N}_i (\\mathbf{x}) u_i`` respectively,
where ``u_i`` are the nodal values of the function.
For a vector valued function with use of `ScalarValues` the gradient is computed as
``\\mathbf{\\nabla} \\mathbf{u}(\\mathbf{x}) = \\sum\\limits_{i = 1}^n \\mathbf{u}_i \\otimes \\mathbf{\\nabla} N_i (\\mathbf{x})``
where ``\\mathbf{u}_i`` are the nodal values of ``\\mathbf{u}``.
"""
function_gradient(::InterfaceValues, ::Int, args...; kwargs...)

"""
    shape_value_average(iv::InterfaceValues, qp::Int, base_function::Int)

Compute the average of the shape function value at the quadrature point on interface.
"""
function shape_value_average end

"""
    shape_value_jump(iv::InterfaceValues, qp::Int, base_function::Int)

Compute the jump of the shape function value at the quadrature point over the interface.

This function uses the definition ``\\llbracket \\vec{v} \\rrbracket=\\vec{v}^\\text{here} -\\vec{v}^\\text{there}``. to obtain the form
``\\llbracket \\vec{v} \\rrbracket=\\vec{v}^\\text{there} ⋅ \\vec{n}^\\text{there} + \\vec{v}^\\text{here} ⋅ \\vec{n}^\\text{here}``
multiply by the outward facing normal to the first element's side of the interface (which is the default normal for [`getnormal`](@ref) with [`InterfaceValues`](@ref)).

"""
function shape_value_jump end

"""
    shape_gradient_average(iv::InterfaceValues, qp::Int, base_function::Int)

Compute the average of the shape function gradient at the quadrature point on the interface.
"""
function shape_gradient_average end

"""
    shape_gradient_jump(iv::InterfaceValues, qp::Int, base_function::Int)

Compute the jump of the shape function gradient at the quadrature point over the interface.

This function uses the definition ``\\llbracket \\vec{v} \\rrbracket=\\vec{v}^\\text{here} -\\vec{v}^\\text{there}``. to obtain the form
``\\llbracket \\vec{v} \\rrbracket=\\vec{v}^\\text{there} ⋅ \\vec{n}^\\text{there} + \\vec{v}^\\text{here} ⋅ \\vec{n}^\\text{here}``
multiply by the outward facing normal to the first element's side of the interface (which is the default normal for [`getnormal`](@ref) with [`InterfaceValues`](@ref)).
"""
function shape_gradient_jump end

for (func,                      f_,              f_type) in (
    (:shape_value,              :shape_value,    :shape_value_type),
    (:shape_gradient,           :shape_gradient, :shape_gradient_type),
)
    @eval begin
        function $(func)(iv::InterfaceValues, qp::Int, i::Int; here::Bool)
            nbf = getnbasefunctions(iv)
            nbf_a = getnbasefunctions(iv.here)
            if i <= nbf_a
                fv = iv.here
                here || return zero($(f_type)(fv))
                f_value = $(f_)(fv, qp, i)
                return f_value
            elseif i <= nbf
                fv = iv.there
                here && return zero($(f_type)(fv))
                f_value = $(f_)(fv, qp, i - nbf_a)
                return f_value
            end
            error("Invalid base function $i. Interface has only $(nbf) base functions")
        end
    end
end

for (func,                      f_,               is_avg) in (
    (:shape_value_average,      :shape_value,     true),
    (:shape_gradient_average,   :shape_gradient,  true),
    (:shape_value_jump,         :shape_value,     false),
    (:shape_gradient_jump,      :shape_gradient,  false),
)
    @eval begin
        function $(func)(iv::InterfaceValues, qp::Int, i::Int)
            f_here = $(f_)(iv, qp, i; here = true)
            f_there = $(f_)(iv, qp, i; here = false)
            return $(is_avg ? :((f_here + f_there) / 2) : :(f_here - f_there))
        end
    end
end

"""
    function_value_average(iv::InterfaceValues, q_point::Int, u)
    function_value_average(iv::InterfaceValues, q_point::Int, u, dof_range_here, dof_range_there)

Compute the average of the function value at the quadrature point on interface.
"""
function function_value_average end

"""
    function_value_jump(iv::InterfaceValues, q_point::Int, u)
    function_value_jump(iv::InterfaceValues, q_point::Int, u, dof_range_here, dof_range_there)

Compute the jump of the function value at the quadrature point over the interface.

This function uses the definition ``\\llbracket \\vec{v} \\rrbracket=\\vec{v}^\\text{here} -\\vec{v}^\\text{there}``. to obtain the form
``\\llbracket \\vec{v} \\rrbracket=\\vec{v}^\\text{there} ⋅ \\vec{n}^\\text{there} + \\vec{v}^\\text{here} ⋅ \\vec{n}^\\text{here}``
multiply by the outward facing normal to the first element's side of the interface (which is the default normal for [`getnormal`](@ref) with [`InterfaceValues`](@ref)).
"""
function function_value_jump end

"""
    function_gradient_average(iv::InterfaceValues, q_point::Int, u)
    function_gradient_average(iv::InterfaceValues, q_point::Int, u, dof_range_here, dof_range_there)

Compute the average of the function gradient at the quadrature point on the interface.
"""
function function_gradient_average end

"""
    function_gradient_jump(iv::InterfaceValues, q_point::Int, u)
    function_gradient_jump(iv::InterfaceValues, q_point::Int, u, dof_range_here, dof_range_there)

Compute the jump of the function gradient at the quadrature point over the interface.

This function uses the definition ``\\llbracket \\vec{v} \\rrbracket=\\vec{v}^\\text{here} -\\vec{v}^\\text{there}``. to obtain the form
``\\llbracket \\vec{v} \\rrbracket=\\vec{v}^\\text{there} ⋅ \\vec{n}^\\text{there} + \\vec{v}^\\text{here} ⋅ \\vec{n}^\\text{here}``
multiply by the outward facing normal to the first element's side of the interface (which is the default normal for [`getnormal`](@ref) with [`InterfaceValues`](@ref)).
"""
function function_gradient_jump end

for (func,                          ) in (
    (:function_value,               ),
    (:function_gradient,            ),
)
    @eval begin
        function $(func)(
                iv::InterfaceValues, q_point::Int, u::AbstractVector;
                here::Bool
            )
            if here
                dof_range_here = 1:getnbasefunctions(iv.here)
                return $(func)(iv.here, q_point, @view(u[dof_range_here]))
            else # there
                dof_range_there = (1:getnbasefunctions(iv.there)) .+ getnbasefunctions(iv.here)
                return $(func)(iv.there, q_point, @view(u[dof_range_there]))
            end
        end
        function $(func)(
                iv::InterfaceValues, q_point::Int,
                u::AbstractVector,
                dof_range_here::AbstractUnitRange{Int}, dof_range_there::AbstractUnitRange{Int};
                here::Bool
            )
            if here
                return $(func)(iv.here, q_point, u, dof_range_here)
            else # there
                return $(func)(iv.there, q_point, u, dof_range_there)
            end
        end
    end
end

for (func,                          f_,                     is_avg) in (
    (:function_value_average,       :function_value,        true ),
    (:function_gradient_average,    :function_gradient,     true ),
    (:function_value_jump,          :function_value,        false),
    (:function_gradient_jump,       :function_gradient,     false),
)
    @eval begin
        function $(func)(iv::InterfaceValues, qp::Int, u::AbstractVector)
            dof_range_here = 1:getnbasefunctions(iv.here)
            dof_range_there = (1:getnbasefunctions(iv.there)) .+ getnbasefunctions(iv.here)
            f_here = $(f_)(iv.here, qp, @view(u[dof_range_here]))
            f_there = $(f_)(iv.there, qp, @view(u[dof_range_there]))
            return $(is_avg ? :((f_here + f_there) / 2) : :(f_here - f_there))
        end
        function $(func)(
                iv::InterfaceValues, qp::Int,
                u::AbstractVector,
                dof_range_here::AbstractUnitRange{Int}, dof_range_there::AbstractUnitRange{Int},
            )
            f_here = $(f_)(iv.here, qp, u, dof_range_here)
            f_there = $(f_)(iv.there, qp, u, dof_range_there)
            return $(is_avg ? :((f_here + f_there) / 2) : :(f_here - f_there))
        end
    end
end

# TODO: Should this be [x_here, x_there], i.e. all coordinates?
function spatial_coordinate(iv::InterfaceValues, q_point::Int, x_here::AbstractVector{<:Vec})
    return spatial_coordinate(iv.here, q_point, x_here)
end


# Transformation of quadrature points

@doc raw"""
    InterfaceTransformation

Orientation information for 1D and 2D interfaces in 2D and 3D elements respectively.
This information is used to construct the transformation matrix to
transform the quadrature points from face_a to face_b achieving synced
spatial coordinates. Face B's orientation relative to Face A's can
possibly flipped (i.e. the vertices indices order is reversed)
and the vertices can be rotated against each other.
The reference orientation of face B is such that the first node
has the lowest vertex index. Thus, this structure also stores the
shift of the lowest vertex index which is used to reorient the face in
case of flipping ["transform_interface_point!"](@ref).
Take for example the faces
```
1           2
| \         | \
|  \        |  \
| A \       | B \
|    \      |    \
2-----3     3-----1
```
which are rotated against each other by 240° after tranfroming to an
equilateral triangle (shift index is 2) or the faces
```
2           2
| \         | \
|  \        |  \
| A \       | B \
|    \      |    \
3-----1     3-----1
```
which are flipped against each other, note that face B has its reference node shifted by 2 indices
so the face is tranformed into an equilateral triangle then rotated 120°, flipped about the x axis then
rotated -120° and tranformed back to the reference triangle.Any combination of these can happen.
"""
struct InterfaceTransformation{RefShapeA, RefShapeB}
    flipped::Bool
    shift_index::Int
    lowest_node_shift_index::Int
    face_a::Int
    face_b::Int
end

"""
    InterfaceTransformation(cell_a::AbstractCell, cell_b::AbstractCell, face_a::Int, face_b::Int)

Return the orientation info for the interface defined by face A and face B.
"""
function InterfaceTransformation(cell_a::AbstractCell, cell_b::AbstractCell, face_a::Int, face_b::Int)
    getdim(cell_a) == 1 && return error("1D elements don't use transformations for interfaces.")

    nodes_a = faces(cell_a)[face_a]
    nodes_b = faces(cell_b)[face_b]

    min_idx_a = argmin(nodes_a)
    min_idx_b = argmin(nodes_b)

    shift_index = min_idx_b - min_idx_a
    flipped = getdim(cell_a) == 2 ? shift_index != 0 : nodes_a[min_idx_a != 1 ? min_idx_a - 1 : end] != nodes_b[min_idx_b != 1 ? min_idx_b - 1 : end]

    return InterfaceTransformation{getrefshape(cell_a), getrefshape(cell_b)}(flipped, shift_index, 1 - min_idx_b, face_a, face_b)
end

# This looks out of place, move it to Tensors.jl or use the one defined there with higher error? *Using sinpi and cospi makes tetrahedon custom quadrature points interface values test pass
"""
    rotation_matrix_pi(x::Float64)

Construct thr 1D 3x3 rotation matrix for θ = xπ more accurately using sinpi and cospi, especially for large x
"""
function rotation_matrix_pi(θ::Float64)
    return SMatrix{3,3}(cospi(θ), sinpi(θ), 0.0, -sinpi(θ), cospi(θ), 0.0, 0.0, 0.0, 1.0)
end

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

    throw(ArgumentError("transformation is not implemented"))
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
