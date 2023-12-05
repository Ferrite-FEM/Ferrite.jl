# Defines InterfaceValues and common methods
"""
    InterfaceValues

An `InterfaceValues` object facilitates the process of evaluating values, averages, jumps
and gradients of shape functions and function on the interfaces between elements.

The first element of the interface is denoted "here" and the second element "there".

**Constructors**
* `InterfaceValues(qr::FaceQuadratureRule, ip::Interpolation)`: same quadrature rule and
  interpolation on both sides, default linear Lagrange geometric interpolation.
* `InterfaceValues(qr::FaceQuadratureRule, ip::Interpolation, ip_geo::Interpolation)`: same
  as above but with given geometric interpolation.
* `InterfaceValues(qr_here::FaceQuadratureRule, ip_here::Interpolation, qr_there::FaceQuadratureRule, ip_there::Interpolation)`:
  different quadrature rule and interpolation on the two sides, default linear Lagrange
  geometric interpolation.
* `InterfaceValues(qr_here::FaceQuadratureRule, ip_here::Interpolation, ip_geo_here::Interpolation, qr_there::FaceQuadratureRule, ip_there::Interpolation, ip_geo_there::Interpolation)`:
  same as above but with given geometric interpolation.
* `InterfaceValues(fv::FaceValues)`: quadrature rule and interpolations from face values
  (same on both sides).
* `InterfaceValues(fv_here::FaceValues, fv_there::FaceValues)`: quadrature rule and
  interpolations from the face values.

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

struct InterfaceValues{FVA <: FaceValues, FVB <: FaceValues} <: AbstractValues
    here::FVA
    there::FVB
    function InterfaceValues{FVA, FVB}(here::FVA, there::FVB) where {FVA, FVB}
        # TODO: check that fields don't alias
        return new{FVA, FVB}(here, there)
    end
end

function InterfaceValues(
        qr_here::FaceQuadratureRule, ip_here::Interpolation, ipg_here::Interpolation,
        qr_there::FaceQuadratureRule, ip_there::Interpolation, ipg_there::Interpolation
        )
    # FaceValues constructor enforces that refshape matches for all arguments
    here = FaceValues(qr_here, ip_here, ipg_here)
    there = FaceValues(qr_there, ip_there, ipg_there)
    return InterfaceValues{typeof(here), typeof(there)}(here, there)
end

# Same on both sides, default geometric mapping
InterfaceValues(qr_here::FaceQuadratureRule, ip_here::Interpolation) =
    InterfaceValues(qr_here, ip_here, deepcopy(qr_here), ip_here)
# Same on both sides, given geometric mapping
InterfaceValues(qr_here::FaceQuadratureRule, ip_here::Interpolation, ipg_here::Interpolation) =
    InterfaceValues(qr_here, ip_here, ipg_here, deepcopy(qr_here), ip_here, ipg_here)
# Different on both sides, default geometric mapping
function InterfaceValues(
        qr_here::FaceQuadratureRule, ip_here::Interpolation,
        qr_there::FaceQuadratureRule, ip_there::Interpolation,
    )
    return InterfaceValues(
        qr_here, ip_here, default_geometric_interpolation(ip_here),
        qr_there, ip_there, default_geometric_interpolation(ip_there),
    )
end
# From FaceValue(s)
InterfaceValues(facevalues_here::FVA, facevalues_there::FVB = deepcopy(facevalues_here)) where {FVA <: FaceValues, FVB <: FaceValues} =
    InterfaceValues{FVA,FVB}(facevalues_here, facevalues_there)

function getnbasefunctions(iv::InterfaceValues)
    return getnbasefunctions(iv.here) + getnbasefunctions(iv.there)
end

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
    interface_transformation = InterfaceOrientationInfo(cell_here, cell_there, face_here, face_there)
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
    # TODO: Remove the `here` kwarg and let user use `- getnormal(iv, qp)` instead?
    return getnormal(here ? iv.here : iv.there, qp)
end

"""
    function_value(iv::InterfaceValues, q_point::Int, u; here::Bool)
    function_value(iv::InterfaceValues, q_point::Int, u, dof_range_here, dof_range_there; here::Bool)

Compute the value of the function in quadrature point `q_point` on the "here" (`here=true`)
or "there" (`here=false`) side of the interface. `u_here` and `u_there` are the values of
the degrees of freedom for the respective element.

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
    shape_value_average(iv::InterfaceValues, qp::Int, i::Int)

Compute the average of the value of shape function `i` at quadrature point `qp` across the
interface.
"""
function shape_value_average end

"""
    shape_value_jump(iv::InterfaceValues, qp::Int, i::Int)

Compute the jump of the value of shape function `i` at quadrature point `qp` across the
interface.

This function uses the definition ``\\llbracket \\vec{v} \\rrbracket=\\vec{v}^\\text{here} -\\vec{v}^\\text{there}``. to obtain the form
``\\llbracket \\vec{v} \\rrbracket=\\vec{v}^\\text{there} ⋅ \\vec{n}^\\text{there} + \\vec{v}^\\text{here} ⋅ \\vec{n}^\\text{here}``
multiply by the outward facing normal to the first element's side of the interface (which is the default normal for [`getnormal`](@ref) with [`InterfaceValues`](@ref)).

"""
function shape_value_jump end

"""
    shape_gradient_average(iv::InterfaceValues, qp::Int, i::Int)

Compute the average of the gradient of shape function `i` at quadrature point `qp` across
the interface.
"""
function shape_gradient_average end

"""
    shape_gradient_jump(iv::InterfaceValues, qp::Int, i::Int)

Compute the jump of the gradient of shape function `i` at quadrature point `qp` across the
interface.

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

Compute the average of the function value at the quadrature point on the interface.
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
            @boundscheck checkbounds(u, 1:getnbasefunctions(iv))
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
            @boundscheck checkbounds(u, dof_range_here)
            @boundscheck checkbounds(u, dof_range_there)
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
            @boundscheck checkbounds(u, getnbasefunctions(iv))
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

function spatial_coordinate(
        iv::InterfaceValues, q_point::Int,
        x_here::AbstractVector{<:Vec}, x_there::AbstractVector{<:Vec}; here::Bool,
    )
    if here
        return spatial_coordinate(iv.here, q_point, x_here)
    else
        return spatial_coordinate(iv.there, q_point, x_there)
    end
end


# Transformation of quadrature points

@doc raw"""
    InterfaceOrientationInfo

Relative orientation information for 1D and 2D interfaces in 2D and 3D elements respectively.
This information is used to construct the transformation matrix to
transform the quadrature points from face_a to face_b achieving synced
spatial coordinates. Face B's orientation relative to Face A's can
possibly be flipped (i.e. the vertices indices order is reversed)
and the vertices can be rotated against each other.
The reference orientation of face B is such that the first node
has the lowest vertex index. Thus, this structure also stores the
shift of the lowest vertex index which is used to reorient the face in
case of flipping ["transform_interface_points!"](@ref).
"""
struct InterfaceOrientationInfo{RefShapeA, RefShapeB}
    flipped::Bool
    shift_index::Int
    lowest_node_shift_index::Int
    face_a::Int
    face_b::Int
end

"""
    InterfaceOrientationInfo(cell_a::AbstractCell, cell_b::AbstractCell, face_a::Int, face_b::Int)

Return the relative orientation info for face B with regards to face A.
Relative orientation is computed using a [`OrientationInfo`](@ref) for each side of the interface.
"""
function InterfaceOrientationInfo(cell_a::AbstractCell{RefShapeA}, cell_b::AbstractCell{RefShapeB}, face_a::Int, face_b::Int) where {RefShapeA <: AbstractRefShape, RefShapeB <: AbstractRefShape}
    OI_a = OrientationInfo(faces(cell_a)[face_a])
    OI_b = OrientationInfo(faces(cell_b)[face_b])
    flipped = OI_a.flipped != OI_b.flipped
    shift_index = OI_b.shift_index - OI_a.shift_index
    return InterfaceOrientationInfo{RefShapeA, RefShapeB}(flipped, shift_index, OI_b.shift_index, face_a, face_b)
end

function InterfaceOrientationInfo(_::AbstractCell{RefShapeA}, _::AbstractCell{RefShapeB}, _::Int, _::Int) where {RefShapeA <: AbstractRefShape{1}, RefShapeB <: AbstractRefShape{1}}
    (error("1D elements don't use transformations for interfaces."))
end

"""
    get_transformation_matrix(interface_transformation::InterfaceOrientationInfo)

Returns the transformation matrix corresponding to the interface orientation information stored in `InterfaceOrientationInfo`.
The transformation matrix is constructed using a combination of affine transformations defined for each interface reference shape.
The transformation for a flipped face is a function of both relative orientation and the orientation of the second face.
If the face is not flipped then the transformation is a function of relative orientation only.
"""
get_transformation_matrix

function get_transformation_matrix(interface_transformation::InterfaceOrientationInfo{RefShapeA}) where RefShapeA <: AbstractRefShape{3}
    face_a = interface_transformation.face_a
    facenodes = reference_faces(RefShapeA)[face_a]
    _get_transformation_matrix(facenodes, interface_transformation)
end

@inline function _get_transformation_matrix(::NTuple{3,Int}, interface_transformation::InterfaceOrientationInfo)
    flipped = interface_transformation.flipped
    shift_index = interface_transformation.shift_index
    lowest_node_shift_index = interface_transformation.lowest_node_shift_index

    θ = 2*shift_index/3
    θpre = 2*lowest_node_shift_index/3

    flipping = SMatrix{3,3}(1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0)

    translate_1 = SMatrix{3,3}(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, -sinpi(2/3)/3, -0.5, 1.0)
    stretch_1 = SMatrix{3,3}(sinpi(2/3), 0.5, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)

    translate_2 = SMatrix{3,3}(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, sinpi(2/3)/3, 0.5, 1.0)
    stretch_2 = SMatrix{3,3}(1/sinpi(2/3), -1/2/sinpi(2/3), 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)

    return flipped ? stretch_2 * translate_2 * rotation_tensor(0,0,θpre*pi) * flipping * rotation_tensor(0,0,(θ - θpre)*pi) * translate_1 * stretch_1 :
        stretch_2 * translate_2 * rotation_tensor(0,0,θ*pi) * translate_1 * stretch_1
end

@inline function _get_transformation_matrix(::NTuple{4,Int}, interface_transformation::InterfaceOrientationInfo)
    flipped = interface_transformation.flipped
    shift_index = interface_transformation.shift_index
    lowest_node_shift_index = interface_transformation.lowest_node_shift_index

    θ = 2*shift_index/4
    θpre = 2*lowest_node_shift_index/4

    flipping = SMatrix{3,3}(0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    return flipped ? rotation_tensor(0,0,θpre*pi) * flipping * rotation_tensor(0,0,(θ - θpre)*pi) :  rotation_tensor(0,0,θ*pi)
end

@inline function _get_transformation_matrix(::NTuple{N,Int}, ::InterfaceOrientationInfo) where N
    throw(ArgumentError("transformation is not implemented"))    
end

@doc raw"""
    transform_interface_points!(dst::Vector{Vec{3, Float64}}, points::Vector{Vec{3, Float64}}, interface_transformation::InterfaceOrientationInfo)

Transform the points from face A to face B using the orientation information of the interface and store it in the vector dst.
For 3D, the faces are transformed into regular polygons such that the rotation angle is the shift in reference node index × 2π ÷ number of edges in face.
If the face is flipped then the flipping is about the axis that preserves the position of the first node (which is the reference node after being rotated to be in the first position,
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
Transforming A to an equilateral triangle and translating it such that {0,0} is equidistant to all nodes
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
Rotating it -270° (or 120°) such that the reference node (the node with the smallest index) is at index 1
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

function transform_interface_points!(dst::Vector{Vec{3, Float64}}, points::Vector{Vec{3, Float64}}, interface_transformation::InterfaceOrientationInfo{RefShapeA, RefShapeB}) where {RefShapeA <: AbstractRefShape{3}, RefShapeB <: AbstractRefShape{3}}
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

function transform_interface_points!(dst::Vector{Vec{2, Float64}}, points::Vector{Vec{2, Float64}}, interface_transformation::InterfaceOrientationInfo{RefShapeA, RefShapeB}) where {RefShapeA <: AbstractRefShape{2}, RefShapeB <: AbstractRefShape{2}}
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
