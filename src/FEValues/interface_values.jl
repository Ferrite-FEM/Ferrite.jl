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
* `geom_interpol_a`: an optional instance of an [`Interpolation`](@ref) which is used to interpolate the geometry for element A.
  It uses the default interpolation of the respective [`RefShape`](@ref) by default.
* `geom_interpol_b`: an optional instance of an [`Interpolation`](@ref) which is used to interpolate the geometry for element B.
  It uses the default interpolation of the respective [`RefShape`](@ref) by default.
 
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
    interface_transformation::InterfaceTransformation
end
function InterfaceValues(quad_rule_a::FaceQuadratureRule, func_interpol_a::Interpolation,
    geom_interpol_a::Interpolation = func_interpol_a; quad_rule_b::FaceQuadratureRule = deepcopy(quad_rule_a),
    func_interpol_b::Interpolation = func_interpol_a, geom_interpol_b::Interpolation = func_interpol_b)
    face_values_a = FaceValues(quad_rule_a, func_interpol_a, geom_interpol_a)
    face_values_b = FaceValues(quad_rule_b, func_interpol_b, geom_interpol_b)
    return InterfaceValues{typeof(face_values_a), typeof(face_values_b)}(face_values_a, face_values_b, InterfaceTransformation())
end

"""
    reinit!(iv::InterfaceValues, face_a::FaceIndex, face_b::FaceIndex, cell_a_coords::AbstractVector{Vec{dim, T}}, cell_b_coords::AbstractVector{Vec{dim, T}}, grid::AbstractGrid) where {dim, T}

Update the [`FaceValues`](@ref) in the interface (A and B) using their corresponding cell coordinates and [`FaceIndex`](@ref). This involved recalculating the transformation matrix [`transform_interface_point`](@ref)
and mutating element B's quadrature points and its [`FaceValues`](@ref) `M, N, dMdξ, dNdξ`.
"""
function reinit!(iv::InterfaceValues, face_a::FaceIndex, face_b::FaceIndex, cell_a_coords::AbstractVector{Vec{dim, T}}, cell_b_coords::AbstractVector{Vec{dim, T}}, grid::AbstractGrid) where {dim, T}
    reinit!(iv.face_values_a, cell_a_coords, face_a[2])
    iv.face_values_b.current_face[] = face_b[2]
    update!(iv.interface_transformation, grid, face_a, face_b)
    quad_points_a = getpoints(iv.face_values_a.qr, face_a[2])
    quad_points_b = getpoints(iv.face_values_b.qr, face_b[2])
    transform_interface_points!(quad_points_b, iv, quad_points_a, grid, face_a, face_b)
    @boundscheck checkface(iv.face_values_b, face_b[2])
    # This is the bottleneck, cache it?
    for (idx, qp) in pairs(quad_points_b)
        gradients_and_values = shape_gradients_and_values(iv.face_values_b.func_interp, qp)
        iv.face_values_b.dNdξ[:,idx, face_b[2]] .= getindex.(gradients_and_values,1)
        iv.face_values_b.N[:, idx, face_b[2]] .= getindex.(gradients_and_values,2)
    end
    for (idx, qp) in pairs(quad_points_b)
        gradients_and_values = shape_gradients_and_values(iv.face_values_b.geo_interp, qp)
        iv.face_values_b.dMdξ[:,idx, face_b[2]] .= getindex.(gradients_and_values,1)
        iv.face_values_b.M[:, idx, face_b[2]] .= getindex.(gradients_and_values,2)
    end
    reinit!(iv.face_values_b, cell_b_coords, face_b[2])
end

"""
    getnormal(iv::InterfaceValues, qp::Int, use_element_a::Bool = true)

Return the normal at the quadrature point `qp` on the interface. 

For `InterfaceValues`, `use_elemet_a` determines which element to use for calculating divergence of the function.
`true` uses the element A's face nomal vector, which is the default, while `false` uses element B's.
"""
getnormal(iv::InterfaceValues, qp::Int, use_element_a::Bool = true) = use_element_a ? iv.face_values_a.normals[qp] : iv.face_values_b.normals[qp]

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

"""
    geometric_value_average(iv::InterfaceValues, qp::Int, base_function::Int)

Compute the average of the geometric interpolation shape function value at the quadrature point on interface.
"""
geometric_value_average

"""
    geometric_value_jump(iv::InterfaceValues, qp::Int, base_function::Int)

Compute the jump of the geometric interpolation shape function value at the quadrature point over the interface.

This function uses the definition ``\\llbracket \\vec{v} \\rrbracket=\\vec{v}^- -\\vec{v}^+``. to obtain the form 
``\\llbracket \\vec{v} \\rrbracket=\\vec{v}^- ⋅ \\vec{n}^- + \\vec{v}^+ ⋅ \\vec{n}^+``one can simple multiply by the normal of face A (which is the default normal for [`getnormal`](@ref) with [`InterfaceValues`](@ref)).
"""
geometric_value_jump

for (func,                      f_,                 multiplier, ) in (
    (:shape_value,              :shape_value,       :(1),       ),
    (:shape_value_average,      :shape_value,       :(0.5),     ),
    (:shape_gradient,           :shape_gradient,    :(1),       ),
    (:shape_gradient_average,   :shape_gradient,    :(0.5),     ),
    (:geometric_value_average,  :geometric_value,   :(0.5),     ),
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
    (:geometric_value_jump,     :geometric_value,   ),
)
    @eval begin
        function $(func)(iv::InterfaceValues, qp::Int, i::Int)
            f_value = $(f_)(iv, qp, i)
            nbf_a = getnbasefunctions(iv.face_values_a)
            return i <= nbf_a ? f_value : -f_value
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
            f_value_here = $(f_)(iv, qp, u_a, dof_range_a, use_element_a = true)
            f_value_there = $(f_)(iv, qp, u_b, dof_range_b, use_element_a = false)
            fv = iv.face_values_a
            result = 0.5 * f_value_here 
            fv = iv.face_values_b
            result += 0.5 * f_value_there
            return result
        end
        # TODO: Deprecate this, nobody is using this in practice...
        function $(func)(iv::InterfaceValues, qp::Int, u_a::AbstractVector{<:Vec}, u_b::AbstractVector{<:Vec})
            f_value_here = $(f_)(iv, qp, u_a, use_element_a = true)
            f_value_there = $(f_)(iv, qp, u_b, use_element_a = false)
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
            f_value_here = $(f_)(iv, qp, u_a, dof_range_a, use_element_a = true)
            f_value_there = $(f_)(iv, qp, u_b, dof_range_b, use_element_a = false)
            return f_value_here - f_value_there 
        end
        # TODO: Deprecate this, nobody is using this in practice...
        function $(func)(iv::InterfaceValues, qp::Int, u_a::AbstractVector{<:Vec}, u_b::AbstractVector{<:Vec})
            f_value_here = $(f_)(iv, qp, u_a, use_element_a = true)
            f_value_there = $(f_)(iv, qp, u_b, use_element_a = false)
            return f_value_here - f_value_there 
        end
    end
end

"""
    transform_interface_points!(dst::Vector{Vec{dim, Float64}}, iv::InterfaceValues, points::Vector{Vec{dim, Float64}}, grid::AbstractGrid, face_a::FaceIndex, face_b::FaceIndex)

Transform the points from face A to face B using the orientation information of the interface and store it in the vecotr dst.
For 3D, the faces are transformed to regular polygons such that the rotation angle is the shift in reference node index × 2π ÷ number of edges in face.
If the face is flipped then the flipping is about the axis that perserves the position of the first node (which is the reference node after being rotated to be in the first position,
it's rotated back in the opposite direction after flipping).
Take for example the interface
```
        2           3
        | \\         | \\
        |  \\        |  \\
y       | A \\       | B \\ 
↑       |    \\      |    \\
→  x    1-----3     1-----2  
```
Transforming A to a equilateral triangle and translating it such that {0,0} is equidistant to all nodes
```
        3
        +
       / \\         
      /   \\
     /  x  \\                          
    /   ↑   \\                        
   /  ←      \\                          
  /  y        \\                            
2+-------------+1                     
```
Rotating it -270° (or 120°) such that the reference node (the node with smallest index) is at index 1
```
        1
        +
       / \\         
      /   \\
     /  x  \\                          
    /   ↑   \\                        
   /  ←      \\                          
  /  y        \\                            
3+-------------+2                     
```
Flipping about the x axis (such that the position of the reference node doesn't change) and rotating 270° (or -120°)
```
        2
        +
       / \\         
      /   \\
     /  x  \\                          
    /   ↑   \\                        
   /  ←      \\                          
  /  y        \\                            
3+-------------+1                     
```
Transforming back to the reference trianle
```
       3           
       | \\
       |  \\
y      |   \\ 
↑      |    \\
→ x    1-----2  
```
"""
function transform_interface_points!(dst::Vector{Vec{dim, Float64}}, iv::InterfaceValues, points::Vector{Vec{dim, Float64}}, grid::AbstractGrid, face_a::FaceIndex, face_b::FaceIndex) where {dim}
    cell = getcells(grid)[face_a[1]]
    face = iv.face_values_a.current_face[]
    flipped = iv.interface_transformation.flipped[]
    shift_index = iv.interface_transformation.shift_index[]
    lowest_node_shift_index = iv.interface_transformation.lowest_node_shift_index[]
    if dim == 3
        if length(faces(cell)[face]) == 3
            θ = 2/3 * shift_index
            θpre = 2/3 * lowest_node_shift_index
            
            flipping = SMatrix{3,3}(1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0)
    
            translate_1 = SMatrix{3,3}(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, -sinpi(2/3)/3, -0.5, 1.0) 
            stretch_1 = SMatrix{3,3}(sinpi(2/3), 0.5, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0) 
    
            translate_2 = SMatrix{3,3}(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, sinpi(2/3)/3, 0.5, 1.0) 
            stretch_2 = SMatrix{3,3}(1/sinpi(2/3), -1/2/sinpi(2/3), 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0) 
    
            M = flipped ? stretch_2 * translate_2 * rotation_matrix_pi(-θpre) * flipping * rotation_matrix_pi(θ + θpre) * translate_1 * stretch_1 : stretch_2 * translate_2 * rotation_matrix_pi(θ) * translate_1 * stretch_1
        else # length(faces(cell)[face]) == 4
            θ = shift_index/2
            θpre = lowest_node_shift_index/2
            flipping = SMatrix{3,3}(0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0)
            M = flipped ? rotation_matrix_pi(-θpre) * flipping * rotation_matrix_pi(θ + θpre) :  rotation_matrix_pi(θ) 
        end
        for (idx, point) in pairs(points)
            point = element_to_face_transformation(point, getrefshape(cell), face)
            result = M * Vec(point[1],point[2], 1.0)
            dst[idx] = face_to_element_transformation(Vec(result[1],result[2]), getrefshape(getcells(grid)[face_b[1]]), iv.face_values_b.current_face[])
        end
    else
        for (idx, point) in pairs(points)
            point = element_to_face_transformation(point, getrefshape(cell), face)
            dim == 2 && flipped && (point *= -1) 
            dst[idx] = face_to_element_transformation(point, getrefshape(getcells(grid)[face_b[1]]), iv.face_values_b.current_face[])
        end
    end
    return nothing
end
