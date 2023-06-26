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

struct InterfaceValues{IP, FV<:FaceValues} <: AbstractValues
    face_values::FV
    face_values_neighbor::FV
    # used for quadrature point syncing
    grid::Grid
    cell_idx::ScalarWrapper{Int}
    cell_idx_neighbor::ScalarWrapper{Int}
    ioi::ScalarWrapper{InterfaceOrientationInfo}
end
function InterfaceValues(grid::AbstractGrid, quad_rule::FaceQuadratureRule, func_interpol::Interpolation,
    geom_interpol::Interpolation = func_interpol)
    #@assert isDiscontinuous(func_interpol) "`InterfaceValues` is designed for discontinuous interpolations. a continuous interpolation is passed" TODO: add this when sparsity_pattern is merged
    face_values = FaceValues(quad_rule, func_interpol, geom_interpol)
    face_values_neighbor = FaceValues(deepcopy(quad_rule), func_interpol, geom_interpol)
    return InterfaceValues{typeof(func_interpol), FaceValues}(face_values, face_values_neighbor, grid, ScalarWrapper(0), ScalarWrapper(0), ScalarWrapper(InterfaceOrientationInfo(false, nothing)))
end
# Maybe move this to common_values.jl?
"""
    shape_value_average(iv::InterfaceValues, qp::Int, base_function::Int)

Compute the average of the shape function value at the quadrature point over interface.
"""
shape_value_average

"""
    shape_value_jump(iv::InterfaceValues, qp::Int, base_function::Int, normal_dotted::Bool = true)

Compute the jump of the shape function value at the quadrature point over the interface.

`normal_dotted::Bool` determines whether to use the definition ``\\llbracket v \\rrbracket=v^- -v^+`` if it's `false`, or
 the definition  ``\\llbracket v \\rrbracket=v^- ⋅ \\vec{n}^- + v^+ ⋅ \\vec{n}^+`` if it's `true`, which is the default.

!!! note
    If `normal_dotted == true` then the jump of scalar shape values is a vector.
"""
shape_value_jump

"""
    shape_gradient_average(iv::InterfaceValues, qp::Int, base_function::Int)

Compute the average of the shape function gradient at the quadrature point over the interface.
"""
shape_gradient_average

"""
    shape_gradient_jump(iv::InterfaceValues, qp::Int, base_function::Int, normal_dotted::Bool = true)

Compute the jump of the shape function gradient at the quadrature point over the interface.

`normal_dotted::Bool` determines whether to use the definition ``\\llbracket \\vec{v} \\rrbracket=\\vec{v}^- -\\vec{v}^+`` if it's `false`, or
 the definition  ``\\llbracket \\vec{v} \\rrbracket=\\vec{v}^- ⋅ \\vec{n}^- + \\vec{v}^+ ⋅ \\vec{n}^+`` if it's `true`, which is the default.

!!! note
    If `normal_dotted == true` then the jump of the gradient vector is a scalar.
"""
shape_gradient_jump

for (func,                      f_,                 multiplier  ) in (
    (:shape_value,              :shape_value,       :(1),       ),
    (:shape_value_average,      :shape_value,       :(0.5),     ),
    (:shape_gradient,           :shape_gradient,    :(1),       ),
    (:shape_gradient_average,   :shape_gradient,    :(0.5),     ),
)
    @eval begin
        function $(func)(iv::InterfaceValues, qp::Int, i::Int)
            nbf = getnbasefunctions(iv)
            if i <= nbf/2
                fv = iv.face_values
                f_value = $(f_)(fv, qp, i)
                return $(multiplier) * f_value
            elseif i <= nbf
                fv = iv.face_values_neighbor
                f_value = $(f_)(fv, qp, i - nbf ÷ 2)
                return $(multiplier) * f_value
            end
            error("Invalid base function $i. Interface has only $(nbf) base functions")
        end
    end
end

for (func,                      f_,             ) in (
    (:shape_value_jump,         :shape_value,   ),
    (:shape_gradient_jump,      :shape_gradient,),
)
    @eval begin
        function $(func)(iv::InterfaceValues, qp::Int, i::Int, normal_dotted::Bool = true)
            nbf = getnbasefunctions(iv)
            if i <= nbf/2
                fv = iv.face_values
                f_value = $(f_)(fv, qp, i)
                normal_dotted || return f_value
                multiplier = getnormal(fv, qp)
                return f_value isa Number ? f_value * multiplier : f_value ⋅ multiplier
            elseif i <= nbf
                fv = iv.face_values_neighbor
                f_value = $(f_)(fv, qp, i - nbf ÷ 2)
                normal_dotted || return -f_value
                multiplier = getnormal(fv, qp)
                return f_value isa Number ? f_value * multiplier : f_value ⋅ multiplier
            end
            error("Invalid base function $i. Interface has only $(nbf) base functions")
        end
    end
end

"""
    function_value_average(iv::InterfaceValues, qp::Int, u::AbstractVector, dof_range = eachindex(u))

Compute the average of the function value at the quadrature point over interface.
"""
function_value_average

"""
    function_value_jump(iv::InterfaceValues, qp::Int, u::AbstractVector, dof_range = eachindex(u), normal_dotted::Bool = true)

Compute the jump of the function value at the quadrature point over the interface.

`normal_dotted::Bool` determines whether to use the definition ``\\llbracket v \\rrbracket=v^- -v^+`` if it's `false`, or
 the definition  ``\\llbracket v \\rrbracket=v^- ⋅ \\vec{n}^- + v^+ ⋅ \\vec{n}^+`` if it's `true`, which is the default.

!!! note
    If `normal_dotted == true` then the jump of scalar function values is a vector.
"""
function_value_jump

"""
    function_gradient_average(iv::InterfaceValues, qp::Int, u::AbstractVector, dof_range = eachindex(u))

Compute the average of the function gradient at the quadrature point over the interface.
"""
function_gradient_average

"""
    function_gradient_jump(iv::InterfaceValues, qp::Int, u::AbstractVector, dof_range = eachindex(u), normal_dotted::Bool = true)

Compute the jump of the function gradient at the quadrature point over the interface.

`normal_dotted::Bool` determines whether to use the definition ``\\llbracket \\vec{v} \\rrbracket=\\vec{v}^- -\\vec{v}^+`` if it's `false`, or
 the definition  ``\\llbracket \\vec{v} \\rrbracket=\\vec{v}^- ⋅ \\vec{n}^- + \\vec{v}^+ ⋅ \\vec{n}^+`` if it's `true`, which is the default.

!!! note
    If `normal_dotted == true` then the jump of the gradient vector is a scalar.
"""
function_gradient_jump

for (func,                          f_,                 ) in (
    (:function_value_average,       :function_value,    ),
    (:function_gradient_average,    :function_gradient, ),
)
    @eval begin
        function $(func)(iv::InterfaceValues, qp::Int, u::AbstractVector, dof_range = eachindex(u))
            dof_range_here = dof_range[dof_range .<= length(eachindex(u)) ÷ 2]
            dof_range_there = dof_range[dof_range .> length(eachindex(u)) ÷ 2]
            f_value_here = $(f_)(iv, qp, u, dof_range_here; here = true)
            f_value_there = $(f_)(iv, qp, u, dof_range_there; here = false)
            fv = iv.face_values
            result = 0.5 * f_value_here 
            fv = iv.face_values_neighbor
            result += 0.5 * f_value_there
            return result
        end
        # TODO: Deprecate this, nobody is using this in practice...
        function $(func)(iv::InterfaceValues, qp::Int, u::AbstractVector{<:Vec})
            f_value_here = $(f_)(iv, qp, u; here = true)
            f_value_there = $(f_)(iv, qp, u; here = false)
            fv = iv.face_values
            result = 0.5 * f_value_here
            fv = iv.face_values_neighbor
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
        function $(func)(iv::InterfaceValues, qp::Int, u::AbstractVector, dof_range = eachindex(u), normal_dotted::Bool = true)
            dof_range_here = dof_range[dof_range .<= length(eachindex(u)) ÷ 2]
            dof_range_there = dof_range[dof_range .> length(eachindex(u)) ÷ 2]
            f_value_here = $(f_)(iv, qp, u, dof_range_here; here = true)
            f_value_there = $(f_)(iv, qp, u, dof_range_there; here = false)
            fv = iv.face_values
            multiplier = getnormal(fv, qp)
            result = f_value_here isa Number || multiplier isa Number ? f_value_here * multiplier : f_value_here ⋅ multiplier
            fv = iv.face_values_neighbor
            multiplier = getnormal(fv, qp)
            result += f_value_there isa Number || multiplier isa Number ? f_value_there * multiplier : f_value_there ⋅ multiplier
            normal_dotted || (result = result ⋅ getnormal(fv, qp))
            return result
        end
        # TODO: Deprecate this, nobody is using this in practice...
        function $(func)(iv::InterfaceValues, qp::Int, u::AbstractVector{<:Vec}, normal_dotted::Bool = true)
            f_value_here = $(f_)(iv, qp, u; here = true)
            f_value_there = $(f_)(iv, qp, u; here = false)
            fv = iv.face_values
            multiplier = getnormal(fv, qp)
            result = f_value_here isa Number || multiplier isa Number ? f_value_here * multiplier : f_value_here ⋅ multiplier
            fv = iv.face_values_neighbor
            multiplier = getnormal(fv, qp)
            result += f_value_there isa Number || multiplier isa Number ? f_value_there * multiplier : f_value_there ⋅ multiplier
            normal_dotted || (result = result ⋅ getnormal(fv, qp))
            return result
        end
    end
end

"""
    transform_interface_point(iv::InterfaceValues, point::AbstractArray)

Transform point from current face in the interface reference coordinates to the neighbor face reference coordinates.
"""
function transform_interface_point(iv::InterfaceValues, point::AbstractArray)
    ioi = iv.ioi[]
    cell = getcells(iv.grid)[iv.cell_idx[]]
    face = iv.face_values.current_face[]
    point = transfer_point_cell_to_face(point, cell, face)
    isnothing(ioi.transformation) || (point = (ioi.transformation * [point..., 1])[1:2])
    ioi.flipped && reverse!(point)
    return transfer_point_face_to_cell(point, getcells(iv.grid)[iv.cell_idx_neighbor[]], iv.face_values_neighbor.current_face[])
end
