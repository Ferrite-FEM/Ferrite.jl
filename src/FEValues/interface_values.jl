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
end
function InterfaceValues(grid::AbstractGrid, quad_rule::FaceQuadratureRule, func_interpol::Interpolation,
    geom_interpol::Interpolation = func_interpol)
    #@assert isDiscontinuous(func_interpol) "`InterfaceValues` is designed for discontinuous interpolations. a continuous interpolation is passed" TODO: add this when sparsity_pattern is merged
    face_values = FaceValues(quad_rule, func_interpol, geom_interpol)
    face_values_neighbor = copy(face_values)
    return InterfaceValues{typeof(func_interpol), FaceValues}(face_values, face_values_neighbor, grid, ScalarWrapper(0), ScalarWrapper(0))
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

for (func,                      f_,                 multiplier,             ) in (
    (:shape_value,              :shape_value,       :(1),                   ),
    (:shape_value_average,      :shape_value,       :(0.5),                 ),
    (:shape_value_jump,         :shape_value,       :(getnormal(fv, qp)),   ),
    (:shape_gradient,           :shape_gradient,    :(1),                   ),
    (:shape_gradient_average,   :shape_gradient,    :(0.5),                 ),
    (:shape_gradient_jump,      :shape_gradient,    :(getnormal(fv, qp)),   ),
)
    @eval begin
        function $(func)(iv::InterfaceValues, qp::Int, i::Int)
            nbf = getnbasefunctions(iv)
            if i <= nbf/2
                fv = iv.face_values
                f_value = $(f_)(fv, qp, i)
                return f_value isa Number || $(multiplier) isa Number ? $(multiplier) * f_value : f_value ⋅ $(multiplier) 
            elseif i <= nbf
                fv = iv.face_values_neighbor
                qp = get_neighbor_quadp(iv, qp)
                f_value = $(f_)(fv, qp, i - nbf ÷ 2)
                return f_value isa Number || $(multiplier) isa Number ? $(multiplier) * f_value : f_value ⋅ $(multiplier) 
            end
            error("Invalid base function $i. Interface has only $(nbf) base functions")
        end
    end
end

for (func,                          f_,                 multiplier,             ) in (
    (:function_value_average,       :function_value,       :(0.5),                 ),
    (:function_value_jump,          :function_value,       :(getnormal(fv, qp)),   ),
    (:function_gradient_average,    :function_gradient,    :(0.5),                 ),
    (:function_gradient_jump,       :function_gradient,    :(getnormal(fv, qp)),   ),
)
    @eval begin
        function $(func)(iv::InterfaceValues, qp::Int, u::AbstractVector, dof_range = eachindex(u))
            dof_range_here = dof_range[dof_range .<= length(eachindex(u)) ÷ 2]
            dof_range_there = dof_range[dof_range .> length(eachindex(u)) ÷ 2]
            f_value_here = $(f_)(iv, qp, u, dof_range_here; here = true)
            f_value_there = $(f_)(iv, qp, u, dof_range_there; here = false)
            fv = iv.face_values
            result = f_value_here isa Number || $(multiplier) isa Number ? $(multiplier) * f_value_here : f_value_here ⋅ $(multiplier)
            fv = iv.face_values_neighbor
            result += f_value_there isa Number || $(multiplier) isa Number ? $(multiplier) * f_value_there : f_value_there ⋅ $(multiplier)
            return result
        end
        # TODO: Deprecate this, nobody is using this in practice...
        function $(func)(iv::InterfaceValues, qp::Int, u::AbstractVector{<:Vec})
            f_value_here = $(f_)(iv, qp, u; here = true)
            f_value_there = $(f_)(iv, qp, u; here = false)
            fv = iv.face_values
            result = f_value_here isa Number || $(multiplier) isa Number ? $(multiplier) * f_value_here : f_value_here ⋅ $(multiplier)
            fv = iv.face_values_neighbor
            result += f_value_there isa Number || $(multiplier) isa Number ? $(multiplier) * f_value_there : f_value_there ⋅ $(multiplier)
            return result
        end
    end
end
"""
    get_neighbor_quadp(iv::InterfaceValues, qpoint::Int)

Find quadrature point index in the neighbor facet.
"""
function get_neighbor_quadp(iv::InterfaceValues, qpoint::Int)
    # TODO: figure out how to use InterfaceOrientationInfo here
    c1 = get_cell_coordinates(iv.grid, iv.cell_idx[])
    c2 = get_cell_coordinates(iv.grid, iv.cell_idx_neighbor[])
    qpcoord = spatial_coordinate(iv.face_values, qpoint, c1)
    neighbor_qp_coords = spatial_coordinate.(Ref(iv.face_values_neighbor), 1:getnquadpoints(iv.face_values_neighbor), Ref(c2))
    return findfirst(i->i ≈ qpcoord, neighbor_qp_coords)
end