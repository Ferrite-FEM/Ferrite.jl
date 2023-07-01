# Defines InterfaceValues and common methods
"""
    InterfaceValues(grid::AbstractGrid, quad_rule::FaceQuadratureRule, func_interpol_a::Interpolation, [geom_interpol_a::Interpolation], [func_interpol_b::Interpolation], [geom_interpol_b::Interpolation])

An `InterfaceValues` object facilitates the process of evaluating values, averages, jumps and gradients of shape functions
and function on the interfaces of finite elements.

**Arguments:**

* `grid` : instance of the current grid.
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

struct InterfaceValues{IP_a, N_t_a, dNdx_t_a, dNdξ_t_a, T_a, dMdξ_t_a, QR_a, Normal_t_a, GIP_a,
    IP_b, N_t_b, dNdx_t_b, dNdξ_t_b, T_b, dMdξ_t_b, QR_b, Normal_t_b, GIP_b,
    dim, C} <: AbstractValues
    face_values_a::FaceValues{IP_a, N_t_a, dNdx_t_a, dNdξ_t_a, T_a, dMdξ_t_a, QR_a, Normal_t_a, GIP_a}
    face_values_b::FaceValues{IP_b, N_t_b, dNdx_t_b, dNdξ_t_b, T_b, dMdξ_t_b, QR_b, Normal_t_b, GIP_b}
    # used for quadrature point syncing
    grid::Grid{dim, C, Float64}
    cell_a_idx::ScalarWrapper{Int}
    cell_b_idx::ScalarWrapper{Int}
    ioi::ScalarWrapper{InterfaceOrientationInfo}
end
function InterfaceValues(grid::AbstractGrid, quad_rule_a::FaceQuadratureRule, func_interpol_a::Interpolation,
    geom_interpol_a::Interpolation = func_interpol_a; quad_rule_b::FaceQuadratureRule = deepcopy(quad_rule_a),
    func_interpol_b::Interpolation = func_interpol_a, geom_interpol_b::Interpolation = func_interpol_b)
    face_values_a = FaceValues(quad_rule_a, func_interpol_a, geom_interpol_a)
    face_values_b = FaceValues(quad_rule_b, func_interpol_b, geom_interpol_b)
    return InterfaceValues{typeof(face_values_a).parameters..., typeof(face_values_b).parameters..., getdim(grid), getcelltype(grid)}(face_values_a, face_values_b, grid, ScalarWrapper(0), ScalarWrapper(0), ScalarWrapper(InterfaceOrientationInfo(false, Float64.(I(1)))))
end

"""
    reinit!(iv::InterfaceValues, face_a::FaceIndex, face_b::FaceIndex, cell_a_coords::AbstractVector{Vec{dim, T}}, cell_b_coords::AbstractVector{Vec{dim, T}}, grid::AbstractGrid) where {dim, T}

Update the [`FaceValues`](@ref) in the interface (A and B) using their corresponding cell coordinates and [`FaceIndex`](@ref). This involved recalculating the transformation matrix [`transform_interface_point`](@ref)
and mutating element B's quadrature points and its [`FaceValues`](@ref) `M, N, dMdξ, dNdξ`.
"""
function reinit!(iv::InterfaceValues, face_a::FaceIndex, face_b::FaceIndex, cell_a_coords::AbstractVector{Vec{dim, T}}, cell_b_coords::AbstractVector{Vec{dim, T}}, grid::AbstractGrid) where {dim, T}
    reinit!(iv.face_values_a, cell_a_coords, face_a[2])
    iv.cell_a_idx[] = face_a[1]
    iv.cell_b_idx[] = face_b[1]
    iv.face_values_b.current_face[] = face_b[2]
    ioi = Ferrite.InterfaceOrientationInfo(grid, face_a, face_b)
    iv.ioi[] = ioi
    getpoints(iv.face_values_b.qr, face_b[2]) .= Vec{dim}.(Ferrite.transform_interface_point.(Ref(iv), getpoints(iv.face_values_a.qr, face_a[2])))
    # Reinit face_b facevalues after the quadrature rule points are mutated
    n_geom_basefuncs = getngeobasefunctions(iv.face_values_b)
    n_func_basefuncs = getnbasefunctions(iv.face_values_b)
    @boundscheck checkface(iv.face_values_b, face_b[2])

    for (qp, ξ) in pairs(getpoints(iv.face_values_b.qr, face_b[2]))
        for basefunc in 1:n_func_basefuncs
            iv.face_values_b.dNdξ[basefunc, qp, face_b[2]], iv.face_values_b.N[basefunc, qp, face_b[2]] = shape_gradient_and_value(iv.face_values_b.func_interp, ξ, basefunc)
        end
        for basefunc in 1:n_geom_basefuncs
            iv.face_values_b.dMdξ[basefunc, qp, face_b[2]], iv.face_values_b.M[basefunc, qp, face_b[2]] = shape_gradient_and_value(iv.face_values_b.geo_interp, ξ, basefunc)
        end
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

Compute the average of the shape function gradient at the quadrature point on the interface.
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

"""
    geometric_value_average(iv::InterfaceValues, qp::Int, base_function::Int)

Compute the average of the geometric interpolation shape function value at the quadrature point on interface.
"""
geometric_value_average

"""
    geometric_value_jump(iv::InterfaceValues, qp::Int, base_function::Int, normal_dotted::Bool = true)

Compute the jump of the geometric interpolation shape function value at the quadrature point over the interface.

`normal_dotted::Bool` determines whether to use the definition ``\\llbracket v \\rrbracket=v^- -v^+`` if it's `false`, or
 the definition  ``\\llbracket v \\rrbracket=v^- ⋅ \\vec{n}^- + v^+ ⋅ \\vec{n}^+`` if it's `true`, which is the default.

!!! note
    If `normal_dotted == true` then the jump of scalar shape values is a vector.
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
        function $(func)(iv::InterfaceValues, qp::Int, i::Int, normal_dotted::Bool = true)
            f_value = $(f_)(iv, qp, i)
            nbf_a = getnbasefunctions(iv.face_values_a)
            if i <= nbf_a
                normal_dotted || return f_value
                multiplier = getnormal(iv, qp, true)
                return f_value isa Number ? f_value * multiplier : f_value ⋅ multiplier
            else
                normal_dotted || return -f_value
                multiplier = getnormal(iv, qp, false)
                return f_value isa Number ? f_value * multiplier : f_value ⋅ multiplier
            end
        end
    end
end

"""
    function_value_average(iv::InterfaceValues, qp::Int, u::AbstractVector, dof_range = eachindex(u))

Compute the average of the function value at the quadrature point on interface.
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

Compute the average of the function gradient at the quadrature point on the interface.
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
            nbf_a = getnbasefunctions(iv.face_values_a)
            dof_range_here = dof_range[dof_range .<= nbf_a]
            dof_range_there = dof_range[dof_range .> nbf_a]
            f_value_here = $(f_)(iv, qp, u, dof_range_here; use_element_a = true)
            f_value_there = $(f_)(iv, qp, u, dof_range_there; use_element_a = false)
            fv = iv.face_values_a
            result = 0.5 * f_value_here 
            fv = iv.face_values_b
            result += 0.5 * f_value_there
            return result
        end
        # TODO: Deprecate this, nobody is using this in practice...
        function $(func)(iv::InterfaceValues, qp::Int, u::AbstractVector{<:Vec})
            f_value_here = $(f_)(iv, qp, u; use_element_a = true)
            f_value_there = $(f_)(iv, qp, u; use_element_a = false)
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
        function $(func)(iv::InterfaceValues, qp::Int, u::AbstractVector, dof_range = eachindex(u), normal_dotted::Bool = true)
            nbf_a = getnbasefunctions(iv.face_values_a)
            dof_range_here = dof_range[dof_range .<= nbf_a]
            dof_range_there = dof_range[dof_range .> nbf_a]
            f_value_here = $(f_)(iv, qp, u, dof_range_here; use_element_a = true)
            f_value_there = $(f_)(iv, qp, u, dof_range_there; use_element_a = false)
            multiplier = getnormal(iv, qp, true)
            result = f_value_here isa Number || multiplier isa Number ? f_value_here * multiplier : f_value_here ⋅ multiplier
            multiplier = getnormal(iv, qp, false)
            result += f_value_there isa Number || multiplier isa Number ? f_value_there * multiplier : f_value_there ⋅ multiplier
            normal_dotted || (result = result ⋅ getnormal(iv, qp))
            return result
        end
        # TODO: Deprecate this, nobody is using this in practice...
        function $(func)(iv::InterfaceValues, qp::Int, u::AbstractVector{<:Vec}, normal_dotted::Bool = true)
            f_value_here = $(f_)(iv, qp, u; use_element_a = true)
            f_value_there = $(f_)(iv, qp, u; use_element_a = false)
            multiplier = getnormal(iv, qp, true)
            result = f_value_here isa Number || multiplier isa Number ? f_value_here * multiplier : f_value_here ⋅ multiplier
            multiplier = getnormal(iv, qp, false)
            result += f_value_there isa Number || multiplier isa Number ? f_value_there * multiplier : f_value_there ⋅ multiplier
            normal_dotted || (result = result ⋅ getnormal(iv, qp))
            return result
        end
    end
end

"""
    transform_interface_point(iv::InterfaceValues, point::AbstractArray)

Transform point from element A's face reference coordinates to element B's face reference coordinates.
"""
function transform_interface_point(iv::InterfaceValues, point::Vec{N, Float64}) where {N}
    ioi = iv.ioi[]
    cell = getcells(iv.grid)[iv.cell_a_idx[]]
    face = iv.face_values_a.current_face[]
    point = transfer_point_cell_to_face(point, cell, face)
    N == 3 && (point = (ioi.transformation * Vec(point[1],point[2], 1.0))[1:2])
    ioi.flipped && (point = reverse(point))
    N == 3 && return transfer_point_face_to_cell(Vec(point[1],point[2]), getcells(iv.grid)[iv.cell_b_idx[]], iv.face_values_b.current_face[])
    return transfer_point_face_to_cell(Vec(point[1]), getcells(iv.grid)[iv.cell_b_idx[]], iv.face_values_b.current_face[])
end

function transform_interface_point(iv::InterfaceValues, point::Vector{Vec{N, Float64}}) where {N}
    ioi = iv.ioi[]
    cell = getcells(iv.grid)[iv.cell_a_idx[]]
    face = iv.face_values_a.current_face[]
    point2 = transfer_point_cell_to_face.(point, Ref(cell), Ref(face))
    if N == 3
        M = ioi.transformation * vcat(reduce(hcat, point2), ones(Float64, 1, length(point)))
        point2 = [(M[1:2,i]) for i in 1:length(point)]
    end
    ioi.flipped && (point2 .= reverse.(point2))
    N == 3 && return transfer_point_face_to_cell.(point2, Ref(getcells(iv.grid)[iv.cell_b_idx[]]), Ref(iv.face_values_b.current_face[]))
    return transfer_point_face_to_cell.(point2, getcells(iv.grid)[iv.cell_b_idx[]], iv.face_values_b.current_face[])
end
