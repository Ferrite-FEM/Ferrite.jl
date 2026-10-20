# Prototype: "extractor" views (cf. deal.II's FEValuesExtractors) for the values of one field,
# with the shape functions indexed by the dofs of the whole element. The shape functions of
# the other fields' dofs are zero, so that e.g. the element matrix of a coupled problem can
# be assembled in a single loop over all the element dofs:
#
#     vu = FieldView(cv, dh, :u)
#     vp = FieldView(cv, dh, :p)
#     reinit!(cv, cell); reinit!(vu); reinit!(vp)
#     for i in 1:ndofs_per_cell(dh), j in 1:ndofs_per_cell(dh)
#         Ke[i, j] += (shape_symmetric_gradient(vu, q, i) ⊡ shape_symmetric_gradient(vu, q, j) -
#                      shape_divergence(vu, q, i) * shape_value(vp, q, j) - ...) * dΩ
#     end

struct FieldView{FV <: AbstractValues, NT, GT} <: AbstractValues
    values::FV                 # the values of the field, e.g. `cv.u` or a single-field `CellValues`
    dof_range::UnitRange{Int}  # the dofs of the field in the element
    # Copies of the field's shape values (gradients) indexed by the dofs of the whole element,
    # zero for the dofs of other fields, so that shape lookups don't branch on the dof.
    # Refreshed with `reinit!(::FieldView)`.
    Nx::Matrix{NT}
    dNdx::GT                   # ::Matrix or `nothing` (no gradients)
    function FieldView(values::FV, dof_range::AbstractUnitRange{<:Integer}, ndofs::Integer) where {FV <: AbstractValues}
        if length(dof_range) != getnbasefunctions(values)
            throw(ArgumentError("the length of the dof range ($(length(dof_range))) does not match the number of base functions of the field ($(getnbasefunctions(values)))"))
        end
        if !isempty(dof_range) && (first(dof_range) < 1 || last(dof_range) > ndofs)
            throw(ArgumentError("the dof range $(dof_range) is not within the dofs of the element (1:$(ndofs))"))
        end
        nqp = getnquadpoints(values)
        Nx = zeros(shape_value_type(values), ndofs, nqp)
        GT = shape_gradient_type(values)
        dNdx = GT === nothing ? nothing : zeros(GT, ndofs, nqp)
        v = new{FV, eltype(Nx), typeof(dNdx)}(values, UnitRange{Int}(dof_range), Nx, dNdx)
        return reinit!(v)
    end
end

# The field `field` of a multi-field CellValues, with the dof range and the number of dofs
# per cell from the (Sub)DofHandler
function FieldView(cv::MultiFieldCellValues, dh::Union{DofHandler, SubDofHandler}, field::Symbol)
    return FieldView(getproperty(cv, field), dof_range(dh, field), ndofs_per_cell(dh))
end

"""
    reinit!(v::FieldView)

Update the field view `v` after its underlying values have been reinitialized, e.g.
`reinit!(cv, cell); reinit!(v)`.
"""
function reinit!(v::FieldView)
    for q in 1:getnquadpoints(v.values), (j, i) in enumerate(v.dof_range)
        @inbounds v.Nx[i, q] = shape_value(v.values, q, j)
        v.dNdx === nothing || @inbounds v.dNdx[i, q] = shape_gradient(v.values, q, j)
    end
    return v
end

getnbasefunctions(v::FieldView) = size(v.Nx, 1)
getnquadpoints(v::FieldView) = size(v.Nx, 2)
shape_value_type(v::FieldView) = eltype(v.Nx)
shape_gradient_type(v::FieldView) = shape_gradient_type(v.values)
function_interpolation(v::FieldView) = function_interpolation(v.values)

@propagate_inbounds shape_value(v::FieldView, q_point::Int, i::Int) = v.Nx[i, q_point]
@propagate_inbounds shape_gradient(v::FieldView, q_point::Int, i::Int) = v.dNdx[i, q_point]

# Function evaluation with the dof values `u` of the whole element: only the field's dofs
# contribute, so evaluate with the field's values and dof range directly. (Restricted to
# scalar dof values to avoid ambiguities with the `AbstractVector{<:Vec}` methods.)
for f in (:function_value, :function_gradient)
    @eval function $f(v::FieldView, q_point::Int, u::AbstractVector{<:Number})
        length(u) == getnbasefunctions(v) || throw_incompatible_dof_length(length(u), getnbasefunctions(v))
        return $f(v.values, q_point, u, v.dof_range)
    end
end
function function_divergence(v::FieldView, q_point::Int, u::AbstractVector{<:Number})
    return divergence_from_gradient(function_gradient(v, q_point, u))
end
function function_curl(v::FieldView, q_point::Int, u::AbstractVector{<:Number})
    return curl_from_gradient(function_gradient(v, q_point, u))
end
