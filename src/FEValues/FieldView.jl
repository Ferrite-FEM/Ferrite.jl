# Prototype: "extractor" views (cf. deal.II's FEValuesExtractors) for the values of one field,
# with the shape functions indexed by the dofs of the whole element. The shape functions of
# the other fields' dofs are zero, so that e.g. the element matrix of a coupled problem can
# be assembled in a single loop over all the element dofs:
#
#     vu = FieldView(cv, dh, :u)
#     vp = FieldView(cv, dh, :p)
#     for i in 1:ndofs_per_cell(dh), j in 1:ndofs_per_cell(dh)
#         Ke[i, j] += (shape_symmetric_gradient(vu, q, i) ⊡ shape_symmetric_gradient(vu, q, j) -
#                      shape_divergence(vu, q, i) * shape_value(vp, q, j) - ...) * dΩ
#     end

struct FieldView{FV <: AbstractValues} <: AbstractValues
    values::FV                 # the values of the field, e.g. `cv.u` or a single-field `CellValues`
    dof_range::UnitRange{Int}  # the dofs of the field in the element
    ndofs::Int                 # the number of dofs of the element
    function FieldView(values::FV, dof_range::AbstractUnitRange{<:Integer}, ndofs::Integer) where {FV <: AbstractValues}
        if length(dof_range) != getnbasefunctions(values)
            throw(ArgumentError("the length of the dof range ($(length(dof_range))) does not match the number of base functions of the field ($(getnbasefunctions(values)))"))
        end
        if !isempty(dof_range) && (first(dof_range) < 1 || last(dof_range) > ndofs)
            throw(ArgumentError("the dof range $(dof_range) is not within the dofs of the element (1:$(ndofs))"))
        end
        return new{FV}(values, UnitRange{Int}(dof_range), Int(ndofs))
    end
end

# The field `field` of a multi-field CellValues, with the dof range and the number of dofs
# per cell from the (Sub)DofHandler
function FieldView(cv::MultiFieldCellValues, dh::Union{DofHandler, SubDofHandler}, field::Symbol)
    return FieldView(getproperty(cv, field), dof_range(dh, field), ndofs_per_cell(dh))
end

getnbasefunctions(v::FieldView) = v.ndofs
getnquadpoints(v::FieldView) = getnquadpoints(v.values)
shape_value_type(v::FieldView) = shape_value_type(v.values)
shape_gradient_type(v::FieldView) = shape_gradient_type(v.values)
function_interpolation(v::FieldView) = function_interpolation(v.values)

# The base function of the field corresponding to dof `i` of the element, or 0 if dof `i`
# belongs to another field
@inline function field_base_function(v::FieldView, i::Int)
    @boundscheck (1 <= i <= v.ndofs) || throw(BoundsError(1:v.ndofs, i))
    j = i - first(v.dof_range) + 1
    return (1 <= j <= length(v.dof_range)) ? j : 0
end

@propagate_inbounds function shape_value(v::FieldView, q_point::Int, i::Int)
    j = field_base_function(v, i)
    return j == 0 ? zero(shape_value_type(v.values)) : shape_value(v.values, q_point, j)
end
@propagate_inbounds function shape_gradient(v::FieldView, q_point::Int, i::Int)
    j = field_base_function(v, i)
    return j == 0 ? zero(shape_gradient_type(v.values)) : shape_gradient(v.values, q_point, j)
end

# Function evaluation with the dof values `u` of the whole element: only the field's dofs
# contribute, so evaluate with the field's values and dof range directly. (Restricted to
# scalar dof values to avoid ambiguities with the `AbstractVector{<:Vec}` methods.)
for f in (:function_value, :function_gradient)
    @eval function $f(v::FieldView, q_point::Int, u::AbstractVector{<:Number})
        length(u) == v.ndofs || throw_incompatible_dof_length(length(u), v.ndofs)
        return $f(v.values, q_point, u, v.dof_range)
    end
end
function function_divergence(v::FieldView, q_point::Int, u::AbstractVector{<:Number})
    return divergence_from_gradient(function_gradient(v, q_point, u))
end
function function_curl(v::FieldView, q_point::Int, u::AbstractVector{<:Number})
    return curl_from_gradient(function_gradient(v, q_point, u))
end
