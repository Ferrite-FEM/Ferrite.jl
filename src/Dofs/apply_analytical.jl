function _geometric_interpolations(dh::DofHandler)
    sdhs = dh.subdofhandlers
    getcelltype(i) = typeof(getcells(get_grid(dh), first(sdhs[i].cellset)))
    return ntuple(i -> geometric_interpolation(getcelltype(i)), length(sdhs))
end

"""
    apply_analytical!(
        a::AbstractVector, dh::AbstractDofHandler, fieldname::Symbol,
        f::Function, cellset = 1:getncells(get_grid(dh))
    )

Apply a solution `f(x)` by modifying the values in the degree of freedom vector `a`
pertaining to the field `fieldname` for all cells in `cellset`.
The function `f(x)` are given the spatial coordinate
of the degree of freedom. For scalar fields, `f(x)::Number`,
for vector fields with dimension `dim`, `f(x)::Vec{dim}`, and for tensor-valued fields
with value type `TB`, `f(x)` should return a tensor convertible to `TB`
(e.g. `f(x)::SymmetricTensor{2, dim}`).

This function can be used to apply initial conditions for time dependent problems.

!!! note

    This function only works for standard nodal finite element interpolations
    when the function value at the (algebraic) node is equal to the corresponding
    degree of freedom value.
    This holds for e.g. Lagrange and Serendipity interpolations, including
    sub- and superparametric elements.
"""
function apply_analytical!(
        a::AbstractVector, dh::DofHandler, fieldname::Symbol, f::Function,
        cellset = 1:getncells(get_grid(dh))
    )

    fieldname ∉ getfieldnames(dh) && error("The fieldname $fieldname was not found in the dof handler")
    ip_geos = _geometric_interpolations(dh)

    for (sdh, ip_geo) in zip(dh.subdofhandlers, ip_geos)
        isnothing(_find_field(sdh, fieldname)) && continue
        field_idx = find_field(sdh, fieldname)
        ip_fun = getfieldinterpolation(sdh, field_idx)
        field_dim = n_components(sdh, field_idx)
        celldofinds = dof_range(sdh, fieldname)
        set_intersection = if length(cellset) == length(sdh.cellset) == getncells(get_grid(dh))
            BitSet(1:getncells(get_grid(dh)))
        else
            intersect(BitSet(sdh.cellset), BitSet(cellset))
        end
        isempty(set_intersection) && continue
        _apply_analytical!(a, dh, celldofinds, field_dim, ip_fun, ip_geo, f, set_intersection)
    end
    return a
end

function _apply_analytical!(
        a::AbstractVector, dh::AbstractDofHandler, celldofinds, field_dim,
        ip_fun::Interpolation{RefShape}, ip_geo::Interpolation, f::Function, cellset
    ) where {dim, RefShape <: AbstractRefShape{dim}}

    coords = getcoordinates(get_grid(dh), first(cellset))
    ref_points = reference_coordinates(ip_fun)
    dummy_weights = zeros(length(ref_points))
    qr = QuadratureRule{RefShape}(dummy_weights, ref_points)
    # Note: Passing ip_geo as the function interpolation here, it is just a dummy.
    cv = CellValues(qr, ip_geo, ip_geo)
    c_dofs = celldofs(dh, first(cellset))
    f_dofs = zeros(Int, length(celldofinds))

    for cellnr in cellset
        getcoordinates!(coords, get_grid(dh), cellnr)
        celldofs!(c_dofs, dh, cellnr)
        for (i, celldofind) in enumerate(celldofinds)
            f_dofs[i] = c_dofs[celldofind]
        end
        _apply_analytical!(a, f_dofs, coords, field_dim, cv, f, ip_fun)
    end
    return a
end

# The dof values (in the interpolation's component order) corresponding to the function
# value `fval`. For scalar/vector fields the value is a collection with one entry per
# component; for tensor-valued fields it is the tensor value, whose independent components
# (the data tuple of the value converted to the field's tensor type) are the dof values.
@inline function _analytical_dof_values(::Interpolation, fval, field_dim::Int)
    length(fval) == field_dim || error("length(f(x)) must be equal to dimension of the field ($field_dim)")
    return fval
end
@inline _analytical_dof_values(::TensorInterpolation{TB}, fval::SecondOrderTensor, ::Int) where {TB <: SecondOrderTensor} = TB(fval).data
function _analytical_dof_values(::TensorInterpolation{TB}, fval, ::Int) where {TB <: SecondOrderTensor}
    return error("f(x) for a tensor-valued field must return the tensor value (convertible to $(TB)), got $(typeof(fval))")
end

function _apply_analytical!(a::AbstractVector, dofs::Vector{Int}, coords::Vector{<:Vec}, field_dim, cv::CellValues, f, ip_fun::Interpolation)
    for i_dof in 1:getnquadpoints(cv)
        x_dof = spatial_coordinate(cv, i_dof, coords)
        for (idim, icval) in enumerate(_analytical_dof_values(ip_fun, f(x_dof), field_dim))
            a[dofs[field_dim * (i_dof - 1) + idim]] = icval
        end
    end
    return a
end
