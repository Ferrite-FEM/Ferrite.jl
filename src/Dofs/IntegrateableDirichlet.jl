@doc raw"""
    IntegrateableDirichlet(field_name::Symbol, facets, f::Function; qr_order = -1)

An `IntegrateableDirichlet` conditions enforces conditions for `field_name` on the boundary for
non-nodal interpolations (e.g. ``H(\mathrm{div})`` or ``H(\mathrm{curl})``) in an integral sense.

For ``H(\mathrm{div})`` interpolations, we have conditions on the form
```math
\boldsymbol{N}^f_i a^f_i \cdot \boldsymbol{n}^f = q_n = f(\boldsymbol{x}, t)
```
These are enforced as
```math
\int_{\Gamma^f} \boldsymbol{N}^f_\alpha a^f_\alpha \cdot \boldsymbol{n}^f\ \mathrm{d}\Gamma
= \int_{\Gamma^f} g_\alpha(\boldsymbol{x}) q_n\ \mathrm{d}\Gamma
```
(no sum on ``\alpha``) where the weighting functions, ``g_\alpha(\boldsymbol{x})``, are defined such that
``\sum_{\alpha} g_\alpha(\boldsymbol{x}) = 1`` at all points on the facet.
These weighting functions are defined by each interpolation.

For ``H(\mathrm{curl})`` interpolations, the conditions are on the form
```math
\boldsymbol{N}^f_i a^f_i \times \boldsymbol{n}^f = \boldsymbol{q}_t = \boldsymbol{f}(\boldsymbol{x}, t)
```
These are similarly enforced as
```math
\int_{\Gamma^f} \boldsymbol{N}^f_\alpha a^f_\alpha \times \boldsymbol{n}^f\ \mathrm{d}\Gamma
= \int_{\Gamma^f} g_\alpha(\boldsymbol{x}) \boldsymbol{q}_t\ \mathrm{d}\Gamma
```
(no sum on ``\alpha``) with equivalent weighting functions as for ``H(\mathrm{div})`` interpolations.
"""
mutable struct IntegrateableDirichlet
    const f::Function
    const facets::OrderedSet{FacetIndex}
    const field_name::Symbol
    const qr_order::Int
    # Created during `add!`
    fv::Union{Nothing, FacetValues}
    facet_dofs::Union{Nothing, ArrayOfVectorViews{Int, 1}}
    ip::Union{Nothing, Interpolation}
end
function IntegrateableDirichlet(field_name::Symbol, facets::AbstractVecOrSet, f::Function; qr_order = -1)
    return IntegrateableDirichlet(f, convert_to_orderedset(facets), field_name, qr_order, nothing, nothing)
end

function _default_bc_qr_order(user_provided::Int, ip::Interpolation)
    user_provided > 0 && return user_provided
    return _default_bc_qr_order(ip)
end
# Q&D default, should be more elaborated
_default_bc_qr_order(::Interpolation{<:Any, order}) where {order} = order

function add!(ch::ConstraintHandler, dbc::IntegrateableDirichlet)
    # Duplicate the Dirichlet constraint for every SubDofHandler
    dbc_added = false
    for sdh in ch.dh.subdofhandlers
        # Skip if the constrained field does not live on this sub domain
        dbc.field_name in sdh.field_names || continue
        # Compute the intersection between dbc.set and the cellset of this
        # SubDofHandler and skip if the set is empty
        filtered_set = filter_dbc_set(get_grid(ch.dh), sdh.cellset, dbc.facets)
        isempty(filtered_set) && continue
        # Fetch information about the field on this SubDofHandler
        field_idx = find_field(sdh, dbc.field_name)
        interpolation = getfieldinterpolation(sdh, field_idx)
        CT = getcelltype(sdh) # Same celltype enforced in SubDofHandler constructor
        qr_order = _default_qr_order(dbc.qr_order, interpolation)
        fqr = FacetQuadratureRule{getrefshape(interpolation)}(qr_order)
        fv = FacetValues(fqr, interpolation, geometric_interpolation(CT))
        local_facet_dofs, local_facet_dofs_offset =
            _local_facet_dofs_for_bc(interpolation, 1, 1, field_offset(sdh, field_idx), dirichlet_facetdof_indices)
        facet_dofs = ArrayOfVectorViews(local_facet_dofs, local_facet_dofs_offset, LinearIndices(1:(length(local_facet_dofs_offset) - 1)))

        filtered_dbc = IntegrateableDirichlet(dbc.f, filtered_set, dbc.field_name, fv, facet_dofs)

        _add!(ch, filtered_dbc, facet_dofs)

        dbc_added = true
    end
    dbc_added || error("No overlap between dbc::Dirichlet and fields in the ConstraintHandler's DofHandler")
    return ch
end

function _add!(ch::ConstraintHandler, dbc::IntegrateableDirichlet, facet_dofs)
    # loop over all the faces in the set and add the global dofs to `constrained_dofs`
    constrained_dofs = Int[]
    cc = CellCache(ch.dh, UpdateFlags(; nodes = false, coords = false, dofs = true))
    for (cellidx, facetidx) in dbc.facets
        reinit!(cc, cellidx)
        local_dofs = facet_dofs[facetidx]
        for d in local_dofs
            push!(constrained_dofs, cc.dofs[d])
        end
    end

    # save it to the ConstraintHandler
    push!(ch.idbcs, dbc)
    for d in constrained_dofs
        add_prescribed_dof!(ch, d, NaN, nothing)
    end
    return ch
end

_update!(ch.inhomogeneities, dbc.f, dbc.facets, dbc.ip, dbc.facet_dofs, ch.dh, ch.dofmapping, ch.dofcoefficients, time)

function _update!(
        inhomogeneities::Vector{T}, f::Function, facets::AbstractVecOrSet{FacetIndex}, fv::FacetValues, facet_dofs::ArrayOfVectorViews,
        dh::AbstractDofHandler, dofmapping::Dict{Int, Int}, dofcoefficients::Vector{Union{Nothing, DofCoefficients{T}}}, time::Real
    ) where {T}

    for fc in FacetIterator(dh, facets)
        reinit!(fv, fc)

        local_dofs = facet_dofs[getcurrentfacet(fv)]
        # local dof-range for this facet
        r = local_facet_dofs_offset[entityidx]:(local_facet_dofs_offset[entityidx + 1] - 1)
        counter = 1
        for location in 1:getnquadpoints(boundaryvalues)
            sign = if mapping_type(ip) isa IdentityMapping
                1
            else
                cell = getcells(cc.grid, cellidx)
                shape_number = local_facet_dofs[r[counter]]
                get_direction(ip, shape_number, cell)
            end
            x = spatial_coordinate(boundaryvalues, location, cc.coords)
            bc_value = f(x, time)
            @assert length(bc_value) == length(components)

            for i in 1:length(components)
                # find the global dof
                globaldof = cc.dofs[local_facet_dofs[r[counter]]]
                counter += 1

                dbc_index = dofmapping[globaldof]
                # Only DBC dofs are currently update!-able so don't modify inhomogeneities
                # for affine constraints
                if dofcoefficients[dbc_index] === nothing
                    inhomogeneities[dbc_index] = sign * bc_value[i]
                    @debug println("prescribing value $(bc_value[i]) on global dof $(globaldof)")
                end
            end
        end
    end
    return
end
