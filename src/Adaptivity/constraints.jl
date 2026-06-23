"""
    This constraint can be passed to the constraint handler when working with non-conforming meshes to
    add the affine constraints required to make the associated interpolation conforming.

    For a full example visit the AMR tutorial.
"""
struct ConformityConstraint
    field_name::Symbol
end

function Ferrite.add!(ch::ConstraintHandler{<:DofHandler{<:Any, <:Grid}}, cc::ConformityConstraint)
    return @warn "Trying to add conformity constraint to $(cc.field_name) on a conforming grid. Skipping."
end

function Ferrite.add!(ch::ConstraintHandler{<:DofHandler{<:Any, <:NonConformingGrid}}, cc::ConformityConstraint)
    @assert length(ch.dh.field_names) == 1 "Multiple fields not supported yet."
    @assert cc.field_name ∈ ch.dh.field_names "Field $(cc.field_name) not found in provided dof handler. Available fields are $(ch.dh.field_names)."
    # One set of linear constraints per hanging node
    for sdh in ch.dh.subdofhandlers
        field_idx = Ferrite._find_field(sdh, cc.field_name)
        field_idx !== nothing && _add_conformity_constraint(ch, field_idx, sdh.field_interpolations[field_idx])
    end
    return
end

function _add_conformity_constraint(ch::ConstraintHandler, field_index::Int, interpolation::ScalarInterpolation)
    # Reached only for a NonConformingGrid, so the entity maps are guaranteed to be present.
    # type annotated for the compiler
    vertices = (ch.dh.entitymaps::Ferrite.EntityMaps).vertices[field_index]
    for (hdof, mdof) in ch.dh.grid.conformity_info
        # A hanging node is the average of its masters: an edge midpoint of its 2 endpoints
        # (weight 1/2), a 3D face centre of its 4 face corners (weight 1/4).
        @debug @assert length(mdof) ∈ (2, 4)
        weight = 1 / length(mdof)
        lc = AffineConstraint(vertices[hdof], [vertices[m] => weight for m in mdof], 0.0)
        add!(ch, lc)
    end
    return
end

function _add_conformity_constraint(ch::ConstraintHandler, field_index::Int, interpolation::VectorizedInterpolation{vdim}) where {vdim}
    # Reached only for a NonConformingGrid, so the entity maps are guaranteed to be present.
    vertices = (ch.dh.entitymaps::Ferrite.EntityMaps).vertices[field_index]
    for (hdof, mdof) in ch.dh.grid.conformity_info
        @debug @assert length(mdof) ∈ (2, 4)
        weight = 1 / length(mdof)
        # One constraint per component
        for vd in 1:vdim
            lc = AffineConstraint(vertices[hdof] + vd - 1, [vertices[m] + vd - 1 => weight for m in mdof], 0.0) # TODO change for other interpolation types than linear
            add!(ch, lc)
        end
    end
    return
end
