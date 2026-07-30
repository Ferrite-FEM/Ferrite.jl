"""
    ConformityConstraint(field_name::Symbol)

This constraint can be passed to the constraint handler when working with non-conforming meshes to
add the affine constraints required to make the associated interpolation conforming. It applies to
the single field `field_name`; combine several `ConformityConstraint`s to constrain several fields
of a multi-field `DofHandler`.

The named field must be defined on every cell of the grid: subdomains (`SubDofHandler`s covering
only part of the grid, e.g. via `L2Projector(...; set = ...)`) are not supported yet and
throw an `ArgumentError`.

For a full example visit the [adaptive heat equation tutorial](@ref tutorial-heat-adaptivity).
"""
struct ConformityConstraint
    field_name::Symbol
end

# Redispatch on the grid type instead of constraining the `ConstraintHandler` type parameters.
function Ferrite.add!(ch::ConstraintHandler{<:DofHandler}, cc::ConformityConstraint)
    return _add_conformity_constraints!(ch, Ferrite.get_grid(ch.dh), cc)
end

function _add_conformity_constraints!(ch::ConstraintHandler, ::Ferrite.AbstractGrid, cc::ConformityConstraint)
    return @warn "Trying to add conformity constraint to $(cc.field_name) on a conforming grid. Skipping."
end

function _add_conformity_constraints!(ch::ConstraintHandler, grid::NonConformingGrid, cc::ConformityConstraint)
    dh = ch.dh
    cc.field_name ∈ dh.field_names || throw(ArgumentError("Field $(cc.field_name) not found in provided dof handler. Available fields are $(dh.field_names)."))
    # The entity maps (vertexdicts) are indexed by the *global* field position, so the
    # hanging-node lookup below must use the global field index. `find_field` returns the
    # SubDofHandler index and the field index *local* to that SubDofHandler, which we use only
    # to fetch the interpolation.
    global_fidx = findfirst(==(cc.field_name), dh.field_names)::Int
    sdh_idx, local_fidx = Ferrite.find_field(dh, cc.field_name)
    interpolation = dh.subdofhandlers[sdh_idx].field_interpolations[local_fidx]
    # The hanging/master node lookup assumes every vertex touched by the conformity relations
    # carries a dof of this field, which does not hold on subdomains (uncovered vertices map
    # to dof 0). Two ways to violate this: a cell belonging to no SubDofHandler at all
    # (`cell_to_subdofhandler == 0`), or the field itself living on only part of the grid.
    vertices = (dh.entitymaps::Ferrite.EntityMaps).vertices[global_fidx]
    if any(iszero, dh.cell_to_subdofhandler) || _has_uncovered_hanging_vertex(grid.conformity_info, vertices)
        throw(ArgumentError("ConformityConstraint requires the field :$(cc.field_name) on every cell of the non-conforming grid, but it covers only part of it. Subdomains (e.g. `L2Projector(...; set = ...)`) are not supported on non-conforming grids yet."))
    end
    # One set of linear constraints per hanging node
    _add_conformity_constraint(ch, global_fidx, interpolation)
    return
end

# A hanging vertex (or one of its masters) that does not carry a dof of the field maps to dof
# 0, which would produce an invalid `AffineConstraint(0, ...)`. Detect that up front.
function _has_uncovered_hanging_vertex(conformity_info, vertices::Vector{Int})
    for (hdof, mdof) in conformity_info
        iszero(vertices[hdof]) && return true
        any(m -> iszero(vertices[m]), mdof) && return true
    end
    return false
end

@noinline function _add_conformity_constraint(ch::ConstraintHandler, field_index::Int, interpolation::Interpolation)
    throw(ArgumentError("ConformityConstraint supports only linear Lagrange interpolations (and their vectorizations), got $interpolation."))
end

function _add_conformity_constraint(ch::ConstraintHandler, field_index::Int, interpolation::Lagrange{<:Any, 1})
    # Reached only for a NonConformingGrid, so the entity maps are guaranteed to be present.
    # type annotated for the compiler
    vertices = (ch.dh.entitymaps::Ferrite.EntityMaps).vertices[field_index]
    for (hdof, mdof) in ch.dh.grid.conformity_info
        # A hanging node is the average of its masters: an edge midpoint of its 2 endpoints
        # (weight 1/2), a 3D face center of its 4 face corners (weight 1/4).
        @debug @assert length(mdof) ∈ (2, 4)
        weight = 1 / length(mdof)
        lc = AffineConstraint(vertices[hdof], [vertices[m] => weight for m in mdof], 0.0)
        add!(ch, lc)
    end
    return
end

function _add_conformity_constraint(ch::ConstraintHandler, field_index::Int, interpolation::VectorizedInterpolation{vdim, <:Any, <:Any, <:Lagrange{<:Any, 1}}) where {vdim}
    # Reached only for a NonConformingGrid, so the entity maps are guaranteed to be present.
    vertices = (ch.dh.entitymaps::Ferrite.EntityMaps).vertices[field_index]
    for (hdof, mdof) in ch.dh.grid.conformity_info
        @debug @assert length(mdof) ∈ (2, 4)
        weight = 1 / length(mdof)
        # One constraint per component
        for vd in 1:vdim
            lc = AffineConstraint(vertices[hdof] + vd - 1, [vertices[m] + vd - 1 => weight for m in mdof], 0.0)
            add!(ch, lc)
        end
    end
    return
end
