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
    for sdh in ch.dh.subdofhandlers
        field_idx = Ferrite._find_field(sdh, cc.field_name)
        field_idx !== nothing && _add_conformity_constraints!(ch, sdh, cc.field_name, sdh.field_interpolations[field_idx])
    end
    return
end

# ---------------------------------------------------------------------------------------
# Generic hanging-interface constraints
#
# The `ConformityInfo` records of the grid are purely topological ((cell, local entity)
# pairs); everything dof-related is derived here, per interpolation:
#  - the interface dof set comes from the interpolation's entity dof layout
#    (`facetdof_indices`/`edgedof_indices`),
#  - the child embedding and relative orientation of each fine sub-entity are derived from
#    the shared corner node ids (each fine entity shares exactly one corner node with the
#    coarse entity, and all fine entities share the hanging center node),
#  - the weights are the coarse entity's trace shape functions evaluated at the fine dofs'
#    reference positions — for nodal interpolations this is fully generic and exact (all
#    evaluation points are dyadic).
# Non-nodal interpolations (Nedelec, RaviartThomas) need their own trace interpolation
# rule and are not supported yet.
# ---------------------------------------------------------------------------------------

_add_conformity_constraints!(ch::ConstraintHandler, sdh, fname::Symbol, ip::ScalarInterpolation) =
    _add_conformity_constraints!(ch, sdh, fname, ip, 1)
_add_conformity_constraints!(ch::ConstraintHandler, sdh, fname::Symbol, vip::VectorizedInterpolation{vdim}) where {vdim} =
    _add_conformity_constraints!(ch, sdh, fname, vip.ip, vdim)

function _add_conformity_constraints!(ch::ConstraintHandler, sdh, fname::Symbol, ip::ScalarInterpolation, vdim::Int)
    dh = ch.dh
    ci = dh.grid.conformity_info
    ci isa ConformityInfo || error("The grid's conformity_info is not a ConformityInfo — was the grid produced by creategrid?")
    ci.complete || error(
        "The hanging-interface records are incomplete: the forest had level jumps > 1 across a tree " *
            "interface. Call balanceforest! before creategrid to use conformity constraints."
    )
    rng = Ferrite.dof_range(sdh, fname)
    refshape = Ferrite.getrefshape(ip)
    rfacets = Ferrite.reference_facets(refshape)
    fdofs = Ferrite.facetdof_indices(ip)
    # A dof can be reachable from several records (e.g. a hanging edge shared by two
    # hanging faces); the constraints coincide, so the first one wins.
    seen = Set{Int}()
    for rec in ci.hanging_facets
        cc, clf = rec.coarse[1], rec.coarse[2]
        _constrain_hanging_entity!(
            ch, dh, rng, ip, vdim,
            cc, rfacets[clf], fdofs[clf],
            [f[1] for f in rec.fine], [rfacets[f[2]] for f in rec.fine], [fdofs[f[2]] for f in rec.fine],
            seen
        )
    end
    if refshape <: Ferrite.AbstractRefShape{3}
        redges = Ferrite.reference_edges(refshape)
        edofs = Ferrite.edgedof_indices(ip)
        for rec in ci.hanging_edges
            cc, cle = rec.coarse[1], rec.coarse[2]
            _constrain_hanging_entity!(
                ch, dh, rng, ip, vdim,
                cc, redges[cle], edofs[cle],
                [f[1] for f in rec.fine], [redges[f[2]] for f in rec.fine], [edofs[f[2]] for f in rec.fine],
                seen
            )
        end
    end
    return
end

# Canonical parameters of an entity's corners: an edge (2 corners) is parametrized by
# t ∈ [-1, 1], a quadrilateral facet (4 corners, cyclic order) by (t₁, t₂) ∈ [-1, 1]².
@inline _entity_corner_params(::Val{2}) = ((-1.0,), (1.0,))
@inline _entity_corner_params(::Val{4}) = ((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0))

# Affine frame of the entity spanned by the reference coordinates of its corner vertices:
# midpoint and half-axes, mapping canonical params to cell reference coordinates. Exact for
# the flat, axis-aligned entities of the hypercube reference shapes.
function _entity_frame(refc, verts::NTuple{2, Int})
    ξ1 = refc[verts[1]]; ξ2 = refc[verts[2]]
    return ((ξ1 + ξ2) / 2, ((ξ2 - ξ1) / 2,))
end
function _entity_frame(refc, verts::NTuple{4, Int})
    ξ1 = refc[verts[1]]; ξ2 = refc[verts[2]]; ξ3 = refc[verts[3]]; ξ4 = refc[verts[4]]
    @debug @assert ξ1 + ξ3 ≈ ξ2 + ξ4 # parallelogram (flat facet)
    return ((ξ1 + ξ3) / 2, ((ξ2 - ξ1) / 2, (ξ4 - ξ1) / 2))
end

@inline _frame_point(fr, t::NTuple{1}) = fr[1] + t[1] * fr[2][1]
@inline _frame_point(fr, t::NTuple{2}) = fr[1] + t[1] * fr[2][1] + t[2] * fr[2][2]
@inline _frame_param(fr, x) = map(a -> ((x - fr[1]) ⋅ a) / (a ⋅ a), fr[2])

"""
    _constrain_hanging_entity!(ch, dh, rng, ip, vdim, ccell, cverts, cdofids, fcells, fverts, fdofids, seen)

Constrain every dof of interpolation `ip` on the fine sub-entities `(fcells, fverts)` of one
hanging interface against the trace of the coarse entity `(ccell, cverts)`:

1. assign coarse-entity parameters to the interface's node ids — corners from the record,
   the hanging center as the node common to all fine entities, (3D) edge midpoints as the
   nodes shared by exactly two fine entities;
2. per fine entity, the corner params define the affine child embedding (position *and*
   relative orientation — derived, never assumed);
3. per fine dof, map its reference coordinate through the embedding and evaluate the coarse
   shape functions there: those are the constraint weights.

Fine dofs that *are* coarse dofs (shared corner nodes) are conforming and skipped; zero
weights are dropped (exact — all evaluation points are dyadic).
"""
function _constrain_hanging_entity!(
        ch::ConstraintHandler, dh, rng, ip::ScalarInterpolation, vdim::Int,
        ccell::Int, cverts::NTuple{NC, Int}, cdofids,
        fcells::Vector{Int}, fverts_all::Vector{<:NTuple{NC, Int}}, fdofids_all::Vector,
        seen::Set{Int}
    ) where {NC}
    grid = dh.grid
    cells = grid.cells
    refshape = Ferrite.getrefshape(ip)
    refc = Ferrite.reference_coordinates(Lagrange{refshape, 1}())
    ipcoords = Ferrite.reference_coordinates(ip)
    P = _entity_corner_params(Val(NC))
    PD = NC == 2 ? 1 : 2

    cglob = ntuple(i -> Ferrite.get_node_ids(cells[ccell])[cverts[i]], Val(NC))
    fglob = [ntuple(i -> Ferrite.get_node_ids(cells[fcells[q]])[fverts_all[q][i]], Val(NC)) for q in eachindex(fcells)]
    fsets = [Set(f) for f in fglob]

    # node id -> coarse entity param
    par = Dict{Int, NTuple{PD, Float64}}()
    for i in 1:NC
        par[cglob[i]] = P[i]
    end
    center = only(intersect(fsets...))                       # the hanging center node
    par[center] = ntuple(_ -> 0.0, PD)
    shared = [only(intersect(fsets[q], cglob)) for q in eachindex(fglob)]  # one coarse corner per fine entity
    if NC == 4
        for q1 in eachindex(fglob), q2 in (q1 + 1):length(fglob)
            s = setdiff(intersect(fsets[q1], fsets[q2]), center)
            length(s) == 1 || continue                       # adjacent quadrants share an edge midpoint
            par[only(s)] = (par[shared[q1]] .+ par[shared[q2]]) ./ 2
        end
    end

    cframe = _entity_frame(refc, cverts)
    cdofs_coarse = Ferrite.celldofs(dh, ccell)[rng]

    for q in eachindex(fcells)
        T = ntuple(i -> par[fglob[q][i]], Val(NC))           # fine corners in coarse params
        fframe = _entity_frame(refc, fverts_all[q])
        cdofs_fine = Ferrite.celldofs(dh, fcells[q])[rng]
        Tm = NC == 2 ? (T[1] .+ T[2]) ./ 2 : (T[1] .+ T[3]) ./ 2
        Ta = (T[2] .- T[1]) ./ 2
        Tb = NC == 4 ? (T[4] .- T[1]) ./ 2 : Ta
        for l in fdofids_all[q]
            s = _frame_param(fframe, ipcoords[l])
            t = NC == 2 ? Tm .+ s[1] .* Ta : Tm .+ s[1] .* Ta .+ s[2] .* Tb
            ξc = _frame_point(cframe, t)
            for c in 1:vdim
                gf = cdofs_fine[(l - 1) * vdim + c]
                gf ∈ seen && continue
                masters = Pair{Int, Float64}[]
                isident = false
                for j in cdofids
                    w = Ferrite.reference_shape_value(ip, ξc, j)
                    iszero(w) && continue
                    gm = cdofs_coarse[(j - 1) * vdim + c]
                    if gm == gf                              # fine dof is a coarse dof: conforming
                        isident = true
                        break
                    end
                    push!(masters, gm => w)
                end
                isident && continue
                push!(seen, gf)
                add!(ch, AffineConstraint(gf, masters, 0.0))
            end
        end
    end
    return
end

# ---------------------------------------------------------------------------------------
# Legacy node-level path (Q1 only, hardcoded weights). Kept during the migration for the
# parity tests against the generic builder above; scheduled for removal.
# ---------------------------------------------------------------------------------------

function _add_conformity_constraint(ch::ConstraintHandler, field_index::Int, interpolation::ScalarInterpolation)
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

function _add_conformity_constraint(ch::ConstraintHandler, field_index::Int, interpolation::VectorizedInterpolation{vdim}) where {vdim}
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
