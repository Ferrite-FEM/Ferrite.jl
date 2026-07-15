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
    @assert cc.field_name ∈ ch.dh.field_names "Field $(cc.field_name) not found in provided dof handler. Available fields are $(ch.dh.field_names)."
    # The constraint kernel reads dofs of both interface sides through one SubDofHandler,
    # so all cells must live in the same one (no subdomain support yet).
    @assert length(ch.dh.subdofhandlers) == 1 "Multiple subdomains not supported yet."
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
#    evaluation points are dyadic). Non-nodal interpolations (Nedelec, RaviartThomas)
#    apply their pullback-invariant dof functionals instead (see further below).
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
    _interface_params(cells, refc, ccell, cverts, fcells, fverts_all) -> (par, cframe, fglob)

Shared geometry derivation of one hanging-interface record: assign coarse-entity
parameters to the interface's node ids — the coarse corners from the record, the hanging
center as the node common to all fine entities, (3D) edge midpoints as the nodes shared by
exactly two fine entities. `par` maps node id → coarse entity param, `cframe` is the
coarse entity's affine frame in cell reference coordinates, `fglob` the fine entities'
corner node ids. Everything is derived from node ids — child position *and* relative
orientation — never assumed.
"""
function _interface_params(cells, refc, ccell::Int, cverts::NTuple{NC, Int}, fcells::Vector{Int}, fverts_all::Vector{<:NTuple{NC, Int}}) where {NC}
    P = _entity_corner_params(Val(NC))
    PD = NC == 2 ? 1 : 2

    cglob = ntuple(i -> Ferrite.get_node_ids(cells[ccell])[cverts[i]], Val(NC))
    fglob = [ntuple(i -> Ferrite.get_node_ids(cells[fcells[q]])[fverts_all[q][i]], Val(NC)) for q in eachindex(fcells)]
    fsets = [Set(f) for f in fglob]

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
    return par, _entity_frame(refc, cverts), fglob
end

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
    par, cframe, fglob = _interface_params(cells, refc, ccell, cverts, fcells, fverts_all)
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
# Non-nodal interpolations: H(curl) (Nedelec) and H(div) (RaviartThomas), lowest order.
#
# The dof functionals are pullback-invariant under the Piola mappings — an edge dof is the
# circulation ∫ u·dx along the edge, a facet dof the flux ∫ u·n ds through the facet, both
# unchanged whether evaluated physically, in the fine cell's reference frame, or in the
# coarse cell's. So the constraint weights are computed exactly in the *coarse* reference
# frame: map the fine sub-entity through the node-id-derived embedding and apply its
# functional to the coarse basis. Ferrite's global sign convention enters through
# `get_direction` (coarse side) and the ascending-node-id orientation of the fine entity.
# The lowest-order traces are constant/linear along the sub-entities, so a midpoint rule
# is exact.
# ---------------------------------------------------------------------------------------

function _add_conformity_constraints!(ch::ConstraintHandler, sdh, fname::Symbol, ip::Nedelec)
    ip isa Nedelec{<:Any, 1} || error("Conformity constraints for Nedelec are only implemented for order 1.")
    return _add_piola_conformity!(ch, sdh, fname, ip)
end
function _add_conformity_constraints!(ch::ConstraintHandler, sdh, fname::Symbol, ip::RaviartThomas)
    ip isa RaviartThomas{<:Any, 1} || error("Conformity constraints for RaviartThomas are only implemented for order 1.")
    return _add_piola_conformity!(ch, sdh, fname, ip)
end

function _add_piola_conformity!(ch::ConstraintHandler, sdh, fname::Symbol, ip::Ferrite.VectorInterpolation)
    dh = ch.dh
    ci = dh.grid.conformity_info
    ci isa ConformityInfo || error("The grid's conformity_info is not a ConformityInfo — was the grid produced by creategrid?")
    ci.complete || error(
        "The hanging-interface records are incomplete: the forest had level jumps > 1 across a tree " *
            "interface. Call balanceforest! before creategrid to use conformity constraints."
    )
    rng = Ferrite.dof_range(sdh, fname)
    refshape = Ferrite.getrefshape(ip)
    cells = dh.grid.cells
    refc = Ferrite.reference_coordinates(Lagrange{refshape, 1}())
    rfacets = Ferrite.reference_facets(refshape)
    fdofs = Ferrite.facetdof_indices(ip)
    hcurl = Ferrite.conformity(ip) isa Ferrite.HcurlConformity
    seen = Set{Int}()
    for rec in ci.hanging_facets
        cc, clf = rec.coarse[1], rec.coarse[2]
        fcells = [f[1] for f in rec.fine]
        fverts = [rfacets[f[2]] for f in rec.fine]
        par, cframe, _ = _interface_params(cells, refc, cc, rfacets[clf], fcells, fverts)
        cdofs_coarse = Ferrite.celldofs(dh, cc)[rng]
        for (q, f) in enumerate(rec.fine)
            if refshape <: Ferrite.AbstractRefShape{2}
                # 2D facet = edge: H(curl) constrains its circulation, H(div) its flux
                if hcurl
                    _constrain_curl_edge!(ch, dh, rng, ip, cells, par, cframe, cc, fdofs[clf], cdofs_coarse, f[1], f[2], seen)
                else
                    _constrain_div_facet!(ch, dh, rng, ip, cells, par, cframe, cc, fdofs[clf], cdofs_coarse, f[1], f[2], seen)
                end
            else
                if hcurl
                    # every edge of the fine sub-facet lies on the coarse facet and hangs
                    for le in Ferrite.reference_face_edgenrs(refshape)[f[2]]
                        _constrain_curl_edge!(ch, dh, rng, ip, cells, par, cframe, cc, fdofs[clf], cdofs_coarse, f[1], le, seen)
                    end
                else
                    _constrain_div_facet!(ch, dh, rng, ip, cells, par, cframe, cc, fdofs[clf], cdofs_coarse, f[1], f[2], seen)
                end
            end
        end
    end
    if refshape <: Ferrite.AbstractRefShape{3} && hcurl
        redges = Ferrite.reference_edges(refshape)
        edofs = Ferrite.edgedof_indices(ip)
        for rec in ci.hanging_edges
            cc, cle = rec.coarse[1], rec.coarse[2]
            fcells = [f[1] for f in rec.fine]
            fverts = [redges[f[2]] for f in rec.fine]
            par, cframe, _ = _interface_params(cells, refc, cc, redges[cle], fcells, fverts)
            cdofs_coarse = Ferrite.celldofs(dh, cc)[rng]
            for f in rec.fine
                _constrain_curl_edge!(ch, dh, rng, ip, cells, par, cframe, cc, edofs[cle], cdofs_coarse, f[1], f[2], seen)
            end
        end
    end
    return
end

"""
    _constrain_curl_edge!(ch, dh, rng, ip, cells, par, cframe, ccell, cdofids, cdofs_coarse, fcell, ledge, seen)

Constrain the H(curl) edge dof of fine edge `(fcell, ledge)` lying on a hanging interface:
its value is the circulation of the coarse trace along the sub-edge, taken from lower to
higher global node id (Ferrite's global dof orientation). The segment is mapped into the
coarse reference frame via `par` (pullback-invariant), and the weight of coarse dof `j`
is `get_direction(ip, j, coarse) · N̂ⱼ(mid) ⋅ Δξ` — midpoint rule, exact for the
lowest-order tangential traces.
"""
function _constrain_curl_edge!(ch::ConstraintHandler, dh, rng, ip, cells, par, cframe, ccell::Int, cdofids, cdofs_coarse, fcell::Int, ledge::Int, seen::Set{Int})
    refshape = Ferrite.getrefshape(ip)
    ldofs = Ferrite.edgedof_interior_indices(ip)[ledge]
    isempty(ldofs) && return
    gdof = Ferrite.celldofs(dh, fcell)[rng][ldofs[1]]
    gdof ∈ seen && return
    enodes = Ferrite.reference_edges(refshape)[ledge]
    ns = Ferrite.get_node_ids(cells[fcell])
    g1 = ns[enodes[1]]; g2 = ns[enodes[2]]
    g2 < g1 && ((g1, g2) = (g2, g1))                     # ascending: the global dof direction
    ξ1 = _frame_point(cframe, par[g1])
    ξ2 = _frame_point(cframe, par[g2])
    ξm = (ξ1 + ξ2) / 2
    Δ = ξ2 - ξ1
    masters = Pair{Int, Float64}[]
    ccell_ = cells[ccell]
    for j in cdofids
        w = Ferrite.get_direction(ip, j, ccell_) * (Ferrite.reference_shape_value(ip, ξm, j) ⋅ Δ)
        iszero(w) && continue
        push!(masters, cdofs_coarse[j] => w)
    end
    push!(seen, gdof)
    add!(ch, AffineConstraint(gdof, masters, 0.0))
    return
end

"""
    _constrain_div_facet!(ch, dh, rng, ip, cells, par, cframe, ccell, cdofids, cdofs_coarse, fcell, lfacet, seen)

Constrain the H(div) facet dof of fine facet `(fcell, lfacet)` covering part of a hanging
interface: its value is the flux of the coarse trace through the sub-facet with Ferrite's
global orientation — 2D: normal is the ascending tangent rotated by -90°; 3D: the
reference corner cycle's right-hand normal times `get_face_direction`. The sub-facet is
mapped into the coarse reference frame via `par` (flux is pullback-invariant under the
contravariant Piola mapping); midpoint rule, exact for the constant lowest-order normal
traces.
"""
function _constrain_div_facet!(ch::ConstraintHandler, dh, rng, ip, cells, par, cframe, ccell::Int, cdofids, cdofs_coarse, fcell::Int, lfacet::Int, seen::Set{Int})
    refshape = Ferrite.getrefshape(ip)
    ldofs = Ferrite.facetdof_interior_indices(ip)[lfacet]
    isempty(ldofs) && return
    gdof = Ferrite.celldofs(dh, fcell)[rng][ldofs[1]]
    gdof ∈ seen && return
    fnodes = Ferrite.reference_facets(refshape)[lfacet]
    ns = Ferrite.get_node_ids(cells[fcell])
    local nA, ξm
    if refshape <: Ferrite.AbstractRefShape{2}
        g1 = ns[fnodes[1]]; g2 = ns[fnodes[2]]
        g2 < g1 && ((g1, g2) = (g2, g1))                 # ascending tangent
        ξ1 = _frame_point(cframe, par[g1])
        ξ2 = _frame_point(cframe, par[g2])
        ξm = (ξ1 + ξ2) / 2
        Δ = ξ2 - ξ1
        nA = Vec(Δ[2], -Δ[1])                            # n ds: ascending tangent rotated by -90°
    else
        ξ = ntuple(i -> _frame_point(cframe, par[ns[fnodes[i]]]), 4)
        ξm = (ξ[1] + ξ[2] + ξ[3] + ξ[4]) / 4
        A = Tensors.cross(ξ[3] - ξ[1], ξ[4] - ξ[2]) / 2  # cycle-oriented area vector
        d = Ferrite.get_face_direction(map(i -> ns[i], fnodes))
        nA = d * A
    end
    masters = Pair{Int, Float64}[]
    ccell_ = cells[ccell]
    for j in cdofids
        w = Ferrite.get_direction(ip, j, ccell_) * (Ferrite.reference_shape_value(ip, ξm, j) ⋅ nA)
        iszero(w) && continue
        push!(masters, cdofs_coarse[j] => w)
    end
    push!(seen, gdof)
    add!(ch, AffineConstraint(gdof, masters, 0.0))
    return
end
