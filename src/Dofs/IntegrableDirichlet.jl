@doc raw"""
    IntegrableDirichlet(field_name::Symbol, facets, f::Function; qr_order = -1)

An `IntegrableDirichlet` conditions enforces conditions for `field_name` on the boundary for
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
a^f_\alpha =
\frac{\int_{\Gamma^f} g_\alpha(\boldsymbol{x}) \boldsymbol{q}_t\ \mathrm{d}\Gamma}{
\int_{\Gamma^f} \boldsymbol{N}^f_\alpha \times \boldsymbol{n}^f\ \mathrm{d}\Gamma}
```
(no sum on ``\alpha``) with equivalent weighting functions as for ``H(\mathrm{div})`` interpolations.
"""
mutable struct IntegrableDirichlet
    const f::Function
    const facets::OrderedSet{FacetIndex}
    const field_name::Symbol
    const qr_order::Int
    # Created during `add!`
    fv::Union{Nothing, FacetValues}
    facet_dofs::Union{Nothing, ArrayOfVectorViews{Int, 1}}
    ip::Union{Nothing, Interpolation}
end
function IntegrableDirichlet(field_name::Symbol, facets::AbstractVecOrSet, f::Function; qr_order = -1)
    return IntegrableDirichlet(f, convert_to_orderedset(facets), field_name, qr_order, nothing, nothing)
end

function _default_bc_qr_order(user_provided::Int, ip::Interpolation)
    user_provided > 0 && return user_provided
    return _default_bc_qr_order(ip)
end
# Q&D default, should be more elaborated
_default_bc_qr_order(::Interpolation{<:Any, order}) where {order} = order

function add!(ch::ConstraintHandler, dbc::IntegrableDirichlet)
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

        filtered_dbc = IntegrableDirichlet(dbc.f, filtered_set, dbc.field_name, fv, facet_dofs)

        _add!(ch, filtered_dbc, facet_dofs)

        dbc_added = true
    end
    dbc_added || error("No overlap between dbc::Dirichlet and fields in the ConstraintHandler's DofHandler")
    return ch
end

function _add!(ch::ConstraintHandler, dbc::IntegrableDirichlet, facet_dofs)
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
    ip = function_interpolation(fv)
    for fc in FacetIterator(dh, facets)
        reinit!(fv, fc)
        local_dofs = facet_dofs[getcurrentfacet(fv)]
        for (idof, shape_nr) in enumerate(dirichlet_facetdof_indices(ip)[getcurrentfacet(fv)])
            bc_value = _integrate_dbc(f, fv, shape_nr, idof, ip, getcoordinates(fc), time)
            globaldof = celldofs(fc)[shape_nr]
            dbc_index = dofmapping[globaldof]
            # Only DBC dofs are currently update!-able so don't modify inhomogeneities
            # for affine constraints
            if dofcoefficients[dbc_index] === nothing
                inhomogeneities[dbc_index] = bc_value
            end
        end
    end
    return nothing
end

# Temp, put in interpolations.jl
"""
edge_parameterization(::Type{<:AbstractRefShape}, ξ, edge_id)

An edge is parameterized by the normalized curve coordinate `s [0, 1]`,
increasing in the positive edge direction.
"""
function edge_parameterization(::Type{RefShape}, ξ, edge_id) where {RefShape <: Ferrite.AbstractRefShape}
    ipg = Lagrange{RefShape, 1}() # Reference shape always described by 1st order Lagrange ip.
    refcoords = Ferrite.reference_coordinates(ipg)
    i1, i2 = Ferrite.edgedof_indices(ipg)[edge_id]
    ξ1, ξ2 = (refcoords[i1], refcoords[i2])
    Δξ = ξ2 - ξ1
    L = norm(Δξ)
    s = (ξ - ξ1) ⋅ normalize(Δξ) / L
    @assert norm(ξ - ξ1) ≈ L * s # Ensure ξ is on the line ξ1 - ξ2
    @assert -eps(L) ≤ s ≤ (1 + eps(L)) # Ensure ξ is between ξ1 and ξ2
    return s
end

function facet_parameterization(::Type{<:Ferrite.AbstractRefShape{3}}, ξ, facet_id)
    # Not implemented (not yet defined in Ferrite what this should be),
    # but to support testing interpolations with a single facedof interior index,
    # we return `nothing` just to allow running the code as long as the output isn't used.
    return nothing
end

function facet_moment(ip::Interpolation{RS}, idof::Int, ξ::Vec, facet_nr)
    s = facet_parameterization(RS, ξ, facet_nr)
    return facet_moment(ip, idof, s)
end

facet_moment(::RaviartThomas{2, RefTriangle, 1}, idof::Int, s::Real) = one(s)
facet_moment(::RaviartThomas{2, RefTriangle, 2}, idof::Int, s::Real) = (idof == 1 ? one(s) - s : s)
facet_moment(::BrezziDouglasMarini{2, RefTriangle, 1}, idof::Int, s::Real) = (idof == 1 ? one(s) - s : s)
facet_moment(::Nedelec{2, RefTriangle, 1}, idof::Int, s::Real) = one(s)
facet_moment(::Nedelec{2, RefTriangle, 2}, idof::Int, s::Real) = (idof == 1 ? one(s) - s : s)

function_space(::Interpolation) = Val(:H1) # Default fallback, should perhaps be explicit...
function_space(::RaviartThomas) = Val(:Hdiv)
function_space(::BrezziDouglasMarini) = Val(:Hdiv)
function_space(::Nedelec) = Val(:Hcurl)
# End of temp that should go in interpolations.jl

function _integrate_dbc(f::Function, args...)
    return _integrate_dbc(function_space(ip), args...)
end

function _integrate_dbc(::Val{:Hdiv}, f::Function, fv::FacetValues, shape_nr, idof, ip, cellcoords, time)
    # Could speed up by having _integrate_facet take two functions...
    f1(N, n, x, t, ξ) = facet_moment(ip, idof, ξ, getcurrentfacet(fv)) * f(x, t)
    top = _integrate_facet(f1, fv, shape_nr, cellcoords, time)
    f2(N, n, x, t, ξ) = N × n
    bot = _integrate_facet(f2, fv, shape_nr, cellcoords, time)
    return top / bot
end

function _integrate_dbc(::Val{:Hcurl}, f::Function, fv::FacetValues, shape_nr, idof, ip, cellcoords, time)
    # Could speed up by having _integrate_facet take two functions...
    f1(N, n, x, t, ξ) = facet_moment(ip, idof, ξ, getcurrentfacet(fv)) * f(x, t)
    top = _integrate_facet(f1, fv, shape_nr, cellcoords, time)
    f2(N, n, x, t, ξ) = N ⋅ n
    bot = _integrate_facet(f2, fv, shape_nr, cellcoords, time)
    return top / bot
end

_integrate_dbc(::Val{:H1}, args...) = ArgumentError("Dirichlet BC for H1 interpolations are not integrable")

function _integrate_facet(f::F, fv::FacetValues, shape_nr, cellcoords, time) where {F}
    function qp_contribution(q_point::Int)
        x = spatial_coordinate(fv, q_point, cellcoords)
        n = getnormal(fv, q_point)
        dΓ = getdetJdV(fv, q_point)
        N = shape_value(fv, q_point, shape_nr)
        ξ = getpoints(fv.fqr, getcurrentfacet(fv))[q_point]
        return f(N, n, x, time, ξ) * dΓ
    end
    retval = qp_contribution(1)
    for q_point in 2:getnquadpoints(fv)
        retval += qp_contribution(q_point)
    end
    return retval
end
