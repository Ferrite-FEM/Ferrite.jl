# abstract type Constraint end
"""
    Dirichlet(u::Symbol, ∂Ω::AbstractVecOrSet, f::Function, components=nothing)

Create a Dirichlet boundary condition on `u` on the `∂Ω` part of
the boundary. `f` is a function of the form `f(x)` or `f(x, t)`
where `x` is the spatial coordinate and `t` is the current time,
and returns the prescribed value. `components` specify the components
of `u` that are prescribed by this condition. By default all components
of `u` are prescribed.

The set, `∂Ω`, can be an `AbstractSet` or `AbstractVector` with elements of
type [`FacetIndex`](@ref), [`FaceIndex`](@ref), [`EdgeIndex`](@ref), [`VertexIndex`](@ref),
or `Int`. For most cases, the element type is `FacetIndex`, as shown below.
To constrain a single point, using `VertexIndex` is recommended, but it is also possible
to constrain a specific nodes by giving the node numbers via `Int` elements.
To constrain e.g. an edge in 3d `EdgeIndex` elements can be given.

For example, here we create a
Dirichlet condition for the `:u` field, on the facetset called
`∂Ω` and the value given by the `sin` function:

*Examples*
```jldoctest
# Obtain the facetset from the grid
∂Ω = getfacetset(grid, "boundary-1")

# Prescribe scalar field :s on ∂Ω to sin(t)
dbc = Dirichlet(:s, ∂Ω, (x, t) -> sin(t))

# Prescribe all components of vector field :v on ∂Ω to 0
dbc = Dirichlet(:v, ∂Ω, x -> 0 * x)

# Prescribe component 2 and 3 of vector field :v on ∂Ω to [sin(t), cos(t)]
dbc = Dirichlet(:v, ∂Ω, (x, t) -> [sin(t), cos(t)], [2, 3])
```

`Dirichlet` boundary conditions are added to a [`ConstraintHandler`](@ref)
which applies the condition via [`apply!`](@ref) and/or [`apply_zero!`](@ref).
"""
struct Dirichlet # <: Constraint
    f::Function # f(x) or f(x,t) -> value(s)
    facets::OrderedSet{T} where {T <: Union{Int, FacetIndex, FaceIndex, EdgeIndex, VertexIndex}}
    field_name::Symbol
    components::Vector{Int} # components of the field
    local_facet_dofs::Vector{Int}
    local_facet_dofs_offset::Vector{Int}
end
function Dirichlet(field_name::Symbol, facets::AbstractVecOrSet, f::Function, components = nothing)
    return Dirichlet(f, convert_to_orderedset(facets), field_name, __to_components(components), Int[], Int[])
end

# components=nothing is default and means that all components should be constrained
# but since number of components isn't known here it will be populated in add!
__to_components(::Nothing) = Int[]
function __to_components(c)
    components = convert(Vector{Int}, vec(collect(Int, c)))
    isempty(components) && error("components are empty: $c")
    issorted(components) || error("components not sorted: $c")
    allunique(components) || error("components not unique: $c")
    return components
end

@doc raw"""
    ProjectedDirichlet(field_name::Symbol, facets::AbstractSet{FacetIndex}, f::Function; qr_order = -1)

A `ProjectedDirichlet` condition enforces conditions for `field_name` on the boundary, `facets`, for
non-nodal interpolations (e.g. ``H(\mathrm{div})`` or ``H(\mathrm{curl})``) by
minimizing the L2 error between the function `f(x, t, n)` and the values described
by the FE approximation on the facet, ``\Gamma^f``. The arguments to the function are
the coordinate, ``\boldsymbol{x}``, the time, ``t``, and the facet normal, ``\boldsymbol{n}``.
The quadrature rule is automatically created, but the default order, `qr_order = 2 * ip_order`
(may be refined in future releases),
where `ip_order` is the order of the interpolation, can be overrided if desired.

# H(div) interpolations
For H(div), we want to prescribe the normal flux, ``q_\mathrm{n} = f(\boldsymbol{x}, t, \boldsymbol{n})``.
To that end, we want to find the degree of freedom values, ``a_j^f``, associated
with the facet that minimizes
```math
U(\boldsymbol{q}(\boldsymbol{x})) = \int_{\Gamma^f}
\left[\boldsymbol{q}(\boldsymbol{x}) \cdot \boldsymbol{n} - f(\boldsymbol{x}, t, \boldsymbol{n}) \right]^2 \mathrm{d}\Gamma
```
with ``\boldsymbol{q}(\boldsymbol{x}) = \boldsymbol{N}^f_j(\boldsymbol{x}) a_j^f``.
Finding the stationary point by taking the directional derivative in all possible directions,
``\delta\boldsymbol{q}(\boldsymbol{x})``, yields,
```math
0 = \lim_{\epsilon\rightarrow 0}\frac{\partial}{\partial \epsilon}
U\big(\boldsymbol{q}(\boldsymbol{x}) + \epsilon \delta\boldsymbol{q}(\boldsymbol{x})\big)
 = 2\int_{\Gamma^f} \left[\boldsymbol{q}(\boldsymbol{x}) \cdot \boldsymbol{n} - f(\boldsymbol{x}, t, \boldsymbol{n})\right]
    \delta\boldsymbol{q}(\boldsymbol{x}) \cdot \boldsymbol{n}\ \mathrm{d}\Gamma
```
Inserting the FE-approximations,
``\boldsymbol{q}(\boldsymbol{x}) = \boldsymbol{N}^f_j(\boldsymbol{x}) a_j^f`` and
``\delta\boldsymbol{q}(\boldsymbol{x}) = \boldsymbol{N}^f_i(\boldsymbol{x}) c_i^f``,
and using the arbitrariness of ``c_i^f``, yields the linear equation system,
```math
\underbrace{\int_{\Gamma^f} \left[\delta\boldsymbol{N}^f_i \cdot \boldsymbol{n}^f\right]\left[\boldsymbol{N}^f_j \cdot \boldsymbol{n}^f\right]\ \mathrm{d}\Gamma}_{K^f_{ij}}\ a_j^f
= \underbrace{\int_{\Gamma^f} \left[\delta\boldsymbol{N}^f_i \cdot \boldsymbol{n}^f\right] f(\boldsymbol{x}, t, \boldsymbol{n})\ \mathrm{d}\Gamma}_{f^f_i}
```
determining the coefficients to be prescribed, ``a_j^f``.

# H(curl) interpolations
For H(curl), we want to prescribe the tangential flux, ``\boldsymbol{q}_\mathrm{t} = \boldsymbol{f}(\boldsymbol{x}, t, \boldsymbol{n})``.
To that end, we want to find the degree of freedom values, ``a_j^f``, associated
with the facet that minimizes
```math
U(\boldsymbol{q}(\boldsymbol{x})) = \int_{\Gamma^f}
\left\vert\left\vert
    \boldsymbol{q}(\boldsymbol{x}) \times \boldsymbol{n} - \boldsymbol{f}(\boldsymbol{x}, t, \boldsymbol{n})
\right\vert\right\vert^2
\mathrm{d}\Gamma
```
with ``\boldsymbol{q}(\boldsymbol{x}) = \boldsymbol{N}^f_j(\boldsymbol{x}) a_j^f``.
Similar to for ``H(\mathrm{div})``, we find the minimum by setting the directional derivative equal to zero,
which after inserting the FE-approximations yields the linear equation system,
```math
\underbrace{\int_{\Gamma^f} \left[\delta\boldsymbol{N}^f_i \times \boldsymbol{n}^f\right]\cdot\left[\boldsymbol{N}^f_j \times \boldsymbol{n}^f\right]\ \mathrm{d}\Gamma}_{K^f_{ij}}\ a_j^f
= \underbrace{\int_{\Gamma^f} \left[\delta\boldsymbol{N}^f_i \times \boldsymbol{n}^f\right]\cdot \boldsymbol{f}(\boldsymbol{x}, t, \boldsymbol{n})\ \mathrm{d}\Gamma}_{f^f_i}
```
determining the coefficients to be prescribed, ``a_j^f``
"""
mutable struct ProjectedDirichlet
    const f::Function
    const facets::OrderedSet{FacetIndex}
    const field_name::Symbol
    const qr_order::Int
    # Created during `add!`
    fv::Union{Nothing, FacetValues}
    facet_dofs::Union{Nothing, ArrayOfVectorViews{Int, 1}}
end
function ProjectedDirichlet(field_name::Symbol, facets::AbstractVecOrSet, f::Function; qr_order = -1)
    return ProjectedDirichlet(f, convert_to_orderedset(facets), field_name, qr_order, nothing, nothing)
end

const DofCoefficients{T} = Vector{Pair{Int, T}}
"""
    AffineConstraint(constrained_dof::Int, entries::Vector{Pair{Int,T}}, b::T) where T

Define an affine/linear constraint to constrain one degree of freedom, `u[i]`,
such that `u[i] = ∑(u[j] * a[j]) + b`,
where `i=constrained_dof` and each element in `entries` are `j => a[j]`
"""
struct AffineConstraint{T}
    constrained_dof::Int
    entries::DofCoefficients{T} # masterdofs and factors
    b::T # inhomogeneity
end

"""
    ConstraintHandler([T=Float64], dh::AbstractDofHandler)

A collection of constraints associated with the dof handler `dh`.
`T` is the numeric type for stored values.
"""
mutable struct ConstraintHandler{DH <: AbstractDofHandler, T}
    const dbcs::Vector{Dirichlet}
    const projbcs::Vector{ProjectedDirichlet}
    const prescribed_dofs::Vector{Int}
    const free_dofs::Vector{Int}
    const inhomogeneities::Vector{T}
    # Store the original constant inhomogeneities for affine constraints used to compute
    # "effective" inhomogeneities in `update!` and then stored in .inhomogeneities.
    const affine_inhomogeneities::Vector{Union{Nothing, T}}
    # `nothing` for pure DBC constraint, otherwise affine constraint
    const dofcoefficients::Vector{Union{Nothing, DofCoefficients{T}}}
    # global dof -> index into dofs and inhomogeneities and dofcoefficients
    const dofmapping::Dict{Int, Int}
    const bcvalues::Vector{BCValues{T}}
    const dh::DH
    closed::Bool
end

ConstraintHandler(dh::AbstractDofHandler) = ConstraintHandler(Float64, dh)

function ConstraintHandler(::Type{T}, dh::AbstractDofHandler) where {T <: Number}
    @assert isclosed(dh)
    return ConstraintHandler(
        Dirichlet[], ProjectedDirichlet[], Int[], Int[], T[], Union{Nothing, T}[],
        Union{Nothing, DofCoefficients{T}}[], Dict{Int, Int}(), BCValues{T}[], dh, false,
    )
end

"""
    RHSData

Stores the constrained columns and mean of the diagonal of stiffness matrix `A`.
"""
struct RHSData{T}
    m::T
    constrained_columns::SparseMatrixCSC{T, Int}
end

"""
    get_rhs_data(ch::ConstraintHandler, A::SparseMatrixCSC) -> RHSData

Returns the needed [`RHSData`](@ref) for [`apply_rhs!`](@ref).

This must be used when the same stiffness matrix is reused for multiple steps,
for example when timestepping, with different non-homogeneouos Dirichlet boundary
conditions.
"""
function get_rhs_data(ch::ConstraintHandler, A::SparseMatrixCSC)
    m = meandiag(A)
    constrained_columns = A[:, ch.prescribed_dofs]
    return RHSData(m, constrained_columns)
end

"""
    apply_rhs!(data::RHSData, f::AbstractVector, ch::ConstraintHandler, applyzero::Bool=false)

Applies the boundary condition to the right-hand-side vector without modifying the stiffness matrix.

See also: [`get_rhs_data`](@ref).
"""
function apply_rhs!(data::RHSData, f::AbstractVector, ch::ConstraintHandler, applyzero::Bool = false)
    K = data.constrained_columns
    @assert length(f) == size(K, 1)
    @boundscheck checkbounds(f, ch.prescribed_dofs)
    m = data.m

    # TODO: Can the loops be combined or does the order matter?
    @inbounds for i in 1:length(ch.inhomogeneities)
        v = ch.inhomogeneities[i]
        if !applyzero && v != 0
            for j in nzrange(K, i)
                f[K.rowval[j]] -= v * K.nzval[j]
            end
        end
    end
    @inbounds for (i, pdof) in pairs(ch.prescribed_dofs)
        dofcoef = ch.dofcoefficients[i]
        b = ch.inhomogeneities[i]
        if dofcoef !== nothing # if affine constraint
            for (d, v) in dofcoef
                f[d] += f[pdof] * v
            end
        end
        bz = applyzero ? zero(eltype(f)) : b
        f[pdof] = bz * m
    end
    return
end

function Base.show(io::IO, ::MIME"text/plain", ch::ConstraintHandler)
    println(io, "ConstraintHandler:")
    if !isclosed(ch)
        print(io, "  Not closed!")
    else
        print(io, "  BCs:")
        for dbc in ch.dbcs
            print(io, "\n    ", "Field: ", dbc.field_name, ", ", "Components: ", dbc.components)
        end
    end
    return
end

isclosed(ch::ConstraintHandler) = ch.closed
free_dofs(ch::ConstraintHandler) = ch.free_dofs
prescribed_dofs(ch::ConstraintHandler) = ch.prescribed_dofs

# Equivalent to `copy!(out, setdiff(1:n_entries, diff))`, but requires that
# `issorted(diff)` and that `1 ≤ diff[1] ≤ diff[end] ≤ n_entries`
function _sorted_setdiff!(out::Vector{Int}, n_entries::Int, diff::Vector{Int})
    n_diff = length(diff)
    resize!(out, n_entries - n_diff)
    diff_ind = out_ind = 1
    for i in 1:n_entries
        if diff_ind ≤ n_diff && i == diff[diff_ind]
            diff_ind += 1
        else
            out[out_ind] = i
            out_ind += 1
        end
    end
    return out
end

"""
    close!(ch::ConstraintHandler)

Close and finalize the `ConstraintHandler`.
"""
function close!(ch::ConstraintHandler)
    @assert(!isclosed(ch))
    @assert(allunique(ch.prescribed_dofs))

    I = sortperm(ch.prescribed_dofs)
    ch.prescribed_dofs .= ch.prescribed_dofs[I]
    ch.inhomogeneities .= ch.inhomogeneities[I]
    ch.affine_inhomogeneities .= ch.affine_inhomogeneities[I]
    ch.dofcoefficients .= ch.dofcoefficients[I]

    _sorted_setdiff!(ch.free_dofs, ndofs(ch.dh), ch.prescribed_dofs)

    for i in 1:length(ch.prescribed_dofs)
        ch.dofmapping[ch.prescribed_dofs[i]] = i
    end

    # TODO: Store index for each affine constraint?
    # affine_mapping = Dict{Int,Int}(pdof => i for (i, pdof) in pairs(cd.prescribed_dofs) if ch.dofcoefficients[i] !== nothing )

    # TODO:
    # Do a bunch of checks to see if the affine constraints are linearly indepented etc.
    # If they are not, it is possible to automatically reformulate the constraints
    # such that they become independent. However, at this point, it is left to
    # the user to assure this.

    # Basic verification of constraints:
    # - `add_prescribed_dof` make sure all prescribed dofs are unique by overwriting the old
    #   constraint when adding a new (TODO: Might change in the future, see comment in
    #   `add_prescribed_dof`.)
    # - We allow affine constraints to have prescribed dofs as master dofs iff those master
    #   dofs are constrained with just an inhomogeneity (i.e. DBC). The effective
    #   inhomogeneity is computed in `update!`.
    for coeffs in ch.dofcoefficients
        coeffs === nothing && continue
        for (d, _) in coeffs
            i = get(ch.dofmapping, d, 0)
            i == 0 && continue
            icoeffs = ch.dofcoefficients[i]
            if !(icoeffs === nothing || isempty(icoeffs))
                error("nested affine constraints currently not supported")
            end
        end
    end

    ch.closed = true

    # Compute the prescribed values by calling update!: This should be cheap, and for the
    # common case where constraints does not depend on time it is annoying and easy to
    # forget to call this on the outside.
    update!(ch)

    return ch
end

"""
    add!(ch::ConstraintHandler, ac::AffineConstraint)

Add the `AffineConstraint` to the `ConstraintHandler`.
"""
function add!(ch::ConstraintHandler, ac::AffineConstraint)
    # TODO: Would be nice to pass nothing if ac.entries is empty, but then we lose the fact
    #       that this constraint is an AffineConstraint which is currently needed in update!
    #       in order to not update inhomogeneities for affine constraints
    add_prescribed_dof!(ch, ac.constrained_dof, ac.b, #=isempty(ac.entries) ? nothing : =# ac.entries)
    return ch
end

"""
    add_prescribed_dof!(ch, constrained_dof::Int, inhomogeneity, dofcoefficients=nothing)

Add a constrained dof directly to the `ConstraintHandler`.
This function checks if the `constrained_dof` is already constrained, and overrides the old
constraint if true.
"""
function add_prescribed_dof!(ch::ConstraintHandler, constrained_dof::Int, inhomogeneity, dofcoefficients = nothing)
    @assert(!isclosed(ch))
    i = get(ch.dofmapping, constrained_dof, 0)
    if i != 0
        @debug @warn "dof $i already prescribed, overriding the old constraint"
        ch.prescribed_dofs[i] = constrained_dof
        ch.inhomogeneities[i] = inhomogeneity
        ch.affine_inhomogeneities[i] = dofcoefficients === nothing ? nothing : inhomogeneity
        ch.dofcoefficients[i] = dofcoefficients
    else
        N = length(ch.dofmapping)
        push!(ch.prescribed_dofs, constrained_dof)
        push!(ch.inhomogeneities, inhomogeneity)
        push!(ch.affine_inhomogeneities, dofcoefficients === nothing ? nothing : inhomogeneity)
        push!(ch.dofcoefficients, dofcoefficients)
        ch.dofmapping[constrained_dof] = N + 1
    end
    return ch
end

# Dirichlet on (facet|face|edge|vertex)set
function _add!(ch::ConstraintHandler, dbc::Dirichlet, bcfacets::AbstractVecOrSet{Index}, interpolation::Interpolation, field_dim::Int, offset::Int, bcvalue::BCValues, _) where {Index <: BoundaryIndex}
    local_facet_dofs, local_facet_dofs_offset =
        _local_facet_dofs_for_bc(interpolation, field_dim, dbc.components, offset, dirichlet_boundarydof_indices(eltype(bcfacets)))
    copy!(dbc.local_facet_dofs, local_facet_dofs)
    copy!(dbc.local_facet_dofs_offset, local_facet_dofs_offset)

    # loop over all the faces in the set and add the global dofs to `constrained_dofs`
    constrained_dofs = Int[]
    cc = CellCache(ch.dh, UpdateFlags(; nodes = false, coords = false, dofs = true))
    for (cellidx, facetidx) in bcfacets
        reinit!(cc, cellidx)
        r = local_facet_dofs_offset[facetidx]:(local_facet_dofs_offset[facetidx + 1] - 1)
        append!(constrained_dofs, cc.dofs[local_facet_dofs[r]]) # TODO: for-loop over r and simply push! to ch.prescribed_dofs
        @debug println("adding dofs $(cc.dofs[local_facet_dofs[r]]) to dbc")
    end

    # save it to the ConstraintHandler
    push!(ch.dbcs, dbc)
    push!(ch.bcvalues, bcvalue)
    for d in constrained_dofs
        add_prescribed_dof!(ch, d, NaN, nothing)
    end
    return ch
end

# Calculate which local dof index live on each facet:
# facet `i` have dofs `local_facet_dofs[local_facet_dofs_offset[i]:local_facet_dofs_offset[i+1]-1]
function _local_facet_dofs_for_bc(interpolation, field_dim, components, offset, boundaryfunc::F = dirichlet_facetdof_indices) where {F}
    @assert issorted(components)
    local_facet_dofs = Int[]
    local_facet_dofs_offset = Int[1]
    for (_, facet) in enumerate(boundaryfunc(interpolation))
        for fdof in facet, d in 1:field_dim
            if d in components
                push!(local_facet_dofs, (fdof - 1) * field_dim + d + offset)
            end
        end
        push!(local_facet_dofs_offset, length(local_facet_dofs) + 1)
    end
    return local_facet_dofs, local_facet_dofs_offset
end

function _add!(ch::ConstraintHandler, dbc::Dirichlet, bcnodes::AbstractVecOrSet{Int}, interpolation::Interpolation, field_dim::Int, offset::Int, bcvalue::BCValues, cellset::AbstractVecOrSet{Int} = OrderedSet{Int}(1:getncells(get_grid(ch.dh))))
    grid = get_grid(ch.dh)
    if interpolation !== geometric_interpolation(getcelltype(grid, first(cellset)))
        @warn("adding constraint to nodeset is not recommended for sub/super-parametric approximations.")
    end

    ncomps = length(dbc.components)
    nnodes = getnnodes(grid)
    interpol_points = getnbasefunctions(interpolation)
    node_dofs = zeros(Int, ncomps, nnodes)
    visited = falses(nnodes)
    for cell in CellIterator(ch.dh, cellset) # only go over cells that belong to current SubDofHandler
        for idx in 1:min(interpol_points, length(cell.nodes))
            node = cell.nodes[idx]
            if !visited[node]
                noderange = (offset + (idx - 1) * field_dim + 1):(offset + idx * field_dim) # the dofs in this node
                for (i, c) in enumerate(dbc.components)
                    node_dofs[i, node] = cell.dofs[noderange[c]]
                    @debug println("adding dof $(cell.dofs[noderange[c]]) to node_dofs")
                end
                visited[node] = true
            end
        end
    end

    constrained_dofs = Int[]
    sizehint!(constrained_dofs, ncomps * length(bcnodes))
    sizehint!(dbc.local_facet_dofs, length(bcnodes))
    for node in bcnodes
        if !visited[node]
            # either the node belongs to another field handler or it does not have dofs in the constrained field
            continue
        end
        for i in 1:ncomps
            push!(constrained_dofs, node_dofs[i, node])
        end
        push!(dbc.local_facet_dofs, node) # use this field to store the node idx for each node
    end

    # save it to the ConstraintHandler
    copy!(dbc.local_facet_dofs_offset, constrained_dofs) # use this field to store the global dofs
    push!(ch.dbcs, dbc)
    push!(ch.bcvalues, bcvalue)
    for d in constrained_dofs
        add_prescribed_dof!(ch, d, NaN, nothing)
    end
    return ch
end

"""
    update!(ch::ConstraintHandler, time::Real=0.0)

Update time-dependent inhomogeneities for the new time. This calls `f(x)` or `f(x, t)` when
applicable, where `f` is the function(s) corresponding to the constraints in the handler, to
compute the inhomogeneities.

Note that this is called implicitly in `close!(::ConstraintHandler)`.
"""
function update!(ch::ConstraintHandler, time::Real = 0.0)
    @assert ch.closed
    for (i, dbc) in pairs(ch.dbcs)
        # If the BC function only accept one argument, i.e. f(x), we create a wrapper
        # g(x, t) = f(x) that discards the second parameter so that _update! can always call
        # the function with two arguments internally.
        wrapper_f = hasmethod(dbc.f, Tuple{get_coordinate_type(get_grid(ch.dh)), typeof(time)}) ? dbc.f : (x, _) -> dbc.f(x)
        # Function barrier
        _update!(
            ch.inhomogeneities, wrapper_f, dbc.facets, dbc.local_facet_dofs, dbc.local_facet_dofs_offset,
            dbc.components, ch.dh, ch.bcvalues[i], ch.dofmapping, ch.dofcoefficients, time
        )
    end
    for bc in ch.projbcs
        _update_projected_dbc!(ch.inhomogeneities, bc.f, bc.facets, bc.fv, bc.facet_dofs, ch.dh, ch.dofmapping, ch.dofcoefficients, time)
    end
    # Compute effective inhomogeneity for affine constraints with prescribed dofs in the
    # RHS. For example, in u2 = w3 * u3 + w4 * u4 + b2 we allow e.g. u3 to be prescribed by
    # a trivial constraint with just an inhomogeneity (e.g. DBC), for example u3 = f(t).
    # This value have now been computed in _update! and we can compute the effective
    # inhomogeneity h2 for u2 which becomes h2 = w3 * u3 + b2 = w3 * f3(t) + b2.
    for i in eachindex(ch.prescribed_dofs, ch.dofcoefficients, ch.inhomogeneities)
        coeffs = ch.dofcoefficients[i]
        coeffs === nothing && continue
        h = ch.affine_inhomogeneities[i]
        @assert h !== nothing
        for (d, w) in coeffs
            j = get(ch.dofmapping, d, 0)
            j == 0 && continue
            # If this dof is prescribed it must only have an inhomogeneity (verified in close!)
            @assert (jcoeffs = ch.dofcoefficients[j]; jcoeffs === nothing || isempty(jcoeffs))
            h += ch.inhomogeneities[j] * w
        end
        ch.inhomogeneities[i] = h
    end
    return nothing
end

# for facets, vertices, faces and edges
function _update!(
        inhomogeneities::Vector{T}, f::Function, boundary_entities::AbstractVecOrSet{<:BoundaryIndex}, local_facet_dofs::Vector{Int}, local_facet_dofs_offset::Vector{Int},
        components::Vector{Int}, dh::AbstractDofHandler, boundaryvalues::BCValues,
        dofmapping::Dict{Int, Int}, dofcoefficients::Vector{Union{Nothing, DofCoefficients{T}}}, time::Real
    ) where {T}

    cc = CellCache(dh, UpdateFlags(; nodes = false, coords = true, dofs = true))
    for (cellidx, entityidx) in boundary_entities
        reinit!(cc, cellidx)

        # no need to reinit!, enough to update current_entity since we only need geometric shape functions M
        boundaryvalues.current_entity = entityidx

        # local dof-range for this facet
        r = local_facet_dofs_offset[entityidx]:(local_facet_dofs_offset[entityidx + 1] - 1)
        counter = 1
        for location in 1:getnquadpoints(boundaryvalues)
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
                    inhomogeneities[dbc_index] = bc_value[i]
                    @debug println("prescribing value $(bc_value[i]) on global dof $(globaldof)")
                end
            end
        end
    end
    return
end

# for nodes
function _update!(
        inhomogeneities::Vector{T}, f::Function, ::AbstractVecOrSet{Int}, nodeidxs::Vector{Int}, globaldofs::Vector{Int},
        components::Vector{Int}, dh::AbstractDofHandler, facetvalues::BCValues,
        dofmapping::Dict{Int, Int}, dofcoefficients::Vector{Union{Nothing, DofCoefficients{T}}}, time::Real
    ) where {T}
    counter = 1
    for nodenumber in nodeidxs
        x = get_node_coordinate(get_grid(dh), nodenumber)
        bc_value = f(x, time)
        @assert length(bc_value) == length(components)
        for v in bc_value
            globaldof = globaldofs[counter]
            counter += 1
            dbc_index = dofmapping[globaldof]
            # Only DBC dofs are currently update!-able so don't modify inhomogeneities
            # for affine constraints
            if dofcoefficients[dbc_index] === nothing
                inhomogeneities[dbc_index] = v
                @debug println("prescribing value $(v) on global dof $(globaldof)")
            end
        end
    end
    return
end

"""

    apply!(K::AbstractSparseMatrix, rhs::AbstractVector, ch::ConstraintHandler)

Adjust the matrix `K` and right hand side `rhs` to account for the Dirichlet boundary
conditions specified in `ch` such that `K \\ rhs` gives the expected solution.

!!! note
    `apply!(K, rhs, ch)` essentially calculates
    ```julia
    rhs[free] = rhs[free] - K[constrained, constrained] * a[constrained]
    ```
    where `a[constrained]` are the inhomogeneities. Consequently, the sign of `rhs` matters
    (in contrast with `apply_zero!`).


```julia
apply!(v::AbstractVector, ch::ConstraintHandler)
```

Apply Dirichlet boundary conditions and affine constraints, specified in `ch`, to the solution vector `v`.

# Examples
```julia
K, f = assemble_system(...) # Assemble system
apply!(K, f, ch)            # Adjust K and f to account for boundary conditions
u = K \\ f                   # Solve the system, u should be "approximately correct"
apply!(u, ch)               # Explicitly make sure bcs are correct
```

!!! note
    The last operation is not strictly necessary since the boundary conditions should
    already be fulfilled after `apply!(K, f, ch)`. However, solvers of linear systems are
    not exact, and thus `apply!(u, ch)` can be used to make sure the boundary conditions
    are fulfilled exactly.
"""
apply!

"""
    apply_zero!(K::SparseMatrixCSC, rhs::AbstractVector, ch::ConstraintHandler)

Adjust the matrix `K` and the right hand side `rhs` to account for prescribed Dirichlet
boundary conditions and affine constraints such that `du = K \\ rhs` gives the expected
result (e.g. `du` zero for all prescribed degrees of freedom).

    apply_zero!(v::AbstractVector, ch::ConstraintHandler)

Zero-out values in `v` corresponding to prescribed degrees of freedom and update values
prescribed by affine constraints, such that if `a` fulfills the constraints,
`a ± v` also will.

These methods are typically used in e.g. a Newton solver where the increment, `du`, should
be prescribed to zero even for non-homogeneouos boundary conditions.

See also: [`apply!`](@ref).

# Examples
```julia
u = un + Δu                 # Current guess
K, g = assemble_system(...) # Assemble residual and tangent for current guess
apply_zero!(K, g, ch)       # Adjust tangent and residual to take prescribed values into account
ΔΔu = K \\ g                # Compute the (negative) increment, prescribed values are "approximately" zero
apply_zero!(ΔΔu, ch)        # Make sure values are exactly zero
Δu .-= ΔΔu                  # Update current guess
```

!!! note
    The last call to `apply_zero!` is only strictly necessary for affine constraints.
    However, even if the Dirichlet boundary conditions should be fulfilled after
    `apply!(K, g, ch)`, solvers of linear systems are not exact.
    `apply!(ΔΔu, ch)` can be used to make sure the values
    for the prescribed degrees of freedom are fulfilled exactly.
"""
apply_zero!

apply_zero!(v::AbstractVector, ch::ConstraintHandler) = _apply_v(v, ch, true)
apply!(v::AbstractVector, ch::ConstraintHandler) = _apply_v(v, ch, false)

function _apply_v(v::AbstractVector, ch::ConstraintHandler, apply_zero::Bool)
    @assert isclosed(ch)
    @assert length(v) >= ndofs(ch.dh)
    v[ch.prescribed_dofs] .= apply_zero ? 0.0 : ch.inhomogeneities
    # Apply affine constraints, e.g u2 = s6*u6 + s3*u3 + h2
    for (dof, dofcoef, h) in zip(ch.prescribed_dofs, ch.dofcoefficients, ch.affine_inhomogeneities)
        dofcoef === nothing && continue
        @assert h !== nothing
        v[dof] = apply_zero ? 0.0 : h
        for (d, s) in dofcoef
            v[dof] += s * v[d]
        end
    end
    return v
end

function apply!(K::Union{AbstractSparseMatrix, Symmetric}, ch::ConstraintHandler)
    return apply!(K, eltype(K)[], ch, true)
end

function apply_zero!(K::Union{AbstractSparseMatrix, Symmetric}, f::AbstractVector, ch::ConstraintHandler)
    return apply!(K, f, ch, true)
end

function apply!(KK::Union{AbstractSparseMatrix, Symmetric{<:Any, <:AbstractSparseMatrix}}, f::AbstractVector, ch::ConstraintHandler, applyzero::Bool = false)
    @assert isclosed(ch)
    sym = isa(KK, Symmetric)
    K = sym ? KK.data : KK
    @assert length(f) == 0 || length(f) == size(K, 1)
    @boundscheck checkbounds(K, ch.prescribed_dofs, ch.prescribed_dofs)
    @boundscheck length(f) == 0 || checkbounds(f, ch.prescribed_dofs)

    m = meandiag(K) # Use the mean of the diagonal here to not ruin things for iterative solver

    # Add inhomogeneities to f: (f - K * ch.inhomogeneities)
    !applyzero && add_inhomogeneities!(f, KK, ch)

    # Condense K := (C' * K * C) and f := (C' * f)
    _condense!(K, f, ch.dofcoefficients, ch.dofmapping, sym)

    # Remove constrained dofs from the matrix
    zero_out_columns!(K, ch)
    zero_out_rows!(K, ch)

    # Add meandiag to constraint dofs
    @inbounds for i in 1:length(ch.inhomogeneities)
        d = ch.prescribed_dofs[i]
        K[d, d] = m
        if length(f) != 0
            vz = applyzero ? zero(eltype(f)) : ch.inhomogeneities[i]
            f[d] = vz * m
        end
    end
    return
end

"""
    add_inhomogeneities!(f::AbstractVector, K::AbstractMatrix, ch::ConstraintHandler)

Compute "f -= K*inhomogeneities".
By default this is a generic version via SpMSpV kernel.
"""
function add_inhomogeneities!(f::AbstractVector, K::AbstractMatrix, ch::ConstraintHandler)
    return mul!(f, K, sparsevec(ch.prescribed_dofs, ch.inhomogeneities, size(K, 2)), -1, 1)
end

# Optimized version for SparseMatrixCSC
add_inhomogeneities!(f::AbstractVector, K::SparseMatrixCSC, ch::ConstraintHandler) = add_inhomogeneities_csc!(f, K, ch, false)
add_inhomogeneities!(f::AbstractVector, K::Symmetric{<:Any, <:SparseMatrixCSC}, ch::ConstraintHandler) = add_inhomogeneities_csc!(f, K.data, ch, true)
function add_inhomogeneities_csc!(f::AbstractVector, K::SparseMatrixCSC, ch::ConstraintHandler, sym::Bool)
    (; inhomogeneities, prescribed_dofs, dofmapping) = ch

    @inbounds for i in 1:length(inhomogeneities)
        d = prescribed_dofs[i]
        v = inhomogeneities[i]
        if v != 0
            for j in nzrange(K, d)
                r = K.rowval[j]
                sym && r > d && break # don't look below diagonal
                f[r] -= v * K.nzval[j]
            end
        end
    end
    if sym
        # In the symmetric case, for a constrained dof `d`, we handle the contribution
        # from `K[1:d, d]` in the loop above, but we are still missing the contribution
        # from `K[(d+1):size(K,1), d]`. These values are not stored, but since the
        # matrix is symmetric we can instead use `K[d, (d+1):size(K,1)]`. Looping over
        # rows is slow, so loop over all columns again, and check if the row is a
        # constrained row.
        @inbounds for col in 1:size(K, 2)
            for ri in nzrange(K, col)
                row = K.rowval[ri]
                row >= col && break
                if (i = get(dofmapping, row, 0); i != 0)
                    f[col] -= inhomogeneities[i] * K.nzval[ri]
                end
            end
        end
    end
    return f
end

# Fetch dof coefficients for a dof prescribed by an affine constraint. Return nothing if the
# dof is not prescribed, or prescribed by DBC.
@inline function coefficients_for_dof(dofmapping, dofcoeffs, dof)
    idx = get(dofmapping, dof, 0)
    idx == 0 && return nothing
    return dofcoeffs[idx]
end

"""
    _condense!(K::AbstractSparseMatrix, f::AbstractVector, dofcoefficients::Vector{Union{Nothing, DofCoefficients{T}}}, dofmapping::Dict{Int, Int}, sym::Bool = false)

Condenses affine constraints K := C'*K*C and f := C'*f in-place, assuming the sparsity pattern is correct.
"""
function _condense!(K::AbstractSparseMatrix, f::AbstractVector, dofcoefficients::Vector{Union{Nothing, DofCoefficients{T}}}, dofmapping::Dict{Int, Int}, sym::Bool = false) where {T}
    # Return early if there are no non-trivial affine constraints
    any(i -> !(i === nothing || isempty(i)), dofcoefficients) || return
    error("condensation of ::$(typeof(K)) matrix not supported")
end

# Condenses K and f: C'*K*C, C'*f, in-place assuming the sparsity pattern is correct
function _condense!(K::SparseMatrixCSC, f::AbstractVector, dofcoefficients::Vector{Union{Nothing, DofCoefficients{T}}}, dofmapping::Dict{Int, Int}, sym::Bool = false) where {T}

    ndofs = size(K, 1)
    condense_f = !(length(f) == 0)
    condense_f && @assert(length(f) == ndofs)

    # Return early if there are no non-trivial affine constraints
    any(i -> !(i === nothing || isempty(i)), dofcoefficients) || return

    # TODO: The rest of this method can't handle K::Symmetric
    if sym
        error("condensation of ::Symmetric matrix not supported")
    end

    for col in 1:ndofs
        col_coeffs = coefficients_for_dof(dofmapping, dofcoefficients, col)
        if col_coeffs === nothing
            for a in nzrange(K, col)
                Kval = K.nzval[a]
                iszero(Kval) && continue
                row = K.rowval[a]
                row_coeffs = coefficients_for_dof(dofmapping, dofcoefficients, row)
                row_coeffs === nothing && continue
                for (d, v) in row_coeffs
                    addindex!(K, v * Kval, d, col)
                end

                # Perform f - K*g. However, this has already been done in outside this functions so we skip this.
                # if condense_f
                #     f[col] -= K.nzval[a] * ac.b;
                # end
            end
        else
            for a in nzrange(K, col)
                Kval = K.nzval[a]
                iszero(Kval) && continue
                row = K.rowval[a]
                row_coeffs = coefficients_for_dof(dofmapping, dofcoefficients, row)
                if row_coeffs === nothing
                    for (d, v) in col_coeffs
                        addindex!(K, v * Kval, row, d)
                    end
                else
                    for (d1, v1) in col_coeffs, (d2, v2) in row_coeffs
                        addindex!(K, v1 * v2 * Kval, d1, d2)
                    end
                end
            end

            if condense_f
                for (d, v) in col_coeffs
                    f[d] += f[col] * v
                end
                f[col] = 0.0
            end
        end
    end
    return
end

function _add_or_grow(cnt::Int, I::Vector{Int}, J::Vector{Int}, dofi::Int, dofj::Int)
    if cnt > length(J)
        resize!(I, trunc(Int, length(I) * 1.5))
        resize!(J, trunc(Int, length(J) * 1.5))
    end
    I[cnt] = dofi
    J[cnt] = dofj
    return
end

"""
    create_constraint_matrix(ch::ConstraintHandler)

Create and return the constraint matrix, `C`, and the inhomogeneities, `g`, from the affine
(linear) and Dirichlet constraints in `ch`.

The constraint matrix relates constrained, `a_c`, and free, `a_f`, degrees of freedom via
`a_c = C * a_f + g`. The condensed system of linear equations is obtained as
`C' * K * C = C' *  (f - K * g)`.
"""
function create_constraint_matrix(ch::ConstraintHandler{dh, T}) where {dh, T}
    @assert(isclosed(ch))

    I = Int[]; J = Int[]; V = T[]
    g = zeros(T, ndofs(ch.dh)) # inhomogeneities

    for (j, d) in enumerate(ch.free_dofs)
        push!(I, d)
        push!(J, j)
        push!(V, 1.0)
    end

    for (i, pdof) in enumerate(ch.prescribed_dofs)
        dofcoef = ch.dofcoefficients[i]
        if dofcoef !== nothing #if affine constraint
            for (d, v) in dofcoef
                push!(I, pdof)
                j = searchsortedfirst(ch.free_dofs, d)
                push!(J, j)
                push!(V, v)
            end
        end
    end
    g[ch.prescribed_dofs] .= ch.inhomogeneities

    C = SparseArrays.sparse!(I, J, V, ndofs(ch.dh), length(ch.free_dofs))

    return C, g
end
"""
    zero_out_columns!(K::AbstractMatrix, ch::ConstraintHandler)
Set the values of all columns associated with constrained dofs to zero.
"""
zero_out_columns!

"""
    zero_out_rows!(K::AbstractMatrix, ch::ConstraintHandler)
Set the values of all rows associated with constrained dofs to zero.
"""
zero_out_rows!

# columns need to be stored entries, this is not checked
function zero_out_columns!(K::AbstractSparseMatrixCSC, ch::ConstraintHandler) # can be removed in 0.7 with #24711 merged
    @debug @assert issorted(ch.prescribed_dofs)
    for col in ch.prescribed_dofs
        r = nzrange(K, col)
        K.nzval[r] .= 0.0
    end
    return
end

function zero_out_rows!(K::AbstractSparseMatrixCSC, ch::ConstraintHandler)
    rowval = K.rowval
    nzval = K.nzval
    @inbounds for i in eachindex(rowval, nzval)
        if haskey(ch.dofmapping, rowval[i])
            nzval[i] = 0
        end
    end
    return
end

function meandiag(K::AbstractMatrix)
    z = zero(eltype(K))
    for i in 1:size(K, 1)
        z += abs(K[i, i])
    end
    return z / size(K, 1)
end

"""
    add!(ch::ConstraintHandler, dbc::Dirichlet)

Add a `Dirichlet` boundary condition to the `ConstraintHandler`.
"""
function add!(ch::ConstraintHandler, dbc::Dirichlet)
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
        # Internally we use the devectorized version
        n_comp = n_dbc_components(interpolation)
        if interpolation isa VectorizedInterpolation
            interpolation = interpolation.ip
        end
        getorder(interpolation) == 0 && error("No dof prescribed for order 0 interpolations")
        # Set up components to prescribe (empty input means prescribe all components)
        components = isempty(dbc.components) ? collect(Int, 1:n_comp) : dbc.components
        if !all(c -> 0 < c <= n_comp, components)
            error("components $(components) not within range of field :$(dbc.field_name) ($(n_comp) dimension(s))")
        end
        # Create BCValues for coordinate evaluation at dof-locations
        EntityType = eltype(dbc.facets) # (Facet|Face|Edge|Vertex)Index
        if EntityType <: Integer
            # BCValues are just dummy for nodesets so set to FacetIndex
            EntityType = FacetIndex
        end
        CT = getcelltype(sdh) # Same celltype enforced in SubDofHandler constructor
        bcvalues = BCValues(interpolation, geometric_interpolation(CT), EntityType)
        # Recreate the Dirichlet(...) struct with the filtered set and call internal add!
        filtered_dbc = Dirichlet(dbc.field_name, filtered_set, dbc.f, components)
        _add!(
            ch, filtered_dbc, filtered_dbc.facets, interpolation, n_comp,
            field_offset(sdh, field_idx), bcvalues, sdh.cellset,
        )
        dbc_added = true
    end
    dbc_added || error("No overlap between dbc::Dirichlet and fields in the ConstraintHandler's DofHandler")
    return ch
end

# Return the intersection of the SubDofHandler set and the Dirichlet BC set
function filter_dbc_set(::AbstractGrid, fhset::AbstractSet{Int}, dbcset::AbstractSet{<:BoundaryIndex})
    ret = empty(dbcset)::typeof(dbcset)
    for x in dbcset
        cellid, _ = x
        cellid in fhset && push!(ret, x)
    end
    return ret
end

function filter_dbc_set(grid::AbstractGrid, fhset::AbstractSet{Int}, dbcset::AbstractSet{Int})
    ret = empty(dbcset)
    nodes_in_fhset = OrderedSet{Int}()
    for cc in CellIterator(grid, fhset, UpdateFlags(; nodes = true, coords = false))
        union!(nodes_in_fhset, cc.nodes)
    end
    for nodeid in dbcset
        nodeid in nodes_in_fhset && push!(ret, nodeid)
    end
    return ret
end

struct PeriodicFacetPair
    mirror::FacetIndex
    image::FacetIndex
    rotation::UInt8 # relative rotation of the mirror facet counter-clockwise the *image* normal (only relevant in 3D)
    mirrored::Bool  # mirrored => opposite normal vectors
end

"""
    PeriodicDirichlet(u::Symbol, facet_mapping, components=nothing)
    PeriodicDirichlet(u::Symbol, facet_mapping, R::AbstractMatrix, components=nothing)
    PeriodicDirichlet(u::Symbol, facet_mapping, f::Function, components=nothing)

Create a periodic Dirichlet boundary condition for the field `u` on the facet-pairs given in
`facet_mapping`. The mapping can be computed with [`collect_periodic_facets`](@ref). The
constraint ensures that degrees-of-freedom on the mirror facet are constrained to the
corresponding degrees-of-freedom on the image facet. `components` specify the components of
`u` that are prescribed by this condition. By default all components of `u` are prescribed.

If the mapping is not aligned with the coordinate axis (e.g. rotated) a rotation matrix `R`
should be passed to the constructor. This matrix rotates dofs on the mirror facet to the
image facet. Note that this is only applicable for vector-valued problems.

To construct an inhomogeneous periodic constraint it is possible to pass a function `f`.
Note that this is currently only supported when the periodicity is aligned with the
coordinate axes.

See the manual section on [Periodic boundary conditions](@ref) for more information.
"""
struct PeriodicDirichlet
    field_name::Symbol
    components::Vector{Int} # components of the field
    facet_pairs::Vector{Pair{String, String}} # legacy that will populate facet_map on add!
    facet_map::Vector{PeriodicFacetPair}
    func::Union{Function, Nothing}
    rotation_matrix::Union{Matrix{Float64}, Nothing}
end

# Default to no inhomogeneity function/rotation
PeriodicDirichlet(fn::Symbol, fp::Union{Vector{<:Pair}, Vector{PeriodicFacetPair}}, c = nothing) =
    PeriodicDirichlet(fn, fp, nothing, c)

# Basic constructor for the simple case where face_map will be populated in
# add!(::ConstraintHandler, ...) instead
function PeriodicDirichlet(fn::Symbol, fp::Vector{<:Pair}, f::Union{Function, Nothing}, c = nothing)
    facet_map = PeriodicFacetPair[] # This will be populated in add!(::ConstraintHandler, ...) instead
    return PeriodicDirichlet(fn, __to_components(c), fp, facet_map, f, nothing)
end

function PeriodicDirichlet(fn::Symbol, fm::Vector{PeriodicFacetPair}, f_or_r::Union{AbstractMatrix, Function, Nothing}, c = nothing)
    f = f_or_r isa Function ? f_or_r : nothing
    rotation_matrix = f_or_r isa AbstractMatrix ? f_or_r : nothing
    components = __to_components(c)
    return PeriodicDirichlet(fn, components, Pair{String, String}[], fm, f, rotation_matrix)
end

function add!(ch::ConstraintHandler, pdbc::PeriodicDirichlet)
    # Legacy code: Might need to build the facet_map
    is_legacy = !isempty(pdbc.facet_pairs) && isempty(pdbc.facet_map)
    if is_legacy
        for (mset, iset) in pdbc.facet_pairs
            collect_periodic_facets!(pdbc.facet_map, get_grid(ch.dh), mset, iset, identity) # TODO: Better transform
        end
    end
    field_idx = find_field(ch.dh, pdbc.field_name)
    interpolation = getfieldinterpolation(ch.dh, field_idx)
    n_comp = n_dbc_components(interpolation)
    if interpolation isa VectorizedInterpolation
        interpolation = interpolation.ip
    end

    if !all(c -> 0 < c <= n_comp, pdbc.components)
        error("components $(pdbc.components) not within range of field :$(pdbc.field_name) ($(n_comp) dimension(s))")
    end

    # Empty components means constrain them all
    isempty(pdbc.components) && append!(pdbc.components, 1:n_comp)

    if pdbc.rotation_matrix === nothing
        dof_map_t = Int
        iterator_f = identity
    else
        @assert pdbc.func === nothing # Verified in constructor
        if is_legacy
            error("legacy mode not supported with rotations")
        end
        nc = length(pdbc.components)
        if !(nc == size(pdbc.rotation_matrix, 1) == size(pdbc.rotation_matrix, 2))
            error("size of rotation matrix does not match the number of components")
        end
        if nc !== n_comp
            error("rotations currently only supported when all components are periodic")
        end
        dof_map_t = Vector{Int}
        iterator_f = x -> Iterators.partition(x, nc)
    end
    _add!(ch, pdbc, interpolation, n_comp, field_offset(ch.dh.subdofhandlers[field_idx[1]], field_idx[2]), is_legacy, pdbc.rotation_matrix, dof_map_t, iterator_f)
    return ch
end

function _add!(
        ch::ConstraintHandler, pdbc::PeriodicDirichlet, interpolation::Interpolation,
        field_dim::Int, offset::Int, is_legacy::Bool, rotation_matrix::Union{Matrix{T}, Nothing}, ::Type{dof_map_t}, iterator_f::F
    ) where {T, dof_map_t, F <: Function}
    grid = get_grid(ch.dh)
    facet_map = pdbc.facet_map

    # Indices of the local dofs for the facets
    local_facet_dofs, local_facet_dofs_offset =
        _local_facet_dofs_for_bc(interpolation, field_dim, pdbc.components, offset)
    mirrored_indices =
        mirror_local_dofs(local_facet_dofs, local_facet_dofs_offset, interpolation, length(pdbc.components))
    rotated_indices = rotate_local_dofs(local_facet_dofs, local_facet_dofs_offset, interpolation, length(pdbc.components))

    # Dof map for mirror dof => image dof
    dof_map = Dict{dof_map_t, dof_map_t}()

    n = ndofs_per_cell(ch.dh, first(facet_map).mirror[1])
    mirror_dofs = zeros(Int, n)
    image_dofs = zeros(Int, n)
    for facet_pair in facet_map
        m = facet_pair.mirror
        i = facet_pair.image
        celldofs!(mirror_dofs, ch.dh, m[1])
        celldofs!(image_dofs, ch.dh, i[1])

        mdof_range = local_facet_dofs_offset[m[2]]:(local_facet_dofs_offset[m[2] + 1] - 1)
        idof_range = local_facet_dofs_offset[i[2]]:(local_facet_dofs_offset[i[2] + 1] - 1)

        for (md, id) in zip(iterator_f(mdof_range), iterator_f(idof_range))
            mdof = image_dofs[local_facet_dofs[id]]
            # Rotate the mirror index
            rotated_md = rotated_indices[md, facet_pair.rotation + 1]
            # Mirror the mirror index (maybe) :)
            mirrored_md = facet_pair.mirrored ? mirrored_indices[rotated_md] : rotated_md
            cdof = mirror_dofs[local_facet_dofs[mirrored_md]]
            if haskey(dof_map, mdof)
                mdof′ = dof_map[mdof]
                # @info "$cdof => $mdof, but $mdof => $mdof′, remapping $cdof => $mdof′."
                # TODO: Is this needed now when untangling below?
                push!(dof_map, cdof => mdof′)
                # elseif haskey(dof_map, cdof) && dof_map[cdof] == mdof
                # @info "$cdof => $mdof already in the set, skipping."
            elseif haskey(dof_map, cdof)
                # @info "$cdof => $mdof, but $cdof => $(dof_map[cdof]) already, skipping."
            elseif cdof == mdof
                # @info "Skipping self-constraint $cdof => $mdof."
            else
                # @info "$cdof => $mdof."
                push!(dof_map, cdof => mdof)
            end
        end
    end

    # Need to untangle in case we have 1 => 2 and 2 => 3 into 1 => 3 and 2 => 3.
    # Note that a single pass is enough (no need to iterate) since all constraints are
    # between just one mirror dof and one image dof.
    remaps = Dict{dof_map_t, dof_map_t}()
    for (k, v) in dof_map
        if haskey(dof_map, v)
            remaps[k] = get(remaps, v, dof_map[v])
        end
    end
    for (k, v) in remaps
        # @info "Remapping $k => $(dof_map[k]) to $k => $v"
        dof_map[k] = v
    end
    @assert isempty(intersect(keys(dof_map), values(dof_map)))

    # For legacy code add Dirichlet conditions in the corners
    if is_legacy
        Base.depwarn(
            "It looks like you are using legacy code for PeriodicDirichlet " *
                "meaning that the solution is automatically locked in the \"corners\"." *
                "This will not be done automatically in the future. Instead add a " *
                "Dirichlet boundary condition on the relevant nodeset.",
            :PeriodicDirichlet
        )
        all_node_idxs = Set{Int}()
        Tx = get_coordinate_type(grid)
        min_x = Tx(i -> typemax(eltype(Tx)))
        max_x = Tx(i -> typemin(eltype(Tx)))
        for facetpair in facet_map, facet_indices in (facetpair.mirror, facetpair.image)
            cellidx, facetidx = facet_indices
            nodes = facets(grid.cells[cellidx])[facetidx]
            union!(all_node_idxs, nodes)
            for n in nodes
                x = get_node_coordinate(grid, n)
                min_x = Tx(i -> min(min_x[i], x[i]))
                max_x = Tx(i -> max(max_x[i], x[i]))
            end
        end
        all_node_idxs_v = collect(all_node_idxs)
        points = construct_cornerish(min_x, max_x)
        tree = KDTree(Tx[get_node_coordinate(grid, i) for i in all_node_idxs_v])
        idxs, _ = NearestNeighbors.nn(tree, points)
        corner_set = OrderedSet{Int}(all_node_idxs_v[i] for i in idxs)

        dbc = Dirichlet(
            pdbc.field_name, corner_set,
            pdbc.func === nothing ? (x, _) -> pdbc.components * eltype(x)(0) : pdbc.func,
            pdbc.components
        )

        # Create a temp constraint handler just to find the dofs in the nodes...
        chtmp = ConstraintHandler(ch.dh)
        add!(chtmp, dbc)
        close!(chtmp)
        # No need to update!(chtmp, t) here since we only care about the dofs
        # TODO: Right? maybe if the user passed f we need to...
        foreach(x -> delete!(dof_map, x), chtmp.prescribed_dofs)

        # Need to reset the internal of this DBC in order to add! it again...
        resize!(dbc.local_facet_dofs, 0)
        resize!(dbc.local_facet_dofs_offset, 0)

        # Add the Dirichlet for the corners
        add!(ch, dbc)
    end

    inhomogeneity_map = nothing
    if pdbc.func !== nothing
        # Create another temp constraint handler if we need to compute inhomogeneities
        chtmp2 = ConstraintHandler(ch.dh)
        all_facets = OrderedSet{FacetIndex}()
        union!(all_facets, (x.mirror for x in facet_map))
        union!(all_facets, (x.image for x in facet_map))
        dbc_all = Dirichlet(pdbc.field_name, all_facets, pdbc.func, pdbc.components)
        add!(chtmp2, dbc_all); close!(chtmp2)
        # Call update! here since we need it to construct the affine constraints...
        # TODO: This doesn't allow for time dependent constraints...
        update!(chtmp2, 0.0)
        inhomogeneity_map = Dict{Int, Float64}()
        for (k, v) in dof_map
            g = chtmp2.inhomogeneities
            push!(
                inhomogeneity_map,
                k => - g[chtmp2.dofmapping[v]] + g[chtmp2.dofmapping[k]]
            )
        end
    end

    # Any remaining mappings are added as homogeneous AffineConstraints
    for (k, v) in dof_map
        if dof_map_t === Int
            ac = AffineConstraint(k, [v => 1.0], inhomogeneity_map === nothing ? 0.0 : inhomogeneity_map[k])
            add!(ch, ac)
        else
            @assert inhomogeneity_map === nothing
            @assert rotation_matrix !== nothing
            for (i, ki) in pairs(k)
                # u_mirror = R ⋅ u_image
                vs = Pair{Int, eltype(T)}[v[j] => rotation_matrix[i, j] for j in 1:length(v)]
                ac = AffineConstraint(ki, vs, 0.0)
                add!(ch, ac)
            end
        end
    end

    return ch
end

function construct_cornerish(min_x::V, max_x::V) where {T, V <: Vec{1, T}}
    lx = max_x - min_x
    max_x += lx
    min_x -= lx
    return V[min_x, max_x]
end
function construct_cornerish(min_x::V, max_x::V) where {T, V <: Vec{2, T}}
    lx = max_x - min_x
    max_x += lx
    min_x -= lx
    return V[
        max_x,
        min_x,
        Vec{2, T}((max_x[1], min_x[2])),
        Vec{2, T}((min_x[1], max_x[2])),
    ]
end
function construct_cornerish(min_x::V, max_x::V) where {T, V <: Vec{3, T}}
    lx = max_x - min_x
    max_x += lx
    min_x -= lx
    return V[
        min_x,
        max_x,
        Vec{3, T}((max_x[1], min_x[2], min_x[3])),
        Vec{3, T}((max_x[1], max_x[2], min_x[3])),
        Vec{3, T}((min_x[1], max_x[2], min_x[3])),
        Vec{3, T}((min_x[1], min_x[2], max_x[3])),
        Vec{3, T}((max_x[1], min_x[2], max_x[3])),
        Vec{3, T}((min_x[1], max_x[2], max_x[3])),
    ]
end

function mirror_local_dofs(_, _, ::Lagrange{RefLine}, ::Int)
    # For 1D there is nothing to do
end
function mirror_local_dofs(local_facet_dofs, local_facet_dofs_offset, ip::Lagrange{<:Union{RefQuadrilateral, RefTriangle}}, n::Int)
    # For 2D we always permute since Ferrite defines dofs counter-clockwise
    ret = collect(1:length(local_facet_dofs))
    for (i, f) in enumerate(dirichlet_facetdof_indices(ip))
        this_offset = local_facet_dofs_offset[i]
        other_offset = this_offset + n
        for d in 1:n
            idx1 = this_offset + (d - 1)
            idx2 = other_offset + (d - 1)
            tmp = ret[idx1]
            ret[idx1] = ret[idx2]
            ret[idx2] = tmp
        end
    end
    return ret
end

# TODO: Can probably be combined with the method above.
function mirror_local_dofs(local_facet_dofs, local_facet_dofs_offset, ip::Lagrange{<:Union{RefHexahedron, RefTetrahedron}, O}, n::Int) where {O}
    @assert 1 <= O <= 2
    N = ip isa Lagrange{RefHexahedron} ? 4 : 3
    ret = collect(1:length(local_facet_dofs))

    # Mirror by changing from counter-clockwise to clockwise
    for (i, f) in enumerate(dirichlet_facetdof_indices(ip))
        r = local_facet_dofs_offset[i]:(local_facet_dofs_offset[i + 1] - 1)
        # 1. Rotate the corners
        vertex_range = r[1:(N * n)]
        vlr = @view ret[vertex_range]
        for i in 1:N
            reverse!(vlr, (i - 1) * n + 1, i * n)
        end
        reverse!(vlr)
        circshift!(vlr, n)
        # 2. Rotate the edge dofs for quadratic interpolation
        if O > 1
            edge_range = r[(N * n + 1):(2N * n)]
            elr = @view ret[edge_range]
            for i in 1:N
                reverse!(elr, (i - 1) * n + 1, i * n)
            end
            reverse!(elr)
            # circshift!(elr, n) # !!! Note: no shift here
        end
    end
    return ret
end

function rotate_local_dofs(local_facet_dofs, local_facet_dofs_offset, ip::Lagrange{<:Union{RefQuadrilateral, RefTriangle}}, ncomponents)
    return collect(1:length(local_facet_dofs)) # TODO: Return range?
end
function rotate_local_dofs(local_facet_dofs, local_facet_dofs_offset, ip::Lagrange{<:Union{RefHexahedron, RefTetrahedron}, O}, ncomponents) where {O}
    @assert 1 <= O <= 2
    N = ip isa Lagrange{RefHexahedron} ? 4 : 3
    ret = similar(local_facet_dofs, length(local_facet_dofs), N)
    ret[:, :] .= 1:length(local_facet_dofs)
    for f in 1:(length(local_facet_dofs_offset) - 1)
        facet_range = local_facet_dofs_offset[f]:(local_facet_dofs_offset[f + 1] - 1)
        for i in 1:(N - 1)
            # 1. Rotate the vertex dofs
            vertex_range = facet_range[1:(N * ncomponents)]
            circshift!(@view(ret[vertex_range, i + 1]), @view(ret[vertex_range, i]), -ncomponents)
            # 2. Rotate the edge dofs
            if O > 1
                edge_range = facet_range[(N * ncomponents + 1):(2N * ncomponents)]
                circshift!(@view(ret[edge_range, i + 1]), @view(ret[edge_range, i]), -ncomponents)
            end
        end
    end
    return ret
end

"""
    collect_periodic_facets(grid::Grid, mset, iset, transform::Union{Function,Nothing}=nothing; tol=1e-12)

Match all mirror facets in `mset` with a corresponding image facet in `iset`. Return a
dictionary which maps each mirror facet to a image facet. The result can then be passed to
[`PeriodicDirichlet`](@ref).

`mset` and `iset` can be given as a `String` (an existing facet set in the grid) or as a
`AbstractSet{FacetIndex}` directly.

By default this function looks for a matching facet in the directions of the coordinate
system. For other types of periodicities the `transform` function can be used. The
`transform` function is applied on the coordinates of the image facet, and is expected to
transform the coordinates to the matching locations in the mirror set.

The keyword `tol` specifies the tolerance (i.e. distance and deviation in facet-normals)
between a image-facet and mirror-facet, for them to be considered matched.

See also: [`collect_periodic_facets!`](@ref), [`PeriodicDirichlet`](@ref).
"""
function collect_periodic_facets(grid::Grid, mset::Union{AbstractSet{FacetIndex}, String}, iset::Union{AbstractSet{FacetIndex}, String}, transform::Union{Function, Nothing} = nothing; tol::Float64 = 1.0e-12)
    return collect_periodic_facets!(PeriodicFacetPair[], grid, mset, iset, transform; tol)
end

"""
    collect_periodic_facets(grid::Grid, all_facets::Union{AbstractSet{FacetIndex},String,Nothing}=nothing; tol=1e-12)

Split all facets in `all_facets` into image and mirror sets. For each matching pair, the facet
located further along the vector `(1, 1, 1)` becomes the image facet.

If no set is given, all facets on the outer boundary of the grid (i.e. all facets that do not
have a neighbor) is used.

See also: [`collect_periodic_facets!`](@ref), [`PeriodicDirichlet`](@ref).
"""
function collect_periodic_facets(grid::Grid, all_facets::Union{AbstractSet{FacetIndex}, String, Nothing} = nothing; tol::Float64 = 1.0e-12)
    return collect_periodic_facets!(PeriodicFacetPair[], grid, all_facets; tol)
end


"""
    collect_periodic_facets!(facet_map::Vector{PeriodicFacetPair}, grid::Grid, mset, iset, transform::Union{Function,Nothing}; tol=1e-12)

Same as [`collect_periodic_facets`](@ref) but adds all matches to the existing `facet_map`.
"""
function collect_periodic_facets!(facet_map::Vector{PeriodicFacetPair}, grid::Grid, mset::Union{AbstractSet{FacetIndex}, String}, iset::Union{AbstractSet{FacetIndex}, String}, transform::Union{Function, Nothing} = nothing; tol::Float64 = 1.0e-12)
    mset = __to_facetset(grid, mset)
    iset = __to_facetset(grid, iset)
    if transform === nothing
        # This method is destructive, hence the copy
        __collect_periodic_facets_bruteforce!(facet_map, grid, copy(mset), copy(iset), #=known_order=# true, tol)
    else
        # This method relies on ordering, hence the collect
        __collect_periodic_facets_tree!(facet_map, grid, collect(mset), collect(iset), transform, tol)
    end
    return facet_map
end

function collect_periodic_facets!(facet_map::Vector{PeriodicFacetPair}, grid::Grid, facetset::Union{AbstractSet{FacetIndex}, String, Nothing}; tol::Float64 = 1.0e-12)
    facetset = facetset === nothing ? __collect_boundary_facets(grid) : copy(__to_facetset(grid, facetset))
    if mod(length(facetset), 2) != 0
        error("uneven number of facets")
    end
    return __collect_periodic_facets_bruteforce!(facet_map, grid, facetset, facetset, #=known_order=# false, tol)
end

__to_facetset(_, set::AbstractSet{FacetIndex}) = set
__to_facetset(grid, set::String) = getfacetset(grid, set)
function __collect_boundary_facets(grid::Grid)
    candidates = Dict{Tuple, FacetIndex}()
    for (ci, c) in enumerate(grid.cells)
        for (fi, fn) in enumerate(facets(c))
            facet = sortfacet_fast(fn)
            if haskey(candidates, facet)
                delete!(candidates, facet)
            else
                candidates[facet] = FacetIndex(ci, fi)
            end
        end
    end
    return OrderedSet{FacetIndex}(values(candidates))
end

function __collect_periodic_facets_tree!(facet_map::Vector{PeriodicFacetPair}, grid::Grid, mset::Vector{FacetIndex}, iset::Vector{FacetIndex}, transformation::F, tol::Float64) where {F <: Function}
    if length(mset) != length(mset)
        error("different number of facets in mirror and image set")
    end
    Tx = get_coordinate_type(grid)

    mirror_mean_x = Tx[]
    for (c, f) in mset
        fn = facets(grid.cells[c])[f]
        push!(mirror_mean_x, sum(get_node_coordinate(grid, i) for i in fn) / length(fn))
    end

    # Same dance for the image
    image_mean_x = Tx[]
    for (c, f) in iset
        fn = facets(grid.cells[c])[f]
        # Apply transformation to all coordinates
        push!(image_mean_x, sum(transformation(get_node_coordinate(grid, i))::Tx for i in fn) / length(fn))
    end

    # Use KDTree to find closest facet
    tree = KDTree(image_mean_x)
    idxs, _ = NearestNeighbors.nn(tree, mirror_mean_x)
    for (midx, iidx) in zip(eachindex(mset), idxs)
        r = __check_periodic_facets_f(grid, mset[midx], iset[iidx], mirror_mean_x[midx], image_mean_x[iidx], transformation, tol)
        if r === nothing
            error("Could not find matching facet for $(mset[midx])")
        end
        push!(facet_map, r)
    end

    # Make sure the mapping is unique
    @assert all(x -> in(x, Set{FacetIndex}(p.mirror for p in facet_map)), mset)
    @assert all(x -> in(x, Set{FacetIndex}(p.image for p in facet_map)), iset)
    if !allunique(Set{FacetIndex}(p.image for p in facet_map))
        error("did not find a unique mapping between facets")
    end

    return facet_map
end

# This method empties mset and iset
function __collect_periodic_facets_bruteforce!(facet_map::Vector{PeriodicFacetPair}, grid::Grid, mset::AbstractSet{FacetIndex}, iset::AbstractSet{FacetIndex}, known_order::Bool, tol::Float64)
    if length(mset) != length(iset)
        error("different facets in mirror and image")
    end
    while length(mset) > 0
        fi = first(mset)
        found = false
        for fj in iset
            fi == fj && continue
            r = __check_periodic_facets(grid, fi, fj, known_order, tol)
            r === nothing && continue
            push!(facet_map, r)
            delete!(mset, fi)
            delete!(iset, fj)
            found = true
            break
        end
        found || error("did not find a corresponding periodic facet")
    end
    @assert isempty(mset) && isempty(iset)
    return facet_map
end

function __periodic_options(::T) where {T <: Vec{2}}
    # (3^2 - 1) / 2 options
    return (
        Vec{2}((1.0, 0.0)),
        Vec{2}((0.0, 1.0)),
        Vec{2}((1.0, 1.0)) / sqrt(2),
        Vec{2}((1.0, -1.0)) / sqrt(2),
    )
end
function __periodic_options(::T) where {T <: Vec{3}}
    # (3^3 - 1) / 2 options
    return (
        Vec{3}((1.0, 0.0, 0.0)),
        Vec{3}((0.0, 1.0, 0.0)),
        Vec{3}((0.0, 0.0, 1.0)),
        Vec{3}((1.0, 1.0, 0.0)) / sqrt(2),
        Vec{3}((0.0, 1.0, 1.0)) / sqrt(2),
        Vec{3}((1.0, 0.0, 1.0)) / sqrt(2),
        Vec{3}((1.0, 1.0, 1.0)) / sqrt(3),
        Vec{3}((1.0, -1.0, 0.0)) / sqrt(2),
        Vec{3}((0.0, 1.0, -1.0)) / sqrt(2),
        Vec{3}((1.0, 0.0, -1.0)) / sqrt(2),
        Vec{3}((1.0, 1.0, -1.0)) / sqrt(3),
        Vec{3}((1.0, -1.0, 1.0)) / sqrt(3),
        Vec{3}((1.0, -1.0, -1.0)) / sqrt(3),
    )
end

function __outward_normal(grid::Grid{2}, nodes, transformation::F = identity) where {F <: Function}
    n1::Vec{2} = transformation(get_node_coordinate(grid, nodes[1]))
    n2::Vec{2} = transformation(get_node_coordinate(grid, nodes[2]))
    n = Vec{2}((n2[2] - n1[2], - n2[1] + n1[1]))
    return n / norm(n)
end

function __outward_normal(grid::Grid{3}, nodes, transformation::F = identity) where {F <: Function}
    n1::Vec{3} = transformation(get_node_coordinate(grid, nodes[1]))
    n2::Vec{3} = transformation(get_node_coordinate(grid, nodes[2]))
    n3::Vec{3} = transformation(get_node_coordinate(grid, nodes[3]))
    n = (n3 - n2) × (n1 - n2)
    return n / norm(n)
end

function circshift_tuple(x::T, n) where {T}
    return Tuple(circshift!(collect(x), n))::T
end

# Check if two facets are periodic. This method assumes that the facets are mirrored and thus
# have opposing normal vectors
function __check_periodic_facets(grid::Grid, fi::FacetIndex, fj::FacetIndex, known_order::Bool, tol::Float64)
    cii, fii = fi
    nodes_i = facets(grid.cells[cii])[fii]
    cij, fij = fj
    nodes_j = facets(grid.cells[cij])[fij]

    # 1. Check that normals are opposite TODO: Should use FacetValues here
    ni = __outward_normal(grid, nodes_i)
    nj = __outward_normal(grid, nodes_j)
    if norm(ni + nj) >= tol
        return nothing
    end

    # 2. Find the periodic direction using the vector between the midpoint of the facets
    xmi = sum(get_node_coordinate(grid, i) for i in nodes_i) / length(nodes_i)
    xmj = sum(get_node_coordinate(grid, j) for j in nodes_j) / length(nodes_j)
    xmij = xmj - xmi
    h = 2 * norm(xmj - get_node_coordinate(grid, nodes_j[1])) # Approximate element size
    TOLh = tol * h
    found = false
    local len
    for o in __periodic_options(xmij)
        len = xmij ⋅ o
        if norm(xmij - len * o) < TOLh
            found = true
            break
        end
    end
    found || return nothing

    # 3. Check that the first node of fj have a corresponding node in fi
    #    In this method facets are mirrored (opposite normal vectors) so reverse the nodes
    nodes_i = circshift_tuple(reverse(nodes_i), 1)
    xj = get_node_coordinate(grid, nodes_j[1])
    node_rot = 0
    found = false
    for i in eachindex(nodes_i)
        xi = get_node_coordinate(grid, nodes_i[i])
        xij = xj - xi
        if norm(xij - xmij) < TOLh
            found = true
            break
        end
        node_rot += 1
    end
    found || return nothing

    # 4. Check the remaining nodes for the same criteria, now with known node_rot
    for j in 2:length(nodes_j)
        xi = get_node_coordinate(grid, nodes_i[mod1(j + node_rot, end)])
        xj = get_node_coordinate(grid, nodes_j[j])
        xij = xj - xi
        if norm(xij - xmij) >= TOLh
            return nothing
        end
    end

    # Rotation is only relevant for 3D
    if getspatialdim(grid) == 3
        node_rot = mod(node_rot, length(nodes_i))
    else
        node_rot = 0
    end

    # 5. Facets match! Facet below the diagonal become the mirror.
    if known_order || len > 0
        return PeriodicFacetPair(fi, fj, node_rot, true)
    else
        return PeriodicFacetPair(fj, fi, node_rot, true)
    end
end

# This method is quite similar to __check_periodic_facets, but is used when user have passed
# a transformation function and we have then used the KDTree to find the matching pair of
# facets. This function only need to i) check whether facets have aligned or opposite normal
# vectors, and ii) compute the relative rotation.
function __check_periodic_facets_f(grid::Grid, fi::FacetIndex, fj::FacetIndex, xmi, xmj, transformation::F, tol::Float64) where {F}
    cii, fii = fi
    nodes_i = facets(grid.cells[cii])[fii]
    cij, fij = fj
    nodes_j = facets(grid.cells[cij])[fij]

    # 1. Check if normals are aligned or opposite TODO: Should use FacetValues here
    ni = __outward_normal(grid, nodes_i)
    nj = __outward_normal(grid, nodes_j, transformation)
    if norm(ni + nj) < tol
        mirror = true
    elseif norm(ni - nj) < tol
        mirror = false
    else
        return nothing
    end

    # 2. Compute the relative rotation
    xmij = xmj - xmi
    h = 2 * norm(xmj - get_node_coordinate(grid, nodes_j[1])) # Approximate element size
    TOLh = tol * h
    nodes_i = mirror ? circshift_tuple(reverse(nodes_i), 1) : nodes_i # reverse if necessary
    xj = transformation(get_node_coordinate(grid, nodes_j[1]))
    node_rot = 0
    found = false
    for i in eachindex(nodes_i)
        xi = get_node_coordinate(grid, nodes_i[i])
        xij = xj - xi
        if norm(xij - xmij) < TOLh
            found = true
            break
        end
        node_rot += 1
    end
    found || return nothing

    # 3. Rotation is only relevant for 3D.
    if getspatialdim(grid) == 3
        node_rot = mod(node_rot, length(nodes_i))
    else
        node_rot = 0
    end

    return PeriodicFacetPair(fi, fj, node_rot, mirror)
end


######################################
## Local application of constraints ##
######################################

"""
    apply_local!(
        local_matrix::AbstractMatrix, local_vector::AbstractVector,
        global_dofs::AbstractVector, ch::ConstraintHandler;
        apply_zero::Bool = false
    )

Similar to [`apply!`](@ref) but perform condensation of constrained degrees-of-freedom
locally in `local_matrix` and `local_vector` *before* they are to be assembled into the
global system.

When the keyword argument `apply_zero` is `true` all inhomogeneities are set to `0` (cf.
[`apply!`](@ref) vs [`apply_zero!`](@ref)).

This method can only be used if all constraints are "local", i.e. no constraint couples with
dofs outside of the element dofs (`global_dofs`) since condensation of such constraints
requires writing to entries in the global matrix/vector. For such a case,
[`apply_assemble!`](@ref) can be used instead.

Note that this method is destructive since it, by definition, modifies `local_matrix` and
`local_vector`.
"""
function apply_local!(
        local_matrix::AbstractMatrix, local_vector::AbstractVector,
        global_dofs::AbstractVector, ch::ConstraintHandler;
        apply_zero::Bool = false
    )
    return _apply_local!(
        local_matrix, local_vector, global_dofs, ch, apply_zero,
        #=global_matrix=# nothing, #=global_vector=# nothing
    )
end

# Element local application of boundary conditions. Global matrix and vectors are necessary
# if there are affine constraints that connect dofs from different elements.
function _apply_local!(
        local_matrix::AbstractMatrix, local_vector::AbstractVector,
        global_dofs::AbstractVector, ch::ConstraintHandler, apply_zero::Bool,
        global_matrix, global_vector
    )
    @assert isclosed(ch)
    # TODO: With apply_zero it shouldn't be required to pass the vector.
    length(global_dofs) == size(local_matrix, 1) == size(local_matrix, 2) == length(local_vector) || error("?")
    # First pass over the dofs check whether there are any constrained dofs at all
    has_constraints = false
    has_nontrivial_affine_constraints = false
    # 1. Adjust local vector
    @inbounds for (local_dof, global_dof) in pairs(global_dofs)
        # Check if this dof is constrained
        pdofs_index = get(ch.dofmapping, global_dof, nothing)
        pdofs_index === nothing && continue # Not constrained, move on
        has_constraints = true
        # Add inhomogeneities to local_vector: local_vector - local_matrix * inhomogeneities
        v = ch.inhomogeneities[pdofs_index]
        if !apply_zero && v != 0
            for j in axes(local_matrix, 1)
                local_vector[j] -= v * local_matrix[j, local_dof]
            end
        end
        # Check if this is an affine constraint
        has_nontrivial_affine_constraints = has_nontrivial_affine_constraints || (
            coeffs = ch.dofcoefficients[pdofs_index];
            !(coeffs === nothing || isempty(coeffs))
        )
    end
    # 2. Compute mean of diagonal before modifying local matrix
    m = has_constraints ? meandiag(local_matrix) : zero(eltype(local_matrix))
    # 3. Condense any affine constraints
    if has_nontrivial_affine_constraints
        # Condense this constraint locally if possible, and otherwise modifies the global arrays.
        _condense_local!(local_matrix, local_vector, global_matrix, global_vector, global_dofs, ch.dofmapping, ch.dofcoefficients)
    end
    # 4. Zero out columns/rows of local matrix and replace diagonal entries with the mean
    if has_constraints
        @inbounds for (local_dof, global_dof) in pairs(global_dofs)
            pdofs_index = get(ch.dofmapping, global_dof, nothing)
            pdofs_index === nothing && continue # Not constrained, move on
            # Zero the column
            for local_row in axes(local_matrix, 1)
                local_matrix[local_row, local_dof] = 0
            end
            # Zero the row
            for local_col in axes(local_matrix, 2)
                local_matrix[local_dof, local_col] = 0
            end
            # Replace diagonal with mean
            local_matrix[local_dof, local_dof] = m
            v = ch.inhomogeneities[pdofs_index]
            local_vector[local_dof] = apply_zero ? zero(eltype(local_vector)) : (v * m)
        end
    end
    return
end

@noinline missing_global() = error("can not condense constraint without the global matrix and vector")

"""
    _condense_local!(local_matrix::AbstractMatrix, local_vector::AbstractVector,
                    global_matrix#=::SparseMatrixCSC=#, global_vector#=::Vector=#,
                    global_dofs::AbstractVector, dofmapping::Dict, dofcoefficients::Vector)

Condensation of affine constraints on element level. If possible this function only
modifies the local arrays.
"""
function _condense_local!(
        local_matrix::AbstractMatrix, local_vector::AbstractVector,
        global_matrix #=::SparseMatrixCSC=#, global_vector #=::Vector=#,
        global_dofs::AbstractVector, dofmapping::Dict, dofcoefficients::Vector
    )
    @assert axes(local_matrix, 1) == axes(local_matrix, 2) ==
        axes(local_vector, 1) == axes(global_dofs, 1)
    has_global_arrays = global_matrix !== nothing && global_vector !== nothing
    for (local_col, global_col) in pairs(global_dofs)
        col_coeffs = coefficients_for_dof(dofmapping, dofcoefficients, global_col)
        if col_coeffs === nothing
            for (local_row, global_row) in pairs(global_dofs)
                m = local_matrix[local_row, local_col]
                iszero(m) && continue # Skip early when zero to avoid remaining lookups
                row_coeffs = coefficients_for_dof(dofmapping, dofcoefficients, global_row)
                row_coeffs === nothing && continue # Neither the column nor the row are constrained: Do nothing
                for (global_mrow, weight) in row_coeffs
                    mw = m * weight
                    local_mrow = findfirst(==(global_mrow), global_dofs)
                    if local_mrow === nothing
                        # Only modify the global array if this isn't prescribed since we
                        # can't zero it out later like with the local matrix.
                        if !haskey(dofmapping, global_col) && !haskey(dofmapping, global_mrow)
                            has_global_arrays || missing_global()
                            addindex!(global_matrix, mw, global_mrow, global_col)
                        end
                    else
                        local_matrix[local_mrow, local_col] += mw
                    end
                end
            end
        else
            for (local_row, global_row) in pairs(global_dofs)
                m = local_matrix[local_row, local_col]
                iszero(m) && continue # Skip early when zero to avoid remaining lookups
                row_coeffs = coefficients_for_dof(dofmapping, dofcoefficients, global_row)
                if row_coeffs === nothing
                    for (global_mcol, weight) in col_coeffs
                        local_mcol = findfirst(==(global_mcol), global_dofs)
                        mw = m * weight
                        if local_mcol === nothing
                            # Only modify the global array if this isn't prescribed since we
                            # can't zero it out later like with the local matrix.
                            if !haskey(dofmapping, global_row) && !haskey(dofmapping, global_mcol)
                                has_global_arrays || missing_global()
                                addindex!(global_matrix, mw, global_row, global_mcol)
                            end
                        else
                            local_matrix[local_row, local_mcol] += mw
                        end
                    end
                else
                    for (global_mcol, weight_col) in col_coeffs
                        local_mcol = findfirst(==(global_mcol), global_dofs)
                        for (global_mrow, weight_row) in row_coeffs
                            mww = m * weight_col * weight_row
                            local_mrow = findfirst(==(global_mrow), global_dofs)
                            if local_mcol === nothing || local_mrow === nothing
                                # Only modify the global array if this isn't prescribed since we
                                # can't zero it out later like with the local matrix.
                                if !haskey(dofmapping, global_mrow) && !haskey(dofmapping, global_mcol)
                                    has_global_arrays || missing_global()
                                    addindex!(global_matrix, mww, global_mrow, global_mcol)
                                end
                            else
                                local_matrix[local_mrow, local_mcol] += mww
                            end
                        end
                    end
                end
            end
            for (global_mcol, weight) in col_coeffs
                vw = local_vector[local_col] * weight
                local_mcol = findfirst(==(global_mcol), global_dofs)
                if local_mcol === nothing
                    has_global_arrays || missing_global()
                    addindex!(global_vector, vw, global_mcol)
                else
                    local_vector[local_mcol] += vw
                end
            end
            local_vector[local_col] = 0
        end
    end
    return
end


function _default_bc_qr_order(user_provided::Int, ip::Interpolation)
    user_provided > 0 && return user_provided
    return _default_bc_qr_order(ip)
end
# Q&D default, should be more elaborated
_default_bc_qr_order(::Interpolation{<:Any, order}) where {order} = 2 * order

function add!(ch::ConstraintHandler, bc::ProjectedDirichlet)
    # Duplicate the Dirichlet constraint for every SubDofHandler
    dbc_added = false
    for sdh in ch.dh.subdofhandlers
        # Skip if the constrained field does not live on this sub domain
        bc.field_name in sdh.field_names || continue
        # Compute the intersection between bc.set and the cellset of this
        # SubDofHandler and skip if the set is empty
        filtered_set = filter_dbc_set(get_grid(ch.dh), sdh.cellset, bc.facets)
        isempty(filtered_set) && continue
        # Fetch information about the field on this SubDofHandler
        field_idx = find_field(sdh, bc.field_name)
        interpolation = getfieldinterpolation(sdh, field_idx)
        CT = getcelltype(sdh) # Same celltype enforced in SubDofHandler constructor
        qr_order = _default_bc_qr_order(bc.qr_order, interpolation)
        fqr = FacetQuadratureRule{getrefshape(interpolation)}(qr_order)
        fv = FacetValues(fqr, interpolation, geometric_interpolation(CT))
        local_facet_dofs, local_facet_dofs_offset =
            _local_facet_dofs_for_bc(get_base_interpolation(interpolation), 1, 1, field_offset(sdh, field_idx), dirichlet_facetdof_indices)
        facet_dofs = ArrayOfVectorViews(local_facet_dofs_offset, local_facet_dofs, LinearIndices(1:(length(local_facet_dofs_offset) - 1)))

        filtered_dbc = ProjectedDirichlet(bc.f, filtered_set, bc.field_name, qr_order, fv, facet_dofs)

        _add!(ch, filtered_dbc, facet_dofs)

        dbc_added = true
    end
    dbc_added || error("No overlap between bc::ProjectedDirichlet and fields in the ConstraintHandler's DofHandler")
    return ch
end

function _add!(ch::ConstraintHandler, bc::ProjectedDirichlet, facet_dofs)
    # loop over all the faces in the set and add the global dofs to `constrained_dofs`
    constrained_dofs = Int[]
    cc = CellCache(ch.dh, UpdateFlags(; nodes = false, coords = false, dofs = true))
    for (cellidx, facetidx) in bc.facets
        reinit!(cc, cellidx)
        local_dofs = facet_dofs[facetidx]
        for d in local_dofs
            push!(constrained_dofs, cc.dofs[d])
        end
    end

    # save it to the ConstraintHandler
    push!(ch.projbcs, bc)
    for d in constrained_dofs
        add_prescribed_dof!(ch, d, NaN, nothing)
    end
    return ch
end


function _update_projected_dbc!(
        inhomogeneities::Vector{T}, f::Function, facets::AbstractVecOrSet{FacetIndex}, fv::FacetValues, facet_dofs::ArrayOfVectorViews,
        dh::AbstractDofHandler, dofmapping::Dict{Int, Int}, dofcoefficients::Vector{Union{Nothing, DofCoefficients{T}}}, time::Real
    ) where {T}
    ip = get_base_interpolation(function_interpolation(fv)) # Ensures getting error message from `integrate_projected_dbc!`
    max_dofs_per_facet = maximum(length, dirichlet_facetdof_indices(ip))
    Kᶠ = zeros(max_dofs_per_facet, max_dofs_per_facet)
    aᶠ = zeros(max_dofs_per_facet)
    fᶠ = zeros(max_dofs_per_facet)
    for fc in FacetIterator(dh, facets)
        reinit!(fv, fc)
        shape_nrs = dirichlet_facetdof_indices(ip)[getcurrentfacet(fv)]
        solve_projected_dbc!(aᶠ, Kᶠ, fᶠ, f, fv, shape_nrs, getcoordinates(fc), time)
        for (idof, shape_nr) in enumerate(shape_nrs)
            globaldof = celldofs(fc)[shape_nr]
            dbc_index = dofmapping[globaldof]
            # Only DBC dofs are currently update!-able so don't modify inhomogeneities
            # for affine constraints
            if dofcoefficients[dbc_index] === nothing
                inhomogeneities[dbc_index] = aᶠ[idof]
            end
        end
    end
    return nothing
end

function solve_projected_dbc!(aᶠ, Kᶠ, fᶠ, bc_fun, fv, shape_nrs, cell_coords, time)
    fill!(Kᶠ, 0)
    fill!(fᶠ, 0)
    # Supporting varying number of facetdofs (for ref shapes with different facet types) requires
    # for i in (length(shape_nrs) + 1):size(Kᶠ, 1)
    #     Kᶠ[i, i] = 1
    # end
    @assert length(shape_nrs) == size(Kᶠ, 1)
    integrate_projected_dbc!(Kᶠ, fᶠ, bc_fun, fv, shape_nrs, cell_coords, time)
    aᶠ .= Kᶠ \ fᶠ # Could be done non-allocating if required, using e.g. SMatrix
    return aᶠ
end

function integrate_projected_dbc!(Kᶠ, fᶠ, bc_fun, fv, shape_nrs, cell_coords, time)
    return integrate_projected_dbc!(conformity(fv), Kᶠ, fᶠ, bc_fun, fv, shape_nrs, cell_coords, time)
end

function integrate_projected_dbc!(::HdivConformity, Kᶠ, fᶠ, bc_fun, fv, shape_nrs, cell_coords, time)
    for q_point in 1:getnquadpoints(fv)
        dΓ = getdetJdV(fv, q_point)
        n = getnormal(fv, q_point)
        x = spatial_coordinate(fv, q_point, cell_coords)
        qn = bc_fun(x, time, n)
        for (i, I) in enumerate(shape_nrs)
            δN_dot_n = shape_value(fv, q_point, I) ⋅ n
            fᶠ[i] += qn * δN_dot_n * dΓ
            for (j, J) in enumerate(shape_nrs)
                N_dot_n = shape_value(fv, q_point, J) ⋅ n
                Kᶠ[i, j] += (δN_dot_n * N_dot_n) * dΓ
            end
        end
    end
    return
end

function integrate_projected_dbc!(::HcurlConformity, Kᶠ, fᶠ, bc_fun, fv, shape_nrs, cell_coords, time)
    if getrefdim(function_interpolation(fv)) == 3
        throw(ArgumentError("ProjectedDirichlet is not implemented for 3D H(curl) conformity"))
    end
    for q_point in 1:getnquadpoints(fv)
        dΓ = getdetJdV(fv, q_point)
        n = getnormal(fv, q_point)
        x = spatial_coordinate(fv, q_point, cell_coords)
        qt = bc_fun(x, time, n)
        for (i, I) in enumerate(shape_nrs)
            δN_cross_n = shape_value(fv, q_point, I) × n
            fᶠ[i] += (δN_cross_n ⋅ qt) * dΓ
            for (j, J) in enumerate(shape_nrs)
                N_cross_n = shape_value(fv, q_point, J) × n
                Kᶠ[i, j] += (δN_cross_n ⋅ N_cross_n) * dΓ
            end
        end
    end
    return
end

function integrate_projected_dbc!(::Union{H1Conformity, L2Conformity}, _, _, _, fv, args...)
    ip_str = sprint(show, function_interpolation(fv))
    throw(ArgumentError("ProjectedDirichlet is not implemented for H¹ and L2 conformities ($ip_str)"))
end
