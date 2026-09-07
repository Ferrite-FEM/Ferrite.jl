"""
    FacetValues([::Type{T}], quad_rule::FacetQuadratureRule, func_interpol::Interpolation, [geom_interpol::Interpolation])
    FacetValues([::Type{T}], quad_rule::FacetQuadratureRule, func_interpols::NamedTuple, [geom_interpol::Interpolation])

A `FacetValues` object facilitates the process of evaluating values of shape functions, gradients of shape functions,
values of nodal functions, gradients and divergences of nodal functions etc. on the facets of finite elements.

**Arguments:**

* `T`: an optional argument (default to `Float64`) to determine the type the internal data is stored as.
* `quad_rule`: an instance of a [`FacetQuadratureRule`](@ref)
* `func_interpol`: an instance of an [`Interpolation`](@ref) used to interpolate the approximated function
* `func_interpols`: a named tuple with entries of type `Interpolation`, one for each field, see
  *Multiple fields* below
* `geom_interpol`: an optional instance of an [`Interpolation`](@ref) which is used to interpolate the geometry.
  By default linear Lagrange interpolation is used.

**Keyword arguments:** The following keyword arguments are experimental and may change in future minor releases

* `update_gradients`: Specifies if the gradients of the shape functions should be updated (default true)
* `update_hessians`: Specifies if the hessians of the shape functions should be updated (default false)

**Common methods:**

* [`reinit!`](@ref)
* [`getnquadpoints`](@ref)
* [`getdetJdV`](@ref)

* [`shape_value`](@ref)
* [`shape_gradient`](@ref)
* [`shape_symmetric_gradient`](@ref)
* [`shape_divergence`](@ref)

* [`function_value`](@ref)
* [`function_gradient`](@ref)
* [`function_symmetric_gradient`](@ref)
* [`function_divergence`](@ref)
* [`spatial_coordinate`](@ref)

# Multiple fields

Constructed with a `NamedTuple` of interpolations, `fv::FacetValues` works like the multi-field
mode of [`CellValues`](@ref): functions related to specific shape functions are called on
`fv.<name>` (of type [`FunctionValues`](@ref Ferrite.FunctionValues)) while geometric queries
(e.g. `getdetJdV`, `getnormal`, and `spatial_coordinate`) are called on `fv` itself. Fields
with equal interpolations share the same `FunctionValues`.
"""
FacetValues

mutable struct FacetValues{NT, FV, GM, FQR, detT, nT, V_FV <: AbstractVector{FV}, V_GM <: AbstractVector{GM}} <: AbstractFacetValues
    const fun_values_nt::NT # NamedTuple mapping field names to Val-indices into the fun_values tuples, or Nothing (single-field mode)
    const fun_values::V_FV  # AbstractVector{<:Tuple}, one tuple of unique FunctionValues per facet
    const geo_mapping::V_GM # AbstractVector{GeometryMapping}
    const fqr::FQR          # FacetQuadratureRule
    # FacetValues are only functional for the types in the comments for the fields below.
    # However, e.g. for GPU support, we allow arrays of one order higher to be passed, allowing this type to be used as a struct-of-arrays (SoA) type.
    # See `soa_utils.jl` for the SoA transformation infrastructure.
    const detJdV::detT      # AbstractVector{<:Number}
    const normals::nT       # AbstractVector{<:Vec}
    current_facet::Int
end

const SingleFieldFacetValues = FacetValues{Nothing}
const MultiFieldFacetValues = FacetValues{<:NamedTuple}

function FacetValues(
        ::Type{T}, fqr::FacetQuadratureRule, ip_fun::Interpolation, ip_geo::VectorizedInterpolation{sdim},
        ::ValuesUpdateFlags{FunDiffOrder, GeoDiffOrder}
    ) where {T, sdim, FunDiffOrder, GeoDiffOrder}

    # max(GeoDiffOrder, 1) ensures that we get the jacobian needed to calculate the normal.
    geo_mapping = map(qr -> GeometryMapping{max(GeoDiffOrder, 1)}(T, ip_geo.ip, qr), fqr.facet_rules)
    fun_values = map(qr -> (FunctionValues{FunDiffOrder}(T, ip_fun, qr, ip_geo),), fqr.facet_rules)
    max_nquadpoints = maximum(qr -> length(getweights(qr)), fqr.facet_rules)
    # detJdV always calculated, since we needed to calculate the jacobian anyways for the normal.
    detJdV = fill(T(NaN), max_nquadpoints)
    normals = fill(zero(Vec{sdim, T}) * T(NaN), max_nquadpoints)
    return FacetValues(nothing, fun_values, geo_mapping, fqr, detJdV, normals, 1)
end

function FacetValues(
        ::Type{T}, fqr::FacetQuadratureRule, ip_funs::NamedTuple, ip_geo::VectorizedInterpolation{sdim},
        ::ValuesUpdateFlags{FunDiffOrder, GeoDiffOrder}
    ) where {T, sdim, FunDiffOrder, GeoDiffOrder}

    geo_mapping = map(qr -> GeometryMapping{max(GeoDiffOrder, 1)}(T, ip_geo.ip, qr), fqr.facet_rules)
    unique_ips = unique(values(ip_funs)) # Not type-stable, but ok for advanced users this should be constructed outside...
    fun_values = map(qr -> tuple((FunctionValues{FunDiffOrder}(T, ip_fun, qr, ip_geo) for ip_fun in unique_ips)...), fqr.facet_rules)
    fun_values_nt = NamedTuple((key => Val(findfirst(unique_ip -> ip == unique_ip, unique_ips)) for (key, ip) in pairs(ip_funs)))
    max_nquadpoints = maximum(qr -> length(getweights(qr)), fqr.facet_rules)
    detJdV = fill(T(NaN), max_nquadpoints)
    normals = fill(zero(Vec{sdim, T}) * T(NaN), max_nquadpoints)
    return FacetValues(fun_values_nt, fun_values, geo_mapping, fqr, detJdV, normals, 1)
end

FacetValues(qr::FacetQuadratureRule, ip::Union{Interpolation, NamedTuple}, args...; kwargs...) = FacetValues(Float64, qr, ip, args...; kwargs...)
function FacetValues(::Type{T}, qr::FacetQuadratureRule, ip::Union{Interpolation, NamedTuple}, ip_geo::ScalarInterpolation; kwargs...) where {T}
    return FacetValues(T, qr, ip, VectorizedInterpolation(ip_geo); kwargs...)
end
function FacetValues(::Type{T}, qr::FacetQuadratureRule, ip::Union{Interpolation, NamedTuple}, ip_geo::VectorizedInterpolation = default_geometric_interpolation(ip); kwargs...) where {T}
    return FacetValues(T, qr, ip, ip_geo, ValuesUpdateFlags(ip; kwargs...))
end

function Base.copy(fv::FacetValues)
    fun_values = map(fvals -> map(copy, fvals), getfield(fv, :fun_values))
    geo_mapping = map(copy, getfield(fv, :geo_mapping))
    return FacetValues(
        getfield(fv, :fun_values_nt), fun_values, geo_mapping,
        copy(getfield(fv, :fqr)), copy(getfield(fv, :detJdV)), copy(getfield(fv, :normals)), getcurrentfacet(fv)
    )
end

getngeobasefunctions(fv::FacetValues) = getngeobasefunctions(get_geo_mapping(fv))
getnquadpoints(fv::FacetValues) = @inbounds getnquadpoints(getfield(fv, :fqr), getcurrentfacet(fv))
@propagate_inbounds getdetJdV(fv::FacetValues, q_point) = getfield(fv, :detJdV)[q_point]

geometric_interpolation(fv::FacetValues) = geometric_interpolation(get_geo_mapping(fv))

get_geo_mapping(fv::FacetValues) = @inbounds getfield(fv, :geo_mapping)[getcurrentfacet(fv)]
@propagate_inbounds geometric_value(fv::FacetValues, args...) = geometric_value(get_geo_mapping(fv), args...)

get_fun_values(fv::FacetValues) = @inbounds getfield(fv, :fun_values)[getcurrentfacet(fv)]
@inline _single_fun_values(fv::SingleFieldFacetValues) = get_fun_values(fv)[1]

getnbasefunctions(fv::SingleFieldFacetValues) = getnbasefunctions(_single_fun_values(fv))
shape_value_type(fv::SingleFieldFacetValues) = shape_value_type(_single_fun_values(fv))
shape_gradient_type(fv::SingleFieldFacetValues) = shape_gradient_type(_single_fun_values(fv))
function_interpolation(fv::SingleFieldFacetValues) = function_interpolation(_single_fun_values(fv))
function_difforder(fv::SingleFieldFacetValues) = function_difforder(_single_fun_values(fv))

@propagate_inbounds shape_value(fv::SingleFieldFacetValues, q_point::Int, i::Int) = shape_value(_single_fun_values(fv), q_point, i)
@propagate_inbounds shape_gradient(fv::SingleFieldFacetValues, q_point::Int, i::Int) = shape_gradient(_single_fun_values(fv), q_point, i)
@propagate_inbounds shape_hessian(fv::SingleFieldFacetValues, q_point::Int, i::Int) = shape_hessian(_single_fun_values(fv), q_point, i)

# In multi-field mode, properties expose (only) the named FunctionValues of the current
# facet, e.g. `fv.u`. Single-field mode keeps the default `getproperty`.
@inline Base.getproperty(fv::MultiFieldFacetValues, key::Symbol) = _tuple_index(get_fun_values(fv), getproperty(getfield(fv, :fun_values_nt), key))
Base.propertynames(fv::MultiFieldFacetValues) = propertynames(getfield(fv, :fun_values_nt))

"""
    getcurrentfacet(fv::FacetValues)

Return the current active facet of the `FacetValues` object (from last `reinit!`).
"""
getcurrentfacet(fv::FacetValues) = getfield(fv, :current_facet)

"""
    getnormal(fv::FacetValues, qp::Int)

Return the normal at the quadrature point `qp` for the active facet of the
`FacetValues` object(from last `reinit!`).
"""
getnormal(fv::FacetValues, qp::Int) = getfield(fv, :normals)[qp]

nfacets(fv::FacetValues) = length(getfield(fv, :geo_mapping))

function set_current_facet!(fv::FacetValues, facet_nr::Int)
    # Checking facet_nr before setting current_facet allows us to use @inbounds
    # when indexing by getcurrentfacet(fv) in other places!
    checkbounds(Bool, 1:nfacets(fv), facet_nr) || throw(ArgumentError("Facet nr is out of range."))
    setfield!(fv, :current_facet, facet_nr)
    return
end

@inline function reinit!(fv::AbstractFacetValues, x::AbstractVector, facet_nr::Int)
    return reinit!(fv, nothing, x, facet_nr)
end

@inline function reinit_needs_cell(fv::FacetValues)
    return any(map(fvals -> !isa(mapping_type(fvals), IdentityMapping), get_fun_values(fv)))
end

function check_reinit_sdim_consistency(fv::FacetValues, ::AbstractVector{VT}) where {VT}
    return map(fvals -> check_reinit_sdim_consistency(:FacetValues, shape_gradient_type(fvals), VT), get_fun_values(fv))
end

function reinit!(fv::FacetValues, cell::Union{AbstractCell, Nothing}, x::AbstractVector{Vec{dim, T}}, facet_nr::Int) where {dim, T}
    check_reinit_sdim_consistency(fv, x)
    set_current_facet!(fv, facet_nr)
    n_geom_basefuncs = getngeobasefunctions(fv)
    if !checkbounds(Bool, x, 1:n_geom_basefuncs) || length(x) != n_geom_basefuncs
        throw_incompatible_coord_length(length(x), n_geom_basefuncs)
    end

    geo_mapping = get_geo_mapping(fv)
    fun_values = get_fun_values(fv)

    if cell === nothing && reinit_needs_cell(fv)
        throw(ArgumentError("The cell::AbstractCell input is required to reinit! non-identity function mappings"))
    end

    detJdV = getfield(fv, :detJdV)
    normals = getfield(fv, :normals)
    @inbounds for (q_point, w) in pairs(getweights(getfield(fv, :fqr), facet_nr))
        mapping = calculate_mapping(geo_mapping, q_point, x)
        J = getjacobian(mapping)
        # See the `Ferrite.embedding_det` docstring for more background
        weight_norm = weighted_normal(J, getrefshape(geo_mapping.ip), facet_nr)
        detJ = norm(weight_norm)
        detJ > 0.0 || throw_detJ_not_pos(detJ)
        @inbounds detJdV[q_point] = detJ * w
        @inbounds normals[q_point] = weight_norm / norm(weight_norm)
        apply_mapping!(fun_values, q_point, mapping, cell)
    end
    return
end

function _show_nqp_per_facet(io::IO, fv::FacetValues)
    nqp = getnquadpoints.(getfield(fv, :fqr).facet_rules)
    if all(n == first(nqp) for n in nqp)
        println(io, first(nqp), " quadrature points per facet")
    else
        println(io, tuple(nqp...), " quadrature points on each facet")
    end
    return
end

function Base.show(io::IO, d::MIME"text/plain", fv::SingleFieldFacetValues)
    ip_geo = geometric_interpolation(fv)
    rdim = getrefdim(ip_geo)
    vdim = isa(shape_value(fv, 1, 1), Vec) ? length(shape_value(fv, 1, 1)) : 0
    GradT = shape_gradient_type(fv)
    sdim = GradT === nothing ? nothing : sdim_from_gradtype(GradT)
    vstr = vdim == 0 ? "scalar" : "vdim=$vdim"
    print(io, "FacetValues(", vstr, ", rdim=$rdim, sdim=$sdim): ")
    _show_nqp_per_facet(io, fv)
    print(io, " Function interpolation: "); show(io, d, function_interpolation(fv))
    print(io, "\nGeometric interpolation: ")
    sdim === nothing ? show(io, d, ip_geo) : show(io, d, ip_geo^sdim)
    return
end

function Base.show(io::IO, d::MIME"text/plain", fv::MultiFieldFacetValues)
    ip_geo = geometric_interpolation(fv)
    GradT = shape_gradient_type(first(get_fun_values(fv)))
    sdim = GradT === nothing ? nothing : sdim_from_gradtype(GradT)
    print(io, "FacetValues with ")
    _show_nqp_per_facet(io, fv)
    print(io, "Geometric interpolation: ")
    sdim === nothing ? show(io, d, ip_geo) : show(io, d, ip_geo^sdim)
    print(io, "\nFunction interpolations")
    for key in propertynames(fv)
        ip = function_interpolation(getproperty(fv, key))
        print(io, "\n  ", key, ": "); show(io, d, ip)
    end
    return
end

# Error messages for functions that should be called on individual `FunctionValues` to help users
function getnbasefunctions(fv::MultiFieldFacetValues)
    k = first(propertynames(fv)) # Pick the first function values to use in example
    throw(ArgumentError("getnbasefunctions isn't applicable to a multi-field FacetValues. Use on `FunctionValues` for the specific field, e.g. getnbasefunctions(fv.$k)"))
end

for f in (:shape_value, :shape_gradient, :shape_symmetric_gradient, :shape_divergence)
    @eval function $f(fv::MultiFieldFacetValues, ::Int, ::Int)
        k = first(propertynames(fv)) # Pick the first function values to use in example
        fun = $f                       # Make the function name available to use in the error message
        throw(ArgumentError("$fun isn't applicable to a multi-field FacetValues. Use on `FunctionValues` for the specific field, e.g. $fun(fv.$k, q_point, shapenr)"))
    end
end
for f in (:function_value, :function_gradient, :function_symmetric_gradient, :function_divergence)
    @eval function $f(fv::MultiFieldFacetValues, ::Int, ::AbstractVector, args...)
        k = first(propertynames(fv)) # Pick the first function values to use in example
        fun = $f                       # Make the function name available to use in the error message
        throw(ArgumentError("$fun isn't applicable to a multi-field FacetValues. Use on `FunctionValues` for the specific field, e.g. $fun(fv.$k, q_point, ae, [dofrange])"))
    end
end

"""
    BCValues(func_interpol::Interpolation, geom_interpol::Interpolation, boundary_type::Union{Type{<:BoundaryIndex}})

`BCValues` stores the shape values at all facet/faces/edges/vertices (depending on `boundary_type`) for the geometric interpolation (`geom_interpol`),
for each dof-position determined by the `func_interpol`. Used mainly by the `ConstraintHandler`.
"""
mutable struct BCValues{Tv, Ti}
    const M::Array{Tv, 3}
    const nqp::Vector{Ti}
    current_entity::Ti
end

BCValues(func_interpol::Interpolation, geom_interpol::Interpolation, boundary_type::Type{<:BoundaryIndex}) =
    BCValues(Float64, func_interpol, geom_interpol, boundary_type)
BCValues(::Type{Tv}, func_interpol::Interpolation{refshape}, geom_interpol::Interpolation{refshape}, boundary_type::Type{<:BoundaryIndex}) where {Tv, dim, refshape <: AbstractRefShape{dim}} =
    BCValues(Tv, Int64, func_interpol, geom_interpol, boundary_type)
function BCValues(::Type{Tv}, ::Type{Ti}, func_interpol::Interpolation{refshape}, geom_interpol::Interpolation{refshape}, boundary_type::Type{<:BoundaryIndex}) where {Tv, Ti, dim, refshape <: AbstractRefShape{dim}}
    # set up quadrature rules for each boundary entity with dof-positions
    # (determined by func_interpol) as the quadrature points
    interpolation_coords = reference_coordinates(func_interpol)

    qrs = QuadratureRule{refshape, Vector{Tv}, Vector{Vec{dim, Tv}}}[]
    for boundarydofs in dirichlet_boundarydof_indices(boundary_type)(func_interpol)
        dofcoords = Vec{dim, Tv}[]
        for boundarydof in boundarydofs
            push!(dofcoords, interpolation_coords[boundarydof])
        end
        qrf = QuadratureRule{refshape}(fill(Tv(NaN), length(dofcoords)), dofcoords) # weights will not be used
        push!(qrs, qrf)
    end

    n_boundary_entities = length(qrs)
    n_qpoints = n_boundary_entities == 0 ? 0 : maximum(qr -> length(getweights(qr)), qrs) # Bound number of qps correctly.
    n_geom_basefuncs = getnbasefunctions(geom_interpol)
    M = fill(zero(Tv) * Tv(NaN), n_geom_basefuncs, n_qpoints, n_boundary_entities)
    nqp = zeros(Ti, n_boundary_entities)

    for n_boundary_entity in 1:n_boundary_entities
        for (qp, ξ) in pairs(qrs[n_boundary_entity].points)
            reference_shape_values!(@view(M[:, qp, n_boundary_entity]), geom_interpol, ξ)
        end
        nqp[n_boundary_entity] = length(qrs[n_boundary_entity].points)
    end

    return BCValues{Tv, Ti}(M, nqp, zero(Ti))
end

getnquadpoints(bcv::BCValues) = bcv.nqp[bcv.current_entity]
function spatial_coordinate(bcv::BCValues, q_point::Int, xh::AbstractVector{Vec{dim, T}}) where {dim, T}
    n_base_funcs = size(bcv.M, 1)
    length(xh) == n_base_funcs || throw_incompatible_coord_length(length(xh), n_base_funcs)
    x = zero(Vec{dim, T})
    facet = bcv.current_entity[]
    @inbounds for i in 1:n_base_funcs
        x += bcv.M[i, q_point, facet] * xh[i] # geometric_value(fe_v, q_point, i) * xh[i]
    end
    return x
end
