"""
    EdgeValues([::Type{T}], quad_rule::EdgeQuadratureRule, func_interpol::Interpolation, [geom_interpol::Interpolation])

A `EdgeValues` object facilitates the process of evaluating values of shape functions, gradients of shape functions,
values of nodal functions, gradients and divergences of nodal functions etc. on the edges of finite elements.

**Arguments:**

* `T`: an optional argument (default to `Float64`) to determine the type the internal data is stored as.
* `quad_rule`: an instance of a [`EdgeQuadratureRule`](@ref)
* `func_interpol`: an instance of an [`Interpolation`](@ref) used to interpolate the approximated function
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
* [`gettangent`](@ref)
"""
EdgeValues

mutable struct EdgeValues{FV, GM, EQR, detT, tT, V_FV <: AbstractVector{FV}, V_GM <: AbstractVector{GM}} <: AbstractValues
    const fun_values::V_FV  # AbstractVector{FunctionValues}
    const geo_mapping::V_GM # AbstractVector{GeometryMapping}
    const eqr::EQR          # EdgeQuadratureRule
    const detJdV::detT      # AbstractVector{<:Number}
    const tangents::tT      # AbstractVector{<:Vec}
    current_edge::Int
end

function EdgeValues(
        ::Type{T}, eqr::EdgeQuadratureRule, ip_fun::Interpolation, ip_geo::VectorizedInterpolation{sdim},
        ::ValuesUpdateFlags{FunDiffOrder, GeoDiffOrder}
    ) where {T, sdim, FunDiffOrder, GeoDiffOrder}

    # max(GeoDiffOrder, 1) ensures that we get the jacobian needed to calculate the tangent.
    geo_mapping = map(qr -> GeometryMapping{max(GeoDiffOrder, 1)}(T, ip_geo.ip, qr), eqr.edge_rules)
    fun_values = map(qr -> FunctionValues{FunDiffOrder}(T, ip_fun, qr, ip_geo), eqr.edge_rules)
    max_nquadpoints = maximum(qr -> length(getweights(qr)), eqr.edge_rules)
    # detJdV always calculated, since we needed to calculate the jacobian anyways for the tangent.
    detJdV = fill(T(NaN), max_nquadpoints)
    tangents = fill(zero(Vec{sdim, T}) * T(NaN), max_nquadpoints)
    return EdgeValues(fun_values, geo_mapping, eqr, detJdV, tangents, 1)
end

EdgeValues(qr::EdgeQuadratureRule, ip::Interpolation, args...; kwargs...) = EdgeValues(Float64, qr, ip, args...; kwargs...)
function EdgeValues(::Type{T}, qr::EdgeQuadratureRule, ip::Interpolation, ip_geo::ScalarInterpolation; kwargs...) where {T}
    return EdgeValues(T, qr, ip, VectorizedInterpolation(ip_geo); kwargs...)
end
function EdgeValues(::Type{T}, qr::EdgeQuadratureRule, ip::Interpolation, ip_geo::VectorizedInterpolation = default_geometric_interpolation(ip); kwargs...) where {T}
    return EdgeValues(T, qr, ip, ip_geo, ValuesUpdateFlags(ip; kwargs...))
end

function Base.copy(ev::EdgeValues)
    fun_values = map(copy, ev.fun_values)
    geo_mapping = map(copy, ev.geo_mapping)
    return EdgeValues(fun_values, geo_mapping, copy(ev.eqr), copy(ev.detJdV), copy(ev.tangents), ev.current_edge)
end

getngeobasefunctions(ev::EdgeValues) = getngeobasefunctions(get_geo_mapping(ev))
getnbasefunctions(ev::EdgeValues) = getnbasefunctions(get_fun_values(ev))
getnquadpoints(ev::EdgeValues) = @inbounds getnquadpoints(ev.eqr, getcurrentedge(ev))
@propagate_inbounds getdetJdV(ev::EdgeValues, q_point) = ev.detJdV[q_point]

shape_value_type(ev::EdgeValues) = shape_value_type(get_fun_values(ev))
shape_gradient_type(ev::EdgeValues) = shape_gradient_type(get_fun_values(ev))
function_interpolation(ev::EdgeValues) = function_interpolation(get_fun_values(ev))
function_difforder(ev::EdgeValues) = function_difforder(get_fun_values(ev))
geometric_interpolation(ev::EdgeValues) = geometric_interpolation(get_geo_mapping(ev))

get_geo_mapping(ev::EdgeValues) = @inbounds ev.geo_mapping[getcurrentedge(ev)]
@propagate_inbounds geometric_value(ev::EdgeValues, args...) = geometric_value(get_geo_mapping(ev), args...)

get_fun_values(ev::EdgeValues) = @inbounds ev.fun_values[getcurrentedge(ev)]

@propagate_inbounds shape_value(ev::EdgeValues, q_point::Int, i::Int) = shape_value(get_fun_values(ev), q_point, i)
@propagate_inbounds shape_gradient(ev::EdgeValues, q_point::Int, i::Int) = shape_gradient(get_fun_values(ev), q_point, i)
@propagate_inbounds shape_hessian(ev::EdgeValues, q_point::Int, i::Int) = shape_hessian(get_fun_values(ev), q_point, i)
@propagate_inbounds shape_symmetric_gradient(ev::EdgeValues, q_point::Int, i::Int) = shape_symmetric_gradient(get_fun_values(ev), q_point, i)

"""
    getcurrentedge(ev::EdgeValues)

Return the current active edge of the `EdgeValues` object (from last `reinit!`).
"""
getcurrentedge(ev::EdgeValues) = ev.current_edge[]

"""
    gettangent(ev::EdgeValues, qp::Int)

Return the edge tangent at the quadrature point `qp` for the active edge of the
`EdgeValues` object(from last `reinit!`).
"""
gettangent(ev::EdgeValues, qp::Int) = ev.tangents[qp]

nedges(ev::EdgeValues) = length(ev.geo_mapping)

function set_current_edge!(ev::EdgeValues, edge_nr::Int)
    # Checking edge_nr before setting current_edge allows us to use @inbounds
    # when indexing by getcurrentedge(ev) in other places!
    checkbounds(Bool, 1:nedges(ev), edge_nr) || throw(ArgumentError("Edge index out of range."))
    ev.current_edge = edge_nr
    return
end

@inline function reinit!(ev::EdgeValues, x::AbstractVector, edge_nr::Int)
    return reinit!(ev, nothing, x, edge_nr)
end

function reinit!(ev::EdgeValues, cell::Union{AbstractCell, Nothing}, x::AbstractVector{Vec{dim, T}}, edge_nr::Int) where {dim, T}
    check_reinit_sdim_consistency(:EdgeValues, shape_gradient_type(ev), eltype(x))
    set_current_edge!(ev, edge_nr)
    n_geom_basefuncs = getngeobasefunctions(ev)
    if !checkbounds(Bool, x, 1:n_geom_basefuncs) || length(x) != n_geom_basefuncs
        throw_incompatible_coord_length(length(x), n_geom_basefuncs)
    end

    geo_mapping = get_geo_mapping(ev)
    fun_values = get_fun_values(ev)

    if cell === nothing && !isa(mapping_type(fun_values), IdentityMapping)
        throw(ArgumentError("The cell::AbstractCell input is required to reinit! non-identity function mappings"))
    end

    @inbounds for (q_point, w) in pairs(getweights(ev.eqr, edge_nr))
        mapping = calculate_mapping(geo_mapping, q_point, x)
        J = getjacobian(mapping)
        Wt = weighted_tangent(J, getrefshape(geo_mapping.ip), edge_nr)
        detJ = norm(Wt)
        @inbounds ev.detJdV[q_point] = detJ * w
        @inbounds ev.tangents[q_point] = Wt / detJ
        apply_mapping!(fun_values, q_point, mapping, cell)
    end
    return
end

function Base.show(io::IO, d::MIME"text/plain", ev::EdgeValues)
    ip_geo = geometric_interpolation(ev)
    rdim = getrefdim(ip_geo)
    vdim = isa(shape_value(ev, 1, 1), Vec) ? length(shape_value(ev, 1, 1)) : 0
    GradT = shape_gradient_type(ev)
    sdim = GradT === nothing ? nothing : sdim_from_gradtype(GradT)
    vstr = vdim == 0 ? "scalar" : "vdim=$vdim"
    print(io, "EdgeValues(", vstr, ", rdim=$rdim, sdim=$sdim): ")
    nqp = getnquadpoints.(ev.eqr.face_rules)
    if all(n == first(nqp) for n in nqp)
        println(io, first(nqp), " quadrature points per edge")
    else
        println(io, tuple(nqp...), " quadrature points on each edge")
    end
    print(io, " Function interpolation: "); show(io, d, function_interpolation(ev))
    print(io, "\nGeometric interpolation: ")
    return sdim === nothing ? show(io, d, ip_geo) : show(io, d, ip_geo^sdim)
    sdim === nothing ? show(io, d, ip_geo) : show(io, d, ip_geo^sdim)
    return
end
