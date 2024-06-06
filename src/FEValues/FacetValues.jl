"""
    FacetValues([::Type{T}], quad_rule::FacetQuadratureRule, func_interpol::Interpolation, [geom_interpol::Interpolation])

A `FacetValues` object facilitates the process of evaluating values of shape functions, gradients of shape functions,
values of nodal functions, gradients and divergences of nodal functions etc. on the facets of finite elements.

**Arguments:**

* `T`: an optional argument (default to `Float64`) to determine the type the internal data is stored as.
* `quad_rule`: an instance of a [`FacetQuadratureRule`](@ref)
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
"""
FacetValues

struct FacetValues{FV, GM, FQR, detT, nT, V_FV<:AbstractVector{FV}, V_GM<:AbstractVector{GM}} <: AbstractFacetValues
    fun_values::V_FV  # AbstractVector{FunctionValues}
    geo_mapping::V_GM # AbstractVector{GeometryMapping}
    fqr::FQR          # FacetQuadratureRule
    detJdV::detT      # AbstractVector{<:Number}
    normals::nT       # AbstractVector{<:Vec}
    current_facet::ScalarWrapper{Int}
end

function FacetValues(::Type{T}, fqr::FacetQuadratureRule, ip_fun::Interpolation, ip_geo::VectorizedInterpolation{sdim},
        update_flags::ValuesUpdateFlags{FunDiffOrder, GeoDiffOrder} = ValuesUpdateFlags{1, required_geo_diff_order(mapping_type(ip_fun), 1), true}()
        ) where {T, sdim, FunDiffOrder, GeoDiffOrder}

    # max(GeoDiffOrder, 1) ensures that we get the jacobian needed to calculate the normal.
    geo_mapping = map(qr -> GeometryMapping{max(GeoDiffOrder, 1)}(T, ip_geo.ip, qr), fqr.face_rules)
    fun_values = map(qr -> FunctionValues{FunDiffOrder}(T, ip_fun, qr, ip_geo), fqr.face_rules)
    max_nquadpoints = maximum(qr -> length(getweights(qr)), fqr.face_rules)
    # detJdV always calculated, since we needed to calculate the jacobian anyways for the normal.
    detJdV  = fill(T(NaN), max_nquadpoints)
    normals = fill(zero(Vec{sdim, T}) * T(NaN), max_nquadpoints)
    return FacetValues(fun_values, geo_mapping, fqr, detJdV, normals, ScalarWrapper(1))
end

FacetValues(qr::FacetQuadratureRule, ip::Interpolation, args...) = FacetValues(Float64, qr, ip, args...)
function FacetValues(::Type{T}, qr::FacetQuadratureRule, ip::Interpolation, ip_geo::ScalarInterpolation, args...) where T
    return FacetValues(T, qr, ip, VectorizedInterpolation(ip_geo), args...)
end
function FacetValues(::Type{T}, qr::FacetQuadratureRule, ip::Interpolation, ip_geo::VectorizedInterpolation = default_geometric_interpolation(ip), args...) where T
    return FacetValues(T, qr, ip, ip_geo, args...)
end

function Base.copy(fv::FacetValues)
    fun_values = map(copy, fv.fun_values)
    geo_mapping = map(copy, fv.geo_mapping)
    return FacetValues(fun_values, geo_mapping, copy(fv.fqr), copy(fv.detJdV), copy(fv.normals), copy(fv.current_facet))
end

getngeobasefunctions(fv::FacetValues) = getngeobasefunctions(get_geo_mapping(fv))
getnbasefunctions(fv::FacetValues) = getnbasefunctions(get_fun_values(fv))
getnquadpoints(fv::FacetValues) = @inbounds getnquadpoints(fv.fqr, getcurrentfacet(fv))
@propagate_inbounds getdetJdV(fv::FacetValues, q_point) = fv.detJdV[q_point]

shape_value_type(fv::FacetValues) = shape_value_type(get_fun_values(fv))
shape_gradient_type(fv::FacetValues) = shape_gradient_type(get_fun_values(fv))
function_interpolation(fv::FacetValues) = function_interpolation(get_fun_values(fv))
function_difforder(fv::FacetValues) = function_difforder(get_fun_values(fv))
geometric_interpolation(fv::FacetValues) = geometric_interpolation(get_geo_mapping(fv))

get_geo_mapping(fv::FacetValues) = @inbounds fv.geo_mapping[getcurrentfacet(fv)]
@propagate_inbounds geometric_value(fv::FacetValues, args...) = geometric_value(get_geo_mapping(fv), args...)

get_fun_values(fv::FacetValues) = @inbounds fv.fun_values[getcurrentfacet(fv)]

@propagate_inbounds shape_value(fv::FacetValues, q_point::Int, i::Int) = shape_value(get_fun_values(fv), q_point, i)
@propagate_inbounds shape_gradient(fv::FacetValues, q_point::Int, i::Int) = shape_gradient(get_fun_values(fv), q_point, i)
@propagate_inbounds shape_hessian(fv::FacetValues, q_point::Int, i::Int) = shape_hessian(get_fun_values(fv), q_point, i)
@propagate_inbounds shape_symmetric_gradient(fv::FacetValues, q_point::Int, i::Int) = shape_symmetric_gradient(get_fun_values(fv), q_point, i)

"""
    getcurrentfacet(fv::FacetValues)

Return the current active facet of the `FacetValues` object (from last `reinit!`).
"""
getcurrentfacet(fv::FacetValues) = fv.current_facet[]

"""
    getnormal(fv::FacetValues, qp::Int)

Return the normal at the quadrature point `qp` for the active facet of the
`FacetValues` object(from last `reinit!`).
"""
getnormal(fv::FacetValues, qp::Int) = fv.normals[qp]

nfacets(fv::FacetValues) = length(fv.geo_mapping)

function set_current_facet!(fv::FacetValues, face_nr::Int)
    # Checking face_nr before setting current_facet allows us to use @inbounds
    # when indexing by getcurrentfacet(fv) in other places!
    checkbounds(Bool, 1:nfacets(fv), face_nr) || throw(ArgumentError("Face index out of range."))
    fv.current_facet[] = face_nr
end

@inline function reinit!(fv::FacetValues, x::AbstractVector, face_nr::Int)
    return reinit!(fv, nothing, x, face_nr)
end

function reinit!(fv::FacetValues, cell::Union{AbstractCell, Nothing}, x::AbstractVector{Vec{dim, T}}, face_nr::Int) where {dim, T}
    check_reinit_sdim_consistency(:FacetValues, shape_gradient_type(fv), eltype(x))
    set_current_facet!(fv, face_nr)
    n_geom_basefuncs = getngeobasefunctions(fv)
    if !checkbounds(Bool, x, 1:n_geom_basefuncs) || length(x) != n_geom_basefuncs
        throw_incompatible_coord_length(length(x), n_geom_basefuncs)
    end

    geo_mapping = get_geo_mapping(fv)
    fun_values = get_fun_values(fv)

    if cell === nothing && !isa(mapping_type(fun_values), IdentityMapping)
        throw(ArgumentError("The cell::AbstractCell input is required to reinit! non-identity function mappings"))
    end

    @inbounds for (q_point, w) in pairs(getweights(fv.fqr, face_nr))
        mapping = calculate_mapping(geo_mapping, q_point, x)
        J = getjacobian(mapping)
        # See the `Ferrite.embedding_det` docstring for more background
        weight_norm = weighted_normal(J, getrefshape(geo_mapping.ip), face_nr)
        detJ = norm(weight_norm)
        detJ > 0.0 || throw_detJ_not_pos(detJ)
        @inbounds fv.detJdV[q_point] = detJ * w
        @inbounds fv.normals[q_point] = weight_norm / norm(weight_norm)
        apply_mapping!(fun_values, q_point, mapping, cell)
    end
end

function Base.show(io::IO, d::MIME"text/plain", fv::FacetValues)
    ip_geo = geometric_interpolation(fv)
    rdim = getrefdim(ip_geo)
    vdim = isa(shape_value(fv, 1, 1), Vec) ? length(shape_value(fv, 1, 1)) : 0
    GradT = shape_gradient_type(fv)
    sdim = GradT === nothing ? nothing : sdim_from_gradtype(GradT)
    vstr = vdim==0 ? "scalar" : "vdim=$vdim"
    print(io, "FacetValues(", vstr, ", rdim=$rdim, sdim=$sdim): ")
    nqp = getnquadpoints.(fv.fqr.face_rules)
    if all(n==first(nqp) for n in nqp)
        println(io, first(nqp), " quadrature points per face")
    else
        println(io, tuple(nqp...), " quadrature points on each face")
    end
    print(io, " Function interpolation: "); show(io, d, function_interpolation(fv))
    print(io, "\nGeometric interpolation: ");
    sdim === nothing ? show(io, d, ip_geo) : show(io, d, ip_geo^sdim)
end

"""
    BCValues(func_interpol::Interpolation, geom_interpol::Interpolation, boundary_type::Union{Type{<:BoundaryIndex}})

`BCValues` stores the shape values at all facet/faces/edges/vertices (depending on `boundary_type`) for the geometric interpolation (`geom_interpol`),
for each dof-position determined by the `func_interpol`. Used mainly by the `ConstraintHandler`.
"""
struct BCValues{T}
    M::Array{T,3}
    nqp::Array{Int}
    current_entity::ScalarWrapper{Int}
end

BCValues(func_interpol::Interpolation, geom_interpol::Interpolation, boundary_type::Type{<:BoundaryIndex} = Ferrite.FaceIndex) =
    BCValues(Float64, func_interpol, geom_interpol, boundary_type)

function BCValues(::Type{T}, func_interpol::Interpolation{refshape}, geom_interpol::Interpolation{refshape}, boundary_type::Type{<:BoundaryIndex} = Ferrite.FaceIndex) where {T,dim,refshape <: AbstractRefShape{dim}}
    # set up quadrature rules for each boundary entity with dof-positions
    # (determined by func_interpol) as the quadrature points
    interpolation_coords = reference_coordinates(func_interpol)

    qrs = QuadratureRule{refshape,T,dim}[]
    for boundarydofs in dirichlet_boundarydof_indices(boundary_type)(func_interpol)
        dofcoords = Vec{dim,T}[]
        for boundarydof in boundarydofs
            push!(dofcoords, interpolation_coords[boundarydof])
        end
        qrf = QuadratureRule{refshape,T}(fill(T(NaN), length(dofcoords)), dofcoords) # weights will not be used
        push!(qrs, qrf)
    end

    n_boundary_entities = length(qrs)
    n_qpoints = n_boundary_entities == 0 ? 0 : maximum(qr->length(getweights(qr)), qrs) # Bound number of qps correctly.
    n_geom_basefuncs = getnbasefunctions(geom_interpol)
    M   = fill(zero(T) * T(NaN), n_geom_basefuncs, n_qpoints, n_boundary_entities)
    nqp = zeros(Int,n_boundary_entities)

    for n_boundary_entity in 1:n_boundary_entities
        for (qp, ξ) in pairs(qrs[n_boundary_entity].points)
            shape_values!(@view(M[:, qp, n_boundary_entity]), geom_interpol, ξ)
        end
        nqp[n_boundary_entity] = length(qrs[n_boundary_entity].points)
    end

    BCValues{T}(M, nqp, ScalarWrapper(0))
end

getnquadpoints(bcv::BCValues) = bcv.nqp[bcv.current_entity.x]
function spatial_coordinate(bcv::BCValues, q_point::Int, xh::AbstractVector{Vec{dim,T}}) where {dim,T}
    n_base_funcs = size(bcv.M, 1)
    length(xh) == n_base_funcs || throw_incompatible_coord_length(length(xh), n_base_funcs)
    x = zero(Vec{dim,T})
    face = bcv.current_entity[]
    @inbounds for i in 1:n_base_funcs
        x += bcv.M[i,q_point,face] * xh[i] # geometric_value(fe_v, q_point, i) * xh[i]
    end
    return x
end
