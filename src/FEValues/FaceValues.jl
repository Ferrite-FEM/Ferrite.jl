"""
    FaceValues([::Type{T}], quad_rule::FaceQuadratureRule, func_interpol::Interpolation, [geom_interpol::Interpolation])

A `FaceValues` object facilitates the process of evaluating values of shape functions, gradients of shape functions,
values of nodal functions, gradients and divergences of nodal functions etc. on the faces of finite elements.

**Arguments:**

* `T`: an optional argument to determine the type the internal data is stored as.
* `quad_rule`: an instance of a [`FaceQuadratureRule`](@ref)
* `func_interpol`: an instance of an [`Interpolation`](@ref) used to interpolate the approximated function
* `geom_interpol`: an optional instance of an [`Interpolation`](@ref) which is used to interpolate the geometry.
  By default linear Lagrange interpolation is used.

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
FaceValues

struct FaceValues{FV, GM, FQR, detT, nT, V_FV<:AbstractVector{FV}, V_GM<:AbstractVector{GM}} <: AbstractFaceValues
    fun_values::V_FV  # AbstractVector{FunctionValues}
    geo_mapping::V_GM # AbstractVector{GeometryMapping}
    fqr::FQR          # FaceQuadratureRule
    detJdV::detT      # AbstractVector{<:Number}
    normals::nT       # AbstractVector{<:Vec}
    current_face::ScalarWrapper{Int}
end

function FaceValues(::Type{T}, fqr::FaceQuadratureRule, ip_fun::Interpolation, ip_geo::VectorizedInterpolation{sdim} = default_geometric_interpolation(ip_fun); FunDiffOrder = 1) where {T,sdim} 
    GeoDiffOrder = max(required_geo_diff_order(get_mapping_type(ip_fun), FunDiffOrder), 1)
    geo_mapping = [GeometryMapping{GeoDiffOrder}(T, ip_geo.ip, qr) for qr in fqr.face_rules]
    fun_values = [FunctionValues{FunDiffOrder}(T, ip_fun, qr, ip_geo) for qr in fqr.face_rules]
    max_nquadpoints = maximum(qr->length(getweights(qr)), fqr.face_rules)
    detJdV  = fill(T(NaN), max_nquadpoints)
    normals = fill(zero(Vec{sdim, T}) * T(NaN), max_nquadpoints)
    return FaceValues(fun_values, geo_mapping, fqr, detJdV, normals, ScalarWrapper(1))
end

FaceValues(qr::FaceQuadratureRule, ip::Interpolation, args...) = FaceValues(Float64, qr, ip, args...)
function FaceValues(::Type{T}, qr::FaceQuadratureRule, ip::Interpolation, ip_geo::ScalarInterpolation) where T
    return FaceValues(T, qr, ip, VectorizedInterpolation(ip_geo))
end

function Base.copy(fv::FaceValues)
    fun_values = map(copy, fv.fun_values)
    geo_mapping = map(copy, fv.geo_mapping)
    return FaceValues(fun_values, geo_mapping, copy(fv.fqr), copy(fv.detJdV), copy(fv.normals), copy(fv.current_face))
end

getngeobasefunctions(fv::FaceValues) = getngeobasefunctions(get_geo_mapping(fv))
getnbasefunctions(fv::FaceValues) = getnbasefunctions(get_fun_values(fv))
getnquadpoints(fv::FaceValues) = @inbounds getnquadpoints(fv.fqr, getcurrentface(fv))
@propagate_inbounds getdetJdV(fv::FaceValues, q_point) = fv.detJdV[q_point]

shape_value_type(fv::FaceValues) = shape_value_type(get_fun_values(fv))
shape_gradient_type(fv::FaceValues) = shape_gradient_type(get_fun_values(fv))
get_function_interpolation(fv::FaceValues) = get_function_interpolation(get_fun_values(fv))
get_function_difforder(fv::FaceValues) = get_function_difforder(get_fun_values(fv))
get_geometric_interpolation(fv::FaceValues) = get_geometric_interpolation(get_geo_mapping(fv))

get_geo_mapping(fv::FaceValues) = @inbounds fv.geo_mapping[getcurrentface(fv)]
@propagate_inbounds geometric_value(fv::FaceValues, args...) = geometric_value(get_geo_mapping(fv), args...)

get_fun_values(fv::FaceValues) = @inbounds fv.fun_values[getcurrentface(fv)]
for op = (:shape_value, :shape_gradient, :shape_symmetric_gradient, :shape_curl)
    @eval @propagate_inbounds $op(fv::FaceValues, i::Int, q_point::Int) = $op(get_fun_values(fv), i, q_point)
end

"""
    getcurrentface(fv::FaceValues)

Return the current active face of the `FaceValues` object (from last `reinit!`).
"""
getcurrentface(fv::FaceValues) = fv.current_face[]

"""
    getnormal(fv::FaceValues, qp::Int)

Return the normal at the quadrature point `qp` for the active face of the
`FaceValues` object(from last `reinit!`).
"""
getnormal(fv::FaceValues, qp::Int) = fv.normals[qp]

nfaces(fv::FaceValues) = length(fv.geo_mapping)

function set_current_face!(fv::FaceValues, face_nr::Int)
    # Checking face_nr before setting current_face allows us to use @inbounds 
    # when indexing by getcurrentface(fv) in other places!
    checkbounds(Bool, 1:nfaces(fv), face_nr) || throw(ArgumentError("Face index out of range."))
    fv.current_face[] = face_nr
end

function reinit!(fv::FaceValues, x::AbstractVector{Vec{dim, T}}, face_nr::Int, cell = nothing) where {dim, T}
    check_reinit_sdim_consistency(:FaceValues, shape_gradient_type(fv), eltype(x))
    set_current_face!(fv, face_nr)
    n_geom_basefuncs = getngeobasefunctions(fv)
    if !checkbounds(Bool, x, 1:n_geom_basefuncs) || length(x) != n_geom_basefuncs
        throw_incompatible_coord_length(length(x), n_geom_basefuncs)
    end
    
    geo_mapping = get_geo_mapping(fv)
    fun_values = get_fun_values(fv)

    if cell === nothing && !isa(get_mapping_type(fun_values), IdentityMapping)
        throw(ArgumentError("The cell::AbstractCell input is required to reinit! non-identity function mappings"))
    end

    @inbounds for (q_point, w) in pairs(getweights(fv.fqr, face_nr))
        mapping = calculate_mapping(geo_mapping, q_point, x)
        J = getjacobian(mapping)
        # See the `Ferrite.embedded_det` docstring for more background
        weight_norm = weighted_normal(J, getrefshape(geo_mapping.ip), face_nr)
        detJ = norm(weight_norm)
        detJ > 0.0 || throw_detJ_not_pos(detJ)
        @inbounds fv.detJdV[q_point] = detJ * w
        @inbounds fv.normals[q_point] = weight_norm / norm(weight_norm)       
        apply_mapping!(fun_values, q_point, mapping, cell)
    end
end

function Base.show(io::IO, d::MIME"text/plain", fv::FaceValues)
    ip_geo = get_geometric_interpolation(fv)
    rdim = getdim(ip_geo)
    vdim = isa(shape_value(fv, 1, 1), Vec) ? length(shape_value(fv, 1, 1)) : 0
    sdim = length(shape_gradient(fv, 1, 1)) ÷ length(shape_value(fv, 1, 1))
    vstr = vdim==0 ? "scalar" : "vdim=$vdim"
    print(io, "FaceValues(", vstr, ", rdim=$rdim, sdim=$sdim): ")
    nqp = getnquadpoints.(fv.fqr.face_rules)
    if all(n==first(nqp) for n in nqp)
        println(io, first(nqp), " quadrature points per face")
    else
        println(io, tuple(nqp...), " quadrature points on each face")
    end
    print(io, " Function interpolation: "); show(io, d, get_function_interpolation(fv))
    print(io, "\nGeometric interpolation: "); show(io, d, ip_geo^sdim)
end

"""
    BCValues(func_interpol::Interpolation, geom_interpol::Interpolation, boundary_type::Union{Type{<:BoundaryIndex}})

`BCValues` stores the shape values at all faces/edges/vertices (depending on `boundary_type`) for the geomatric interpolation (`geom_interpol`),
for each dof-position determined by the `func_interpol`. Used mainly by the `ConstrainHandler`.
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
        for (qp, ξ) in enumerate(qrs[n_boundary_entity].points), i in 1:n_geom_basefuncs
            M[i, qp, n_boundary_entity] = shape_value(geom_interpol, ξ, i)
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
