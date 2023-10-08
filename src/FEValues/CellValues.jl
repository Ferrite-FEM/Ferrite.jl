
include("GeometryValues.jl")
include("FunctionValues.jl")

function default_geometric_interpolation(::Interpolation{shape}) where {dim, shape <: AbstractRefShape{dim}}
    return VectorizedInterpolation{dim}(Lagrange{shape, 1}())
end

struct CellValues{FV, GV, QR, detT<:AbstractVector} <: AbstractCellValues
    fun_values::FV # FunctionValues
    geo_values::GV # GeometryValues
    qr::QR         # QuadratureRule
    detJdV::detT   # AbstractVector{<:Number}
end
function CellValues(::Type{T}, qr::QuadratureRule, ip_fun::Interpolation, ip_geo::VectorizedInterpolation) where T 
    geo_values = GeometryValues(T, ip_geo.ip, qr, RequiresHessian(ip_fun, ip_geo))
    fun_values = FunctionValues(T, ip_fun, qr, ip_geo)
    detJdV = fill(T(NaN), length(getweights(qr)))
    return CellValues(fun_values, geo_values, qr, detJdV)
end

CellValues(qr::QuadratureRule, ip::Interpolation, args...) = CellValues(Float64, qr, ip, args...)
function CellValues(::Type{T}, qr, ip::Interpolation, ip_geo::ScalarInterpolation=default_geometric_interpolation(ip)) where T
    return CellValues(T, qr, ip, VectorizedInterpolation(ip_geo))
end

# Access geometry values
for op = (:getngeobasefunctions, :geometric_value)
    eval(quote
        @propagate_inbounds $op(cv::CellValues, args...) = $op(cv.geo_values, args...)
    end)
end
getdetJdV(cv::CellValues, q_point::Int) = cv.detJdV[q_point]

# Accessors for function values 
getnbasefunctions(cv::CellValues) = getnbasefunctions(cv.fun_values)
get_function_interpolation(cv::CellValues) = get_function_interpolation(cv.fun_values)
get_geometric_interpolation(cv::CellValues) = get_geometric_interpolation(cv.geo_values)
shape_value_type(cv::CellValues) = shape_value_type(cv.fun_values)
shape_gradient_type(cv::CellValues) = shape_gradient_type(cv.fun_values)

for op = (:shape_value, :shape_gradient, :shape_symmetric_gradient)
    eval(quote
        @propagate_inbounds $op(cv::CellValues, i::Int, q_point::Int) = $op(cv.fun_values, i, q_point)
    end)
end
# Access quadrature rule values 
getnquadpoints(cv::CellValues) = getnquadpoints(cv.qr)

function reinit!(cv::CellValues, x::AbstractVector{<:Vec}, cell=nothing)
    geo_values = cv.geo_values
    fun_values = cv.fun_values
    n_geom_basefuncs = getngeobasefunctions(geo_values)
    if !checkbounds(Bool, x, 1:n_geom_basefuncs) || length(x)!=n_geom_basefuncs
        throw_incompatible_coord_length(length(x), n_geom_basefuncs)
    end
    @inbounds for (q_point, w) in enumerate(getweights(cv.qr))
        mapping = calculate_mapping(geo_values, q_point, x)
        detJ = calculate_detJ(getjacobian(mapping))
        detJ > 0.0 || throw_detJ_not_pos(detJ)
        @inbounds cv.detJdV[q_point] = detJ*w
        apply_mapping!(fun_values, q_point, mapping, cell)
    end
    return nothing
end

function Base.show(io::IO, d::MIME"text/plain", cv::CellValues)
    rdim = getdim(cv.geo_values.ip)
    vdim = isa(shape_value(cv, 1, 1), Vec) ? length(shape_value(cv, 1, 1)) : 0
    sdim = length(shape_gradient(cv, 1, 1)) ÷ length(shape_value(cv, 1, 1))
    vstr = vdim==0 ? "scalar" : "vdim=$vdim"
    print(io, "CellValues(", vstr, ", rdim=$rdim, and sdim=$sdim): ")
    print(io, getnquadpoints(cv), " quadrature points")
    print(io, "\n Function interpolation: "); show(io, d, cv.fun_values.ip)
    print(io, "\nGeometric interpolation: "); show(io, d, cv.geo_values.ip^sdim)
end

# Temporary for benchmark/test
include("cell_values.jl")
function OldCellValues(cv::CellValues)
    ip = cv.fun_values.ip 
    sdim = length(shape_gradient(cv, 1, 1)) ÷ length(shape_value(cv, 1, 1))
    ip_geo = cv.geo_values.ip^sdim
    return OldCellValues(cv.qr, ip, ip_geo)
end