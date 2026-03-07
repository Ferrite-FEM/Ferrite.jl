# ------------------------------ User-facing part -------------------------------
struct CellValuesContainer{ReturnedFEVType, InnerFEVType} <: AbstractVector{ReturnedFEVType}
    values::InnerFEVType
end

Base.size(cv::CellValuesContainer) = size(cv.values.detJdV, 1)
Base.axes(cv::CellValuesContainer) = (size(cv.values.detJdV, 1),)
Base.getindex(cv::CellValuesContainer, i::Int) = get_substruct(i, cv.values)

function CellValuesContainer(backend, outer_dim, cv::CellValues)
    inner_values = as_structure_of_arrays(backend, outer_dim, cv)
    return CellValuesContainer{typeof(get_substruct(1, inner_values)), typeof(inner_values)}(inner_values)
end

function CellValuesContainer(backend, outer_dim, args...; kwargs...)
    inner_values = as_structure_of_arrays(backend, outer_dim, CellValues, args...; kwargs...)
    return CellValuesContainer{typeof(get_substruct(1, inner_values)), typeof(inner_values)}(inner_values)
end

function FacetValuesContainer(backend, outer_dim, args...; kwargs...)
    inner_values = as_structure_of_arrays(backend, outer_dim, FacetValues, args...; kwargs...)
    return error("TODO: Implement FacetValuesContainer")
end

struct CellCacheContainer{ReturnedCCType, InnerCCType} <: AbstractVector{ReturnedCCType}
    values::InnerCCType
end
Base.size(cc::CellCacheContainer) = size(cc.values.coords, 1)
Base.axes(cc::CellCacheContainer) = (size(cc.values.coords, 1),)
Base.getindex(cc::CellCacheContainer, i::Int) = get_substruct(i, cc.values, -1)

function CellCacheContainer(backend, outer_dim, args...; kwargs...)
    inner_values = as_structure_of_arrays(backend, outer_dim, CellCache, args...; kwargs...)
    return CellCacheContainer{typeof(get_substruct(1, inner_values, -1)), typeof(inner_values)}(inner_values)
end

# ------------------------------ Internal part --------------------------------------
function as_structure_of_arrays(d, outer_dim, ::Type{ThingType}, args...; kwargs...) where {ThingType}
    error("Structure of Arrays transformation not defined for object of type $(ThingType) device $d . Are all extensions loaded?")
end

function as_structure_of_arrays(d, outer_dim, thing)
    error("Structure of Arrays transformation not defined for object of type $(typeof(thing)) device $d . Are all extensions loaded?")
end

# Extract the i-th worker's local slice from batched device data
function get_substruct(i, cv::CellValues)
    return CellValues(
        get_substruct(i, cv.fun_values), get_substruct(i, cv.geo_mapping),
        cv.qr, view(cv.detJdV, i, :)
    )
end

function get_substruct(i, fv::FunctionValues)
    Nx = fv.Nξ === fv.Nx ? fv.Nx : view(fv.Nx, i, :, :)
    dNdx = fv.dNdx === nothing ? nothing : view(fv.dNdx, i, :, :)
    return FunctionValues(
        fv.ip, Nx, fv.Nξ,
        dNdx, fv.dNdξ, nothing, nothing
    )
end

function get_substruct(i, fv::GeometryMapping)
    return GeometryMapping(fv.ip, view(fv.M, i, :, :), fv.dMdξ, fv.d2Mdξ2)
end
