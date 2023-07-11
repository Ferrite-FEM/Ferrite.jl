# Temporary to check performance of "new" way of separating geometry values from cell values

struct SingleCellValues{T, FVS, QR, GVS} <: AbstractCellValues
    geo_values::GVS     # GeometryValues
    fun_values::FVS     # FunctionValues
    detJdV::Vector{T}   # 
    qr::QR
end

function SingleCellValues(cv::CellValues)
    geo_values = GeometryValues(cv)
    fun_values = FunctionValues(cv)
    return SingleCellValues(geo_values, fun_values, cv.detJdV, cv.qr)
end

function reinit!(cv::SingleCellValues, x::AbstractVector{<:Vec})
    geo_values = cv.geo_values
    checkbounds(Bool, x, 1:getngeobasefunctions(geo_values)) || throw_incompatible_coord_length(length(x), getngeobasefunctions(geo_values))
    @inbounds for (q_point, w) in enumerate(getweights(cv.qr))
        detJ, Jinv = calculate_mapping(geo_values, q_point, x)
        cv.detJdV[q_point] = detJ*w
        apply_mapping!(cv.fun_values, q_point, Jinv)
    end
    return nothing
end