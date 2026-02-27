struct TaskDescriptor{D}
    device::D
    num_workers::Int
end

# Extract the i-th worker's local slice from batched GPU data
function get_worker_part(i, cv::CellValues)
    return CellValues(
        get_worker_part(i, cv.fun_values), get_worker_part(i, cv.geo_mapping),
        cv.qr, view(cv.detJdV, i, :)
    )
end
function get_worker_part(i, fv::FunctionValues)
    return FunctionValues(
        fv.ip, view(fv.Nx, i, :, :), fv.Nξ,
        view(fv.dNdx, i, :, :), fv.dNdξ, nothing, nothing
    )
end
function get_worker_part(i, fv::GeometryMapping)
    return GeometryMapping(fv.ip, view(fv.M, i, :, :), fv.dMdξ, fv.d2Mdξ2)
end
