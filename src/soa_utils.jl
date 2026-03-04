function as_structure_of_arrays(d, outer_dim, ::Type{ThingType}, args...; kwargs...) where {ThingType}
    error("Structure of Arrays transformation not defined for object of type $(ThingType) device $d . Are all extensions loaded?")
end

function as_structure_of_arrays(d, outer_dim, thing)
    error("Structure of Arrays transformation not defined for object of type $(typeof(thing)) device $d . Are all extensions loaded?")
end

# Extract the i-th worker's local slice from batched GPU data
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

materialize(thing) = thing
