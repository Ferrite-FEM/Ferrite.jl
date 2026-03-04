function as_structure_of_arrays(d, outer_dim, ::Type{CellValues}, args...; kwargs...)
    cv = CellValues(args...; kwargs...)
    return as_structure_of_arrays(d, outer_dim, cv)
end

function as_structure_of_arrays(d, N, cv::CellValues)
    return CellValues(
        as_structure_of_arrays(d, N, cv.fun_values),
        as_structure_of_arrays(d, N, cv.geo_mapping),
        adapt(d, cv.qr),
        KA.zeros(d, eltype(cv.detJdV), N, length(cv.detJdV)),
    )
end

function as_structure_of_arrays(d, N, fv::Ferrite.FunctionValues)
    Nξ = adapt(d, fv.Nξ)
    return Ferrite.FunctionValues(
        adapt(d, fv.ip),
        fv.Nξ === fv.Nx ? Nξ : KA.zeros(d, eltype(fv.Nx), N, size(fv.Nx, 1), size(fv.Nx, 2)), # Ensure proper aliasing,
        Nξ,
        fv.dNdx === nothing ? nothing : KA.zeros(d, eltype(fv.dNdx), N, size(fv.dNdx, 1), size(fv.dNdx, 2)),
        adapt(d, fv.dNdξ),
        fv.d2Ndx2 === nothing ? nothing : KA.zeros(d, eltype(fv.d2Ndx2), N, size(fv.d2Ndx2, 1), size(fv.d2Ndx2, 2)),
        fv.d2Ndξ2 === nothing ? nothing : adapt(d, fv.d2Ndξ2),
    )
end

function as_structure_of_arrays(d, N, fv::Ferrite.GeometryMapping)
    return Ferrite.GeometryMapping(
        adapt(d, fv.ip),
        KA.zeros(d, eltype(fv.M), N, size(fv.M, 1), size(fv.M, 2)),
        fv.dMdξ === nothing ? nothing : adapt(d, fv.dMdξ),
        fv.d2Mdξ2 === nothing ? nothing : adapt(d, fv.d2Mdξ2),
    )
end
