# This file contains adapt rules for all relevant data structures in Ferrite.jl which do
# not need customized GPU data structures.

#Adapt.@adapt_structure CellValues
#Adapt.@adapt_structure Ferrite.GeometryMapping
#Adapt.@adapt_structure Ferrite.FunctionValues
#=
function adapt_structure(to, ccc::Ferrite.CellValuesContainer)
    inner_values = adapt(to, ccc.values)
    return Ferrite.CellValuesContainer{typeof(get_substruct(1, inner_values)), typeof(inner_values)}(inner_values)
end

adapt(to, ip::Ferrite.Interpolation) = ip
=#

function adapt(d, cv::CellValues)
    return CellValues(
        adapt(d, cv.fun_values),
        adapt(d, cv.geo_mapping),
        adapt(d, cv.qr),
        adapt(d, cv.detJdV),
    )
end

function adapt(d, fv::Ferrite.FunctionValues)
    Nξ = adapt(d, fv.Nξ)
    return Ferrite.FunctionValues(
        adapt(d, fv.ip),
        fv.Nξ === fv.Nx ? Nξ : adapt(fv.Nx), # Ensure proper aliasing
        Nξ,
        adapt(d, fv.dNdx),
        adapt(d, fv.dNdξ),
        fv.d2Ndx2 === nothing ? nothing : as_shared_array(d, N, fv.d2Ndx2),
        fv.d2Ndξ2 === nothing ? nothing : adapt(d, collect(fv.d2Ndξ2)),
    )
end

function adapt(d, fv::Ferrite.GeometryMapping)
    return Ferrite.GeometryMapping(
        adapt(d, fv.ip),
        adapt(d, fv.M),
        fv.dMdξ === nothing ? nothing : adapt(d, fv.dMdξ),
        fv.d2Mdξ2 === nothing ? nothing : adapt(d, fv.d2Mdξ2),
    )
end

function adapt(to, qr::QuadratureRule{shape}) where {shape}
    return QuadratureRule{shape}(adapt(to, qr.weights), adapt(to, qr.points))
end

# Adapt.@adapt_structure QuadratureRule does not work here due to the type parameter ctor.
function adapt_structure(to, qr::QuadratureRule{shape}) where {shape}
    return QuadratureRule{shape}(adapt_structure(to, qr.weights), adapt_structure(to, qr.points))
end
