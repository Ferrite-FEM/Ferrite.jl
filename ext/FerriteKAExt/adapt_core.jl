# This file contains adapt rules for all relevant data structures in Ferrite.jl which do
# not need customized GPU data structures.

Adapt.@adapt_structure CellValues
Adapt.@adapt_structure Ferrite.GeometryMapping
function adapt_structure(to, ccc::Ferrite.CellValuesContainer)
    inner_values = adapt(to, ccc.values)
    return Ferrite.CellValuesContainer{typeof(get_substruct(1, inner_values)), typeof(inner_values)}(inner_values)
end

adapt_structure(to, ip::Ferrite.Interpolation) = ip

function adapt_structure(d, fv::Ferrite.FunctionValues)
    Nξ = adapt(d, fv.Nξ)
    return Ferrite.FunctionValues(
        adapt(d, fv.ip),
        fv.Nξ === fv.Nx ? Nξ : adapt(fv.Nx), # Ensure proper aliasing
        Nξ,
        adapt(d, fv.dNdx),
        adapt(d, fv.dNdξ),
        adapt(d, fv.d2Ndx2),
        adapt(d, fv.d2Ndξ2),
    )
end

function adapt_structure(to, qr::QuadratureRule{shape}) where {shape}
    return QuadratureRule{shape}(adapt(to, qr.weights), adapt(to, qr.points))
end
