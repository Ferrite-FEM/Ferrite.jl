# This file contains adapt rules for all relevant data structures in Ferrite.jl.
# During setup, these rules are typically called for a `KA.Backend` (e.g. `CUDABackend()`),
# and later during kernel construction, these are called for the specific kernel,
# e.g. CUDA.KernelAdaptor(). Please consult Adapt.jl for further details.

Adapt.@adapt_structure CellCache
Adapt.@adapt_structure CellValues
Adapt.@adapt_structure Ferrite.GeometryMapping
Adapt.@adapt_structure Ferrite.SoAContainer

# Wildcard adapt
adapt_structure(to, ip::Ferrite.Interpolation) = ip

# This is adapted manually to ensure the aliasing is kept correctly
function adapt_structure(d, fv::Ferrite.FunctionValues)
    Nξ = adapt(d, fv.Nξ)
    return Ferrite.FunctionValues(
        adapt(d, fv.ip),
        fv.Nξ === fv.Nx ? Nξ : adapt(d, fv.Nx), # Ensure proper aliasing
        Nξ,
        adapt(d, fv.dNdx),
        adapt(d, fv.dNdξ),
        adapt(d, fv.d2Ndx2),
        adapt(d, fv.d2Ndξ2),
    )
end

# This must be done manually, because of the custom constructor
function adapt_structure(to, qr::QuadratureRule{shape}) where {shape}
    return QuadratureRule{shape}(adapt(to, qr.weights), adapt(to, qr.points))
end
