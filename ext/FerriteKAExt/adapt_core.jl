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

# The `atomic` type parameter of the assemblers cannot be inferred from the fields, so this
# cannot use `Adapt.@adapt_structure`. Only scratch-free assemblers can be adapted, since
# the sorting scratch of a host assembler has no device counterpart.
for AT in (:CSCAssembler, :CSRAssembler)
    @eval function adapt_structure(to, a::Ferrite.$AT{Tv, Ti, <:Any, atomic, <:Any, Nothing}) where {Tv, Ti, atomic}
        K = adapt(to, a.K)
        f = adapt(to, a.f)
        return Ferrite.$AT{Tv, Ti, typeof(K), atomic, typeof(f), Nothing}(K, f, nothing, nothing, nothing, nothing)
    end
end
