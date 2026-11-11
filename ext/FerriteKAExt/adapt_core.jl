# This file contains adapt rules for all relevant data structures in Ferrite.jl.
# During setup, these rules are typically called for a `KA.Backend` (e.g. `CUDABackend()`),
# and later during kernel construction, these are called for the specific kernel,
# e.g. CUDA.KernelAdaptor(). Please consult Adapt.jl for further details.

Adapt.@adapt_structure CellCache
Adapt.@adapt_structure Ferrite.GeometryMapping
Adapt.@adapt_structure Ferrite.SoAContainer

# Wildcard adapt
adapt_structure(to, ip::Ferrite.Interpolation) = ip

# Adapted manually (instead of @adapt_structure) since getproperty is overloaded
function adapt_structure(to, cv::CellValues)
    fun_values = map(fv -> adapt(to, fv), Ferrite.get_fun_values(cv))
    return CellValues(
        getfield(cv, :fun_values_nt), fun_values,
        adapt(to, Ferrite.get_geo_mapping(cv)),
        adapt(to, Ferrite.get_quadrature_rule(cv)),
        adapt(to, Ferrite.getdetJdVs(cv)),
    )
end

function adapt_structure(to, fv::FacetValues)
    fun_values = map(facet_fvs -> map(f -> adapt(to, f), facet_fvs), getfield(fv, :fun_values))
    geo_mapping = map(gm -> adapt(to, gm), getfield(fv, :geo_mapping))
    return FacetValues(
        getfield(fv, :fun_values_nt),
        fun_values,
        geo_mapping,
        adapt(to, getfield(fv, :fqr)),
        adapt(to, Ferrite.getdetJdVs(fv)),
        adapt(to, getfield(fv, :normals)),
        Ferrite.getcurrentfacet(fv),
    )
end

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

function adapt_structure(to, fqr::FacetQuadratureRule{shape}) where {shape}
    return FacetQuadratureRule{shape}(adapt(to, fqr.facet_rules))
end
