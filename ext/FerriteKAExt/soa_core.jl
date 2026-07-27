function Ferrite.distribute_to_workers(backend::KA.Backend, obj, num_workers) # Could also be KA.GPU <: KA.Backend, but nice to test logic on CPU probably...
    num_workers < 1 && throw(ArgumentError("num_workers must be strictly positive"))
    soa = as_structure_of_arrays(backend, num_workers, obj)
    return Ferrite.SoAContainer(soa, num_workers)
end

zeros_shared(::Any, ::Nothing, ::Integer) = nothing
function zeros_shared(backend, a::AbstractArray{T}, N::Integer) where {T}
    return KA.zeros(backend, T, N, size(a)...)
end

function as_structure_of_arrays(d, N, cv::CellValues)
    return CellValues(
        as_structure_of_arrays(d, N, cv.fun_values),
        as_structure_of_arrays(d, N, cv.geo_mapping),
        adapt(d, cv.qr),
        zeros_shared(d, cv.detJdV, N),
    )
end

function as_structure_of_arrays(d, N, fv::Ferrite.FunctionValues)
    Nξ = adapt(d, fv.Nξ)
    return Ferrite.FunctionValues(
        adapt(d, fv.ip),
        fv.Nξ === fv.Nx ? Nξ : KA.zeros(d, eltype(fv.Nx), N, size(fv.Nx, 1), size(fv.Nx, 2)), # Ensure proper aliasing,
        Nξ,
        zeros_shared(d, fv.dNdx, N),
        adapt(d, fv.dNdξ),
        zeros_shared(d, fv.d2Ndx2, N),
        adapt(d, fv.d2Ndξ2),
    )
end

function as_structure_of_arrays(d, N, fv::Ferrite.GeometryMapping)
    return Ferrite.GeometryMapping(
        adapt(d, fv.ip),
        adapt(d, fv.M),
        adapt(d, fv.dMdξ),
        adapt(d, fv.d2Mdξ2),
    )
end

# We do not need a soa container for the CPU variant.
function Ferrite.distribute_to_workers(backend::KA.CPU, a::AT, num_workers) where {AT <: Ferrite.AbstractAssembler}
    return [
        a; [start_assemble(a.K, a.f; fillzero = false)::AT for _ in 2:num_workers]
    ]
end

# Only supported for thread safe assemblers
function Ferrite.distribute_to_workers(backend::KA.GPU, a::Ferrite.AbstractThreadSafeAssembler, num_workers)
    return Ferrite.SoAContainer(a, num_workers)
end
