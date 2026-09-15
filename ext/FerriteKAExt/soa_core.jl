# We intentionally do not restrict the path to KA.GPU <: KA.Backend, as the CPU backend
# can be helpful to isolate problems in the assembly kernels.
function Ferrite.distribute_to_workers(backend::KA.Backend, obj, num_workers)
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
        fv.Nξ === fv.Nx ? Nξ : KA.zeros(d, eltype(fv.Nx), N, size(fv.Nx, 1), size(fv.Nx, 2)), # Ensure proper aliasing
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

# Assemblers without sorting scratch (created by `start_assemble` on device arrays) hold no
# per-worker state, so all workers share a single instance.
const ScratchFreeAssembler = Union{
    Ferrite.CSCAssembler{<:Any, <:Any, <:Any, <:Any, <:Any, Nothing},
    Ferrite.CSRAssembler{<:Any, <:Any, <:Any, <:Any, <:Any, Nothing},
}

function Ferrite.distribute_to_workers(backend::KA.Backend, a::ScratchFreeAssembler, num_workers)
    return Ferrite.SoAContainer(a, num_workers)
end

get_substruct(a::ScratchFreeAssembler, i) = a

# Assemblers with sorting scratch need one instance per worker, which we can only allocate
# on the host.
function Ferrite.distribute_to_workers(backend::KA.Backend, a::AT, num_workers) where {AT <: Ferrite.AbstractAssembler}
    if !(backend isa KA.CPU)
        throw(ArgumentError("$AT cannot be used on $backend, create the assembler with `start_assemble` on device arrays instead"))
    end
    return [
        a; [start_assemble(a.K, a.f; fillzero = false, atomic = Ferrite._is_atomic(a))::AT for _ in 2:num_workers]
    ]
end
