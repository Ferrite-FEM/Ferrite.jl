function Ferrite.distribute_to_tasks(backend::KA.Backend, obj, num_tasks) # Could also be KA.GPU <: KA.Backend, but nice to test logic on CPU probably...
    num_tasks < 1 && throw(ArgumentError("num_tasks must be strictly positive"))
    soa = as_structure_of_arrays(backend, num_tasks, obj)
    return Ferrite.SoAContainer(soa, num_tasks)
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
        fv.Nξ === fv.Nx ? Nξ : zeros_shared(d, fv.Nx, N), # Ensure proper aliasing,
        Nξ,
        zeros_shared(d, fv.dNdx, N),
        adapt(d, fv.dNdξ),
        zeros_shared(d, fv.d2Ndx2, N),
        adapt(d, fv.d2Ndξ2),
    )
end

function as_structure_of_arrays(d, N, fv::Ferrite.GeometryMapping)
    return Ferrite.GeometryMapping(
        fv.ip, zeros_shared(d, fv.M, N),
        adapt(d, fv.dMdξ), adapt(d, fv.d2Mdξ2),
    )
end

function as_structure_of_arrays(_, _, ::Ferrite.ImmutableCellCache{<:Any, <:Union{<:Ferrite.DofHandler, <:Ferrite.SubDofHandler}})
    throw(ArgumentError("Distributing ImmutableCellCache for GPU requires that it has a GPU-compatible (sub)dofhandler"))
end

function as_structure_of_arrays(d, N, cc::Ferrite.ImmutableCellCache{<:Any, <:HostDofHandler})
    @assert length(cc.sdh.subdofhandlers) == 1 "Distributed ImmutableCellCache only works on HostDofHandler's with a single subdomain. Please call the ImmutableCellCache adaptation on the DeviceSubDofHandler."
    sdh = only(cc.sdh.subdofhandlers)
    sub_cc = Ferrite.ImmutableCellCache(cc.flags, cc.grid, cc.cellid, cc.nodes, cc.coords, sdh, cc.dofs)
    return as_structure_of_arrays(d, N, sub_cc)
end

function as_structure_of_arrays(d, N, cc::Ferrite.ImmutableCellCache{<:Any, <:DeviceSubDofHandler})
    return Ferrite.ImmutableCellCache(
        cc.flags, cc.grid, -1, 
        zeros_shared(d, cc.nodes, N),
        zeros_shared(d, cc.coords, N),
        cc.sdh,
        zeros_shared(d, cc.dofs, N)
    )
end
