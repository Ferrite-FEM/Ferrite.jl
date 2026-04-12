# This file contains adapt rules for all relevant data structures in Ferrite.jl.
# During setup, these rules are typically called for a `KA.Backend` (e.g. `CUDABackend()`), 
# and later during kernel construction, these are called for the specific kernel, 
# e.g. CUDA.KernelAdaptor(). Please consult Adapt.jl for further details.

function adapt_structure(to, c::Ferrite.SoAContainer)
    return Ferrite.SoAContainer(adapt(to, c.soa), c.nels)
end

function adapt_structure(d, cv::CellValues)
    return CellValues(
        adapt(d, cv.fun_values),
        adapt(d, cv.geo_mapping),
        adapt(d, cv.qr),
        adapt(d, cv.detJdV),
    )
end

function adapt_structure(d, fv::Ferrite.FunctionValues)
    Nξ = adapt(d, fv.Nξ)
    return Ferrite.FunctionValues(
        fv.ip,
        fv.Nξ === fv.Nx ? Nξ : adapt(d, fv.Nx), # Ensure proper aliasing
        Nξ,
        adapt(d, fv.dNdx),
        adapt(d, fv.dNdξ),
        adapt(d, fv.d2Ndx2),
        adapt(d, fv.d2Ndξ2),
    )
end

function adapt_structure(d, gm::Ferrite.GeometryMapping)
    return Ferrite.GeometryMapping(
        adapt(d, gm.ip),
        adapt(d, gm.M),
        adapt(d, gm.dMdξ),
        adapt(d, gm.d2Mdξ2),
    )
end

function adapt_structure(to, qr::QuadratureRule{shape}) where {shape}
    return QuadratureRule{shape}(adapt(to, qr.weights), adapt(to, qr.points))
end

function adapt_structure(to, grid::DeviceGrid)
    return DeviceGrid(adapt(to, grid.cells), adapt(to, grid.nodes))
end

function adapt_structure(to, sdh::DeviceSubDofHandler)
    return DeviceSubDofHandler(
        adapt(to, sdh.cellset),
        adapt(to, sdh.cell_dofs),
        adapt(to, sdh.cell_dofs_offset),
        sdh.ndofs_per_cell,
        sdh.nnodes_per_cell,
        adapt(to, sdh.dof_ranges),
        adapt(to, sdh.grid),
    )
end

function adapt_structure(to::KA.Backend, dh::DofHandler)
    return HostDofHandler(to, dh)
end

function adapt_structure(backend, cc::Ferrite.ImmutableCellCache)
    return Ferrite.ImmutableCellCache(
        cc.flags,
        adapt(backend, cc.grid),
        -1,
        adapt(backend, cc.nodes),
        adapt(backend, cc.coords),
        adapt(backend, cc.sdh),
        adapt(backend, cc.dofs),
    )
end
