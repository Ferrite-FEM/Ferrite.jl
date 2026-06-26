function as_structure_of_arrays(backend, N, cc::CellCache)
    return Ferrite.CellCache(
        cc.flags, cc.grid, 
        KA.zeros(backend, eltype(cc.cellid), N),
        zeros_shared(backend, cc.nodes, N),
        zeros_shared(backend, cc.coords, N),
        cc.dh,
        zeros_shared(backend, cc.dofs, N)
    )
end

function Ferrite.CellCache(dh::HostDofHandler{dim}, flags::UpdateFlags = UpdateFlags()) where {dim}
    @assert length(dh.subdofhandlers) == 1 "CellCache only works on HostDofHandler's with a single subdomain. Please call the CellCache on the corresponding DeviceSubDofHandler."
    return CellCache(first(dh.subdofhandlers), flags)
end

function Ferrite.CellCache(sdh::DeviceSubDofHandler{dim}, flags::UpdateFlags = UpdateFlags()) where {dim}
    backend = get_backend(sdh.cellset)
    grid = get_grid(sdh)
    @allowscalar N = Ferrite.nnodes_per_cell(grid, first(sdh.cellset))
    nodes = KA.zeros(backend, Int, N)
    coords = KA.zeros(backend, get_coordinate_type(grid), N)
    cellid = KA.zeros(backend, Int, 1)

    n = Ferrite.ndofs_per_cell(sdh)
    dofs = KA.zeros(backend, Int, n)
    return CellCache(flags, grid, cellid, nodes, coords, sdh, dofs)
end
