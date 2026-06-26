# NOTE CellCache is mutable and hence inherently incompatible with GPU. So here is the
# immutable variant. Making the CellCache immutable is considered breaking due to the reinit! API integration.
struct ImmutableCellCache{G <: AbstractGrid, SDH, IVT, VX}
    flags::UpdateFlags
    grid::G
    cellid::Int
    nodes::IVT
    coords::VX
    sdh::SDH
    dofs::IVT
end
function reinit(cc::ImmutableCellCache, cellid::Int)
    cc2 = ImmutableCellCache(cc.flags, cc.grid, cellid, cc.nodes, cc.coords, cc.sdh, cc.dofs)
    cc2.flags.nodes  && Ferrite.cellnodes!(cc2.nodes, cc2.grid, cellid)
    cc2.flags.coords && Ferrite.getcoordinates!(cc2.coords, cc2.grid, cellid)
    cc2.sdh !== nothing && cc2.flags.dofs && Ferrite.celldofs!(cc2.dofs, cc2.sdh, cellid)
    return cc2
end

@inline Ferrite.celldofs(cc::ImmutableCellCache) = cc.dofs
@inline Ferrite.reinit!(cv::Ferrite.AbstractCellValues, cc::ImmutableCellCache) = reinit!(cv, cc.coords)
@inline Ferrite.getcoordinates(cc::ImmutableCellCache) = cc.coords
@inline Ferrite.cellid(cc::ImmutableCellCache) = cc.cellid

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

function as_structure_of_arrays(backend, N, cc::ImmutableCellCache)
    return ImmutableCellCache(
        cc.flags, cc.grid, -1, 
        zeros_shared(backend, cc.nodes, N),
        zeros_shared(backend, cc.coords, N),
        cc.sdh,
        zeros_shared(backend, cc.dofs, N)
    )
end

function Ferrite.CellCache(dh::HostDofHandler{dim}, flags::UpdateFlags = UpdateFlags(), immutable = false) where {dim}
    @assert length(dh.subdofhandlers) == 1 "CellCache only works on HostDofHandler's with a single subdomain. Please call the CellCache on the corresponding DeviceSubDofHandler."
    return CellCache(first(dh.subdofhandlers), flags, immutable)
end

function Ferrite.CellCache(sdh::DeviceSubDofHandler{dim}, flags::UpdateFlags = UpdateFlags(), immutable = false) where {dim}
    backend = get_backend(sdh.cellset)
    grid = get_grid(sdh)
    @allowscalar N = Ferrite.nnodes_per_cell(grid, first(sdh.cellset))
    nodes = KA.zeros(backend, Int, N)
    coords = KA.zeros(backend, Vec{dim, get_coordinate_eltype(grid)}, N)
    cellid = KA.zeros(backend, Int, 1)

    n = Ferrite.ndofs_per_cell(sdh)
    dofs = KA.zeros(backend, Int, n)
    return immutable ? ImmutableCellCache(flags, grid, -1, nodes, coords, sdh, dofs) : CellCache(flags, grid, cellid, nodes, coords, sdh, dofs)
end

function adapt_structure(backend, cc::ImmutableCellCache)
    return ImmutableCellCache(
        cc.flags,
        adapt(backend, cc.grid),
        -1,
        adapt(backend, cc.nodes),
        adapt(backend, cc.coords),
        adapt(backend, cc.sdh),
        adapt(backend, cc.dofs),
    )
end

function get_substruct(i, cc::ImmutableCellCache)
    return ImmutableCellCache(
        cc.flags, cc.grid, -1,
        view(cc.nodes, i, :), view(cc.coords, i, :), cc.sdh, view(cc.dofs, i, :)
    )
end

function Ferrite.reinit!(cc_i::ImmutableCellCache, cellid::Integer)
    cc_i.flags.nodes  && Ferrite.cellnodes!(cc_i.nodes, cc_i.grid, cellid)
    cc_i.flags.coords && Ferrite.getcoordinates!(cc_i.coords, cc_i.grid, cellid)
    cc_i.sdh !== nothing && cc_i.flags.dofs && Ferrite.celldofs!(cc_i.dofs, cc_i.sdh, cellid)
    return nothing
end

#############

function get_substruct(i, cc::CellCache)
    return CellCache(
        cc.flags, cc.grid, view(cc.cellid, i:i),
        view(cc.nodes, i, :), view(cc.coords, i, :), cc.dh, view(cc.dofs, i, :)
    )
end
