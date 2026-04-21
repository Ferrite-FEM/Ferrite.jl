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
function (cc::ImmutableCellCache)(cellid::Int)
    cc2 = ImmutableCellCache(cc.flags, cc.grid, cellid, cc.nodes, cc.coords, cc.sdh, cc.dofs)
    reinit!(cc2, cellid)
    return cc2
end
Adapt.@adapt_structure ImmutableCellCache
function adapt_structure(to, ccc::CellCacheContainer)
    inner_values = adapt(to, ccc.values)
    return CellCacheContainer{typeof(get_substruct(1, inner_values, -1)), typeof(inner_values)}(inner_values)
end

@inline Ferrite.celldofs(cc::ImmutableCellCache) = cc.dofs
@inline Ferrite.reinit!(cv::Ferrite.AbstractCellValues, cc::ImmutableCellCache) = reinit!(cv, cc.coords)
@inline Ferrite.getcoordinates(cc::ImmutableCellCache) = cc.coords
@inline Ferrite.cellid(cc::ImmutableCellCache) = cc.cellid

function as_structure_of_arrays(backend, outer_dim, ::Type{CellCache}, dh::HostDofHandler, flags::UpdateFlags = UpdateFlags())
    @assert length(dh.subdofhandlers) == 1 "ImmutableCellCache only works on HostDofHandler's with a single subdomain. Please call the ImmutableCellCache adaptation on the DeviceSubDofHandler."
    return as_structure_of_arrays(backend, outer_dim, CellCache, first(dh.subdofhandlers), flags)
end

function as_structure_of_arrays(backend, outer_dim, ::Type{CellCache}, sdh::DeviceSubDofHandler{dim}, flags::UpdateFlags = UpdateFlags()) where {dim}
    grid = get_grid(sdh)
    begin
        n = Ferrite.ndofs_per_cell(sdh)
        N = Ferrite.nnodes_per_cell(sdh)
        nodes = KA.zeros(backend, Int, outer_dim, N)
        coords = KA.zeros(backend, Vec{dim, get_coordinate_eltype(grid)}, outer_dim, N)
        dofs = KA.zeros(backend, Int, outer_dim, n)
    end
    return ImmutableCellCache(flags, grid, -1, nodes, coords, sdh, dofs)
end

function Ferrite.CellCache(backend, dh::HostDofHandler{dim}, flags::UpdateFlags = UpdateFlags()) where {dim}
    @assert length(dh.subdofhandlers) == 1 "ImmutableCellCache only works on HostDofHandler's with a single subdomain. Please call the ImmutableCellCache adaptation on the DeviceSubDofHandler."
    return CellCache(backend, first(dh.subdofhandlers), flags)
end

function Ferrite.CellCache(backend, sdh::DeviceSubDofHandler{dim}, flags::UpdateFlags = UpdateFlags()) where {dim}
    grid = get_grid(sdh)
    N = Ferrite.nnodes_per_cell(grid, first(sdh.cellset))
    nodes = KA.zeros(backend, Int, N)
    coords = KA.zeros(backend, Vec{dim, get_coordinate_eltype(grid)}, N)

    n = Ferrite.ndofs_per_cell(sdh)
    dofs = KA.zeros(backend, Int, n)
    return ImmutableCellCache(flags, grid, -1, nodes, coords, sdh, dofs)
end

function adapt(backend, cc::ImmutableCellCache)
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

function get_substruct(i, cc::ImmutableCellCache, cellid)
    return ImmutableCellCache(
        cc.flags, cc.grid, cellid,
        view(cc.nodes, i, :), view(cc.coords, i, :), cc.sdh, view(cc.dofs, i, :)
    )
end

function Ferrite.reinit!(cc_i::ImmutableCellCache, cellid::Integer)
    cc_i.flags.nodes  && Ferrite.cellnodes!(cc_i.nodes, cc_i.grid, cellid)
    cc_i.flags.coords && Ferrite.getcoordinates!(cc_i.coords, cc_i.grid, cellid)
    cc_i.sdh !== nothing && cc_i.flags.dofs && Ferrite.celldofs!(cc_i.dofs, cc_i.sdh, cellid)
    return nothing
end
