# This file defines the adapt_structure function, which is used to adapt custom structures to be used on the GPU.

function Adapt.adapt_structure(to, cv::CellValues)
    fv = Adapt.adapt(to, StaticInterpolationValues(cv.fun_values))
    gm = Adapt.adapt(to, StaticInterpolationValues(cv.geo_mapping))
    weights = Adapt.adapt(to, ntuple(i -> getweights(cv.qr)[i], getnquadpoints(cv)))
    return Ferrite.StaticCellValues(fv, gm, weights)
end


function Adapt.adapt_structure(to, iter::Ferrite.QuadratureValuesIterator)
    cv = Adapt.adapt_structure(to, iter.v)
    cell_coords = Adapt.adapt_structure(to, iter.cell_coords)
    return Ferrite.QuadratureValuesIterator(cv, cell_coords)
end

function Adapt.adapt_structure(to, qv::Ferrite.StaticQuadratureValues)
    det = Adapt.adapt_structure(to, qv.detJdV)
    N = Adapt.adapt_structure(to, qv.N)
    dNdx = Adapt.adapt_structure(to, qv.dNdx)
    M = Adapt.adapt_structure(to, qv.M)
    return Ferrite.StaticQuadratureValues(det, N, dNdx, M)
end
function Adapt.adapt_structure(to, qv::StaticQuadratureView)
    mapping = Adapt.adapt_structure(to, qv.mapping)
    cell_coords = Adapt.adapt_structure(to, qv.cell_coords |> cu)
    q_point = Adapt.adapt_structure(to, qv.q_point)
    cv = Adapt.adapt_structure(to, qv.cv)
    return StaticQuadratureView(mapping, cell_coords, q_point, cv)
end

function Adapt.adapt_structure(to, grid::Grid)
    # map Int64 to Int32 to reduce number of registers
    cu_cells = grid.cells .|> (x -> Int32.(x.nodes)) .|> Quadrilateral |> cu
    cells = Adapt.adapt_structure(to, cu_cells)
    nodes = Adapt.adapt_structure(to, cu(grid.nodes))
    return GPUGrid(cells, nodes)
end


function Adapt.adapt_structure(to, iterator::CUDACellIterator)
    grid = Adapt.adapt_structure(to, iterator.grid)
    dh = Adapt.adapt_structure(to, iterator.dh)
    ncells = Adapt.adapt_structure(to, iterator.n_cells)
    return GPUCellIterator(dh, grid, ncells)
end


function _get_ndofs_cell(dh::DofHandler)
    ndofs_cell = [Int32(Ferrite.ndofs_per_cell(dh, i)) for i in 1:(dh |> Ferrite.get_grid |> Ferrite.getncells)]
    return ndofs_cell
end

# function Adapt.adapt_structure(to, dh::DofHandler)
#     cell_dofs = Adapt.adapt_structure(to, dh.cell_dofs .|> Int32 |> cu)
#     cells = Adapt.adapt_structure(to, dh.grid.cells |> cu)
#     offsets = Adapt.adapt_structure(to, dh.cell_dofs_offset .|> Int32 |> cu)
#     nodes = Adapt.adapt_structure(to, dh.grid.nodes |> cu)
#     ndofs_cell = Adapt.adapt_structure(to, get_ndofs_cell(dh) |> cu)
#     GPUDofHandler(cell_dofs, GPUGrid(cells,nodes),offsets, Ferrite.isclosed(dh), ndofs_cell)
# end

function Adapt.adapt_structure(to, dh::DofHandler)
    cell_dofs = Adapt.adapt_structure(to, dh.cell_dofs .|> Int32 |> cu)
    cells = Adapt.adapt_structure(to, dh.grid.cells |> cu)
    offsets = Adapt.adapt_structure(to, dh.cell_dofs_offset .|> Int32 |> cu)
    nodes = Adapt.adapt_structure(to, dh.grid.nodes |> cu)
    ndofs_cell = Adapt.adapt_structure(to, _get_ndofs_cell(dh) |> cu)
    return GPUDofHandler(cell_dofs, GPUGrid(cells, nodes), offsets, ndofs_cell)
end


function Adapt.adapt_structure(to, dh::LocalsGPUDofHandler)
    dh_ = Adapt.adapt_structure(to, dh |> dofhandler)
    Kes = Adapt.adapt_structure(to, dh |> localkes)
    fes = Adapt.adapt_structure(to, dh |> localfes)
    return LocalsGPUDofHandler(dh_, Kes, fes)
end


function Adapt.adapt_structure(to, assembler::GPUAssemblerSparsityPattern)
    K = Adapt.adapt_structure(to, assembler.K)
    f = Adapt.adapt_structure(to, assembler.f)
    return Ferrite.GPUAssemblerSparsityPattern(K, f)
end
