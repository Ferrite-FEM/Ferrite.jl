# This file defines the adapt_structure function, which is used to adapt custom structures to be used on the GPU.

Adapt.@adapt_structure GPUGrid
Adapt.@adapt_structure GPUDofHandler
Adapt.@adapt_structure GlobalMemAlloc

function _adapt_args(args)
    return tuple(((_adapt(arg) for arg in args) |> collect)...)
end


function _adapt(kgpu::CUSPARSE.CuSparseMatrixCSC)
    # custom adaptation
    return Adapt.adapt_structure(CUSPARSE.CuSparseDeviceMatrixCSC, kgpu)
end


function _adapt(obj::Any)
    # fallback to the default implementation
    return Adapt.adapt_structure(CuArray, obj)
end

## Adapt GlobalMemAlloc
function Adapt.adapt_structure(to, mem_alloc::GlobalMemAlloc)
    kes = Adapt.adapt_structure(to, mem_alloc.Kes)
    fes = Adapt.adapt_structure(to, mem_alloc.fes)
    return GlobalMemAlloc(kes, fes)
end


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

_symbols_to_int32(symbols) = 1:length(symbols) .|> (sym -> convert(Int32, sym))

function Adapt.adapt_structure(to, sdh::SubDofHandler)
    cellset = Adapt.adapt_structure(to, sdh.cellset |> collect .|> (x -> convert(Int32, x)) |> cu)
    field_names = Adapt.adapt_structure(to, _symbols_to_int32(sdh.field_names) |> cu)
    field_interpolations = sdh.field_interpolations .|> (ip -> Adapt.adapt_structure(to, ip)) |> cu
    ndofs_per_cell = Adapt.adapt_structure(to, sdh.ndofs_per_cell)
    return GPUSubDofHandler(cellset, field_names, field_interpolations, ndofs_per_cell)
end

function Adapt.adapt_structure(to, dh::DofHandler)
    subdofhandlers = dh.subdofhandlers .|> (sdh -> Adapt.adapt_structure(to, sdh)) |> cu
    cell_dofs = Adapt.adapt_structure(to, dh.cell_dofs .|> (x -> convert(Int32, x)) |> cu)
    cells = Adapt.adapt_structure(to, dh.grid.cells |> cu)
    offsets = Adapt.adapt_structure(to, dh.cell_dofs_offset .|> Int32 |> cu)
    nodes = Adapt.adapt_structure(to, dh.grid.nodes |> cu)
    #ndofs_cell = Adapt.adapt_structure(to, _get_ndofs_cell(dh) |> cu)
    cell_to_subdofhandler = Adapt.adapt_structure(to, dh.cell_to_subdofhandler .|> (x -> convert(Int32, x)) |> cu)
    return GPUDofHandler(subdofhandlers, cell_dofs, GPUGrid(cells, nodes), offsets, cell_to_subdofhandler)
end

## TODO: remove this...not needed
function Adapt.adapt_structure(to, assembler::GPUAssemblerSparsityPattern)
    K = Adapt.adapt_structure(to, assembler.K)
    f = Adapt.adapt_structure(to, assembler.f)
    return Ferrite.GPUAssemblerSparsityPattern(K, f)
end
