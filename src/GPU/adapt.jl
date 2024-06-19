# This file defines the adapt_structure function, which is used to adapt custom structures to be used on the GPU.

function Adapt.adapt_structure(to,cv::CellValues)
    fv = Adapt.adapt(to,StaticInterpolationValues(cv.fun_values))
    gm =Adapt.adapt(to,StaticInterpolationValues(cv.geo_mapping))
    weights =Adapt.adapt(to, ntuple(i -> getweights(cv.qr)[i], getnquadpoints(cv)))
    Ferrite.StaticCellValues(fv,gm, weights)
end


function Adapt.adapt_structure(to, iter::Ferrite.QuadratureValuesIterator)
    cv = Adapt.adapt_structure(to, iter.v)
    cell_coords = Adapt.adapt_structure(to, iter.cell_coords)
    Ferrite.QuadratureValuesIterator(cv,cell_coords)
end

function Adapt.adapt_structure(to, qv::Ferrite.StaticQuadratureValues)
    det = Adapt.adapt_structure(to, qv.detJdV)
    N = Adapt.adapt_structure(to, qv.N)
    dNdx = Adapt.adapt_structure(to, qv.dNdx)
    M = Adapt.adapt_structure(to, qv.M)
    Ferrite.StaticQuadratureValues(det,N,dNdx,M)
end
function Adapt.adapt_structure(to, qv::StaticQuadratureView)
    mapping = Adapt.adapt_structure(to, qv.mapping)
    cell_coords = Adapt.adapt_structure(to, qv.cell_coords)
    q_point = Adapt.adapt_structure(to, qv.q_point)
    cv = Adapt.adapt_structure(to, qv.cv)
    StaticQuadratureView(mapping,cell_coords,q_point,cv)
end

function Adapt.adapt_structure(to, grid::Grid)
    cells = Adapt.adapt_structure(to, cu(grid.cells))
    nodes = Adapt.adapt_structure(to, cu(grid.nodes))
    GPUGrid(cells,nodes)
end

# function Adapt.adapt_structure(to, dh::DofHandler)
#     cell_dofs = Adapt.adapt_structure(to, Int32.(dh.cell_dofs) |> cu)
#     cells = Adapt.adapt_structure(to, cu(dh.grid.cells))
#     nodes = Adapt.adapt_structure(to, cu(dh.grid.nodes))
#     GPUDofHandler(cell_dofs, GPUGrid(cells,nodes))
# end


function get_ndofs_cell(dh::DofHandler)
    ndofs_cell = [ndofs_per_cell(dh, i) for i in 1:(dh |> get_grid |> getncells)]
    return ndofs_cell
end

function Adapt.adapt_structure(to, dh::DofHandler)
    cell_dofs = Adapt.adapt_structure(to, dh.cell_dofs |> cu)
    cells = Adapt.adapt_structure(to, dh.grid.cells |> cu)
    offsets = Adapt.adapt_structure(to, dh.cell_dofs_offset |> cu)
    nodes = Adapt.adapt_structure(to, dh.grid.nodes |> cu)
    ndofs_cell = Adapt.adapt_structure(to, get_ndofs_cell(dh) |> cu)
    GPUDofHandler(cell_dofs, GPUGrid(cells,nodes),offsets, Ferrite.isclosed(dh), ndofs_cell)
end


function Adapt.adapt_structure(to, K::SparseMatrixCSC)
    m = Adapt.adapt_structure(to, Int32(K.m))
    n = Adapt.adapt_structure(to, Int32(K.n))
    colptr = Adapt.adapt_structure(to, Int32.(K.colptr)|>cu)
    rowval = Adapt.adapt_structure(to, Int32.(K.rowval)|>cu)
    nzval = Adapt.adapt_structure(to, Float32.(K.nzval) |> cu)
    GPUSparseMatrixCSC(m, n, colptr, rowval, nzval)
end

function Adapt.adapt_structure(to, K::GPUSparseMatrixCSC)
    m = Adapt.adapt_structure(to, K.m)
    n = Adapt.adapt_structure(to, K.n)
    colptr = Adapt.adapt_structure(to, K.colptr)
    rowval = Adapt.adapt_structure(to, K.rowval)
    nzval = Adapt.adapt_structure(to, K.nzval)
    GPUSparseMatrixCSC(m, n, colptr, rowval, nzval)
end



