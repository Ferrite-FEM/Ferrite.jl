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

function Adapt.adapt_structure(to, grid::Grid)
    cells = Adapt.adapt_structure(to, cu(grid.cells))
    nodes = Adapt.adapt_structure(to, cu(grid.nodes))
    GPUGrid(cells,nodes)
end

function Adapt.adapt_structure(to, dh::DofHandler)
    cell_dofs = Adapt.adapt_structure(to, cu(dh.cell_dofs))
    cells = Adapt.adapt_structure(to, cu(dh.grid.cells))
    nodes = Adapt.adapt_structure(to, cu(dh.grid.nodes))
    GPUDofHandler(cell_dofs, GPUGrid(cells,nodes))
end