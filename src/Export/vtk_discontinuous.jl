function create_discontinuous_vtk_griddata(grid::Grid{dim, C, T}) where {dim, C, T}
    cls = Vector{WriteVTK.MeshCell}(undef, getncells(grid))
    cellnodes = Vector{UnitRange{Int}}(undef, getncells(grid))
    ncoords = sum(nnodes, getcells(grid))
    coords = zeros(T, dim, ncoords)
    icoord = 0
    for cell in CellIterator(grid)
        CT = getcelltype(grid, cellid(cell))
        vtk_celltype = cell_to_vtkcell(CT)
        cell_coords = getcoordinates(cell)
        n = length(cell_coords)
        cellnodes[cellid(cell)] = (1:n) .+ icoord
        vtk_cellnodes = nodes_to_vtkorder(CT((ntuple(i -> i + icoord, n))))
        cls[cellid(cell)] = WriteVTK.MeshCell(vtk_celltype, vtk_cellnodes)
        for x in cell_coords
            icoord += 1
            coords[:, icoord] = x
        end
    end
    return coords, cls, cellnodes
end

function evaluate_at_discontinuous_vtkgrid_nodes(dh::DofHandler{sdim}, u::Vector{T}, fieldname::Symbol, cellnodes) where {sdim, T}
    # Make sure the field exists
    fieldname ∈ getfieldnames(dh) || error("Field $fieldname not found.")
    # Figure out the return type (scalar or vector)
    field_idx = find_field(dh, fieldname)
    ip = getfieldinterpolation(dh, field_idx)

    get_vtk_dim(::ScalarInterpolation, ::AbstractVector{<:Number}) = 1
    get_vtk_dim(::ScalarInterpolation, ::AbstractVector{<:Vec{dim}}) where {dim} = dim == 2 ? 3 : dim
    get_vtk_dim(::VectorInterpolation{vdim}, ::AbstractVector{<:Number}) where {vdim} = vdim == 2 ? 3 : vdim

    vtk_dim = get_vtk_dim(ip, u)
    n_vtk_nodes = maximum(maximum, cellnodes)
    data = fill(NaN * zero(eltype(T)), vtk_dim, n_vtk_nodes)
    # Loop over the subdofhandlers
    for sdh in dh.subdofhandlers
        # Check if this sdh contains this field, otherwise continue to the next
        field_idx = _find_field(sdh, fieldname)
        field_idx === nothing && continue

        # Set up CellValues with the local node coords as quadrature points
        CT = getcelltype(sdh)
        ip = getfieldinterpolation(sdh, field_idx)
        ip_geo = geometric_interpolation(CT)
        local_node_coords = reference_coordinates(ip_geo)
        qr = QuadratureRule{getrefshape(ip)}(zeros(length(local_node_coords)), local_node_coords)
        cv = CellValues(qr, ip, ip_geo^sdim; update_gradients = false, update_hessians = false, update_detJdV = false)
        drange = dof_range(sdh, field_idx)
        # Function barrier
        _evaluate_at_discontinuous_vtkgrid_nodes!(data, sdh, u, cv, drange, cellnodes)
    end
    return data
end

function _evaluate_at_discontinuous_vtkgrid_nodes!(
        data::Matrix, sdh::SubDofHandler,
        u::Vector{T}, cv::CellValues, drange::UnitRange, cellnodes
    ) where {T}
    ue = zeros(T, length(drange))
    for cell in CellIterator(sdh)
        reinit!(cv, cell)
        @assert getnquadpoints(cv) == length(cell.nodes)
        for (i, I) in pairs(drange)
            ue[i] = u[cell.dofs[I]]
        end
        for (qp, nodeid) in pairs(cellnodes[cellid(cell)])
            val = function_value(cv, qp, ue)
            data[1:length(val), nodeid] .= val
            data[(length(val) + 1):end, nodeid] .= 0 # purge the NaN
        end
    end
    return data
end
