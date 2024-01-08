function create_discontinuous_vtk_griddata(grid::Grid{dim, C, T}) where {dim, C, T}
    cls = Vector{WriteVTK.MeshCell}(undef, getncells(grid))
    cellnodes = Vector{UnitRange{Int}}(undef, getncells(grid))
    ncoords = sum(nnodes, getcells(grid))
    coords = zeros(T, dim, ncoords)
    icoord = 0
    for cell in CellIterator(grid)
        CT = getcelltype(grid, cellid(cell))
        vtk_celltype = Ferrite.cell_to_vtkcell(CT)
        cell_coords = getcoordinates(cell)
        n = length(cell_coords)
        cellnodes[cellid(cell)] = (1:n) .+ icoord
        vtk_cellnodes = nodes_to_vtkorder(CT((ntuple(i->i+icoord, n))))
        cls[cellid(cell)] = WriteVTK.MeshCell(vtk_celltype, vtk_cellnodes)
        for x in cell_coords
            icoord += 1
            coords[:, icoord] = x
        end
    end
    return coords, cls, cellnodes
end

function evaluate_at_discontinuous_vtkgrid_nodes(dh::DofHandler, u::Vector{T}, fieldname::Symbol, cellnodes) where T
    # Make sure the field exists
    fieldname ∈ getfieldnames(dh) || error("Field $fieldname not found.")
    # Figure out the return type (scalar or vector)
    field_idx = find_field(dh, fieldname)
    ip = getfieldinterpolation(dh, field_idx)
    RT = ip isa ScalarInterpolation ? T : Vec{n_components(ip),T}
    n_c = n_components(ip)
    vtk_dim = n_c == 2 ? 3 : n_c # VTK wants vectors padded to 3D
    n_vtk_nodes = maximum(maximum, cellnodes)
    data = fill(NaN * zero(T), vtk_dim, n_vtk_nodes)
    # Loop over the subdofhandlers
    for sdh in dh.subdofhandlers
        # Check if this sdh contains this field, otherwise continue to the next
        field_idx = _find_field(sdh, fieldname)
        field_idx === nothing && continue

        # Set up CellValues with the local node coords as quadrature points
        CT = getcelltype(sdh)
        ip = getfieldinterpolation(sdh, field_idx)
        ip_geo = default_interpolation(CT)
        local_node_coords = reference_coordinates(ip_geo)
        qr = QuadratureRule{getrefshape(ip)}(zeros(length(local_node_coords)), local_node_coords)
        if ip isa VectorizedInterpolation
            # TODO: Remove this hack when embedding works...
            cv = CellValues(qr, ip.ip, ip_geo)
        else
            cv = CellValues(qr, ip, ip_geo)
        end
        drange = dof_range(sdh, field_idx)
        # Function barrier
        _evaluate_at_discontinuous_vtkgrid_nodes!(data, sdh, u, cv, drange, RT, cellnodes)
    end
    return data
end

function _evaluate_at_discontinuous_vtkgrid_nodes!(data::Matrix, sdh::SubDofHandler,
    u::Vector{T}, cv::CellValues, drange::UnitRange, ::Type{RT}, cellnodes) where {T, RT}
    ue = zeros(T, length(drange))
    # TODO: Remove this hack when embedding works...
    if RT <: Vec && function_interpolation(cv) isa ScalarInterpolation
        uer = reinterpret(RT, ue)
    else
        uer = ue
    end
    for cell in CellIterator(sdh)
        # Note: We are only using the shape functions: no reinit!(cv, cell) necessary
        @assert getnquadpoints(cv) == length(cell.nodes)
        for (i, I) in pairs(drange)
            ue[i] = u[cell.dofs[I]]
        end
        for (qp, nodeid) in pairs(cellnodes[cellid(cell)])
            val = function_value(cv, qp, uer)
            data[1:length(val), nodeid] .= val
            data[(length(val)+1):end, nodeid] .= 0 # purge the NaN
        end
    end
    return data
end
