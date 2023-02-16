"""
"""
function WriteVTK.vtk_grid(filename::AbstractString, dgrid::DistributedGrid{dim,C,T}; compress::Bool=true) where {dim,C,T}
    part   = MPI.Comm_rank(global_comm(dgrid))+1
    nparts = MPI.Comm_size(global_comm(dgrid))
    cls = MeshCell[]
    for cell in getcells(dgrid)
        celltype = Ferrite.cell_to_vtkcell(typeof(cell))
        push!(cls, MeshCell(celltype, nodes_to_vtkorder(cell)))
    end
    coords = reshape(reinterpret(T, getnodes(dgrid)), (dim, getnnodes(dgrid)))
    return pvtk_grid(filename, coords, cls; part=part, nparts=nparts, compress=compress)
end

"""
"""
function WriteVTK.vtk_point_data(vtk, dh::Ferrite.AbstractDofHandler, u::PVector)
    map_parts(local_view(u, u.rows)) do u_local
        vtk_point_data(pvtkwrapper(vtk), dh, u_local)
    end
end

"""
Enrich the VTK file with meta information about shared vertices.
"""
function vtk_shared_vertices(vtk, dgrid::DistributedGrid)
    u = Vector{Float64}(undef, getnnodes(dgrid))
    my_rank = MPI.Comm_rank(global_comm(dgrid))+1
    for rank ∈ 1:MPI.Comm_size(global_comm(dgrid))
        fill!(u, 0.0)
        for sv ∈ values(get_shared_vertices(dgrid))
            if haskey(sv.remote_vertices, rank)
                (cellidx, i) = sv.local_idx
                cell = getcells(dgrid, cellidx)
                u[vertices(cell)[i]] = my_rank
            end
        end
        vtk_point_data(pvtkwrapper(vtk), u, "shared vertices with $rank")
    end
end


"""
Enrich the VTK file with meta information about shared faces.
"""
function vtk_shared_faces(vtk, dgrid::DistributedGrid)
    u = Vector{Float64}(undef, getnnodes(dgrid))
    my_rank = MPI.Comm_rank(global_comm(dgrid))+1
    for rank ∈ 1:MPI.Comm_size(global_comm(dgrid))
        fill!(u, 0.0)
        for sf ∈ values(get_shared_faces(dgrid))
            if haskey(sf.remote_faces, rank)
                (cellidx, i) = sf.local_idx
                cell = getcells(dgrid, cellidx)
                facenodes = faces(cell)[i]
                u[[facenodes...]] .= my_rank
            end
        end
        vtk_point_data(pvtkwrapper(vtk), u, "shared faces with $rank")
    end
end


"""
Enrich the VTK file with meta information about shared edges.
"""
function vtk_shared_edges(vtk, dgrid::DistributedGrid)
    u = Vector{Float64}(undef, getnnodes(dgrid))
    my_rank = MPI.Comm_rank(global_comm(dgrid))+1
    for rank ∈ 1:MPI.Comm_size(global_comm(dgrid))
        fill!(u, 0.0)
        for se ∈ values(get_shared_edges(dgrid))
            if haskey(se.remote_edges, rank)
                (cellidx, i) = se.local_idx
                cell = getcells(dgrid, cellidx)
                edgenodes = edges(cell)[i]
                u[[edgenodes...]] .= my_rank
            end
        end
        vtk_point_data(pvtkwrapper(vtk), u, "shared edges with $rank")
    end
end

"""
Enrich the VTK file with partitioning meta information.
"""
function vtk_partitioning(vtk, dgrid::DistributedGrid)
    u  = Vector{Float64}(undef, getncells(dgrid))
    u .= MPI.Comm_rank(global_comm(dgrid))+1
    vtk_cell_data(pvtkwrapper(vtk), u, "partitioning")
end
