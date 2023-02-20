"""
    DistributedDofHandler(grid::AbstractDistributedGrid)

Construct a `DistributedDofHandler` based on `grid`.

Distributed version of [`DofHandler`](@docs). 

Supports:
- `Grid`s with a single concrete cell type.
- One or several fields on the whole domaine.
"""
struct DistributedDofHandler{dim,T,G<:Ferrite.AbstractDistributedGrid{dim}} <: Ferrite.AbstractDofHandler
    field_names::Vector{Symbol}
    field_dims::Vector{Int}
    # TODO: field_interpolations can probably be better typed: We should at least require
    #       all the interpolations to have the same dimension and reference shape
    field_interpolations::Vector{Interpolation}
    bc_values::Vector{Ferrite.BCValues{T}} # TODO: BcValues is created/handeld by the constrainthandler, so this can be removed
    cell_dofs::Vector{Int}
    cell_dofs_offset::Vector{Int}
    closed::Ferrite.ScalarWrapper{Bool}
    grid::G
    ndofs::Ferrite.ScalarWrapper{Int}

    ldof_to_gdof::Vector{Int}
    ldof_to_rank::Vector{Int32}
end

"""
Compute the global dof range of the dofs owned by the calling process. It is guaranteed to be continuous.
"""
function local_dof_range(dh::DistributedDofHandler)
    my_rank = global_rank(getglobalgrid(dh))
    ltdofs = dh.ldof_to_gdof[dh.ldof_to_rank .== my_rank]
    return minimum(ltdofs):maximum(ltdofs)
end

"""
Construct the correct distributed dof handler from a given distributed grid.
"""
function Ferrite.DofHandler(grid::Ferrite.AbstractDistributedGrid{dim}) where {dim}
    isconcretetype(getcelltype(grid)) || error("Grid includes different celltypes. DistributedMixedDofHandler not implemented yet.")
    DistributedDofHandler(Symbol[], Int[], Interpolation[], Ferrite.BCValues{Float64}[], Int[], Int[], Ferrite.ScalarWrapper(false), grid, Ferrite.ScalarWrapper(-1), Int[], Int32[])
end

function Base.show(io::IO, ::MIME"text/plain", dh::DistributedDofHandler)
    println(io, "DistributedDofHandler")
    println(io, "  Fields:")
    for i in 1:num_fields(dh)
        println(io, "    ", repr(dh.field_names[i]), ", interpolation: ", dh.field_interpolations[i],", dim: ", dh.field_dims[i])
    end
    if !Ferrite.isclosed(dh)
        print(io, "  Not closed!")
    else
        println(io, "  Dofs per cell: ", ndofs_per_cell(dh))
        print(io, "  Total local dofs: ", ndofs(dh))
    end
end

Ferrite.getdim(dh::DistributedDofHandler{dim}) where {dim} = dim 

Ferrite.getlocalgrid(dh::DistributedDofHandler) = Ferrite.getlocalgrid(dh.grid)
getglobalgrid(dh::DistributedDofHandler) = dh.grid

# Compat layer against serial code
Ferrite.getgrid(dh::DistributedDofHandler) = getlocalgrid(dh)

# TODO problem here is that the reorder has to be synchronized. We also cannot arbitrary reorder dofs, 
# because some distributed matrix data structures have strict requirements on the orderings.
Ferrite.renumber!(dh::DistributedDofHandler, perm::AbstractVector{<:Integer}) = error("Not implemented.")

"""
TODO fix for shells
"""
function compute_dof_ownership(dh::DistributedDofHandler)
    dgrid = getglobalgrid(dh)
    my_rank = global_rank(dgrid)

    dof_owner = Vector{Int}(undef,ndofs(dh))
    fill!(dof_owner, my_rank)

    for (lvi, sv) ∈ get_shared_vertices(dgrid)
        for field_idx in 1:num_fields(dh)
            if Ferrite.has_vertex_dofs(dh, field_idx, lvi)
                local_dofs = Ferrite.vertex_dofs(dh, field_idx, lvi)
                dof_owner[local_dofs] .= compute_owner(dgrid, sv)
            end
        end
    end

    for (lfi, sf) ∈ get_shared_faces(dgrid)
        for field_idx in 1:num_fields(dh)
            if Ferrite.has_face_dofs(dh, field_idx, lfi)
                local_dofs = Ferrite.face_dofs(dh, field_idx, lfi)
                dof_owner[local_dofs] .= compute_owner(dgrid, sf)
            end
        end
    end

    for (lei, se) ∈ get_shared_edges(dgrid)
        for field_idx in 1:num_fields(dh)
            if Ferrite.has_edge_dofs(dh, field_idx, lei)
                local_dofs = Ferrite.edge_dofs(dh, field_idx, lei)
                dof_owner[local_dofs] .= compute_owner(dgrid, se)
            end
        end
    end

    return dof_owner
end

"""
Compute the number of dofs owned by the current process.
"""
num_local_true_dofs(dh::DistributedDofHandler) = sum(dh.ldof_to_rank .== global_rank(getglobalgrid(dh)))

"""
Compute the number of dofs visible to the current process.
"""
num_local_dofs(dh::DistributedDofHandler) = length(dh.ldof_to_gdof)

"""
Compute the number of dofs in the global system.
"""
num_global_dofs(dh::DistributedDofHandler) = MPI.Allreduce(num_local_true_dofs(dh), MPI.SUM, global_comm(getglobalgrid(dh)))

"""
Renumber the dofs in local ordering to their corresponding global numbering.

TODO: Refactor for MixedDofHandler integration
"""
function local_to_global_numbering(dh::DistributedDofHandler)
    dgrid = getglobalgrid(dh)
    dim = getdim(dgrid)
    # MPI rank starting with 1 to match Julia's index convention
    my_rank = global_rank(dgrid)

    local_to_global = Vector{Int}(undef,ndofs(dh))
    fill!(local_to_global,0) # 0 is the invalid index!
    # Start by numbering local dofs only from 1:#local_dofs

    # Lookup for synchronization in the form (Remote Rank,Shared Entity)
    # @TODO replace dict with vector and tie to MPI neighborhood graph of the mesh
    vertices_send = Dict{Int,Vector{VertexIndex}}()
    n_vertices_recv = Dict{Int,Int}()
    faces_send = Dict{Int,Vector{FaceIndex}}()
    n_faces_recv = Dict{Int,Int}()
    edges_send = Dict{Int,Vector{EdgeIndex}}()
    edges_recv = Dict{Int,Vector{EdgeIndex}}()

    # We start by assigning a local dof to all owned entities.
    # An entity is owned if:
    # 1. *All* topological neighbors are on the local process
    # 2. If the rank of the local process it lower than the rank of *all* topological neighbors
    # A topological neighbor in this context is hereby defined per entity:
    # * vertex: All elements whose vertex is the vertex in question
    # * cell: Just the cell itself
    # * All other entities: All cells for which one of the corresponding entities interior intersects 
    #                       with the interior of the entity in question.
    next_local_idx = 1
    for (ci, cell) in enumerate(getcells(getgrid(dh)))
        Ferrite.@debug println("cell #$ci (R$my_rank)")
        for field_idx in 1:num_fields(dh)
            Ferrite.@debug println("  field: $(dh.field_names[field_idx]) (R$my_rank)")
            interpolation_info = Ferrite.InterpolationInfo(Ferrite.getfieldinterpolation(dh, field_idx))
            if interpolation_info.nvertexdofs > 0
                for (vi,vertex) in enumerate(Ferrite.vertices(cell))
                    Ferrite.@debug println("    vertex#$vertex (R$my_rank)")
                    lvi = VertexIndex(ci,vi)
                    # Dof is owned if it is local or if my rank is the smallest in the neighborhood
                    if !is_shared_vertex(dgrid, lvi) || (compute_owner(dgrid, get_shared_vertex(dgrid, lvi)) == my_rank)
                        # Update dof assignment
                        dof_local_indices = Ferrite.vertex_dofs(dh, field_idx, lvi)
                        if local_to_global[dof_local_indices[1]] == 0
                            for d in 1:getfielddim(dh, field_idx)
                                Ferrite.@debug println("      mapping vertex dof#$dof_local_indices[d] to $next_local_idx (R$my_rank)")
                                local_to_global[dof_local_indices[d]] = next_local_idx
                                next_local_idx += 1
                            end
                        else
                            for d in 1:getfielddim(dh, field_idx)
                                Ferrite.@debug println("      vertex dof#$(dof_local_indices[d]) already mapped to $(local_to_global[dof_local_indices[d]]) (R$my_rank)")
                            end
                        end
                    end

                    # Update shared vertex lookup table
                    if is_shared_vertex(dgrid, lvi)
                        master_rank = my_rank
                        remote_vertex_dict = remote_entities(get_shared_vertex(dgrid, lvi))
                        for master_rank_new ∈ keys(remote_vertex_dict)
                            master_rank = min(master_rank, master_rank_new)
                        end
                        for (remote_rank, svs) ∈ remote_vertex_dict
                            if master_rank == my_rank # I own the dof - we have to send information
                                if !haskey(vertices_send,remote_rank)
                                    vertices_send[remote_rank] = Vector{VertexIndex}()
                                end
                                Ferrite.@debug println("      prepare sending vertex #$(lvi) to $remote_rank (R$my_rank)")
                                for i ∈ svs
                                    push!(vertices_send[remote_rank],lvi)
                                end
                            elseif master_rank == remote_rank  # dof is owned by remote - we have to receive information
                                if !haskey(n_vertices_recv,remote_rank)
                                    n_vertices_recv[remote_rank] = length(svs)
                                else
                                    n_vertices_recv[remote_rank] += length(svs)
                                end
                                Ferrite.@debug println("      prepare receiving vertex #$(lvi) from $remote_rank (R$my_rank)")
                            end
                        end
                    end
                end
            end

            if dim > 2 # edges only in 3D
                if interpolation_info.nedgedofs > 0
                    for (ei,edge) in enumerate(Ferrite.edges(cell))
                        Ferrite.@debug println("    edge#$edge (R$my_rank)")
                        lei = EdgeIndex(ci,ei)
                        # Dof is owned if it is local or if my rank is the smallest in the neighborhood
                        if !is_shared_edge(dgrid, lei) || (compute_owner(dgrid, get_shared_edge(dgrid, lei)) == my_rank)
                            # Update dof assignment
                            dof_local_indices = Ferrite.edge_dofs(dh, field_idx, lei)
                            if local_to_global[dof_local_indices[1]] == 0
                                for d in 1:getfielddim(dh, field_idx)
                                    Ferrite.@debug println("      mapping edge dof#$(dof_local_indices[d]) to $next_local_idx (R$my_rank)")
                                    local_to_global[dof_local_indices[d]] = next_local_idx
                                    next_local_idx += 1
                                end
                            else
                                for d in 1:getfielddim(dh, field_idx)
                                    Ferrite.@debug println("      edge dof#$(dof_local_indices[d]) already mapped to $(local_to_global[dof_local_indices[d]]) (R$my_rank)")
                                end
                            end
                        end

                        # Update shared edge lookup table
                        if is_shared_edge(dgrid, lei)
                            master_rank = my_rank
                            remote_edge_dict = remote_entities(get_shared_edge(dgrid, lei))
                            for master_rank_new ∈ keys(remote_edge_dict)
                                master_rank = min(master_rank, master_rank_new)
                            end
                            for (remote_rank, svs) ∈ remote_edge_dict
                                if master_rank == my_rank # I own the dof - we have to send information
                                    if !haskey(edges_send,remote_rank)
                                        edges_send[remote_rank] = EdgeIndex[]
                                    end
                                    Ferrite.@debug println("      prepare sending edge #$(lei) to $remote_rank (R$my_rank)")
                                    for i ∈ svs
                                        push!(edges_send[remote_rank], lei)
                                    end
                                elseif master_rank == remote_rank  # dof is owned by remote - we have to receive information
                                    if !haskey(edges_recv,remote_rank)
                                        edges_recv[remote_rank] = EdgeIndex[]
                                    end
                                    push!(edges_recv[remote_rank], lei)
                                    Ferrite.@debug println("      prepare receiving edge #$(lei) from $remote_rank (R$my_rank)")
                                end
                            end
                        end
                    end
                end
            end

            if interpolation_info.nfacedofs > 0 && (interpolation_info.dim == dim)
                for (fi,face) in enumerate(Ferrite.faces(cell))
                    Ferrite.@debug println("    face#$face (R$my_rank)")
                    lfi = FaceIndex(ci,fi)
                    # Dof is owned if it is local or if my rank is the smallest in the neighborhood
                    if !is_shared_face(dgrid, lfi) || (compute_owner(dgrid, get_shared_face(dgrid, lfi)) == my_rank)
                        # Update dof assignment
                        dof_local_indices = Ferrite.face_dofs(dh, field_idx, lfi)
                        if local_to_global[dof_local_indices[1]] == 0
                            for d in 1:getfielddim(dh, field_idx)
                                Ferrite.@debug println("      mapping face dof#$(dof_local_indices[d]) to $next_local_idx (R$my_rank)")
                                local_to_global[dof_local_indices[d]] = next_local_idx
                                next_local_idx += 1
                            end
                        else
                            for d in 1:getfielddim(dh, field_idx)
                                Ferrite.@debug println("      face dof#$(dof_local_indices[d]) already mapped to $(local_to_global[dof_local_indices[d]]) (R$my_rank)")
                            end
                        end
                    end

                    # Update shared face lookup table
                    if is_shared_face(dgrid, lfi)
                        master_rank = my_rank
                        remote_face_dict = remote_entities(get_shared_face(dgrid, lfi))
                        for master_rank_new ∈ keys(remote_face_dict)
                            master_rank = min(master_rank, master_rank_new)
                        end
                        for (remote_rank, svs) ∈ remote_face_dict
                            if master_rank == my_rank # I own the dof - we have to send information
                                if !haskey(faces_send,remote_rank)
                                    faces_send[remote_rank] = FaceIndex[]
                                end
                                Ferrite.@debug println("      prepare sending face #$(lfi) to $remote_rank (R$my_rank)")
                                for i ∈ svs
                                    push!(faces_send[remote_rank],lfi)
                                end
                            elseif master_rank == remote_rank  # dof is owned by remote - we have to receive information
                                if !haskey(n_faces_recv,remote_rank)
                                    n_faces_recv[remote_rank] = length(svs)
                                else
                                    n_faces_recv[remote_rank] += length(svs)
                                end
                                Ferrite.@debug println("      prepare receiving face #$(lfi) from $remote_rank (R$my_rank)")
                            end
                        end
                    end
                end # face loop
            end

            if interpolation_info.ncelldofs > 0 # always distribute new dofs for cell
                Ferrite.@debug println("    cell#$ci")
                if interpolation_info.ncelldofs > 0
                    # Update dof assignment
                    dof_local_indices = Ferrite.cell_dofs(dh, field_idx, ci)
                    if local_to_global[dof_local_indices[1]] == 0
                        for d in 1:getfielddim(dh, field_idx)
                            Ferrite.@debug println("      mapping cell dof#$(dof_local_indices[d]) to $next_local_idx (R$my_rank)")
                            local_to_global[dof_local_indices[d]] = next_local_idx
                            next_local_idx += 1
                        end
                    else
                        for d in 1:getfielddim(dh, field_idx)
                            # Should never happen...
                            Ferrite.@debug println("      WARNING! cell dof#$(dof_local_indices[d]) already mapped to $(local_to_global[dof_local_indices[d]]) (R$my_rank)")
                        end
                    end
                end # cell loop
            end
        end # field loop
    end

    #
    num_true_local_dofs = next_local_idx-1
    Ferrite.@debug println("#true local dofs $num_true_local_dofs (R$my_rank)")

    # @TODO optimize the following synchronization with MPI line graph topology 
    # and allgather
    # Set true local indices
    local_offset = 0
    if my_rank > 1
        local_offset = MPI.Recv(Int, global_comm(dgrid); source=my_rank-1-1)
    end
    if my_rank < MPI.Comm_size(global_comm(dgrid))
        MPI.Send(local_offset+num_true_local_dofs, global_comm(dgrid); dest=my_rank+1-1)
    end
    Ferrite.@debug println("#shifted local dof range $(local_offset+1):$(local_offset+num_true_local_dofs) (R$my_rank)")

    # Shift assigned local dofs (dofs with value >0) into the global range
    # At this point in the algorithm the dofs with value 0 are the dofs owned of neighboring processes
    for i ∈ 1:length(local_to_global)
        if local_to_global[i] != 0
            local_to_global[i] += local_offset
        end
    end

    # Sync non-owned dofs with neighboring processes.
    # TODO: Use MPI graph primitives to simplify this code
    # TODO: Simplify with dimension-agnostic code...
    for sending_rank ∈ 1:MPI.Comm_size(global_comm(dgrid))
        if my_rank == sending_rank
            for remote_rank ∈ 1:MPI.Comm_size(global_comm(dgrid))
                if haskey(vertices_send, remote_rank)
                    n_vertices = length(vertices_send[remote_rank])
                    Ferrite.@debug println("Sending $n_vertices vertices to rank $remote_rank (R$my_rank)")
                    remote_cells = Array{Int64}(undef,n_vertices)
                    remote_cell_vis = Array{Int64}(undef,n_vertices)
                    next_buffer_idx = 1
                    for lvi ∈ vertices_send[remote_rank]
                        sv = dgrid.shared_vertices[lvi]
                        @assert haskey(sv.remote_vertices, remote_rank)
                        for (cvi, llvi) ∈ sv.remote_vertices[remote_rank][1:1] # Just don't ask :)
                            remote_cells[next_buffer_idx] = cvi
                            remote_cell_vis[next_buffer_idx] = llvi
                            next_buffer_idx += 1
                        end
                    end
                    MPI.Send(remote_cells, global_comm(dgrid); dest=remote_rank-1)
                    MPI.Send(remote_cell_vis, global_comm(dgrid); dest=remote_rank-1)
                    for field_idx ∈ 1:num_fields(dh)
                        next_buffer_idx = 1
                        ip = Ferrite.getfieldinterpolation(dh, field_idx)
                        if Ferrite.nvertexdofs(ip) == 0
                            Ferrite.@debug println("Skipping send vertex on field $(dh.field_names[field_idx]) (R$my_rank)")
                            continue
                        end
                        # fill correspondence array
                        corresponding_global_dofs = Array{Int64}(undef,n_vertices)
                        for vertex ∈ vertices_send[remote_rank]
                            if Ferrite.has_vertex_dofs(dh, field_idx, vertex)
                                # We just put the first dof into the array to reduce communication
                                vdofs = Ferrite.vertex_dofs(dh, field_idx, vertex)
                                corresponding_global_dofs[next_buffer_idx] = local_to_global[vdofs[1]]
                            end
                            next_buffer_idx += 1
                        end
                        MPI.Send(corresponding_global_dofs, global_comm(dgrid); dest=remote_rank-1)
                    end
                end

                if haskey(faces_send, remote_rank)
                    n_faces = length(faces_send[remote_rank])
                    Ferrite.@debug println("Sending $n_faces faces to rank $remote_rank (R$my_rank)")
                    remote_cells = Array{Int64}(undef,n_faces)
                    remote_cell_vis = Array{Int64}(undef,n_faces)
                    next_buffer_idx = 1
                    for lvi ∈ faces_send[remote_rank]
                        sv = dgrid.shared_faces[lvi]
                        @assert haskey(sv.remote_faces, remote_rank)
                        for (cvi, llvi) ∈ sv.remote_faces[remote_rank][1:1] # Just don't ask :)
                            remote_cells[next_buffer_idx] = cvi
                            remote_cell_vis[next_buffer_idx] = llvi 
                            next_buffer_idx += 1
                        end
                    end
                    MPI.Send(remote_cells, global_comm(dgrid); dest=remote_rank-1)
                    MPI.Send(remote_cell_vis, global_comm(dgrid); dest=remote_rank-1)
                    for field_idx ∈ 1:num_fields(dh)
                        next_buffer_idx = 1
                        ip = Ferrite.getfieldinterpolation(dh, field_idx)
                        if Ferrite.nfacedofs(ip) == 0
                            Ferrite.@debug println("Skipping send faces on field $(dh.field_names[field_idx]) (R$my_rank)")
                            continue
                        end
                        # fill correspondence array
                        corresponding_global_dofs = Array{Int64}(undef,n_faces)
                        for face ∈ faces_send[remote_rank]
                            if Ferrite.has_face_dofs(dh, field_idx, face)
                                # We just put the first dof into the array to reduce communication
                                fdofs = Ferrite.face_dofs(dh, field_idx, face)
                                corresponding_global_dofs[next_buffer_idx] = local_to_global[fdofs[1]]
                            end
                            next_buffer_idx += 1
                        end
                        MPI.Send(corresponding_global_dofs, global_comm(dgrid); dest=remote_rank-1)
                    end
                end

                if haskey(edges_send, remote_rank)
                    # Well .... that some hotfix straight outta hell.
                    edges_send_unique_set = Set{Tuple{Int,Int}}()
                    edges_send_unique = Set{EdgeIndex}()
                    for lei ∈ edges_send[remote_rank]
                        edge = Ferrite.toglobal(dgrid, lei)
                        if edge ∉ edges_send_unique_set
                            push!(edges_send_unique_set, edge)
                            push!(edges_send_unique, lei)
                        end
                    end
                    n_edges = length(edges_send_unique)
                    Ferrite.@debug println("Sending $n_edges edges to rank $remote_rank (R$my_rank)")
                    remote_cells = Array{Int64}(undef,n_edges)
                    remote_cell_vis = Array{Int64}(undef,n_edges)
                    next_buffer_idx = 1
                    for lvi ∈ edges_send_unique
                        sv = dgrid.shared_edges[lvi]
                        @assert haskey(sv.remote_edges, remote_rank)
                        for (cvi, llvi) ∈ sv.remote_edges[remote_rank][1:1] # Just don't ask :)
                            remote_cells[next_buffer_idx] = cvi
                            remote_cell_vis[next_buffer_idx] = llvi 
                            next_buffer_idx += 1
                        end
                    end
                    MPI.Send(remote_cells, global_comm(dgrid); dest=remote_rank-1)
                    MPI.Send(remote_cell_vis, global_comm(dgrid); dest=remote_rank-1)
                    for field_idx ∈ 1:num_fields(dh)
                        next_buffer_idx = 1
                        ip = Ferrite.getfieldinterpolation(dh, field_idx)
                        if Ferrite.nedgedofs(ip) == 0
                            Ferrite.@debug println("Skipping send edges on field $(dh.field_names[field_idx]) (R$my_rank)")
                            continue
                        end
                        # fill correspondence array
                        corresponding_global_dofs = Array{Int64}(undef,n_edges)
                        for edge ∈ edges_send_unique
                            if Ferrite.has_edge_dofs(dh, field_idx, edge)
                                # We just put the first dof into the array to reduce communication
                                edofs = Ferrite.edge_dofs(dh, field_idx, edge)
                                corresponding_global_dofs[next_buffer_idx] = local_to_global[edofs[1]]
                            end
                            next_buffer_idx += 1
                        end
                        MPI.Send(corresponding_global_dofs, global_comm(dgrid); dest=remote_rank-1)
                    end
                end
            end
        else
            if haskey(n_vertices_recv, sending_rank)
                n_vertices = n_vertices_recv[sending_rank]
                Ferrite.@debug println("Receiving $n_vertices vertices from rank $sending_rank (R$my_rank)")
                local_cells = Array{Int64}(undef,n_vertices)
                local_cell_vis = Array{Int64}(undef,n_vertices)
                MPI.Recv!(local_cells, global_comm(dgrid); source=sending_rank-1)
                MPI.Recv!(local_cell_vis, global_comm(dgrid); source=sending_rank-1)
                for field_idx in 1:num_fields(dh)
                    ip = Ferrite.getfieldinterpolation(dh, field_idx)
                    if Ferrite.nvertexdofs(ip) == 0
                        Ferrite.@debug println("  Skipping recv of vertices on field $(dh.field_names[field_idx]) (R$my_rank)")
                        continue
                    end
                    corresponding_global_dofs = Array{Int64}(undef,n_vertices)
                    MPI.Recv!(corresponding_global_dofs, global_comm(dgrid); source=sending_rank-1)
                    for (cdi,vertex) ∈ enumerate(VertexIndex.(zip(local_cells,local_cell_vis)))
                        if Ferrite.has_vertex_dofs(dh, field_idx, vertex)
                            vdofs = Ferrite.vertex_dofs(dh, field_idx, vertex)
                            for d in 1:getfielddim(dh, field_idx)
                                local_to_global[vdofs[d]] = corresponding_global_dofs[cdi]+d-1
                                Ferrite.@debug println("  Updating field $(dh.field_names[field_idx]) vertex $vertex to $(corresponding_global_dofs[cdi]+d-1) (R$my_rank)")
                            end
                        else
                            Ferrite.@debug println("  Skipping recv on field $(dh.field_names[field_idx]) vertex $vertex (R$my_rank)")
                        end
                    end
                end
            end

            if haskey(n_faces_recv, sending_rank)
                n_faces = n_faces_recv[sending_rank]
                Ferrite.@debug println("Receiving $n_faces faces from rank $sending_rank (R$my_rank)")
                local_cells = Array{Int64}(undef,n_faces)
                local_cell_vis = Array{Int64}(undef,n_faces)
                MPI.Recv!(local_cells, global_comm(dgrid); source=sending_rank-1)
                MPI.Recv!(local_cell_vis, global_comm(dgrid); source=sending_rank-1)
                for field_idx in 1:num_fields(dh)
                    ip = Ferrite.getfieldinterpolation(dh, field_idx)
                    if Ferrite.nfacedofs(ip) == 0
                        Ferrite.@debug println("  Skipping recv of faces on field $(dh.field_names[field_idx]) (R$my_rank)")
                        continue
                    end
                    corresponding_global_dofs = Array{Int64}(undef,n_faces)
                    MPI.Recv!(corresponding_global_dofs, global_comm(dgrid); source=sending_rank-1)
                    for (cdi,face) ∈ enumerate(FaceIndex.(zip(local_cells,local_cell_vis)))
                        if Ferrite.has_face_dofs(dh, field_idx, face)
                            fdofs = Ferrite.face_dofs(dh, field_idx, face)
                            for d in 1:getfielddim(dh, field_idx)
                                local_to_global[fdofs[d]] = corresponding_global_dofs[cdi]+d-1
                                Ferrite.@debug println("  Updating field $(dh.field_names[field_idx]) face $face to $(corresponding_global_dofs[cdi]) (R$my_rank)")
                            end
                        else
                            Ferrite.@debug println("  Skipping recv on field $(dh.field_names[field_idx]) face $face (R$my_rank)")
                        end
                    end
                end
            end

            if haskey(edges_recv, sending_rank)
                edges_recv_unique_set = Set{Tuple{Int,Int}}()
                for lei ∈ edges_recv[sending_rank]
                    edge = Ferrite.toglobal(dgrid, lei)
                    push!(edges_recv_unique_set, edge)
                end
                n_edges = length(edges_recv_unique_set)
                Ferrite.@debug println("Receiving $n_edges edges from rank $sending_rank (R$my_rank)")
                local_cells = Array{Int64}(undef,n_edges)
                local_cell_vis = Array{Int64}(undef,n_edges)
                MPI.Recv!(local_cells, global_comm(dgrid); source=sending_rank-1)
                MPI.Recv!(local_cell_vis, global_comm(dgrid); source=sending_rank-1)
                for field_idx in 1:num_fields(dh)
                    ip = Ferrite.getfieldinterpolation(dh, field_idx)
                    if Ferrite.nedgedofs(ip) == 0
                        Ferrite.@debug println("  Skipping recv on field $(dh.field_names[field_idx]) (R$my_rank)")
                        continue
                    end
                    corresponding_global_dofs = Array{Int64}(undef,n_edges)
                    MPI.Recv!(corresponding_global_dofs, global_comm(dgrid); source=sending_rank-1)
                    Ferrite.@debug println("   Received $corresponding_global_dofs edge dofs from $sending_rank (R$my_rank)")
                    for (cdi,edge) ∈ enumerate(EdgeIndex.(zip(local_cells,local_cell_vis)))
                        if Ferrite.has_edge_dofs(dh, field_idx, edge)
                            edofs = Ferrite.edge_dofs(dh, field_idx, edge)
                            for d in 1:getfielddim(dh, field_idx)
                                local_to_global[edofs[d]] = corresponding_global_dofs[cdi]+d-1
                                Ferrite.@debug println("  Updating field $(dh.field_names[field_idx]) edge $edge to $(corresponding_global_dofs[cdi]) (R$my_rank)")
                            end
                        else
                            Ferrite.@debug println("  Skipping recv on field $(dh.field_names[field_idx]) edge $edge (R$my_rank)")
                        end
                    end
                end
            end
        end
    end

    # Postcondition: All local dofs need a corresponding global dof!
    Ferrite.@debug println("Local to global mapping: $local_to_global (R$my_rank)")
    @assert findfirst(local_to_global .== 0) === nothing

    return local_to_global
end

function Ferrite.close!(dh::DistributedDofHandler)
    # We could merge these functions into an optimized one if we want.
    Ferrite.__close!(dh)
    append!(dh.ldof_to_rank, compute_dof_ownership(dh))
    append!(dh.ldof_to_gdof, local_to_global_numbering(dh))
    return dh
end


# Hypre to Ferrite vector
function hypre_to_ferrite!(u, x, dh)
    my_rank = global_rank(getglobalgrid(dh))

    # Helper to gather which global dof and values have to be send to which process
    gdof_value_send = [Dict{Int,Float64}() for i ∈ 1:MPI.Comm_size(MPI.COMM_WORLD)]
    # Helper to get the global dof to local dof mapping
    rank_recv_count = [0 for i∈1:MPI.Comm_size(MPI.COMM_WORLD)]
    gdof_to_ldof = Dict{Int,Int}()

    next_dof = 1
    for (ldof,rank) ∈ enumerate(dh.ldof_to_rank)
        if rank == my_rank
            u[ldof] = x[next_dof]
            next_dof += 1
        else 
            # We have to sync these later.
            gdof_to_ldof[dh.ldof_to_gdof[ldof]] = ldof
            rank_recv_count[rank] += 1
        end
    end

    # TODO speed this up and better API
    dgrid = FerritePartitionedArrays.getglobalgrid(dh)
    for (lvi, sv) ∈ get_shared_vertices(dgrid)
        my_rank != FerritePartitionedArrays.compute_owner(dgrid, sv) && continue
        for field_idx in 1:num_fields(dh)
            if Ferrite.has_vertex_dofs(dh, field_idx, lvi)
                local_dofs = Ferrite.vertex_dofs(dh, field_idx, lvi)
                global_dofs = dh.ldof_to_gdof[local_dofs]
                for receiver_rank ∈ keys(FerritePartitionedArrays.remote_entities(sv))
                    for i ∈ 1:length(global_dofs)
                        # Note that u already has the correct values for all locally owned dofs due to the loop above!
                        gdof_value_send[receiver_rank][global_dofs[i]] = u[local_dofs[i]]
                    end
                end
            end
        end
    end

    for (lvi, se) ∈ get_shared_edges(dgrid)
        my_rank != FerritePartitionedArrays.compute_owner(dgrid, se) && continue
        for field_idx in 1:num_fields(dh)
            if Ferrite.has_edge_dofs(dh, field_idx, lvi)
                local_dofs = Ferrite.edge_dofs(dh, field_idx, lvi)
                global_dofs = dh.ldof_to_gdof[local_dofs]
                for receiver_rank ∈ keys(FerritePartitionedArrays.remote_entities(se))
                    for i ∈ 1:length(global_dofs)
                        # Note that u already has the correct values for all locally owned dofs due to the loop above!
                        gdof_value_send[receiver_rank][global_dofs[i]] = u[local_dofs[i]]
                    end
                end
            end
        end
    end
    
    for (lvi, sf) ∈ get_shared_faces(dgrid)
        my_rank != FerritePartitionedArrays.compute_owner(dgrid, sf) && continue
        for field_idx in 1:num_fields(dh)
            if Ferrite.has_face_dofs(dh, field_idx, lvi)
                local_dofs = Ferrite.face_dofs(dh, field_idx, lvi)
                global_dofs = dh.ldof_to_gdof[local_dofs]
                for receiver_rank ∈ keys(FerritePartitionedArrays.remote_entities(sf))
                    for i ∈ 1:length(global_dofs)
                        # Note that u already has the correct values for all locally owned dofs due to the loop above!
                        gdof_value_send[receiver_rank][global_dofs[i]] = u[local_dofs[i]]
                    end
                end
            end
        end
    end

    Ferrite.@debug println("preparing to distribute $gdof_value_send (R$my_rank)")

    # TODO precompute graph at it is static
    graph_source   = Cint[my_rank-1]
    graph_dest   = Cint[]
    for r ∈ 1:MPI.Comm_size(MPI.COMM_WORLD)
        !isempty(gdof_value_send[r]) && push!(graph_dest, r-1)
    end

    graph_degree = Cint[length(graph_dest)]
    graph_comm = MPI.Dist_graph_create(MPI.COMM_WORLD, graph_source, graph_degree, graph_dest)
    indegree, outdegree, _ = MPI.Dist_graph_neighbors_count(graph_comm)

    inranks = Vector{Cint}(undef, indegree)
    outranks = Vector{Cint}(undef, outdegree)
    MPI.Dist_graph_neighbors!(graph_comm, inranks, outranks)

    send_count = [length(gdof_value_send[outrank+1]) for outrank ∈ outranks]
    recv_count = [rank_recv_count[inrank+1] for inrank ∈ inranks]

    send_gdof = Cint[]
    for outrank ∈ outranks
        append!(send_gdof, Cint.(keys(gdof_value_send[outrank+1])))
    end
    recv_gdof = Vector{Cint}(undef, sum(recv_count))
    MPI.Neighbor_alltoallv!(VBuffer(send_gdof,send_count), VBuffer(recv_gdof,recv_count), graph_comm)

    send_val = Cdouble[]
    for outrank ∈ outranks
        append!(send_val, Cdouble.(values(gdof_value_send[outrank+1])))
    end
    recv_val = Vector{Cdouble}(undef, sum(recv_count))
    MPI.Neighbor_alltoallv!(VBuffer(send_val,send_count), VBuffer(recv_val,recv_count), graph_comm)

    for (gdof, val) ∈ zip(recv_gdof, recv_val)
        u[gdof_to_ldof[gdof]] = val
    end

    return u
end



function WriteVTK.vtk_grid(filename::AbstractString, dh::DistributedDofHandler; compress::Bool=true)
    vtk_grid(filename, getglobalgrid(dh); compress=compress)
end
