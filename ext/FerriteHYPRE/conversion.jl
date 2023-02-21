# Hypre to Ferrite vector
function hypre_to_ferrite!(u::Vector{T}, uh::HYPREVector, dh::Ferrite.AbstractDofHandler) where {T}
    # Copy solution from HYPRE to Julia
    uj = Vector{Float64}(undef, num_local_true_dofs(dh))
    copy!(uj, uh)

    my_rank = global_rank(getglobalgrid(dh))

    # Helper to gather which global dof and values have to be send to which process
    gdof_value_send = [Dict{Int,Float64}() for i ∈ 1:MPI.Comm_size(MPI.COMM_WORLD)]
    # Helper to get the global dof to local dof mapping
    rank_recv_count = [0 for i∈1:MPI.Comm_size(MPI.COMM_WORLD)]
    gdof_to_ldof = Dict{Int,Int}()

    next_dof = 1
    for (ldof,rank) ∈ enumerate(dh.ldof_to_rank)
        if rank == my_rank
            u[ldof] = uj[next_dof]
            next_dof += 1
        else 
            # We have to sync these later.
            gdof_to_ldof[dh.ldof_to_gdof[ldof]] = ldof
            rank_recv_count[rank] += 1
        end
    end

    # TODO speed this up and better API
    dgrid = getglobalgrid(dh)
    for sv ∈ get_shared_vertices(dgrid)
        lvi = sv.local_idx
        my_rank != compute_owner(dgrid, sv) && continue
        for field_idx in 1:num_fields(dh)
            if Ferrite.has_vertex_dofs(dh, field_idx, lvi)
                local_dofs = Ferrite.vertex_dofs(dh, field_idx, lvi)
                global_dofs = dh.ldof_to_gdof[local_dofs]
                for receiver_rank ∈ keys(remote_entities(sv))
                    for i ∈ 1:length(global_dofs)
                        # Note that u already has the correct values for all locally owned dofs due to the loop above!
                        gdof_value_send[receiver_rank][global_dofs[i]] = u[local_dofs[i]]
                    end
                end
            end
        end
    end

    for se ∈ get_shared_edges(dgrid)
        lei = se.local_idx
        my_rank != compute_owner(dgrid, se) && continue
        for field_idx in 1:num_fields(dh)
            if Ferrite.has_edge_dofs(dh, field_idx, lei)
                local_dofs = Ferrite.edge_dofs(dh, field_idx, lei)
                global_dofs = dh.ldof_to_gdof[local_dofs]
                for receiver_rank ∈ keys(remote_entities(se))
                    for i ∈ 1:length(global_dofs)
                        # Note that u already has the correct values for all locally owned dofs due to the loop above!
                        gdof_value_send[receiver_rank][global_dofs[i]] = u[local_dofs[i]]
                    end
                end
            end
        end
    end
    
    for sf ∈ get_shared_faces(dgrid)
        lfi = sf.local_idx
        my_rank != compute_owner(dgrid, sf) && continue
        for field_idx in 1:num_fields(dh)
            if Ferrite.has_face_dofs(dh, field_idx, lfi)
                local_dofs = Ferrite.face_dofs(dh, field_idx, lfi)
                global_dofs = dh.ldof_to_gdof[local_dofs]
                for receiver_rank ∈ keys(remote_entities(sf))
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
