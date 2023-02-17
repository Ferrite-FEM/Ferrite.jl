
"""
Simplest partitioned assembler in COO format to obtain a PSparseMatrix and a PVector.
"""
struct COOAssembler{T}
    I::Vector{Int}
    J::Vector{Int}
    V::Vector{T}

    cols
    rows
    f::PVector

    👻remotes
    dh

    # TODO PartitionedArrays backend as additional input arg
    COOAssembler(dh::DistributedDofHandler) = COOAssembler{Float64}(dh)

    # TODO PartitionedArrays backend as additional input arg
    function COOAssembler{T}(dh::DistributedDofHandler) where {T}
        ldof_to_gdof = dh.ldof_to_gdof
        ldof_to_rank = dh.ldof_to_rank
        nldofs = num_local_dofs(dh)
        ngdofs = num_global_dofs(dh)
        dgrid = getglobalgrid(dh)
        dim = getdim(dgrid)

        I = Int[]
        J = Int[]
        V = T[]
        sizehint!(I, nldofs)
        sizehint!(J, nldofs)
        sizehint!(V, nldofs)

        # @TODO the code below can be massively simplified by introducing a ghost layer to the
        #       distributed grid, which can efficiently precompute some of the values below.
        comm = global_comm(dgrid)
        np = MPI.Comm_size(comm)
        my_rank = global_rank(dgrid)

        Ferrite.@debug println("starting assembly... (R$my_rank)")

        # Neighborhood graph
        # @TODO cleanup old code below and use graph primitives instead.
        (source_len, destination_len, _) = MPI.Dist_graph_neighbors_count(vertex_comm(dgrid))
        sources = Vector{Cint}(undef, source_len)
        destinations = Vector{Cint}(undef, destination_len)
        MPI.Dist_graph_neighbors!(vertex_comm(dgrid), sources, destinations)

        # Adjust to Julia index convention
        sources .+= 1
        destinations .+= 1

        Ferrite.@debug println("Neighborhood | $sources | $destinations (R$my_rank)")

        # Invert the relations to clarify the code
        source_index = Dict{Cint, Int}()
        for (i,remote_rank) ∈ enumerate(sources)
            source_index[remote_rank] = i
        end
        destination_index = Dict{Int, Cint}()
        for (i,remote_rank) ∈ enumerate(destinations)
            destination_index[remote_rank] = i
        end

        # Note: We assume a symmetric neighborhood for now... this may not be true in general.
        neighbors = MPIData(Int32.(sources), comm, (np,))

        # Extract locally owned dofs
        ltdof_indices = ldof_to_rank.==my_rank
        ltdof_to_gdof = ldof_to_gdof[ltdof_indices]

        Ferrite.@debug println("ltdof_to_gdof $ltdof_to_gdof (R$my_rank)")
        Ferrite.@debug println("ldof_to_gdof $ldof_to_gdof (R$my_rank)")
        Ferrite.@debug println("ldof_to_rank $ldof_to_rank (R$my_rank)")

        # Process owns rows of owned dofs. The process also may write to some remote dofs,
        # which correspond to non-owned share entities. Here we construct the rows for the
        # distributed matrix.
        # We decide for row (i.e. test function) ownership, because it the image of
        # SpMV is process local.
        row_indices = PartitionedArrays.IndexSet(my_rank, ldof_to_gdof, Int32.(ldof_to_rank))
        #FIXME: This below must be fixed before we can assemble to HYPRE IJ. Problem seems to be that rows and cols must be continuously assigned.
        #row_indices = PartitionedArrays.IndexRange(my_rank, length(ltdof_indices), ltdof_to_gdof[1], ldof_to_gdof[.!ltdof_indices], Int32.(ldof_to_rank[.!ltdof_indices]))
        row_data = MPIData(row_indices, comm, (np,))
        row_exchanger = Exchanger(row_data)
        rows = PRange(ngdofs,row_data,row_exchanger)

        Ferrite.@debug println("rows done (R$my_rank)")

        # For the locally visible columns we also have to take into account that remote
        # processes will write their data in some of these, because their remotely
        # owned trial functions overlap with the locally owned test functions.
        ghost_dof_to_global = Int[]
        ghost_dof_rank = Int32[]

        # ------------ Ghost dof synchronization ----------
        # Prepare sending ghost dofs to neighbors 👻
        #@TODO move relevant parts into dof handler
        #@TODO communication can be optimized by deduplicating entries in, and compressing the following arrays
        #@TODO reorder communication by field to eliminate need for `ghost_dof_field_index_to_send`
        ghost_dof_to_send = [Int[] for i ∈ 1:destination_len] # global dof id
        ghost_rank_to_send = [Int[] for i ∈ 1:destination_len] # rank of dof
        ghost_dof_field_index_to_send = [Int[] for i ∈ 1:destination_len]
        ghost_dof_owner = [Int[] for i ∈ 1:destination_len] # corresponding owner
        ghost_dof_pivot_to_send = [Int[] for i ∈ 1:destination_len] # corresponding dof to interact with
        for (pivot_vertex, pivot_shared_vertex) ∈ dgrid.shared_vertices
            # Start by searching shared entities which are not owned
            pivot_vertex_owner_rank = compute_owner(dgrid, pivot_shared_vertex)
            pivot_cell_idx = pivot_vertex[1]

            if my_rank != pivot_vertex_owner_rank
                sender_slot = destination_index[pivot_vertex_owner_rank]

                Ferrite.@debug println("$pivot_vertex may require synchronization (R$my_rank)")
                # Note: We have to send ALL dofs on the element to the remote.
                cell_dofs_upper_bound = (pivot_cell_idx == getncells(dh.grid)) ? length(dh.cell_dofs) : dh.cell_dofs_offset[pivot_cell_idx+1]
                cell_dofs = dh.cell_dofs[dh.cell_dofs_offset[pivot_cell_idx]:cell_dofs_upper_bound]

                for (field_idx, field_name) in zip(1:num_fields(dh), getfieldnames(dh))
                    !Ferrite.has_vertex_dofs(dh, field_idx, pivot_vertex) && continue
                    pivot_vertex_dofs = Ferrite.vertex_dofs(dh, field_idx, pivot_vertex)
                    
                    for d ∈ 1:dh.field_dims[field_idx]
                        Ferrite.@debug println("  adding dof $(pivot_vertex_dofs[d]) to ghost sync synchronization on slot $sender_slot (R$my_rank)")

                        # Extract dofs belonging to the current field
                        #cell_field_dofs = cell_dofs[dof_range(dh, field_name)]
                        #for cell_field_dof ∈ cell_field_dofs
                        for cell_dof ∈ cell_dofs
                            append!(ghost_dof_pivot_to_send[sender_slot], ldof_to_gdof[pivot_vertex_dofs[d]])
                            append!(ghost_dof_to_send[sender_slot], ldof_to_gdof[cell_dof])
                            append!(ghost_rank_to_send[sender_slot], ldof_to_rank[cell_dof])
                            append!(ghost_dof_field_index_to_send[sender_slot], field_idx)
                        end
                    end
                end
            end
        end

        if dim > 1
            for (pivot_face, pivot_shared_face) ∈ dgrid.shared_faces
                # Start by searching shared entities which are not owned
                pivot_face_owner_rank = compute_owner(dgrid, pivot_shared_face)
                pivot_cell_idx = pivot_face[1]

                if my_rank != pivot_face_owner_rank
                    sender_slot = destination_index[pivot_face_owner_rank]

                    Ferrite.@debug println("$pivot_face may require synchronization (R$my_rank)")
                    # Note: We have to send ALL dofs on the element to the remote.
                    cell_dofs_upper_bound = (pivot_cell_idx == getncells(dh.grid)) ? length(dh.cell_dofs) : dh.cell_dofs_offset[pivot_cell_idx+1]
                    cell_dofs = dh.cell_dofs[dh.cell_dofs_offset[pivot_cell_idx]:cell_dofs_upper_bound]

                    for (field_idx, field_name) in zip(1:num_fields(dh), getfieldnames(dh))
                        !Ferrite.has_face_dofs(dh, field_idx, pivot_face) && continue
                        pivot_face_dofs = Ferrite.face_dofs(dh, field_idx, pivot_face)
                        
                        for d ∈ 1:dh.field_dims[field_idx]
                            Ferrite.@debug println("  adding dof $(pivot_face_dofs[d]) to ghost sync synchronization on slot $sender_slot (R$my_rank)")
                            
                            # Extract dofs belonging to the current field
                            #cell_field_dofs = cell_dofs[dof_range(dh, field_name)]
                            #for cell_field_dof ∈ cell_field_dofs
                            for cell_dof ∈ cell_dofs
                                append!(ghost_dof_pivot_to_send[sender_slot], ldof_to_gdof[pivot_face_dofs[d]])
                                append!(ghost_dof_to_send[sender_slot], ldof_to_gdof[cell_dof])
                                append!(ghost_rank_to_send[sender_slot], ldof_to_rank[cell_dof])
                                append!(ghost_dof_field_index_to_send[sender_slot], field_idx)
                            end
                        end
                    end
                end
            end
        end

        if dim > 2
            for (pivot_edge, pivot_shared_edge) ∈ dgrid.shared_edges
                # Start by searching shared entities which are not owned
                pivot_edge_owner_rank = compute_owner(dgrid, pivot_shared_edge)
                pivot_cell_idx = pivot_edge[1]

                if my_rank != pivot_edge_owner_rank
                    sender_slot = destination_index[pivot_edge_owner_rank]

                    Ferrite.@debug println("$pivot_edge may require synchronization (R$my_rank)")
                    # Note: We have to send ALL dofs on the element to the remote.
                    cell_dofs_upper_bound = (pivot_cell_idx == getncells(dh.grid)) ? length(dh.cell_dofs) : dh.cell_dofs_offset[pivot_cell_idx+1]
                    cell_dofs = dh.cell_dofs[dh.cell_dofs_offset[pivot_cell_idx]:cell_dofs_upper_bound]

                    for (field_idx, field_name) in zip(1:num_fields(dh), getfieldnames(dh))
                        !Ferrite.has_edge_dofs(dh, field_idx, pivot_edge) && continue
                        pivot_edge_dofs = Ferrite.edge_dofs(dh, field_idx, pivot_edge)

                        for d ∈ 1:dh.field_dims[field_idx]
                            Ferrite.@debug println("  adding dof $(pivot_edge_dofs[d]) to ghost sync synchronization on slot $sender_slot (R$my_rank)")
                            # Extract dofs belonging to the current field
                            #cell_field_dofs = cell_dofs[dof_range(dh, field_name)]
                            #for cell_field_dof ∈ cell_field_dofs
                            for cell_dof ∈ cell_dofs
                                append!(ghost_dof_pivot_to_send[sender_slot], ldof_to_gdof[pivot_edge_dofs[d]])
                                append!(ghost_dof_to_send[sender_slot], ldof_to_gdof[cell_dof])
                                append!(ghost_rank_to_send[sender_slot], ldof_to_rank[cell_dof])
                                append!(ghost_dof_field_index_to_send[sender_slot], field_idx)
                            end
                        end
                    end
                end
            end
        end

        ghost_send_buffer_lengths = Int[length(i) for i ∈ ghost_dof_to_send]
        ghost_recv_buffer_lengths = zeros(Int, destination_len)
        MPI.Neighbor_alltoall!(UBuffer(ghost_send_buffer_lengths,1), UBuffer(ghost_recv_buffer_lengths,1), vertex_comm(dgrid));
        Ferrite.@debug for (i,ghost_recv_buffer_length) ∈ enumerate(ghost_recv_buffer_lengths)
            println("receiving $ghost_recv_buffer_length ghosts from $(sources[i])  (R$my_rank)")
        end

        # Communicate ghost information 👻
        # @TODO coalesce communication
        ghost_send_buffer_dofs = vcat(ghost_dof_to_send...)
        ghost_recv_buffer_dofs = zeros(Int, sum(ghost_recv_buffer_lengths))
        MPI.Neighbor_alltoallv!(VBuffer(ghost_send_buffer_dofs,ghost_send_buffer_lengths), VBuffer(ghost_recv_buffer_dofs,ghost_recv_buffer_lengths), vertex_comm(dgrid))

        ghost_send_buffer_fields = vcat(ghost_dof_field_index_to_send...)
        ghost_recv_buffer_fields = zeros(Int, sum(ghost_recv_buffer_lengths))
        MPI.Neighbor_alltoallv!(VBuffer(ghost_send_buffer_fields,ghost_send_buffer_lengths), VBuffer(ghost_recv_buffer_fields,ghost_recv_buffer_lengths), vertex_comm(dgrid))

        ghost_send_buffer_ranks = vcat(ghost_rank_to_send...)
        ghost_recv_buffer_ranks = zeros(Int, sum(ghost_recv_buffer_lengths))
        MPI.Neighbor_alltoallv!(VBuffer(ghost_send_buffer_ranks,ghost_send_buffer_lengths), VBuffer(ghost_recv_buffer_ranks,ghost_recv_buffer_lengths), vertex_comm(dgrid))

        ghost_send_buffer_dofs_piv = vcat(ghost_dof_pivot_to_send...)
        ghost_recv_buffer_dofs_piv = zeros(Int, sum(ghost_recv_buffer_lengths))
        MPI.Neighbor_alltoallv!(VBuffer(ghost_send_buffer_dofs_piv,ghost_send_buffer_lengths), VBuffer(ghost_recv_buffer_dofs_piv,ghost_recv_buffer_lengths), vertex_comm(dgrid))

        # Reconstruct source ranks
        ghost_recv_buffer_source_ranks = Int[]
        for (source_idx, recv_len) ∈ enumerate(ghost_recv_buffer_lengths)
            append!(ghost_recv_buffer_source_ranks, ones(recv_len)*sources[source_idx])
        end

        Ferrite.@debug println("received $ghost_recv_buffer_dofs with owners $ghost_recv_buffer_ranks (R$my_rank)")

        unique_ghosts_dr = sort(unique(first,zip(ghost_recv_buffer_dofs,ghost_recv_buffer_ranks)))
        # unzip manually and make sure we do not add duplicate entries to our columns
        for (dof,rank) ∈ unique_ghosts_dr
            if rank != my_rank && dof ∉ ldof_to_gdof
                push!(ghost_dof_to_global, dof)
                push!(ghost_dof_rank, rank)
            end
        end

        # ------------- Construct rows and cols of distributed matrix --------
        all_local_cols = Int[ldof_to_gdof; ghost_dof_to_global]
        all_local_col_ranks = Int32[ldof_to_rank; ghost_dof_rank]
        Ferrite.@debug println("all_local_cols $all_local_cols (R$my_rank)")
        Ferrite.@debug println("all_local_col_ranks $all_local_col_ranks (R$my_rank)")

        col_indices = PartitionedArrays.IndexSet(my_rank, all_local_cols, all_local_col_ranks)
        #FIXME: This below must be fixed before we can assemble to HYPRE IJ. Problem seems to be that rows and cols must be continuously assigned.
        #col_indices = PartitionedArrays.IndexRange(my_rank, length(ltdof_indices), ltdof_to_gdof[1], all_local_cols[all_local_col_ranks .!= my_rank], Int32.(all_local_col_ranks[all_local_col_ranks .!= my_rank]))
        col_data = MPIData(col_indices, comm, (np,))
        col_exchanger = Exchanger(col_data)
        cols = PRange(ngdofs,col_data,col_exchanger)

        Ferrite.@debug println("cols and rows constructed (R$my_rank)")
        f = PartitionedArrays.PVector(0.0,rows)
        Ferrite.@debug println("f constructed (R$my_rank)")

        👻remotes = zip(ghost_recv_buffer_dofs_piv, ghost_recv_buffer_dofs, ghost_recv_buffer_ranks,ghost_recv_buffer_fields)
        Ferrite.@debug println("👻remotes $👻remotes (R$my_rank)")

        return new(I, J, V, cols, rows, f, 👻remotes, dh)
    end
end

@propagate_inbounds function Ferrite.assemble!(a::COOAssembler{T}, edof::AbstractVector{Int}, Ke::AbstractMatrix{T}) where {T}
    n_dofs = length(edof)
    append!(a.V, Ke)
    @inbounds for j in 1:n_dofs
        append!(a.I, edof)
        for i in 1:n_dofs
            push!(a.J, edof[j])
        end
    end
end

@propagate_inbounds function Ferrite.assemble!(a::COOAssembler{T}, dofs::AbstractVector{Int}, fe::AbstractVector{T}, Ke::AbstractMatrix{T}) where {T}
    Ferrite.assemble!(a, dofs, Ke)
    map_parts(local_view(a.f, a.f.rows)) do f_local
        Ferrite.assemble!(f_local, dofs, fe)
    end
end

function Ferrite.end_assemble(assembler::COOAssembler{T}) where {T}
    dgrid = getglobalgrid(assembler.dh)
    comm = global_comm(dgrid)
    np = MPI.Comm_size(comm)
    my_rank = global_rank(dgrid)

    # --------------------- Add ghost entries in IJ 👻 --------------------
    I = map(i->assembler.dh.ldof_to_gdof[i], assembler.I)
    J = map(j->assembler.dh.ldof_to_gdof[j], assembler.J)
    V = map(v->v, assembler.V)

    # Fix ghost layer 👻! Note that the locations for remote processes to write their
    # data into are missing up to this point.
    # TODO here still the interaction between fields is missing...
    for (i, (pivot_dof, global_ghost_dof, ghost_owner_rank, ghost_field_idx)) ∈ enumerate(assembler.👻remotes)
        for dᵢ ∈ 1:1#assembler.dh.field_dims[ghost_field_idx]
            for dⱼ ∈ 1:1#assembler.dh.field_dims[ghost_field_idx]
                push!(I, pivot_dof+dᵢ-1)
                push!(J, global_ghost_dof+dⱼ-1)
                push!(V, 0.0)
            end
        end
    end

    Ferrite.@debug println("I=$(I) (R$my_rank)")
    Ferrite.@debug println("J=$(J) (R$my_rank)")
    K = PartitionedArrays.PSparseMatrix(
        MPIData(I, comm, (np,)),
        MPIData(J, comm, (np,)),
        MPIData(V, comm, (np,)),
        assembler.rows, assembler.cols, ids=:global
    )

    PartitionedArrays.assemble!(K)
    PartitionedArrays.assemble!(assembler.f)

    return K, assembler.f
end
