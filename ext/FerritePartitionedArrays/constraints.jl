function meandiag(K::PartitionedArrays.PSparseMatrix)
    # Get local portion of z
    z_pa = map_parts(local_view(K, K.rows, K.cols)) do K_local
        z = zero(eltype(K_local))
        for i in 1:size(K_local, 1)
            z += abs(K_local[i, i])
        end
        return z;
    end
    # z = get_part(z_pa, MPI.Comm_rank(z_pa.comm)+1) # Crashes :)
    return MPI.Allreduce(z_pa.part, MPI.SUM, z_pa.comm) / size(K, 1)
end

"""
Poor man's Dirichlet BC application for PartitionedArrays. :)

    TODO integrate with constraints.
"""
function apply_zero!(K::PartitionedArrays.PSparseMatrix, f::PartitionedArrays.PVector, ch::ConstraintHandler)
    map_parts(local_view(f, f.rows), f.rows.partition) do f_local, partition
        f_local[ch.prescribed_dofs] .= 0.0
    end

    map_parts(local_view(K, K.rows, K.cols), local_view(f, f.rows)) do K_local, f_local
        for cdof in ch.prescribed_dofs
            K_local[cdof, :] .= 0.0
            K_local[:, cdof] .= 0.0
            K_local[cdof, cdof] = 1.0
        end
    end
end

"""
Poor man's Dirichlet BC application for PartitionedArrays. :)

    TODO integrate with constraints.
    TODO optimize.
"""
function apply!(K::PartitionedArrays.PSparseMatrix, f::PartitionedArrays.PVector, ch::ConstraintHandler)
    # Start by substracting the inhomogeneous solution from the right hand side
    u_constrained = PartitionedArrays.PVector(0.0, K.cols)
    map_parts(local_view(u_constrained, u_constrained.rows)) do u_local
        u_local[ch.prescribed_dofs] .= ch.inhomogeneities
    end
    f .-= K*u_constrained

    m = meandiag(K)

    # Then fix the 
    map_parts(local_view(f, f.rows), f.rows.partition) do f_local, partition
        # Note: RHS only non-zero for owned RHS entries
        f_local[ch.prescribed_dofs] .= ch.inhomogeneities .* map(p -> p == partition.part, partition.lid_to_part[ch.prescribed_dofs]) * m
    end

    # Zero out locally visible rows and columns
    map_parts(local_view(K, K.rows, K.cols)) do K_local
        for cdof ∈ ch.prescribed_dofs
            K_local[cdof, :] .= 0.0
            K_local[:, cdof] .= 0.0
            K_local[cdof, cdof] = m
        end
    end

    # Zero out columns associated to the ghost dofs constrained on a remote process
    # TODO optimize. If we assume that the sparsity pattern is symmetric, then we can constrain
    #      via the column information of the matrix.

    # Step 1: Send out all local ghosts to all other processes...
    remote_ghost_gdofs, remote_ghost_parts = map_parts(K.cols.partition) do partition
        remote_ghost_ldofs = partition.hid_to_lid
        remote_ghost_parts = partition.lid_to_part[remote_ghost_ldofs]
        remote_ghost_gdofs = partition.lid_to_gid[remote_ghost_ldofs]
        return (remote_ghost_gdofs, remote_ghost_parts)
    end

    comm = remote_ghost_parts.comm
    my_rank = MPI.Comm_rank(comm)+1
    buffer_sizes_send = zeros(Cint, MPI.Comm_size(comm))
    buffer_sizes_recv = Vector{Cint}(undef, MPI.Comm_size(comm))
    for part ∈ remote_ghost_parts.part
        buffer_sizes_send[part] += 1
    end
    MPI.Alltoall!(UBuffer(buffer_sizes_send, 1), UBuffer(buffer_sizes_recv, 1), comm)
    @debug println("Got $buffer_sizes_recv (R$my_rank)")

    remote_ghosts_recv = Vector{Int}(undef, sum(buffer_sizes_recv))
    MPI.Alltoallv!(VBuffer(remote_ghost_gdofs.part, buffer_sizes_send), VBuffer(remote_ghosts_recv, buffer_sizes_recv), comm)
    @debug println("Got $remote_ghosts_recv (R$my_rank)")

    # Step 2: Union with all locally constrained dofs
    @debug println("$my_rank : Step 2....")
    remote_ghosts_constrained_send = copy(remote_ghosts_recv)
    for (i, remote_ghost_dof) ∈ enumerate(remote_ghosts_recv)
        remote_ghosts_constrained_send[i] = remote_ghost_dof ∈ K.cols.partition.part.lid_to_gid[ch.prescribed_dofs]
    end

    # Step 3: Send trash back
    @debug println("$my_rank : Step 3....")
    remote_ghosts_constrained_recv = Vector{Int}(undef, sum(buffer_sizes_send))
    MPI.Alltoallv!(VBuffer(remote_ghosts_constrained_send, buffer_sizes_recv), VBuffer(remote_ghosts_constrained_recv, buffer_sizes_send), comm)

    @debug println("$my_rank : remote constraints on $(remote_ghost_gdofs.part[remote_ghosts_constrained_recv .== 1])")

    # Step 4: Constrain remaining columns
    map_parts(local_view(K, K.rows, K.cols), K.cols.partition) do K_local, partition
        for cdof ∈ partition.hid_to_lid[remote_ghosts_constrained_recv .== 1]
            K_local[:, cdof] .= 0.0
        end
    end
end
