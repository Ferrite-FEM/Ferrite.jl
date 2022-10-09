# # Heat Equation
#
# ![](heat_square.png)
#
# *Figure 1*: Temperature field on the unit square with an internal uniform heat source
# solved with homogeneous Dirichlet boundary conditions on the boundary.
#
#
# ## Introduction
#
# The heat equation is the "Hello, world!" equation of finite elements.
# Here we solve the equation on a unit square, with a uniform internal source.
# The strong form of the (linear) heat equation is given by
#
# ```math
#  -\nabla \cdot (k \nabla u) = f  \quad x \in \Omega,
# ```
#
# where $u$ is the unknown temperature field, $k$ the heat conductivity,
# $f$ the heat source and $\Omega$ the domain. For simplicity we set $f = 1$
# and $k = 1$. We will consider homogeneous Dirichlet boundary conditions such that
# ```math
# u(x) = 0 \quad x \in \partial \Omega,
# ```
# where $\partial \Omega$ denotes the boundary of $\Omega$.
#
# The resulting weak form is given by
# ```math
# \int_{\Omega} \nabla v \cdot \nabla u \ d\Omega = \int_{\Omega} v \ d\Omega,
# ```
# where $v$ is a suitable test function.
#-
# ## Commented Program
#
# Now we solve the problem in Ferrite. What follows is a program spliced with comments.
#md # The full program, without comments, can be found in the next [section](@ref heat_equation-plain-program).
#
# First we load Ferrite, and some other packages we need
using Ferrite, SparseArrays, MPI, PartitionedArrays, IterativeSolvers, HYPRE

macro debug(ex)
    return :($(esc(ex)))
end

# @TODO contribute diagnostics upstream
function PartitionedArrays.matrix_exchanger(values,row_exchanger,row_lids,col_lids)
    part = get_part_ids(row_lids)
    parts_rcv = row_exchanger.parts_rcv
    parts_snd = row_exchanger.parts_snd

    function setup_rcv(part,parts_rcv,row_lids,col_lids,values)
        owner_to_i = Dict(( owner=>i for (i,owner) in enumerate(parts_rcv) ))
        ptrs = zeros(Int32,length(parts_rcv)+1)
        for (li,lj,v) in nziterator(values)
            owner = row_lids.lid_to_part[li]
            if owner != part
            ptrs[owner_to_i[owner]+1] +=1
            end
        end
        length_to_ptrs!(ptrs)
        k_rcv_data = zeros(Int,ptrs[end]-1)
        gi_rcv_data = zeros(Int,ptrs[end]-1)
        gj_rcv_data = zeros(Int,ptrs[end]-1)
        for (k,(li,lj,v)) in enumerate(nziterator(values))
            owner = row_lids.lid_to_part[li]
            if owner != part
            p = ptrs[owner_to_i[owner]]
            k_rcv_data[p] = k
            gi_rcv_data[p] = row_lids.lid_to_gid[li]
            gj_rcv_data[p] = col_lids.lid_to_gid[lj]
            ptrs[owner_to_i[owner]] += 1
            end
        end
        rewind_ptrs!(ptrs)
        k_rcv = Table(k_rcv_data,ptrs)
        gi_rcv = Table(gi_rcv_data,ptrs)
        gj_rcv = Table(gj_rcv_data,ptrs)
        k_rcv, gi_rcv, gj_rcv
    end

    k_rcv, gi_rcv, gj_rcv = map_parts(setup_rcv,part,parts_rcv,row_lids,col_lids,values)

    gi_snd = exchange(gi_rcv,parts_snd,parts_rcv)
    gj_snd = exchange(gj_rcv,parts_snd,parts_rcv)

    function setup_snd(part,row_lids,col_lids,gi_snd,gj_snd,values)
        ptrs = gi_snd.ptrs
        k_snd_data = zeros(Int,ptrs[end]-1)
        for p in 1:length(gi_snd.data)
            gi = gi_snd.data[p]
            gj = gj_snd.data[p]
            li = row_lids.gid_to_lid[gi]
            lj = col_lids.gid_to_lid[gj]
            k = nzindex(values,li,lj)
            PartitionedArrays.@check k > 0 "The sparsity pattern of the ghost layer is inconsistent - $part | ($li, $lj) | ($gi, $gj)"
            k_snd_data[p] = k
        end
        k_snd = Table(k_snd_data,ptrs)
        k_snd
    end

    k_snd = map_parts(setup_snd,part,row_lids,col_lids,gi_snd,gj_snd,values)

    Exchanger(parts_rcv,parts_snd,k_rcv,k_snd)
end

# Launch MPI
MPI.Init()

# We start  generating a simple grid with 20x20 quadrilateral elements
# using `generate_grid`. The generator defaults to the unit square,
# so we don't need to specify the corners of the domain.
dgrid = generate_distributed_grid(Quadrilateral, (20, 20));

# TODO refactor this into a utility function
@debug vtk_grid("grid", dgrid; compress=false) do vtk
    u = Vector{Float64}(undef,length(dgrid.local_grid.nodes))
    for rank ∈ 1:MPI.Comm_size(global_comm(dgrid))
        fill!(u, 0.0)
        for sv ∈ values(dgrid.shared_vertices)
            if haskey(sv.remote_vertices,rank)
                (cellidx,i) = sv.local_idx
                nodeidx = dgrid.local_grid.cells[cellidx].nodes[i]
                u[nodeidx] = rank
            end
        end
        vtk_point_data(vtk, u,"sv $rank")
    end
end

# ### Trial and test functions
# A `CellValues` facilitates the process of evaluating values and gradients of
# test and trial functions (among other things). Since the problem
# is a scalar problem we will use a `CellScalarValues` object. To define
# this we need to specify an interpolation space for the shape functions.
# We use Lagrange functions (both for interpolating the function and the geometry)
# based on the reference "cube". We also define a quadrature rule based on the
# same reference cube. We combine the interpolation and the quadrature rule
# to a `CellScalarValues` object.
dim = 2
ip = Lagrange{dim, RefCube, 1}()
ip_geo = Lagrange{dim, RefCube, 1}()
qr = QuadratureRule{dim, RefCube}(3)
cellvalues = CellScalarValues(qr, ip, ip_geo);

# ### Degrees of freedom
# Next we need to define a `DofHandler`, which will take care of numbering
# and distribution of degrees of freedom for our approximated fields.
# We create the `DofHandler` and then add a single field called `u`.
# Lastly we `close!` the `DofHandler`, it is now that the dofs are distributed
# for all the elements.
dh = DistributedDofHandler(dgrid)
push!(dh, :u, 1, ip)
close!(dh);

# ### Boundary conditions
# In Ferrite constraints like Dirichlet boundary conditions
# are handled by a `ConstraintHandler`.
ch = ConstraintHandler(dh);

# Next we need to add constraints to `ch`. For this problem we define
# homogeneous Dirichlet boundary conditions on the whole boundary, i.e.
# the `union` of all the face sets on the boundary.
∂Ω = union(getfaceset.((getlocalgrid(dgrid), ), ["left", "right", "top", "bottom"])...);

# Now we are set up to define our constraint. We specify which field
# the condition is for, and our combined face set `∂Ω`. The last
# argument is a function which takes the spatial coordinate $x$ and
# the current time $t$ and returns the prescribed value. In this case
# it is trivial -- no matter what $x$ and $t$ we return $0$. When we have
# specified our constraint we `add!` it to `ch`.
dbc = Dirichlet(:u, ∂Ω, (x, t) -> 1)
add!(ch, dbc);

# We also need to `close!` and `update!` our boundary conditions. When we call `close!`
# the dofs which will be constrained by the boundary conditions are calculated and stored
# in our `ch` object. Since the boundary conditions are, in this case,
# independent of time we can `update!` them directly with e.g. $t = 0$.
close!(ch)
update!(ch, 0.0);

# ### Assembling the linear system
# Now we have all the pieces needed to assemble the linear system, $K u = f$.
# We define a function, `doassemble` to do the assembly, which takes our `cellvalues`,
# the sparse matrix and our DofHandler as input arguments. The function returns the
# assembled stiffness matrix, and the force vector.
function doassemble(cellvalues::CellScalarValues{dim}, dh::DistributedDofHandler) where {dim}
    ldof_to_gdof = dh.ldof_to_gdof
    ldof_to_rank = dh.ldof_to_rank
    ngdofs = num_global_dofs(dh)
    dgrid = getglobalgrid(dh)

    # We allocate the element stiffness matrix and element force vector
    # just once before looping over all the cells instead of allocating
    # them every time in the loop.
    #+
    n_basefuncs = getnbasefunctions(cellvalues)
    Ke = zeros(n_basefuncs, n_basefuncs)
    fe = zeros(n_basefuncs)

    # @TODO put the code below into a "distributed assembler" struct and functions
    # @TODO the code below can be massively simplified by introducing a ghost layer to the 
    #       distributed grid, which can efficiently precompute some of the values below.
    comm = global_comm(dgrid)
    np = MPI.Comm_size(comm)
    my_rank = MPI.Comm_rank(comm)+1

    @debug println("starting assembly... (R$my_rank)")

    # Neighborhood graph
    # @TODO cleanup old code below and use graph primitives instead.
    (source_len, destination_len, _) = MPI.Dist_graph_neighbors_count(vertex_comm(dgrid))
    sources = Vector{Cint}(undef, source_len)
    destinations = Vector{Cint}(undef, destination_len)
    MPI.Dist_graph_neighbors!(vertex_comm(dgrid), sources, destinations)

    # Adjust to Julia index convention
    sources .+= 1
    destinations .+= 1

    @debug println("Neighborhood | $sources | $destinations (R$my_rank)")

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

    @debug println("ltdof_to_gdof $ltdof_to_gdof (R$my_rank)")
    @debug println("ldof_to_gdof $ldof_to_gdof (R$my_rank)")
    @debug println("ldof_to_rank $ldof_to_rank (R$my_rank)")

    # Process owns rows of owned dofs. The process also may write to some remote dofs, 
    # which correspond to non-owned share entities. Here we construct the rows for the
    # distributed matrix.
    # We decide for row (i.e. test function) ownership, because it the image of
    # SpMV is process local.
    row_indices = PartitionedArrays.IndexSet(my_rank, ldof_to_gdof, Int32.(ldof_to_rank))
    #FIXME: This below must be fixed before we can assemble to HYPRE IJ. Problem seems to be that rows and cols must be continuously assigned.
    #row_indices = PartitionedArrays.IndexRange(my_rank, length(ltdof_indices), ltdof_to_gdof[1], ldof_to_gdof[.!ltdof_indices], Int32.(ldof_to_rank[.!ltdof_indices]))
    row_data = MPIData(row_indices, comm, (np,))
    row_exchanger = Exchanger(row_data,neighbors)
    rows = PRange(ngdofs,row_data,row_exchanger)

    @debug println("rows done (R$my_rank)")

    # For the locally visible columns we also have to take into account that remote
    # processes will write their data in some of these, because their remotely 
    # owned trial functions overlap with the locally owned test functions.
    ghost_dof_to_global = Int[]
    ghost_dof_element_index = Int[]
    ghost_dof_rank = Int32[]

    # ------------ Ghost dof synchronization ----------   
    # Prepare sending ghost dofs to neighbors
    #@TODO communication can be optimized by deduplicating entries in, and compressing the following arrays
    #@TODO reorder communication by field to eliminate need for `ghost_dof_field_index_to_send`
    ghost_dof_to_send = [Int[] for i ∈ 1:destination_len] # global dof id
    ghost_rank_to_send = [Int[] for i ∈ 1:destination_len] # rank of dof
    ghost_dof_field_index_to_send = [Int[] for i ∈ 1:destination_len]
    ghost_element_to_send = [Int[] for i ∈ 1:destination_len] # corresponding element
    ghost_dof_owner = [Int[] for i ∈ 1:destination_len] # corresponding owner
    ghost_dof_pivot_to_send = [Int[] for i ∈ 1:destination_len] # corresponding dof to interact with
    for shared_entity_set ∈ [dgrid.shared_vertices, dgrid.shared_faces, dgrid.shared_edges]
        for (pivot_entity, pivot_shared_entity) ∈ shared_entity_set
            # Start by searching shared entities which are not owned
            pivot_entity_owner_rank = Ferrite.compute_owner(dgrid, pivot_shared_entity)
            pivot_cell_idx = pivot_entity[1]

            if my_rank != pivot_entity_owner_rank
                sender_slot = destination_index[pivot_entity_owner_rank]

                @debug println("$pivot_entity may require synchronization (R$my_rank)")
                # Note: We have to send ALL dofs on the element to the remote.
                cell_dofs_upper_bound = (pivot_cell_idx == getncells(dh.grid)) ? length(dh.cell_dofs) : dh.cell_dofs_offset[pivot_cell_idx+1]
                cell_dofs = dh.cell_dofs[dh.cell_dofs_offset[pivot_cell_idx]:cell_dofs_upper_bound]

                pivot_entity_global = Ferrite.toglobal(getlocalgrid(dgrid), pivot_entity)

                for (field_idx, field_name) in zip(1:Ferrite.nfields(dh), Ferrite.getfieldnames(dh))
                    pivot_entity_dof = Ferrite.entity_dofs(dh, field_idx, pivot_entity_global)
                    # Extract dofs belonging to the current field
                    cell_field_dofs = cell_dofs[Ferrite.dof_range(dh, field_name)]
                    for cell_field_dof ∈ cell_field_dofs
                        append!(ghost_dof_pivot_to_send[sender_slot], ldof_to_gdof[pivot_entity_dof])
                        append!(ghost_dof_to_send[sender_slot], ldof_to_gdof[cell_field_dof])
                        append!(ghost_rank_to_send[sender_slot], ldof_to_rank[cell_field_dof])
                        append!(ghost_dof_field_index_to_send[sender_slot], field_idx)
                        append!(ghost_element_to_send[sender_slot], pivot_cell_idx)
                    end
                end
            end
        end
    end

    ghost_send_buffer_lengths = Int[length(i) for i ∈ ghost_element_to_send]
    ghost_recv_buffer_lengths = zeros(Int, destination_len)
    MPI.Neighbor_alltoall!(UBuffer(ghost_send_buffer_lengths,1), UBuffer(ghost_recv_buffer_lengths,1), vertex_comm(dgrid));
    @debug for (i,ghost_recv_buffer_length) ∈ enumerate(ghost_recv_buffer_lengths)
        println("receiving $ghost_recv_buffer_length ghosts from $(sources[i])  (R$my_rank)")
    end

    # Communicate ghost information
    # @TODO coalesce communication
    ghost_send_buffer_dofs = vcat(ghost_dof_to_send...)
    ghost_recv_buffer_dofs = zeros(Int, sum(ghost_recv_buffer_lengths))
    MPI.Neighbor_alltoallv!(VBuffer(ghost_send_buffer_dofs,ghost_send_buffer_lengths), VBuffer(ghost_recv_buffer_dofs,ghost_recv_buffer_lengths), vertex_comm(dgrid))

    ghost_send_buffer_elements = vcat(ghost_element_to_send...)
    ghost_recv_buffer_elements = zeros(Int, sum(ghost_recv_buffer_lengths))
    MPI.Neighbor_alltoallv!(VBuffer(ghost_send_buffer_elements,ghost_send_buffer_lengths), VBuffer(ghost_recv_buffer_elements,ghost_recv_buffer_lengths), vertex_comm(dgrid))

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

    @debug println("received $ghost_recv_buffer_dofs with owners $ghost_recv_buffer_ranks (R$my_rank)")

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
    @debug println("all_local_cols $all_local_cols (R$my_rank)")
    @debug println("all_local_col_ranks $all_local_col_ranks (R$my_rank)")

    col_indices = PartitionedArrays.IndexSet(my_rank, all_local_cols, all_local_col_ranks)
    #FIXME: This below must be fixed before we can assemble to HYPRE IJ. Problem seems to be that rows and cols must be continuously assigned.
    #col_indices = PartitionedArrays.IndexRange(my_rank, length(ltdof_indices), ltdof_to_gdof[1], all_local_cols[all_local_col_ranks .!= my_rank], Int32.(all_local_col_ranks[all_local_col_ranks .!= my_rank]))
    col_data = MPIData(col_indices, comm, (np,))
    col_exchanger = Exchanger(col_data,neighbors)
    cols = PRange(ngdofs,col_data,col_exchanger)

    # --------------------- Local assembly --------------------
    # Next we define the global force vector `f` and use that and
    # the stiffness matrix `K` and create an assembler. The assembler
    # is just a thin wrapper around `f` and `K` and some extra storage
    # to make the assembling faster.
    #+
    @debug println("cols and rows constructed (R$my_rank)")
    f = PartitionedArrays.PVector(0.0,rows)
    @debug println("f constructed (R$my_rank)")
    assembler = start_assemble()
    @debug println("starting assembly (R$my_rank)")

    # It is now time to loop over all the cells in our grid. We do this by iterating
    # over a `CellIterator`. The iterator caches some useful things for us, for example
    # the nodal coordinates for the cell, and the local degrees of freedom.
    #+
    for cell in CellIterator(dh)
        @debug println("assembling cell #$(cell.current_cellid.x) (R$my_rank)")

        # Always remember to reset the element stiffness matrix and
        # force vector since we reuse them for all elements.
        #+
        fill!(Ke, 0)
        fill!(fe, 0)

        # For each cell we also need to reinitialize the cached values in `cellvalues`.
        #+
        reinit!(cellvalues, cell)

        # It is now time to loop over all the quadrature points in the cell and
        # assemble the contribution to `Ke` and `fe`. The integration weight
        # can be queried from `cellvalues` by `getdetJdV`.
        #+
        for q_point in 1:getnquadpoints(cellvalues)
            @debug println("assembling qp $q_point (R$my_rank)")
            dΩ = getdetJdV(cellvalues, q_point)
            # For each quadrature point we loop over all the (local) shape functions.
            # We need the value and gradient of the testfunction `v` and also the gradient
            # of the trial function `u`. We get all of these from `cellvalues`.
            #+
            for i in 1:n_basefuncs
                v  = shape_value(cellvalues, q_point, i)
                ∇v = shape_gradient(cellvalues, q_point, i)
                fe[i] += v * dΩ
                for j in 1:n_basefuncs
                    ∇u = shape_gradient(cellvalues, q_point, j)
                    Ke[i, j] += (∇v ⋅ ∇u) * dΩ
                end
            end
        end

        # The last step in the element loop is to assemble `Ke` and `fe`
        # into the global `K` and `f` with `assemble!`.
        #+
        @debug println("assembling cell finished local (R$my_rank)")
        Ferrite.assemble!(assembler, celldofs(cell), Ke)
        @debug println("assembling cell finished global (R$my_rank)")
        map_parts(local_view(f, f.rows)) do f_local
            Ferrite.assemble!(f_local, celldofs(cell), fe)
        end
    end
    @debug println("done assembling (R$my_rank)")

    # --------------------- Add ghost entries in IJ --------------------
    # Fix ghost layer - the locations for remote processes to write their data into
    unique_ghosts_dre = zip(ghost_recv_buffer_dofs_piv, ghost_recv_buffer_dofs, ghost_recv_buffer_ranks)
    @debug println("unique_ghosts_dre $unique_ghosts_dre (R$my_rank)")
    IJfix = []
    for (i,(pivot_dof, global_ghost_dof, ghost_owner_rank)) ∈ enumerate(unique_ghosts_dre)
        push!(IJfix, (pivot_dof, global_ghost_dof))
    end
    @debug println("IJfix $IJfix (R$my_rank)")

    I = map(i->ldof_to_gdof[i], assembler.I)
    J = map(j->ldof_to_gdof[j], assembler.J)
    V = map(v->v, assembler.V)

    for (i,j) ∈ IJfix
        push!(I, i)
        push!(J, j)
        push!(V, 0.0)
    end

    @debug println("I=$(I) (R$my_rank)")
    @debug println("J=$(J) (R$my_rank)")
    K = PartitionedArrays.PSparseMatrix(
        MPIData(I, comm, (np,)), 
        MPIData(J, comm, (np,)), 
        MPIData(V, comm, (np,)), 
        rows, cols, ids=:global
    )

    PartitionedArrays.assemble!(K)
    PartitionedArrays.assemble!(f)

    return K, f
end
#md nothing # hide

my_rank = MPI.Comm_rank(global_comm(dgrid))+1

# ### Solution of the system
# The last step is to solve the system. First we call `doassemble`
# to obtain the global stiffness matrix `K` and force vector `f`.
K, f = doassemble(cellvalues, dh, local_to_global, dof_owner, ndofs_total, dgrid);

# To account for the boundary conditions we use the `apply!` function.
# This modifies elements in `K` and `f` respectively, such that
# we can get the correct solution vector `u` by using a parallel 
# iterative solver.
"""
Poor man's Dirichlet BC application for PartitionedArrays. :)
"""
function apply_zero!(K::PartitionedArrays.PSparseMatrix, f::PartitionedArrays.PVector, ch::ConstraintHandler)
    map_parts(local_view(f, f.rows), f.rows.partition) do f_local, partition
        f_local[ch.prescribed_dofs] .= 0.0
    end

    map_parts(local_view(K, K.rows, K.cols)) do K_local
        for cdof in ch.prescribed_dofs
            K_local[cdof, :] .= 0.0
            K_local[:, cdof] .= 0.0
            K_local[cdof, cdof] = 1.0
        end
    end
end

function apply!(K::PartitionedArrays.PSparseMatrix, f::PartitionedArrays.PVector, ch::ConstraintHandler)
    map_parts(local_view(f, f.rows), f.rows.partition) do f_local, partition
        # Note: RHS only non-zero for owned RHS entries
        f_local[ch.prescribed_dofs] .= ch.inhomogeneities .* map(p -> p == partition.part, partition.lid_to_part[ch.prescribed_dofs])
    end

    # Zero out locally visible rows and columns
    map_parts(local_view(K, K.rows, K.cols)) do K_local
        for cdof ∈ ch.prescribed_dofs
            K_local[cdof, :] .= 0.0
            K_local[:, cdof] .= 0.0
            K_local[cdof, cdof] = 1.0
        end
    end

    # Zero out columns associated to the ghost dofs constrained on a remote process
    # TODO optimize

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
    remote_ghosts_constrained_send = copy(remote_ghosts_recv)
    for (i, remote_ghost_dof) ∈ enumerate(remote_ghosts_recv)
        remote_ghosts_constrained_send[i] = remote_ghost_dof ∈ K.cols.partition.part.lid_to_gid[ch.prescribed_dofs]
    end

    # Step 3: Send trash back
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

apply!(K, f, ch)

# Compute the solution
# Note: At the moment of writing this we have no good preconditioners for PSparseMatrix in Julia, 
# partly due to unimplemented multiplication operators for the matrix data type.
u = cg(K, f)

# Compute the solution with HYPRE (needs the hotfix in https://github.com/fredrikekre/HYPRE.jl/pull/4 to function)
# u_ = HYPRE.solve(
#     HYPRE.PCG(
#         global_comm(dgrid);
#         Precond = HYPRE.BoomerAMG()
#     ),
#     HYPRE.HYPREMatrix(K),
#     HYPRE.HYPREVector(f)
# )

# Convert back to PartitionedArrays vector
# u = PVector(0.0, K.cols)
# copy!(u, u_)
# PartitionedArrays.assemble!(u)

# ### Exporting to VTK
# To visualize the result we export the grid and our field `u`
# to a VTK-file, which can be viewed in e.g. [ParaView](https://www.paraview.org/).
vtk_grid("heat_equation_distributed-$my_rank", dh) do vtk
    map_parts(local_view(u, u.rows)) do u_local
        vtk_point_data(vtk, dh, u_local)
    end
end

## test the result                #src
using Test                        #src
@test norm(u) ≈ 9.536307974872432 #src

# Shutdown MPI
MPI.Finalize()

#md # ## [Plain program](@id distributed-assembly-plain-program)
#md #
#md # Here follows a version of the program without any comments.
#md # The file is also available here: [`distributed_assembly.jl`](distributed_assembly.jl).
#md #
#md # ```julia
#md # @__CODE__
#md # ```
