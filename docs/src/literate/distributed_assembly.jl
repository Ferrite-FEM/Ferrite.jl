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
using Ferrite, SparseArrays, MPI, PartitionedArrays

macro debug(ex)
    return :($(esc(ex)))
end

# Launch MPI
MPI.Init()

# We start  generating a simple grid with 20x20 quadrilateral elements
# using `generate_grid`. The generator defaults to the unit square,
# so we don't need to specify the corners of the domain.
grid = generate_grid(Quadrilateral, (2, 2));

dgrid = DistributedGrid(grid)

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
qr = QuadratureRule{dim, RefCube}(2)
cellvalues = CellScalarValues(qr, ip);

# ### Degrees of freedom
# Next we need to define a `DofHandler`, which will take care of numbering
# and distribution of degrees of freedom for our approximated fields.
# We create the `DofHandler` and then add a single field called `u`.
# Lastly we `close!` the `DofHandler`, it is now that the dofs are distributed
# for all the elements.
dh = DofHandler(dgrid.local_grid)
push!(dh, :u, 1)
close!(dh);

# Renumber the dofs in local ordering to their corresponding global numbering.
# TODO: Refactor for MixedDofHandler integration
function local_to_global_numbering(dh::DofHandler, dgrid)
    # MPI rank starting with 1 to match Julia's index convention
    my_rank = MPI.Comm_rank(global_comm(dgrid))+1

    local_to_global = Vector{Int}(undef,ndofs(dh))
    fill!(local_to_global,0) # 0 is the invalid index!
    # Start by numbering local dofs only from 1:#local_dofs

    # Lookup for synchronization in the form (Remote Rank,Shared Entity)
    # @TODO replace dict with vector and tie to MPI neighborhood graph of the mesh
    vertices_send = Dict{Int,Vector{VertexIndex}}()
    n_vertices_recv = Dict{Int,Int}()

    # We start by assigning a local dof to all owned entities.
    # An entity is owned if:
    # 1. *All* topological neighbors are on the local process
    # 2. If the rank of the local process it lower than the rank of *all* topological neighbors
    # A topological neighbor in this context is hereby defined per entity:
    # * vertex: All elements whose vertex is the vertex in question
    # * cell: Just the cell itself
    # * All other entities: All cells for which one of the corresponding entities interior intersects 
    #                       with the interior of the entity in question.
    # TODO: implement for entitied with dim > 0
    next_local_idx = 1
    for (ci, cell) in enumerate(getcells(dh.grid))
        @debug println("cell #$ci (R$my_rank)")
        for fi in 1:Ferrite.nfields(dh)
            @debug println("  field: $(dh.field_names[fi]) (R$my_rank)")
            interpolation_info = Ferrite.InterpolationInfo(dh.field_interpolations[fi])
            if interpolation_info.nvertexdofs > 0
                for (vi,vertex) in enumerate(Ferrite.vertices(cell))
                    @debug println("    vertex#$vertex (R$my_rank)")
                    # Dof is owned if it is local or if my rank is the smallest in the neighborhood
                    if !haskey(dgrid.shared_vertices,VertexIndex(ci,vi)) || all(keys(dgrid.shared_vertices[VertexIndex(ci,vi)].remote_vertices) .> my_rank)
                        # Update dof assignment
                        dof_local_idx = dh.vertexdicts[fi][vertex]
                        if local_to_global[dof_local_idx] == 0
                            @debug println("      mapping vertex dof#$dof_local_idx to $next_local_idx (R$my_rank)")
                            local_to_global[dof_local_idx] = next_local_idx
                            next_local_idx += 1
                        else
                            @debug println("      vertex dof#$dof_local_idx already mapped to $(local_to_global[dof_local_idx]) (R$my_rank)")
                        end
                    end

                    # Update shared vertex lookup table
                    if haskey(dgrid.shared_vertices,VertexIndex(ci,vi))
                        master_rank = my_rank
                        for master_rank_new ∈ keys(dgrid.shared_vertices[VertexIndex(ci,vi)].remote_vertices)
                            master_rank = min(master_rank, master_rank_new)
                        end
                        for (remote_rank, svs) ∈ dgrid.shared_vertices[VertexIndex(ci,vi)].remote_vertices
                            if master_rank == my_rank # I own the dof - we have to send information
                                if !haskey(vertices_send,remote_rank)
                                    vertices_send[remote_rank] = Vector{Ferrite.VertexIndex}()
                                end
                                @debug println("      prepare sending vertex #$(VertexIndex(ci,vi)) to $remote_rank (R$my_rank)")
                                for i ∈ svs
                                    push!(vertices_send[remote_rank],VertexIndex(ci,vi))
                                end
                            elseif master_rank == remote_rank  # dof is owned by remote - we have to receive information
                                if !haskey(n_vertices_recv,remote_rank)
                                    n_vertices_recv[remote_rank] = length(svs)
                                else
                                    n_vertices_recv[remote_rank] += length(svs)
                                end
                                @debug println("      prepare receiving vertex #$(VertexIndex(ci,vi)) from $remote_rank (R$my_rank)")
                            end
                        end
                    end
                end
            end
        end
    end

    #
    num_true_local_dofs = next_local_idx-1
    @debug println("#true local dofs $num_true_local_dofs (R$my_rank)")

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
    @debug println("#shifted local dof range $(local_offset+1):$(local_offset+num_true_local_dofs) (R$my_rank)")

    # Shift assigned local dofs (dofs with value >0) into the global range
    # At this point in the algorithm the dofs with value 0 are the dofs owned of neighboring processes
    for i ∈ 1:length(local_to_global)
        if local_to_global[i] != 0
            local_to_global[i] += local_offset
        end
    end

    # Sync non-owned dofs with neighboring processes.
    # TODO: implement for entitied with dim > 0
    # TODO: Use MPI graph primitives to simplify this code
    for sending_rank ∈ 1:MPI.Comm_size(global_comm(dgrid))
        if my_rank == sending_rank
            for remote_rank ∈ 1:MPI.Comm_size(global_comm(dgrid))
                if haskey(vertices_send, remote_rank)
                    n_vertices = length(vertices_send[remote_rank])
                    @debug println("Sending $n_vertices vertices to rank $remote_rank (R$my_rank)")
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
                    for fi ∈ 1:Ferrite.nfields(dh)
                        next_buffer_idx = 1
                        if length(dh.vertexdicts[fi]) == 0
                            @debug println("Skipping send on field $(dh.field_names[fi]) (R$my_rank)")
                            continue
                        end
                        # fill correspondence array
                        corresponding_global_dofs = Array{Int64}(undef,n_vertices)
                        for (lci,lclvi) ∈ vertices_send[remote_rank]
                            vi = Ferrite.vertices(getcells(getgrid(dh),lci))[lclvi]
                            if haskey(dh.vertexdicts[fi], vi)
                                corresponding_global_dofs[next_buffer_idx] = local_to_global[dh.vertexdicts[fi][vi]]
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
                @debug println("Receiving $n_vertices vertices from rank $sending_rank (R$my_rank)")
                local_cells = Array{Int64}(undef,n_vertices)
                local_cell_vis = Array{Int64}(undef,n_vertices)
                MPI.Recv!(local_cells, global_comm(dgrid); source=sending_rank-1)
                MPI.Recv!(local_cell_vis, global_comm(dgrid); source=sending_rank-1)
                for fi in 1:Ferrite.nfields(dh)
                    if length(dh.vertexdicts[fi]) == 0
                        @debug println("  Skipping recv on field $(dh.field_names[fi]) (R$my_rank)")
                        continue
                    end
                    corresponding_global_dofs = Array{Int64}(undef,n_vertices)
                    MPI.Recv!(corresponding_global_dofs, global_comm(dgrid); source=sending_rank-1)
                    for (cdi,(lci,lclvi)) ∈ enumerate(zip(local_cells,local_cell_vis))
                        vi = Ferrite.vertices(getcells(getgrid(dh),lci))[lclvi]
                        if haskey(dh.vertexdicts[fi], vi)
                            local_to_global[dh.vertexdicts[fi][vi]] = corresponding_global_dofs[cdi]
                            @debug println("  Updating field $(dh.field_names[fi]) vertex $(VertexIndex(lci,lclvi)) to $(corresponding_global_dofs[cdi]) (R$my_rank)")
                        else
                            @debug println("  Skipping recv on field $(dh.field_names[fi]) vertex $vi (R$my_rank)")
                        end
                    end
                end
            end
        end
    end

    # Postcondition: All local dofs need a corresponding global dof!
    @assert findfirst(local_to_global .== 0) == nothing

    @debug vtk_grid("dofs", dgrid; compress=false) do vtk
        u = Vector{Float64}(undef,length(dgrid.local_grid.nodes))
        fill!(u, 0.0)
        for i=1:length(u)
            u[i] = local_to_global[dh.vertexdicts[1][i]]
        end
        vtk_point_data(vtk, u,"dof")
    end

    return local_to_global
end
local_to_global = local_to_global_numbering(dh, dgrid);

function compute_dof_ownership(dh, dgrid)
    my_rank = MPI.Comm_rank(global_comm(dgrid))+1

    dof_owner = Vector{Int}(undef,ndofs(dh))
    fill!(dof_owner, my_rank)

    for ((lci, lclvi),sv) ∈ dgrid.shared_vertices
        owner_rank = minimum([collect(keys(sv.remote_vertices));my_rank])

        if owner_rank != my_rank
            for fi in 1:Ferrite.nfields(dh)
                vi = Ferrite.vertices(getcells(getgrid(dh),lci))[lclvi]
                if haskey(dh.vertexdicts[fi], vi)
                    local_dof_idx = dh.vertexdicts[fi][vi]
                    dof_owner[local_dof_idx] = owner_rank
                end
            end
        end
    end

    return dof_owner
end
dof_owner = compute_dof_ownership(dh, dgrid);

nltdofs = sum(dof_owner.==(MPI.Comm_rank(global_comm(dgrid))+1))
ndofs_total = MPI.Allreduce(nltdofs, MPI.SUM, global_comm(dgrid))

# Now that we have distributed all our dofs we can create our tangent matrix,
# using `create_sparsity_pattern`. This function returns a sparse matrix
# with the correct elements stored.
#K = create_sparsity_pattern(dh)

# ### Boundary conditions
# In Ferrite constraints like Dirichlet boundary conditions
# are handled by a `ConstraintHandler`.
ch = ConstraintHandler(dh);

# Next we need to add constraints to `ch`. For this problem we define
# homogeneous Dirichlet boundary conditions on the whole boundary, i.e.
# the `union` of all the face sets on the boundary.
∂Ω = union(getfaceset.((grid, ), ["left", "right", "top", "bottom"])...);

# Now we are set up to define our constraint. We specify which field
# the condition is for, and our combined face set `∂Ω`. The last
# argument is a function which takes the spatial coordinate $x$ and
# the current time $t$ and returns the prescribed value. In this case
# it is trivial -- no matter what $x$ and $t$ we return $0$. When we have
# specified our constraint we `add!` it to `ch`.
dbc = Dirichlet(:u, ∂Ω, (x, t) -> 0)
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
function doassemble(cellvalues::CellScalarValues{dim}, dh::DofHandler, ldof_to_gdof, ldof_to_rank, ngdofs, dgrid) where {dim}
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
    #@TODO comminication can be optimized by deduplicating entries in the following arrays
    #@TODO reorder communication by field to eliminate need for `ghost_dof_field_index_to_send`
    ghost_dof_to_send = [Int[] for i ∈ 1:destination_len] # global dof id
    ghost_rank_to_send = [Int[] for i ∈ 1:destination_len] # rank of dof
    ghost_dof_field_index_to_send = [Int[] for i ∈ 1:destination_len]
    ghost_element_to_send = [Int[] for i ∈ 1:destination_len] # corresponding element
    ghost_dof_owner = [Int[] for i ∈ 1:destination_len] # corresponding owner
    for (pivot_vi, pivot_sv) ∈ dgrid.shared_vertices
        # We start by searching shared vertices which are not owned by us
        pivot_vertex_owner_rank = Ferrite.compute_owner(dgrid, pivot_sv)
        pivot_cell_idx = pivot_vi[1]

        if my_rank != pivot_vertex_owner_rank
            sender_slot = destination_index[pivot_vertex_owner_rank]

            @debug println("$pivot_vi may require synchronization (R$my_rank)")
            # We have to send ALL dofs on the element to the remote.
            # @TODO send actually ALL dofs (currently only vertex dofs for a first version...)
            pivot_cell = getcells(dgrid, pivot_cell_idx)
            for (other_vertex_idx, other_vertex) ∈ enumerate(Ferrite.vertices(pivot_cell))
                # Skip self
                other_vi = VertexIndex(pivot_cell_idx, other_vertex_idx)
                if other_vi == pivot_vi
                    continue
                end

                if is_shared_vertex(dgrid, other_vi)
                    #@TODO We should be able to remove more redundant communication is many cases.
                    other_sv = dgrid.shared_vertices[other_vi]
                    other_vertex_owner_rank = Ferrite.compute_owner(dgrid, other_sv)
                    # Also skip if the "other vertex" is already owned by the process owning the pivot vertex
                    if other_vertex_owner_rank == pivot_vertex_owner_rank
                        continue;
                    end
                    # A vertex is also not a ghost vertex if it touches the domain of the rank of the pivot
                    if pivot_vertex_owner_rank ∈ keys(other_sv.remote_vertices)
                        continue
                    end
                else
                    other_vertex_owner_rank = my_rank
                end

                # Now we have to sync all fields separately
                @debug println("  Ghost candidate $other_vi for $pivot_vi (R$my_rank)")
                for field_idx in 1:Ferrite.nfields(dh)
                    pivot_vertex = Ferrite.toglobal(getlocalgrid(dgrid), pivot_vi)
                    # If any of the two vertices is not defined on the current field, just skip.
                    if !haskey(dh.vertexdicts[field_idx], pivot_vertex) || !haskey(dh.vertexdicts[field_idx], other_vertex)
                        continue
                    end
                    @debug println("    $other_vi is ghost for $pivot_vi in field $field_idx (R$my_rank)")

                    other_vertex_dof = dh.vertexdicts[field_idx][other_vertex]

                    append!(ghost_dof_to_send[sender_slot], ldof_to_gdof[other_vertex_dof])
                    append!(ghost_rank_to_send[sender_slot], other_vertex_owner_rank)
                    append!(ghost_dof_field_index_to_send[sender_slot], field_idx)
                    append!(ghost_element_to_send[sender_slot], pivot_cell_idx)
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

    # Reconstruct source ranks
    ghost_recv_buffer_source_ranks = Int[]
    for (source_idx, recv_len) ∈ enumerate(ghost_recv_buffer_lengths)
        append!(ghost_recv_buffer_source_ranks, ones(recv_len)*sources[source_idx])
    end
    
    println("received $ghost_recv_buffer_dofs with owners $ghost_recv_buffer_ranks (R$my_rank)")

    # #TODO obtain ghosts algorithmic
    # if np == 2
    #     if my_rank == 1
    #         append!(ghost_dof_to_global, collect(7:9))
    #         append!(ghost_dof_rank, [2,2,2])
    #     else
    #         # no ghosts
    #     end
    # elseif np == 3
    #     if my_rank == 1
    #         append!(ghost_dof_to_global, [5,6,7,8,9])
    #         append!(ghost_dof_rank, [2,2,2,3,3])
    #     elseif my_rank == 2
    #         append!(ghost_dof_to_global, [1,3,8,9])
    #         append!(ghost_dof_rank, [1,1,3,3])
    #     else
    #         # no ghosts
    #     end        
    # end

    unique_ghosts_dr = sort(unique(first,zip(ghost_recv_buffer_dofs,ghost_recv_buffer_ranks)))
    # unzip manually
    for (dof,rank) ∈ unique_ghosts_dr
        push!(ghost_dof_to_global, dof)
        push!(ghost_dof_rank, rank)
    end

    # ------------- Construct rows and cols of distributed matrix --------
    all_local_cols = Int[ldof_to_gdof; ghost_dof_to_global]
    all_local_col_ranks = Int32[ldof_to_rank; ghost_dof_rank]
    @debug println("all_local_cols $all_local_cols (R$my_rank)")
    @debug println("all_local_col_ranks $all_local_col_ranks (R$my_rank)")

    col_indices = PartitionedArrays.IndexSet(my_rank, all_local_cols, all_local_col_ranks)
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
        #Ferrite.assemble!(f, celldofs(cell), fe)
    end

    @debug println("done assembling (R$my_rank)")

    # --------------------- Add ghost entries in IJ --------------------
    # Fix ghost layer - the locations for remote processes to write their data into
    unique_ghosts_dre = zip(ghost_recv_buffer_dofs,ghost_recv_buffer_ranks,ghost_recv_buffer_elements)
    @debug println("unique_ghosts_dre $unique_ghosts_dre (R$my_rank)")

    # @TODO this is dead slow. optimize.
    IJ = []
    for (pivot_vi, pivot_sv) ∈ dgrid.shared_vertices
        # Loop over owned shared vertices
        if my_rank == Ferrite.compute_owner(dgrid, pivot_sv)
            @debug println("Fixing ghost layer $pivot_vi (R$my_rank)")
            pivot_vertex = Ferrite.toglobal(getlocalgrid(dgrid), pivot_vi)
            # Now compare the vertex against EVERY ghost dof...
            for (i,(global_ghost_dof,ghost_owner_rank,ghost_cell_idx)) ∈ enumerate(unique_ghosts_dre)
                source_rank = ghost_recv_buffer_source_ranks[i]
                pivot_remotes = Ferrite.remote_entities(pivot_sv)
                local_ghost_dof = findfirst(x->x==global_ghost_dof, all_local_cols)
                # ...where we have to check that the ghost is on the correct rank ...
                if haskey(pivot_remotes, source_rank)
                    @debug println("  $pivot_vi found for rank $source_rank with dof $global_ghost_dof (R$my_rank)")
                    for (remote_cell_idx,_) ∈ pivot_remotes[source_rank]
                        # ... and that it is the correct element.
                        if remote_cell_idx == ghost_cell_idx
                            @debug println("    $pivot_vi synced against remote cell $remote_cell_idx (R$my_rank)")
                            for field_idx in 1:Ferrite.nfields(dh)
                                if !haskey(dh.vertexdicts[field_idx], pivot_vertex)
                                    continue
                                end
                                @debug println("      handling field $field_idx (R$my_rank)")
                                pivot_dof = dh.vertexdicts[field_idx][pivot_vertex]
                                push!(IJ, (pivot_dof, local_ghost_dof))
                            end
                        end
                    end
                end
            end
        end
    end
    # Deduplicate entries.
    unique!(IJ)

    for (i,j) ∈ IJ
        push!(assembler.I, i)
        push!(assembler.J, j)
        push!(assembler.V, 0.0)
    end

    # if np == 2
    #     if my_rank == 1
    #         append!(assembler.I, [3,3,3, 4,4, 6,6])
    #         append!(assembler.J, [7,8,9, 7,8, 7,9])
    #         append!(assembler.V, zeros(7))
    #     else
    #         # no ghost layer
    #     end
    # elseif np == 3
    #     if my_rank == 1
    #         append!(assembler.I, [3,3, 1,1, 4,4,4,4,4])
    #         append!(assembler.J, [9,6, 5,8, 8,5,7,6,9])
    #         append!(assembler.V, zeros(9))
    #     elseif my_rank == 2
    #         # all_local_cols [5, 4, 6, 7, 1, 3, 8, 9] (R2)
    #         append!(assembler.I, [1,1, 3,3])
    #         append!(assembler.J, [5,7, 6,8])
    #         append!(assembler.V, zeros(4))
    #     else
    #         # no ghost layer
    #     end
    # end
    I_ = MPIData(assembler.I, comm, (np,))
    J_ = MPIData(assembler.J, comm, (np,))
    V_ = MPIData(assembler.V, comm, (np,))
    @debug println("I=$(assembler.I) (R$my_rank)")
    @debug println("J=$(assembler.J) (R$my_rank)")
    K = PartitionedArrays.PSparseMatrix(I_, J_, V_, rows, cols, ids=:local)
    return K, f
end
#md nothing # hide

# ### Solution of the system
# The last step is to solve the system. First we call `doassemble`
# to obtain the global stiffness matrix `K` and force vector `f`.
K, f = doassemble(cellvalues, dh, local_to_global, dof_owner, ndofs_total, dgrid);

# Shutdown MPI
MPI.Finalize()

# Early out for testing.
exit(0)

# To account for the boundary conditions we use the `apply!` function.
# This modifies elements in `K` and `f` respectively, such that
# we can get the correct solution vector `u` by using `\`.
apply!(K, f, ch)
#u = PartitionedArray...
cg!(u, K, f);

# ### Exporting to VTK
# To visualize the result we export the grid and our field `u`
# to a VTK-file, which can be viewed in e.g. [ParaView](https://www.paraview.org/).
vtk_grid("heat_equation_distributed", dh) do vtk
    vtk_point_data(vtk, dh, u)
end

## test the result                #src
using Test                        #src
@test norm(u) ≈ 3.307743912641305 #src

#md # ## [Plain program](@id distributed-assembly-plain-program)
#md #
#md # Here follows a version of the program without any comments.
#md # The file is also available here: [`distributed_assembly.jl`](distributed_assembly.jl).
#md #
#md # ```julia
#md # @__CODE__
#md # ```
