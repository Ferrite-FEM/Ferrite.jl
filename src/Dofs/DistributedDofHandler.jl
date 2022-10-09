"""
    DistributedDofHandler(grid::AbstractDistributedGrid)

Construct a `DistributedDofHandler` based on `grid`.

Distributed version of [`DofHandler`](@docs). 

Supports:
- `Grid`s with a single concrete cell type.
- One or several fields on the whole domaine.
"""
struct DistributedDofHandler{dim,T,G<:AbstractDistributedGrid{dim}} <: AbstractDofHandler
    field_names::Vector{Symbol}
    field_dims::Vector{Int}
    # TODO: field_interpolations can probably be better typed: We should at least require
    #       all the interpolations to have the same dimension and reference shape
    field_interpolations::Vector{Interpolation}
    bc_values::Vector{BCValues{T}} # TODO: BcValues is created/handeld by the constrainthandler, so this can be removed
    cell_dofs::Vector{Int}
    cell_dofs_offset::Vector{Int}
    closed::ScalarWrapper{Bool}
    grid::G
    ndofs::ScalarWrapper{Int}

    vertexdicts::Vector{Dict{Int,Int}}
    edgedicts::Vector{Dict{Tuple{Int,Int},Tuple{Int,Bool}}}
    facedicts::Vector{Dict{NTuple{dim,Int},Int}}

    ldof_to_gdof::Vector{Int}
    ldof_to_rank::Vector{Int32}
end

function DistributedDofHandler(grid::AbstractDistributedGrid{dim}) where {dim}
    isconcretetype(getcelltype(grid)) || error("Grid includes different celltypes. DistributedMixedDofHandler not implemented yet.")
    DistributedDofHandler(Symbol[], Int[], Interpolation[], BCValues{Float64}[], Int[], Int[], ScalarWrapper(false), grid, Ferrite.ScalarWrapper(-1), Dict{Int,Int}[], Dict{Tuple{Int,Int},Tuple{Int,Bool}}[],Dict{NTuple{dim,Int},Int}[], Int[], Int32[])
end

function Base.show(io::IO, ::MIME"text/plain", dh::DistributedDofHandler)
    println(io, "DistributedDofHandler")
    println(io, "  Fields:")
    for i in 1:num_fields(dh)
        println(io, "    ", repr(dh.field_names[i]), ", interpolation: ", dh.field_interpolations[i],", dim: ", dh.field_dims[i])
    end
    if !isclosed(dh)
        print(io, "  Not closed!")
    else
        println(io, "  Dofs per cell: ", ndofs_per_cell(dh))
        print(io, "  Total local dofs: ", ndofs(dh))
    end
end

getlocalgrid(dh::DistributedDofHandler) = getlocalgrid(dh.grid)
getglobalgrid(dh::DistributedDofHandler) = dh.grid

# Compat layer against serial code
getgrid(dh::DistributedDofHandler) = getlocalgrid(dh)

# TODO this is copy pasta from DofHandler.jl
function celldofs!(global_dofs::Vector{Int}, dh::DistributedDofHandler, i::Int)
    @assert isclosed(dh)
    @assert length(global_dofs) == ndofs_per_cell(dh, i)
    unsafe_copyto!(global_dofs, 1, dh.cell_dofs, dh.cell_dofs_offset[i], length(global_dofs))
    return global_dofs
end

# TODO this is copy pasta from DofHandler.jl
cellcoords!(global_coords::Vector{<:Vec}, dh::DistributedDofHandler, i::Int) = cellcoords!(global_coords, getgrid(dh), i)

# TODO this is copy pasta from DofHandler.jl
function celldofs(dh::DistributedDofHandler, i::Int)
    @assert isclosed(dh)
    n = ndofs_per_cell(dh, i)
    global_dofs = zeros(Int, n)
    unsafe_copyto!(global_dofs, 1, dh.cell_dofs, dh.cell_dofs_offset[i], n)
    return global_dofs
end

renumber!(dh::DistributedDofHandler, perm::AbstractVector{<:Integer}) = (@assert false) && "Unimplemented"

function compute_dof_ownership(dh, dgrid)
    my_rank = MPI.Comm_rank(global_comm(dgrid))+1

    dof_owner = Vector{Int}(undef,ndofs(dh))
    fill!(dof_owner, my_rank)

    for ((lci, lclvi),sv) ∈ dgrid.shared_vertices
        owner_rank = minimum([collect(keys(sv.remote_vertices));my_rank])

        if owner_rank != my_rank
            for fi in 1:Ferrite.num_fields(dh)
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

"""
Compute the number of dofs owned by the current process.
"""
num_local_true_dofs(dh::DistributedDofHandler) = sum(dof_owner.==(MPI.Comm_rank(global_comm(dgrid))+1))

"""
Compute the number of dofs visible to the current process.
"""
num_local_dofs(dh::DistributedDofHandler) = length(dh.ldof_to_gdof)

"""
Compute the number of dofs in the global system.
"""
num_global_dofs(dh::DistributedDofHandler) = MPI.Allreduce(num_local_true_dofs(dh), MPI.SUM, global_comm(dgrid))

"""
Renumber the dofs in local ordering to their corresponding global numbering.

TODO: Refactor for MixedDofHandler integration
"""
function local_to_global_numbering(dh::DistributedDofHandler, dgrid::AbstractDistributedGrid)
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
    for (ci, cell) in enumerate(getcells(getgrid(dh)))
        @debug println("cell #$ci (R$my_rank)")
        for fi in 1:Ferrite.num_fields(dh)
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
                    for fi ∈ 1:Ferrite.num_fields(dh)
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
                for fi in 1:Ferrite.num_fields(dh)
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
    @assert findfirst(local_to_global .== 0) === nothing

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

function close!(dh::DistributedDofHandler)
    __close!(dh)
    append!(dh.ldof_to_gdof, local_to_global_numbering(dh, getglobalgrid(dh)))
    append!(dh.ldof_to_rank, compute_dof_ownership(dh, getglobalgrid(dh)))
end

# TODO this is copy pasta from DofHandler.jl
# close the DofHandler and distribute all the dofs
function __close!(dh::DistributedDofHandler{dim}) where {dim}
    @assert !isclosed(dh)

    # `vertexdict` keeps track of the visited vertices. We store the global vertex
    # number and the first dof we added to that vertex.
    resize!(dh.vertexdicts, num_fields(dh))
    for i in 1:num_fields(dh)
        dh.vertexdicts[i] = Dict{Tuple{Int,Int},Tuple{Int,Bool}}()
    end

    # `edgedict` keeps track of the visited edges, this will only be used for a 3D problem
    # An edge is determined from two vertices, but we also need to store the direction
    # of the first edge we encounter and add dofs too. When we encounter the same edge
    # the next time we check if the direction is the same, otherwise we reuse the dofs
    # in the reverse order
    resize!(dh.edgedicts, num_fields(dh))
    for i in 1:num_fields(dh)
        dh.edgedicts[i] = Dict{Tuple{Int,Int},Tuple{Int,Bool}}()
    end

    # `facedict` keeps track of the visited faces. We only need to store the first dof we
    # added to the face; if we encounter the same face again we *always* reverse the order
    # In 2D a face (i.e. a line) is uniquely determined by 2 vertices, and in 3D a
    # face (i.e. a surface) is uniquely determined by 3 vertices.
    resize!(dh.facedicts, num_fields(dh))
    for i in 1:num_fields(dh)
        dh.facedicts[i] = Dict{NTuple{dim,Int},Int}()
    end

    # celldofs are never shared between different cells so there is no need
    # for a `celldict` to keep track of which cells we have added dofs too.

    # We create the `InterpolationInfo` structs with precomputed information for each
    # interpolation since that allows having the cell loop as the outermost loop,
    # and the interpolation loop inside without using a function barrier
    interpolation_infos = InterpolationInfo[]
    for interpolation in dh.field_interpolations
        # push!(dh.interpolation_info, InterpolationInfo(interpolation))
        push!(interpolation_infos, InterpolationInfo(interpolation))
    end

    # not implemented yet: more than one facedof per face in 3D
    dim == 3 && @assert(!any(x->x.nfacedofs > 1, interpolation_infos))

    nextdof = 1 # next free dof to distribute
    push!(dh.cell_dofs_offset, 1) # dofs for the first cell start at 1

    # loop over all the cells, and distribute dofs for all the fields
    for (ci, cell) in enumerate(getcells(getgrid(dh)))
        @debug println("cell #$ci")
        for fi in 1:num_fields(dh)
            interpolation_info = interpolation_infos[fi]
            @debug println("  field: $(dh.field_names[fi])")
            if interpolation_info.nvertexdofs > 0
                for vertex in vertices(cell)
                    @debug println("    vertex#$vertex")
                    token = Base.ht_keyindex2!(dh.vertexdicts[fi], vertex)
                    if token > 0 # haskey(dh.vertexdicts[fi], vertex) # reuse dofs
                        reuse_dof = dh.vertexdicts[fi].vals[token] # dh.vertexdicts[fi][vertex]
                        for d in 1:dh.field_dims[fi]
                            @debug println("      reusing dof #$(reuse_dof + (d-1))")
                            push!(dh.cell_dofs, reuse_dof + (d-1))
                        end
                    else # token <= 0, distribute new dofs
                        for vertexdof in 1:interpolation_info.nvertexdofs
                            Base._setindex!(dh.vertexdicts[fi], nextdof, vertex, -token) # dh.vertexdicts[fi][vertex] = nextdof
                            for d in 1:dh.field_dims[fi]
                                @debug println("      adding dof#$nextdof")
                                push!(dh.cell_dofs, nextdof)
                                nextdof += 1
                            end
                        end
                    end
                end # vertex loop
            end
            if dim == 3 # edges only in 3D
                if interpolation_info.nedgedofs > 0
                    for edge in edges(cell)
                        sedge, dir = sortedge(edge)
                        @debug println("    edge#$sedge dir: $(dir)")
                        token = Base.ht_keyindex2!(dh.edgedicts[fi], sedge)
                        if token > 0 # haskey(dh.edgedicts[fi], sedge), reuse dofs
                            startdof, olddir = dh.edgedicts[fi].vals[token] # dh.edgedicts[fi][sedge] # first dof for this edge (if dir == true)
                            for edgedof in (dir == olddir ? (1:interpolation_info.nedgedofs) : (interpolation_info.nedgedofs:-1:1))
                                for d in 1:dh.field_dims[fi]
                                    reuse_dof = startdof + (d-1) + (edgedof-1)*dh.field_dims[fi]
                                    @debug println("      reusing dof#$(reuse_dof)")
                                    push!(dh.cell_dofs, reuse_dof)
                                end
                            end
                        else # token <= 0, distribute new dofs
                            Base._setindex!(dh.edgedicts[fi], (nextdof, dir), sedge, -token) # dh.edgedicts[fi][sedge] = (nextdof, dir),  store only the first dof for the edge
                            for edgedof in 1:interpolation_info.nedgedofs
                                for d in 1:dh.field_dims[fi]
                                    @debug println("      adding dof#$nextdof")
                                    push!(dh.cell_dofs, nextdof)
                                    nextdof += 1
                                end
                            end
                        end
                    end # edge loop
                end
            end
            if interpolation_info.nfacedofs > 0 && (interpolation_info.dim == dim)
                for face in faces(cell)
                    sface = sortface(face) # TODO: faces(cell) may as well just return the sorted list
                    @debug println("    face#$sface")
                    token = Base.ht_keyindex2!(dh.facedicts[fi], sface)
                    if token > 0 # haskey(dh.facedicts[fi], sface), reuse dofs
                        startdof = dh.facedicts[fi].vals[token] # dh.facedicts[fi][sface]
                        for facedof in interpolation_info.nfacedofs:-1:1 # always reverse (YOLO)
                            for d in 1:dh.field_dims[fi]
                                reuse_dof = startdof + (d-1) + (facedof-1)*dh.field_dims[fi]
                                @debug println("      reusing dof#$(reuse_dof)")
                                push!(dh.cell_dofs, reuse_dof)
                            end
                        end
                    else # distribute new dofs
                        Base._setindex!(dh.facedicts[fi], nextdof, sface, -token)# dh.facedicts[fi][sface] = nextdof,  store the first dof for this face
                        for facedof in 1:interpolation_info.nfacedofs
                            for d in 1:dh.field_dims[fi]
                                @debug println("      adding dof#$nextdof")
                                push!(dh.cell_dofs, nextdof)
                                nextdof += 1
                            end
                        end
                    end
                end # face loop
            end
            if interpolation_info.ncelldofs > 0 # always distribute new dofs for cell
                @debug println("    cell#$ci")
                for celldof in 1:interpolation_info.ncelldofs
                    for d in 1:dh.field_dims[fi]
                        @debug println("      adding dof#$nextdof")
                        push!(dh.cell_dofs, nextdof)
                        nextdof += 1
                    end
                end # cell loop
            end
        end # field loop
        # push! the first index of the next cell to the offset vector
        push!(dh.cell_dofs_offset, length(dh.cell_dofs)+1)
    end # cell loop
    dh.ndofs[] = maximum(dh.cell_dofs)
    dh.closed[] = true

    return dh
end
