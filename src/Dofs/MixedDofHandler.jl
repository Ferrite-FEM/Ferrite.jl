
#import JuAFEM.BCValues, JuAFEM.ScalarWrapper
#import JuAFEM.isclosed
#import JuAFEM.create_sparsity_pattern, JuAFEM.celldofs!

struct MixedDofHandler{dim,C,T} <: JuAFEM.AbstractDofHandler
    field_names::Vector{Symbol}
    field_dims::Vector{Int}
    # TODO: field_interpolations can probably be better typed: We should at least require
    #       all the interpolations to have the same dimension and reference shape
    field_interpolations::Vector{Interpolation}
    bc_values::Vector{BCValues{T}} # for boundary conditions
    cell_dofs::Vector{Int}
    cell_dofs_offset::Vector{Int}
    closed::ScalarWrapper{Bool}
    grid::MixedGrid{dim,C,T}
end

# TODO bad style of using the constructor name of another type?
function DofHandler(grid::MixedGrid)
    MixedDofHandler(Symbol[], Int[], Interpolation[], BCValues{Float64}[], Int[], Int[], ScalarWrapper(false), grid)
end

function Base.push!(dh::MixedDofHandler, name::Symbol, dim::Int, ip::Interpolation)
    @assert !isclosed(dh)
    @assert !in(name, dh.field_names)
    push!(dh.field_names, name)
    push!(dh.field_dims, dim)
    push!(dh.field_interpolations, ip)

    # TODO BCValues needs to be dealt with
    push!(dh.bc_values, BCValues(ip, ip))
    #push!(dh.bc_values, BCValues(ip, default_interpolation(getcelltype(dh.grid))))
    return dh
end

# close the DofHandler and distribute all the dofs
function close!(dh::MixedDofHandler{dim}) where {dim}
    @assert !isclosed(dh)

    # `vertexdict` keeps track of the visited vertices. We store the global vertex
    # number and the first dof we added to that vertex.
    vertexdicts = [Dict{Int,Int}() for _ in 1:nfields(dh)]

    # `edgedict` keeps track of the visited edges, this will only be used for a 3D problem
    # An edge is determined from two vertices, but we also need to store the direction
    # of the first edge we encounter and add dofs too. When we encounter the same edge
    # the next time we check if the direction is the same, otherwise we reuse the dofs
    # in the reverse order
    edgedicts = [Dict{Tuple{Int,Int},Tuple{Int,Bool}}() for _ in 1:nfields(dh)]

    # `facedict` keeps track of the visited faces. We only need to store the first dof we
    # added to the face; if we encounter the same face again we *always* reverse the order
    # In 2D a face (i.e. a line) is uniquely determined by 2 vertices, and in 3D a
    # face (i.e. a surface) is uniquely determined by 3 vertices.
    facedicts = [Dict{NTuple{dim,Int},Int}() for _ in 1:nfields(dh)]

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
    for (ci, cell) in enumerate(getcells(dh.grid))
        @debug println("cell #$ci")
        for fi in 1:length(dh.field_names)
            interpolation_info = interpolation_infos[fi]
            @debug println("  field: $(dh.field_names[fi])")
            if interpolation_info.nvertexdofs > 0
                for vertex in vertices(cell)
                    @debug println("    vertex#$vertex")
                    token = Base.ht_keyindex2!(vertexdicts[fi], vertex)
                    if token > 0 # haskey(vertexdicts[fi], vertex) # reuse dofs
                        reuse_dof = vertexdicts[fi].vals[token] # vertexdicts[fi][vertex]
                        for d in 1:dh.field_dims[fi]
                            @debug println("      reusing dof #$(reuse_dof + (d-1))")
                            push!(dh.cell_dofs, reuse_dof + (d-1))
                        end
                    else # token <= 0, distribute new dofs
                        for vertexdof in 1:interpolation_info.nvertexdofs
                            Base._setindex!(vertexdicts[fi], nextdof, vertex, -token) # vertexdicts[fi][vertex] = nextdof
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
                        token = Base.ht_keyindex2!(edgedicts[fi], sedge)
                        if token > 0 # haskey(edgedicts[fi], sedge), reuse dofs
                            startdof, olddir = edgedicts[fi].vals[token] # edgedicts[fi][sedge] # first dof for this edge (if dir == true)
                            for edgedof in (dir == olddir ? (1:interpolation_info.nedgedofs) : (interpolation_info.nedgedofs:-1:1))
                                for d in 1:dh.field_dims[fi]
                                    reuse_dof = startdof + (d-1) + (edgedof-1)*dh.field_dims[fi]
                                    @debug println("      reusing dof#$(reuse_dof)")
                                    push!(dh.cell_dofs, reuse_dof)
                                end
                            end
                        else # token <= 0, distribute new dofs
                            Base._setindex!(edgedicts[fi], (nextdof, dir), sedge, -token) # edgedicts[fi][sedge] = (nextdof, dir),  store only the first dof for the edge
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
            if interpolation_info.nfacedofs > 0 # nfacedofs(interpolation) > 0
                for face in faces(cell)
                    sface = sortface(face) # TODO: faces(cell) may as well just return the sorted list
                    @debug println("    face#$sface")
                    token = Base.ht_keyindex2!(facedicts[fi], sface)
                    if token > 0 # haskey(facedicts[fi], sface), reuse dofs
                        startdof = facedicts[fi].vals[token] # facedicts[fi][sface]
                        for facedof in interpolation_info.nfacedofs:-1:1 # always reverse (YOLO)
                            for d in 1:dh.field_dims[fi]
                                reuse_dof = startdof + (d-1) + (facedof-1)*dh.field_dims[fi]
                                @debug println("      reusing dof#$(reuse_dof)")
                                push!(dh.cell_dofs, reuse_dof)
                            end
                        end
                    else # distribute new dofs
                        Base._setindex!(facedicts[fi], nextdof, sface, -token)# facedicts[fi][sface] = nextdof,  store the first dof for this face
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
    dh.closed[] = true
    return dh
end

function _create_sparsity_pattern(dh::MixedDofHandler, sym::Bool)
    ncells = getncells(dh.grid)
    #n = ndofs_per_cell(dh)
    N::Int = 0
    for element_id = 1:ncells
        N += ndofs_per_cell(dh, element_id)
    end

    #N = sym ? div(n*(n+1), 2) * ncells : n^2 * ncells
    N += ndofs(dh) # always add the diagonal elements
    I = Int[]; resize!(I, N)
    J = Int[]; resize!(J, N)
    #global_dofs = zeros(Int, n)
    cnt = 0
    for element_id in 1:ncells
        # TODO inefficient but general
        n = ndofs_per_cell(dh, element_id)
        global_dofs = zeros(Int,n)
        celldofs!(global_dofs, dh, element_id)
        @inbounds for j in 1:n, i in 1:n
            dofi = global_dofs[i]
            dofj = global_dofs[j]
            sym && (dofi > dofj && continue)
            cnt += 1
            if cnt > length(J)
                resize!(I, trunc(Int, length(I) * 1.5))
                resize!(J, trunc(Int, length(J) * 1.5))
            end
            I[cnt] = dofi
            J[cnt] = dofj

        end
    end
    @inbounds for d in 1:ndofs(dh)
        cnt += 1
        if cnt > length(J)
            resize!(I, trunc(Int, length(I) + ndofs(dh)))
            resize!(J, trunc(Int, length(J) + ndofs(dh)))
        end
        I[cnt] = d
        J[cnt] = d
    end
    resize!(I, cnt)
    resize!(J, cnt)
    V = zeros(length(I))
    K = sparse(I, J, V)
    return K

end

@inline create_sparsity_pattern(dh::MixedDofHandler) = _create_sparsity_pattern(dh, false)
