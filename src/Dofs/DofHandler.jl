

struct Field
    name::Symbol
    interpolation::Interpolation
    dim::Int
end

struct DofHandler{dim,N,T,M}
    fields::Vector{Field}
    # TODO: field_interpolations can probably be better typed: We should at least require
    #       all the interpolations to have the same dimension and reference shape
    bc_values::Vector{BCValues{T}} # for boundary conditions
    cell_dofs::Vector{Int}
    cell_dofs_offset::Vector{Int}
    closed::ScalarWrapper{Bool}
    grid::Grid{dim,N,T,M}
end

function DofHandler(grid::Grid)
    DofHandler(Field[], BCValues{Float64}[], Int[], Int[], ScalarWrapper(false), grid)
end

function Base.show(io::IO, dh::DofHandler)
    println(io, "DofHandler")
    println(io, "  Fields:")
    for field in dh.fields
        println(io, "    ", repr(field.name), " interpolation: ", field.interpolation,", dim: ", field.dim)
    end
    if !isclosed(dh)
        print(io, "  Not closed!")
    else
        println(io, "  Dofs per cell: ", ndofs_per_cell(dh))
        print(io, "  Total dofs: ", ndofs(dh))
    end
end

# TODO: This is not very nice, worth storing ndofs explicitly?
#       How often do you actually call ndofs though...
ndofs(dh::DofHandler) = maximum(dh.cell_dofs)
ndofs_per_cell(dh::DofHandler, cell::Int=1) = dh.cell_dofs_offset[cell+1] - dh.cell_dofs_offset[cell]
isclosed(dh::DofHandler) = dh.closed[]
nfields(dh::DofHandler) = length(dh.fields)
ndim(dh::DofHandler, field_name::Symbol) = dh.fields[find_field(dh, field_name)].dim
function find_field(dh::DofHandler, field_name::Symbol)
    for i in 1:length(dh.fields)
        if field_name == dh.fields[i].name
            return i
        end
    end
    error("did not find field $field_name")
end

# Calculate the offset to the first local dof of a field
function field_offset(dh::DofHandler, field_name::Symbol)
    offset = 0
    for i in 1:find_field(dh, field_name)-1
        offset += getnbasefunctions(f.interpolation) * dh.fields[i].dim
    end
    return offset
end

function Base.push!(dh::DofHandler, name::Symbol, dim::Int, ip::Interpolation=default_interpolation(getcelltype(dh.grid)))
    @assert !isclosed(dh)
    @assert !in(name, (x.name for x in dh.fields))
    push!(dh.fields, Field(name,ip,dim))
    push!(dh.bc_values, BCValues(ip, default_interpolation(getcelltype(dh.grid))))
    return dh
end

# sort and return true (was already sorted) or false (if we had to sort)
function sortedge(edge::Tuple{Int,Int})
    a, b = edge
    a < b ? (return (edge, true)) : (return ((b, a), false))
end

sortface(face::Tuple{Int,Int}) = minmax(face[1], face[2])
function sortface(face::Tuple{Int,Int,Int})
    a, b, c = face
    b, c = minmax(b, c)
    a, c = minmax(a, c)
    a, b = minmax(a, b)
    return (a, b, c)
end

# close the DofHandler and distribute all the dofs
function close!(dh::DofHandler{dim}) where {dim}
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
    for field in dh.fields
        # push!(dh.interpolation_info, InterpolationInfo(interpolation))
        push!(interpolation_infos, InterpolationInfo(field.interpolation))
    end

    # not implemented yet: more than one facedof per face in 3D
    dim == 3 && assert(!any(x->x.nfacedofs > 1, interpolation_infos))

    nextdof = 1 # next free dof to distribute
    push!(dh.cell_dofs_offset, 1) # dofs for the first cell start at 1

    # loop over all the cells, and distribute dofs for all the fields
    for (ci, cell) in enumerate(getcells(dh.grid))
        @debug println("cell #$ci")
        for (i,fi) in enumerate(dh.fields)
            interpolation_info = interpolation_infos[i]
            @debug println("  field: $(fi.name)")
            if interpolation_info.nvertexdofs > 0
                for vertex in vertices(cell)
                    @debug println("    vertex#$vertex")
                    token = ht_keyindex2!(vertexdicts[i], vertex)
                    if token > 0 # haskey(vertexdicts[i], vertex) # reuse dofs
                        reuse_dof = vertexdicts[i].vals[token] # vertexdicts[i][vertex]
                        for d in 1:fi.dim
                            @debug println("      reusing dof #$(reuse_dof + (d-1))")
                            push!(dh.cell_dofs, reuse_dof + (d-1))
                        end
                    else # token <= 0, distribute new dofs
                        for vertexdof in 1:interpolation_info.nvertexdofs
                            Base._setindex!(vertexdicts[i], nextdof, vertex, -token) # vertexdicts[i][vertex] = nextdof
                            for d in 1:fi.dim
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
                        token = ht_keyindex2!(edgedicts[i], sedge)
                        if token > 0 # haskey(edgedicts[i], sedge), reuse dofs
                            startdof, olddir = edgedicts[i].vals[token] # edgedicts[i][sedge] # first dof for this edge (if dir == true)
                            for edgedof in (dir == olddir ? 1:interpolation_info.nedgedofs : interpolation_info.nedgedofs:-1:1)
                                for d in 1:fi.dim
                                    reuse_dof = startdof + (d-1) + (edgedof-1)*fi.dim
                                    @debug println("      reusing dof#$(reuse_dof)")
                                    push!(dh.cell_dofs, reuse_dof)
                                end
                            end
                        else # token <= 0, distribute new dofs
                            Base._setindex!(edgedicts[i], (nextdof, dir), sedge, -token) # edgedicts[i][sedge] = (nextdof, dir),  store only the first dof for the edge
                            for edgedof in 1:interpolation_info.nedgedofs
                                for d in 1:fi.dim
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
                    token = ht_keyindex2!(facedicts[i], sface)
                    if token > 0 # haskey(facedicts[i], sface), reuse dofs
                        startdof = facedicts[i].vals[token] # facedicts[i][sface]
                        for facedof in interpolation_info.nfacedofs:-1:1 # always reverse (YOLO)
                            for d in 1:fi.dim
                                reuse_dof = startdof + (d-1) + (facedof-1)*fi.dim
                                @debug println("      reusing dof#$(reuse_dof)")
                                push!(dh.cell_dofs, reuse_dof)
                            end
                        end
                    else # distribute new dofs
                        Base._setindex!(facedicts[i], nextdof, sface, -token)# facedicts[i][sface] = nextdof,  store the first dof for this face
                        for facedof in 1:interpolation_info.nfacedofs
                            for d in 1:fi.dim
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
                    for d in 1:fi.dim
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

function celldofs!(global_dofs::Vector{Int}, dh::DofHandler, i::Int)
    @assert isclosed(dh)
    @assert length(global_dofs) == ndofs_per_cell(dh, i)
    unsafe_copy!(global_dofs, 1, dh.cell_dofs, dh.cell_dofs_offset[i], length(global_dofs))
    return global_dofs
end

# Creates a sparsity pattern from the dofs in a DofHandler.
# Returns a sparse matrix with the correct storage pattern.
@inline create_sparsity_pattern(dh::DofHandler) = _create_sparsity_pattern(dh, false)
@inline create_symmetric_sparsity_pattern(dh::DofHandler) = Symmetric(_create_sparsity_pattern(dh, true), :U)

function _create_sparsity_pattern(dh::DofHandler, sym::Bool)
    ncells = getncells(dh.grid)
    n = ndofs_per_cell(dh)
    N = sym ? div(n*(n+1), 2) * ncells : n^2 * ncells
    N += ndofs(dh) # always add the diagonal elements
    I = Int[]; sizehint!(I, N)
    J = Int[]; sizehint!(J, N)
    global_dofs = zeros(Int, n)
    for element_id in 1:ncells
        celldofs!(global_dofs, dh, element_id)
        @inbounds for j in 1:n, i in 1:n
            dofi = global_dofs[i]
            dofj = global_dofs[j]
            sym && (dofi > dofj && continue)
            push!(I, dofi)
            push!(J, dofj)
        end
    end
    for d in 1:ndofs(dh)
        push!(I, d)
        push!(J, d)
    end
    V = zeros(length(I))
    K = sparse(I, J, V)
    return K
end

WriteVTK.vtk_grid(filename::AbstractString, dh::DofHandler) = vtk_grid(filename, dh.grid)

# Exports the FE field `u` to `vtkfile`
function WriteVTK.vtk_point_data(vtkfile, dh::DofHandler, u::Vector)
    for field in dh.fields
        @debug println("exporting field $(field.name)")
        space_dim = field.dim == 2 ? 3 : field.dim
        data = fill(0.0, space_dim, getnnodes(dh.grid))
        offset = field_offset(dh, field.name)
        for cell in CellIterator(dh)
            _celldofs = celldofs(cell)
            counter = 1
            for node in getnodes(cell)
                for d in 1:field.dim
                    data[d, node] = u[_celldofs[counter + offset]]
                    @debug println("  exporting $(u[_celldofs[counter + offset]]) for dof#$(_celldofs[counter + offset])")
                    counter += 1
                end
            end
        end
        vtk_point_data(vtkfile, data, string(field.name))
    end
    return vtkfile
end
