abstract type AbstractDofHandler end

"""
    DofHandler(grid::Grid)

Construct a `DofHandler` based on `grid`.

Operates slightly faster than [`MixedDofHandler`](@docs). Supports:
- `Grid`s with a single concrete cell type.
- One or several fields on the whole domaine.
"""
struct DofHandler{dim,T,G<:AbstractGrid{dim}} <: AbstractDofHandler
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
end

function DofHandler(grid::AbstractGrid{dim}) where {dim}
    isconcretetype(getcelltype(grid)) || error("Grid includes different celltypes. Use MixedDofHandler instead of DofHandler")
    DofHandler(Symbol[], Int[], Interpolation[], BCValues{Float64}[], Int[], Int[], ScalarWrapper(false), grid, Ferrite.ScalarWrapper(-1), Dict{Int,Int}[], Dict{Tuple{Int,Int},Tuple{Int,Bool}}[],Dict{NTuple{dim,Int},Int}[])
end

function Base.show(io::IO, ::MIME"text/plain", dh::DofHandler)
    println(io, "DofHandler")
    println(io, "  Fields:")
    for i in 1:num_fields(dh)
        println(io, "    ", repr(dh.field_names[i]), ", interpolation: ", dh.field_interpolations[i],", dim: ", dh.field_dims[i])
    end
    if !isclosed(dh)
        print(io, "  Not closed!")
    else
        println(io, "  Dofs per cell: ", ndofs_per_cell(dh))
        print(io, "  Total dofs: ", ndofs(dh))
    end
end

# has_entity_dof(dh::AbstractDofHandler, field_idx::Int, vertex::Int) = haskey(dh.vertexdicts[field_idx], vertex)
# has_entity_dof(dh::AbstractDofHandler, field_idx::Int, edge::Tuple{Int,Int}) = haskey(dh.edgedicts[field_idx], edge)
# has_entity_dof(dh::AbstractDofHandler, field_idx::Int, face::NTuple{dim,Int}) where {dim} = haskey(dh.facedicts[field_idx], face)

has_cell_dofs(dh::AbstractDofHandler, field_idx::Int, cell::Int) = haskey(dh.celldicts[field_idx], cell)
has_vertex_dofs(dh::AbstractDofHandler, field_idx::Int, vertex::Int) = haskey(dh.vertexdicts[field_idx], vertex)
has_edge_dofs(dh::AbstractDofHandler, field_idx::Int, edge::Tuple{Int,Int}) = haskey(dh.edgedicts[field_idx], edge)
has_face_dofs(dh::AbstractDofHandler, field_idx::Int, face::NTuple{dim,Int}) where {dim} = haskey(dh.facedicts[field_idx], face)

# entity_dofs(dh::AbstractDofHandler, field_idx::Int, vertex::Int) = dh.vertexdicts[field_idx][vertex]
# entity_dofs(dh::AbstractDofHandler, field_idx::Int, edge::Tuple{Int,Int}) = dh.edgedicts[field_idx][edge]
# entity_dofs(dh::AbstractDofHandler, field_idx::Int, face::NTuple{dim,Int}) where {dim} = dh.facedicts[field_idx][face]

cell_dofs(dh::AbstractDofHandler, field_idx::Int, cell::Int) = dh.celldicts[field_idx][cell]
vertex_dofs(dh::AbstractDofHandler, field_idx::Int, vertex::Int) = dh.vertexdicts[field_idx][vertex]
edge_dofs(dh::AbstractDofHandler, field_idx::Int, edge::Tuple{Int,Int}) = dh.edgedicts[field_idx][edge]
face_dofs(dh::AbstractDofHandler, field_idx::Int, face::NTuple{dim,Int}) where {dim} = dh.facedicts[field_idx][face]

"""
    ndofs(dh::AbstractDofHandler)

Return the number of degrees of freedom in `dh`
"""
ndofs(dh::AbstractDofHandler) = dh.ndofs[]
ndofs_per_cell(dh::AbstractDofHandler, cell::Int=1) = dh.cell_dofs_offset[cell+1] - dh.cell_dofs_offset[cell]
isclosed(dh::AbstractDofHandler) = dh.closed[]
num_fields(dh::AbstractDofHandler) = length(dh.field_names)
getfieldnames(dh::AbstractDofHandler) = dh.field_names
getfieldinterpolation(dh::AbstractDofHandler, field_idx::Int) = dh.field_interpolations[field_idx]
getfielddim(dh::AbstractDofHandler, field_idx::Int) = dh.field_dims[field_idx]
getbcvalue(dh::AbstractDofHandler, field_idx::Int) = dh.bc_values[field_idx]
getgrid(dh::AbstractDofHandler) = dh.grid

function find_field(dh::AbstractDofHandler, field_name::Symbol)
    j = findfirst(i->i == field_name, getfieldnames(dh))
    j == 0 && error("did not find field $field_name")
    return j
end

# Calculate the offset to the first local dof of a field
function field_offset(dh::AbstractDofHandler, field_name::Symbol)
    offset = 0
    for i in 1:find_field(dh, field_name)-1
        offset += getnbasefunctions(getfieldinterpolation(dh,i))::Int * getfielddim(dh, i)
    end
    return offset
end

function getfielddim(dh::AbstractDofHandler, name::Symbol)
    field_pos = findfirst(i->i == name, getfieldnames(dh))
    field_pos === nothing && error("did not find field $name")
    return getfielddim(dh, field_pos)
end

"""
    dof_range(dh:DofHandler, field_name)

Return the local dof range for `field_name`. Example:

```jldoctest
julia> grid = generate_grid(Triangle, (3, 3))
Grid{2, Triangle, Float64} with 18 Triangle cells and 16 nodes

julia> dh = DofHandler(grid); push!(dh, :u, 3); push!(dh, :p, 1); close!(dh);

julia> dof_range(dh, :u)
1:9

julia> dof_range(dh, :p)
10:12
```
"""
function dof_range(dh::AbstractDofHandler, field_name::Symbol)
    f = find_field(dh, field_name)
    offset = field_offset(dh, field_name)
    n_field_dofs = getnbasefunctions(dh.field_interpolations[f])::Int * getfielddim(dh, f)
    return (offset+1):(offset+n_field_dofs)
end

"""
    push!(dh::AbstractDofHandler, name::Symbol, dim::Int[, ip::Interpolation])

Add a `dim`-dimensional `Field` called `name` which is approximated by `ip` to `dh`.

The field is added to all cells of the underlying grid. In case no interpolation `ip` is given,
the default interpolation of the grid's celltype is used. 
If the grid uses several celltypes, [`push!(dh::MixedDofHandler, fh::FieldHandler)`](@ref) must be used instead.
"""
function Base.push!(dh::AbstractDofHandler, name::Symbol, dim::Int, ip::Interpolation=default_interpolation(getcelltype(getgrid(dh))))
    @assert !isclosed(dh)
    @assert !in(name, dh.field_names)
    push!(dh.field_names, name)
    push!(dh.field_dims, dim)
    push!(dh.field_interpolations, ip)
    push!(dh.bc_values, BCValues(ip, default_interpolation(getcelltype(getgrid(dh)))))
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

function sortface(face::Tuple{Int,Int,Int,Int})
    a, b, c, d = face
    c, d = minmax(c, d)
    b, d = minmax(b, d)
    a, d = minmax(a, d)
    b, c = minmax(b, c)
    a, c = minmax(a, c)
    a, b = minmax(a, b)
    return (a, b, c)
end

function close!(dh::DofHandler)
    return __close!(dh)
end

# close the DofHandler and distribute all the dofs
function __close!(dh::DofHandler{dim}) where {dim}
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

function celldofs!(global_dofs::Vector{Int}, dh::DofHandler, i::Int)
    @assert isclosed(dh)
    @assert length(global_dofs) == ndofs_per_cell(dh, i)
    unsafe_copyto!(global_dofs, 1, dh.cell_dofs, dh.cell_dofs_offset[i], length(global_dofs))
    return global_dofs
end

function cellnodes!(global_nodes::Vector{Int}, grid::AbstractGrid{dim}, i::Int) where {dim,C}
    nodes = getcells(grid,i).nodes
    N = length(nodes)
    @assert length(global_nodes) == N
    for j in 1:N
        global_nodes[j] = nodes[j]
    end
    return global_nodes
end

function cellcoords!(global_coords::Vector{Vec{dim,T}}, grid::AbstractGrid{dim}, i::Int) where {dim,C,T}
    nodes = getcells(grid,i).nodes
    N = length(nodes)
    @assert length(global_coords) == N
    for j in 1:N
        global_coords[j] = getcoordinates(getnodes(grid,nodes[j]))
    end
    return global_coords
end

cellcoords!(global_coords::Vector{<:Vec}, dh::DofHandler, i::Int) = cellcoords!(global_coords, getgrid(dh), i)

function celldofs(dh::DofHandler, i::Int)
    @assert isclosed(dh)
    n = ndofs_per_cell(dh, i)
    global_dofs = zeros(Int, n)
    unsafe_copyto!(global_dofs, 1, dh.cell_dofs, dh.cell_dofs_offset[i], n)
    return global_dofs
end

# Creates a sparsity pattern from the dofs in a DofHandler.
# Returns a sparse matrix with the correct storage pattern.
"""
    create_sparsity_pattern(dh::AbstractDofHandler)

Create the sparsity pattern corresponding to the degree of freedom
numbering in the [`AbstractDofHandler`](@ref). Return a `SparseMatrixCSC`
with stored values in the correct places.

See the [Sparsity Pattern](@ref) section of the manual.
"""
create_sparsity_pattern(dh::AbstractDofHandler) = _create_sparsity_pattern(dh, nothing, false)

"""
    create_symmetric_sparsity_pattern(dh::AbstractDofHandler)

Create the symmetric sparsity pattern corresponding to the degree of freedom
numbering in the [`AbstractDofHandler`](@ref) by only considering the upper
triangle of the matrix. Return a `Symmetric{SparseMatrixCSC}`.

See the [Sparsity Pattern](@ref) section of the manual.
"""
create_symmetric_sparsity_pattern(dh::AbstractDofHandler) = Symmetric(_create_sparsity_pattern(dh, nothing, true), :U)

function _create_sparsity_pattern(dh::DofHandler, ch#=::Union{ConstraintHandler, Nothing}=#, sym::Bool)
    ncells = getncells(getgrid(dh))
    n = ndofs_per_cell(dh)
    N = sym ? div(n*(n+1), 2) * ncells : n^2 * ncells
    N += ndofs(dh) # always add the diagonal elements
    I = Int[]; resize!(I, N)
    J = Int[]; resize!(J, N)
    global_dofs = zeros(Int, n)
    cnt = 0
    for element_id in 1:ncells
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

    # Add entries to K corresponding to condensation due the linear constraints
    # Note, this requires the K matrix, which is why we can't push!() to the I,J,V
    # triplet directly.
    if ch !== nothing
        @assert isclosed(ch)
        _condense_sparsity_pattern!(K, ch.acs)
    end

    return K
end

# dof renumbering
"""
    renumber!(dh::DofHandler, perm)

Renumber the degrees of freedom in the DofHandler according to the
permuation `perm`.

!!! warning
    Remember to do renumbering *before* adding boundary conditions,
    otherwise the mapping for the dofs will be wrong.
"""
function renumber!(dh::AbstractDofHandler, perm::AbstractVector{<:Integer})
    @assert isperm(perm) && length(perm) == ndofs(dh)
    cell_dofs = dh.cell_dofs
    for i in eachindex(cell_dofs)
        cell_dofs[i] = perm[cell_dofs[i]]
    end
    return dh
end

function WriteVTK.vtk_grid(filename::AbstractString, dh::AbstractDofHandler; compress::Bool=true)
    vtk_grid(filename, getgrid(dh); compress=compress)
end

"""
    reshape_to_nodes(dh::AbstractDofHandler, u::Vector{T}, fieldname::Symbol) where T

Reshape the entries of the dof-vector `u` which correspond to the field `fieldname` in nodal order.
Return a matrix with a column for every node and a row for every dimension of the field.
For superparametric fields only the entries corresponding to nodes of the grid will be returned. Do not use this function for subparametric approximations.
"""
function reshape_to_nodes(dh::DofHandler, u::Vector{T}, fieldname::Symbol) where T
    # make sure the field exists
    fieldname ∈ Ferrite.getfieldnames(dh) || error("Field $fieldname not found.")

    field_idx = findfirst(i->i==fieldname, getfieldnames(dh))
    offset = field_offset(dh, fieldname)
    field_dim = getfielddim(dh, field_idx)

    space_dim = field_dim == 2 ? 3 : field_dim
    data = fill(zero(T), space_dim, getnnodes(getgrid(dh)))

    reshape_field_data!(data, dh, u, offset, field_dim)

    return data
end

function reshape_field_data!(data::Matrix{T}, dh::AbstractDofHandler, u::Vector{T}, field_offset::Int, field_dim::Int, cellset=Set{Int}(1:getncells(getgrid(dh)))) where T

    _celldofs = Vector{Int}(undef, ndofs_per_cell(dh, first(cellset)))
    for cell in CellIterator(dh, collect(cellset))
        celldofs!( _celldofs, cell)
        counter = 1
        for node in getnodes(cell)
            for d in 1:field_dim
                data[d, node] = u[_celldofs[counter + field_offset]]
                @debug println("  exporting $(u[_celldofs[counter + field_offset]]) for dof#$(_celldofs[counter + field_offset])")
                counter += 1
            end
            if field_dim == 2
                # paraview requires 3D-data so pad with zero
                data[3, node] = 0
            end
        end
    end
    return data
end
