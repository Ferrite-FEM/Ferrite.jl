struct Field
    name::Symbol
    interpolation::Interpolation
    dim::Int
end

abstract type AbstractDofHandler end

"""
    DofHandler(grid::Grid)

Construct a `DofHandler` based on the grid `grid`.
"""
struct DofHandler{dim,C,T} <: AbstractDofHandler
    field_names::Vector{Symbol}
    field_dims::Vector{Int}
    # TODO: field_interpolations can probably be better typed: We should at least require
    #       all the interpolations to have the same dimension and reference shape
    field_interpolations::Vector{Interpolation}
    bc_values::Vector{BCValues{T}} # TODO: BcValues is created/handeld by the constrainthandler, so this can be removed
    cell_dofs::Vector{Int}
    cell_dofs_offset::Vector{Int}
    closed::ScalarWrapper{Bool}
    grid::Grid{dim,C,T}
    ndofs::ScalarWrapper{Int}
end

function DofHandler(grid::Grid)
    isconcretetype(getcelltype(grid)) || error("Grid includes different celltypes. Use MixedDofHandler instead of DofHandler")
    DofHandler(Symbol[], Int[], Interpolation[], BCValues{Float64}[], Int[], Int[], ScalarWrapper(false), grid, Ferrite.ScalarWrapper(-1))
end

function Base.show(io::IO, ::MIME"text/plain", dh::DofHandler)
    println(io, "DofHandler")
    println(io, "  Fields:")
    for i in 1:nfields(dh)
        println(io, "    ", repr(dh.field_names[i]), ", interpolation: ", dh.field_interpolations[i],", dim: ", dh.field_dims[i])
    end
    if !isclosed(dh)
        print(io, "  Not closed!")
    else
        println(io, "  Dofs per cell: ", ndofs_per_cell(dh))
        print(io, "  Total dofs: ", ndofs(dh))
    end
end

ndofs(dh::AbstractDofHandler) = dh.ndofs[]
ndofs_per_cell(dh::AbstractDofHandler, cell::Int=1) = dh.cell_dofs_offset[cell+1] - dh.cell_dofs_offset[cell]
isclosed(dh::AbstractDofHandler) = dh.closed[]
nfields(dh::AbstractDofHandler) = length(dh.field_names)
getfieldnames(dh::AbstractDofHandler) = dh.field_names
ndim(dh::AbstractDofHandler, field_name::Symbol) = dh.field_dims[find_field(dh, field_name)]
function find_field(dh::DofHandler, field_name::Symbol)
    j = findfirst(i->i == field_name, dh.field_names)
    j == 0 && error("did not find field $field_name")
    return j
end

# Calculate the offset to the first local dof of a field
function field_offset(dh::DofHandler, field_name::Symbol)
    offset = 0
    for i in 1:find_field(dh, field_name)-1
        offset += getnbasefunctions(dh.field_interpolations[i])::Int * dh.field_dims[i]
    end
    return offset
end

getfieldinterpolation(dh::DofHandler, field_idx::Int) = dh.field_interpolations[field_idx]
getfielddim(dh::DofHandler, field_idx::Int) = dh.field_dims[field_idx]
getbcvalue(dh::DofHandler, field_idx::Int) = dh.bc_values[field_idx]

"""
    dof_range(dh:DofHandler, field_name)

Return the local dof range for `field_name`. Example:

```jldoctest
julia> grid = generate_grid(Triangle, (3, 3));

julia> dh = DofHandler(grid); push!(dh, :u, 3); push!(dh, :p, 1); close!(dh);

julia> dof_range(dh, :u)
1:9

julia> dof_range(dh, :p)
10:12
```
"""
function dof_range(dh::DofHandler, field_name::Symbol)
    f = find_field(dh, field_name)
    offset = field_offset(dh, field_name)
    n_field_dofs = getnbasefunctions(dh.field_interpolations[f])::Int * dh.field_dims[f]
    return (offset+1):(offset+n_field_dofs)
end

function Base.push!(dh::DofHandler, name::Symbol, dim::Int, ip::Interpolation=default_interpolation(getcelltype(dh.grid)))
    @assert !isclosed(dh)
    @assert !in(name, dh.field_names)
    push!(dh.field_names, name)
    push!(dh.field_dims, dim)
    push!(dh.field_interpolations, ip)
    push!(dh.bc_values, BCValues(ip, default_interpolation(getcelltype(dh.grid)), FaceIndex))
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

function close!(dh::DofHandler)
    dh, _, _, _ = __close!(dh)
    return dh
end

# close the DofHandler and distribute all the dofs
function __close!(dh::DofHandler{dim}) where {dim}
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
        for fi in 1:nfields(dh)
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
            if interpolation_info.nfacedofs > 0 && (interpolation_info.dim == dim)
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
    dh.ndofs[] = maximum(dh.cell_dofs)
    dh.closed[] = true

    return dh, vertexdicts, edgedicts, facedicts

end

function celldofs!(global_dofs::Vector{Int}, dh::DofHandler, i::Int)
    @assert isclosed(dh)
    @assert length(global_dofs) == ndofs_per_cell(dh, i)
    unsafe_copyto!(global_dofs, 1, dh.cell_dofs, dh.cell_dofs_offset[i], length(global_dofs))
    return global_dofs
end

function cellnodes!(global_nodes::Vector{Int}, grid::Grid{dim,C}, i::Int) where {dim,C}
    @assert length(global_nodes) == nnodes(C)
    for j in 1:nnodes(C) # Currently assuming that DofHandler only has one celltype
        global_nodes[j] = grid.cells[i].nodes[j]
    end
    return global_nodes
end

function cellcoords!(global_coords::Vector{Vec{dim,T}}, grid::Grid{dim,C}, i::Int) where {dim,C,T}
    @assert length(global_coords) == nnodes(C)
    for j in 1:nnodes(C) # Currently assuming that DofHandler only has one celltype
        nodeid = grid.cells[i].nodes[j]
        global_coords[j] = grid.nodes[nodeid].x
    end
    return global_coords
end
cellcoords!(global_coords::Vector{<:Vec}, dh::DofHandler, i::Int) = cellcoords!(global_coords, dh.grid, i)

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
    create_sparsity_pattern(dh::DofHandler)

Create the sparsity pattern corresponding to the degree of freedom
numbering in the [`DofHandler`](@ref). Return a `SparseMatrixCSC`
with stored values in the correct places.

See the [Sparsity Pattern](@ref) section of the manual.
"""
@inline create_sparsity_pattern(dh::DofHandler) = _create_sparsity_pattern(dh, false)

"""
    create_symmetric_sparsity_pattern(dh::DofHandler)

Create the symmetric sparsity pattern corresponding to the degree of freedom
numbering in the [`DofHandler`](@ref) by only considering the upper
triangle of the matrix. Return a `Symmetric{SparseMatrixCSC}`.

See the [Sparsity Pattern](@ref) section of the manual.
"""
@inline create_symmetric_sparsity_pattern(dh::DofHandler) = Symmetric(_create_sparsity_pattern(dh, true), :U)

function _create_sparsity_pattern(dh::DofHandler, sym::Bool)
    ncells = getncells(dh.grid)
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
    vtk_grid(filename, dh.grid; compress=compress)
end

# Exports the FE field `u` to `vtkfile`
function WriteVTK.vtk_point_data(vtkfile, dh::DofHandler, u::Vector, suffix="")
    for f in 1:nfields(dh)
        @debug println("exporting field $(dh.field_names[f])")
        field_dim = dh.field_dims[f]
        space_dim = field_dim == 2 ? 3 : field_dim
        data = fill(0.0, space_dim, getnnodes(dh.grid))
        offset = field_offset(dh, dh.field_names[f])
        for cell in CellIterator(dh)
            _celldofs = celldofs(cell)
            counter = 1
            for node in getnodes(cell)
                for d in 1:dh.field_dims[f]
                    data[d, node] = u[_celldofs[counter + offset]]
                    @debug println("  exporting $(u[_celldofs[counter + offset]]) for dof#$(_celldofs[counter + offset])")
                    counter += 1
                end
            end
        end
        vtk_point_data(vtkfile, data, string(dh.field_names[f], suffix))
    end
    return vtkfile
end
