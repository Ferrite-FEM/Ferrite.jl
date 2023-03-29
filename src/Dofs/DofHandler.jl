"""
    DofHandler(grid::Grid)

Construct a `DofHandler` based on `grid`.

Operates slightly faster than [`MixedDofHandler`](@ref). Supports:
- `Grid`s with a single concrete cell type.
- One or several fields on the whole domaine.
"""
struct DofHandler{dim,G<:AbstractGrid{dim}} <: AbstractDofHandler
    field_names::Vector{Symbol}
    field_dims::Vector{Int}
    # TODO: field_interpolations can probably be better typed: We should at least require
    #       all the interpolations to have the same dimension and reference shape
    field_interpolations::Vector{Interpolation}
    cell_dofs::Vector{Int}
    cell_dofs_offset::Vector{Int}
    closed::ScalarWrapper{Bool}
    grid::G
    ndofs::ScalarWrapper{Int}
end

function DofHandler(grid::AbstractGrid)
    isconcretetype(getcelltype(grid)) || error("Grid includes different celltypes. Use MixedDofHandler instead of DofHandler")
    DofHandler(Symbol[], Int[], Interpolation[], Int[], Int[], ScalarWrapper(false), grid, Ferrite.ScalarWrapper(-1))
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

ndofs_per_cell(dh::AbstractDofHandler, cell::Int=1) = dh.cell_dofs_offset[cell+1] - dh.cell_dofs_offset[cell]
nfields(dh::AbstractDofHandler) = length(dh.field_names)
getfieldnames(dh::AbstractDofHandler) = dh.field_names
ndim(dh::AbstractDofHandler, field_name::Symbol) = dh.field_dims[find_field(dh, field_name)]
function find_field(dh::DofHandler, field_name::Symbol)
    j = findfirst(i->i == field_name, dh.field_names)
    j === nothing && error("could not find field :$field_name in DofHandler (existing fields: $(getfieldnames(dh)))")
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

function getfielddim(dh::DofHandler, name::Symbol)
    field_pos = findfirst(i->i == name, getfieldnames(dh))
    field_pos === nothing && error("did not find field $name")
    return dh.field_dims[field_pos]
end

"""
    dof_range(dh:DofHandler, field_name)

Return the local dof range for `field_name`. Example:

```jldoctest
julia> grid = generate_grid(Triangle, (3, 3))
Grid{2, Triangle, Float64} with 18 Triangle cells and 16 nodes

julia> dh = DofHandler(grid); add!(dh, :u, 3); add!(dh, :p, 1); close!(dh);

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

"""
    add!(dh::AbstractDofHandler, name::Symbol, dim::Int[, ip::Interpolation])

Add a `dim`-dimensional `Field` called `name` which is approximated by `ip` to `dh`.

The field is added to all cells of the underlying grid. In case no interpolation `ip` is
given, the default interpolation of the grid's celltype is used. If the grid uses several
celltypes, [`add!(dh::MixedDofHandler, fh::FieldHandler)`](@ref) must be used instead.
"""
function add!(dh::DofHandler, name::Symbol, dim::Int, ip::Interpolation=default_interpolation(getcelltype(dh.grid)))
    @assert !isclosed(dh)
    @assert !in(name, dh.field_names)
    push!(dh.field_names, name)
    push!(dh.field_dims, dim)
    push!(dh.field_interpolations, ip)
    return dh
end

# Method for supporting dim=1 default
function add!(dh::DofHandler, name::Symbol, ip::Interpolation=default_interpolation(getcelltype(dh.grid)))
    return add!(dh, name, 1, ip)
end

function close!(dh::DofHandler)
    dh, _, _, _ = __close!(dh)
    return dh
end

# close the DofHandler and distribute all the dofs
function __close!(dh::DofHandler{dim}) where {dim}
    @assert !isclosed(dh)

    # `vertexdict` keeps track of the visited vertices. The first dof added to vertex v is
    # stored in vertexdict[v]
    # TODO: No need to allocate this vector for fields that don't have vertex dofs
    vertexdicts = [zeros(Int, getnnodes(dh.grid)) for _ in 1:nfields(dh)]

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
    dim == 3 && @assert(!any(x->any(y->y > 1, x.nfacedofs), interpolation_infos))

    nextdof = 1 # next free dof to distribute
    push!(dh.cell_dofs_offset, 1) # dofs for the first cell start at 1

    # loop over all the cells, and distribute dofs for all the fields
    for (ci, cell) in enumerate(getcells(dh.grid))
        @debug println("cell #$ci")
        for field_idx in 1:nfields(dh)
            interpolation_info = interpolation_infos[field_idx]
            @debug println("  field: $(dh.field_names[field_idx])")
            for (vi, vertex) in enumerate(vertices(cell))
                if interpolation_info.nvertexdofs[vi] > 0
                    @debug println("    vertex#$vertex")
                    reuse_dof = vertexdicts[field_idx][vertex]
                    if reuse_dof > 0 # reuse dofs
                        for d in 1:dh.field_dims[field_idx]
                            @debug println("      reusing dof #$(reuse_dof + (d-1))")
                            push!(dh.cell_dofs, reuse_dof + (d-1))
                        end
                    else # distribute new dofs
                        for vertexdof in 1:interpolation_info.nvertexdofs[vi]
                            vertexdicts[field_idx][vertex] = nextdof
                            for d in 1:dh.field_dims[field_idx]
                                @debug println("      adding dof#$nextdof")
                                push!(dh.cell_dofs, nextdof)
                                nextdof += 1
                            end
                        end
                    end
                end
            end # vertex loop
            if dim == 3 # edges only in 3D
                for (ei, edge) in enumerate(edges(cell))
                    if interpolation_info.dim == 3 && interpolation_info.nedgedofs[ei] > 0
                        sedge, dir = sortedge(edge)
                        @debug println("    edge#$sedge dir: $(dir)")
                        token = Base.ht_keyindex2!(edgedicts[field_idx], sedge)
                        if token > 0 # haskey(edgedicts[field_idx], sedge), reuse dofs
                            startdof, olddir = edgedicts[field_idx].vals[token] # edgedicts[field_idx][sedge] # first dof for this edge (if dir == true)
                            for edgedof in (dir == olddir ? (1:interpolation_info.nedgedofs[ei]) : (interpolation_info.nedgedofs[ei]:-1:1))
                                for d in 1:dh.field_dims[field_idx]
                                    reuse_dof = startdof + (d-1) + (edgedof-1)*dh.field_dims[field_idx]
                                    @debug println("      reusing dof#$(reuse_dof)")
                                    push!(dh.cell_dofs, reuse_dof)
                                end
                            end
                        else # token <= 0, distribute new dofs
                            Base._setindex!(edgedicts[field_idx], (nextdof, dir), sedge, -token) # edgedicts[field_idx][sedge] = (nextdof, dir),  store only the first dof for the edge
                            for edgedof in 1:interpolation_info.nedgedofs[ei]
                                for d in 1:dh.field_dims[field_idx]
                                    @debug println("      adding dof#$nextdof")
                                    push!(dh.cell_dofs, nextdof)
                                    nextdof += 1
                                end
                            end
                        end
                    elseif interpolation_info.dim == 2 && interpolation_info.nfacedofs[ei] > 0 # hotfix for current embedded elements
                        sedge, dir = sortedge(edge)
                        @debug println("    edge#$sedge dir: $(dir)")
                        token = Base.ht_keyindex2!(edgedicts[field_idx], sedge)
                        if token > 0 # haskey(edgedicts[field_idx], sedge), reuse dofs
                            startdof, olddir = edgedicts[field_idx].vals[token] # edgedicts[field_idx][sedge] # first dof for this edge (if dir == true)
                            for edgedof in (dir == olddir ? (1:interpolation_info.nfacedofs[ei]) : (interpolation_info.nfacedofs[ei]:-1:1))
                                for d in 1:dh.field_dims[field_idx]
                                    reuse_dof = startdof + (d-1) + (edgedof-1)*dh.field_dims[field_idx]
                                    @debug println("      reusing dof#$(reuse_dof)")
                                    push!(dh.cell_dofs, reuse_dof)
                                end
                            end
                        else # token <= 0, distribute new dofs
                            Base._setindex!(edgedicts[field_idx], (nextdof, dir), sedge, -token) # edgedicts[field_idx][sedge] = (nextdof, dir),  store only the first dof for the edge
                            for edgedof in 1:interpolation_info.nfacedofs[ei]
                                for d in 1:dh.field_dims[field_idx]
                                    @debug println("      adding dof#$nextdof")
                                    push!(dh.cell_dofs, nextdof)
                                    nextdof += 1
                                end
                            end
                        end
                    end 
                end # edge loop
            end
            for (fi, face) in enumerate(faces(cell))
                if interpolation_info.nfacedofs[fi] > 0 && (interpolation_info.dim == dim)
                    sface = sortface(face)
                    @debug println("    face#$sface")
                    token = Base.ht_keyindex2!(facedicts[field_idx], sface)
                    if token > 0 # haskey(facedicts[field_idx], sface), reuse dofs
                        startdof = facedicts[field_idx].vals[token] # facedicts[field_idx][sface]
                        for facedof in interpolation_info.nfacedofs[fi]:-1:1 # always reverse (YOLO)
                            for d in 1:dh.field_dims[field_idx]
                                reuse_dof = startdof + (d-1) + (facedof-1)*dh.field_dims[field_idx]
                                @debug println("      reusing dof#$(reuse_dof)")
                                push!(dh.cell_dofs, reuse_dof)
                            end
                        end
                    else # distribute new dofs
                        Base._setindex!(facedicts[field_idx], nextdof, sface, -token)# facedicts[field_idx][sface] = nextdof,  store the first dof for this face
                        for facedof in 1:interpolation_info.nfacedofs[fi]
                            for d in 1:dh.field_dims[field_idx]
                                @debug println("      adding dof#$nextdof")
                                push!(dh.cell_dofs, nextdof)
                                nextdof += 1
                            end
                        end
                    end
                end
            end # face loop
            if interpolation_info.ncelldofs > 0 # always distribute new dofs for cell
                @debug println("    cell#$ci")
                for celldof in 1:interpolation_info.ncelldofs
                    for d in 1:dh.field_dims[field_idx]
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
    dh.ndofs[] = maximum(dh.cell_dofs, init=0)
    dh.closed[] = true

    return dh, vertexdicts, edgedicts, facedicts
end

function celldofs!(global_dofs::Vector{Int}, dh::DofHandler, i::Int)
    @assert isclosed(dh)
    @assert length(global_dofs) == ndofs_per_cell(dh, i)
    unsafe_copyto!(global_dofs, 1, dh.cell_dofs, dh.cell_dofs_offset[i], length(global_dofs))
    return global_dofs
end

cellcoords!(global_coords::Vector{<:Vec}, dh::DofHandler, i::Int) = cellcoords!(global_coords, dh.grid, i)

function reshape_to_nodes(dh::DofHandler, u::Vector{T}, fieldname::Symbol) where T
    # make sure the field exists
    fieldname ∈ Ferrite.getfieldnames(dh) || error("Field $fieldname not found.")

    field_idx = findfirst(i->i==fieldname, getfieldnames(dh))
    offset = field_offset(dh, fieldname)
    field_dim = getfielddim(dh, field_idx)

    space_dim = field_dim == 2 ? 3 : field_dim
    data = fill(zero(T), space_dim, getnnodes(dh.grid))

    reshape_field_data!(data, dh, u, offset, field_dim)

    return data
end
