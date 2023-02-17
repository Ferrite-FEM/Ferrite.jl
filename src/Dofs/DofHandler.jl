abstract type AbstractDofHandler end

"""
    DofHandler(grid::Grid)

Construct a `DofHandler` based on `grid`.

Operates slightly faster than [`MixedDofHandler`](@ref). Supports:
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
end

function DofHandler(grid::AbstractGrid{dim}) where {dim}
    isconcretetype(getcelltype(grid)) || error("Grid includes different celltypes. Use MixedDofHandler instead of DofHandler")
    DofHandler(Symbol[], Int[], Interpolation[], BCValues{Float64}[], Int[], Int[], ScalarWrapper(false), grid, Ferrite.ScalarWrapper(-1))
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

"""
Get the spatial dimension of a dofhandler.
"""
getdim(dh::DofHandler{dim}) where {dim} = dim 

# has_entity_dof(dh::AbstractDofHandler, field_idx::Int, vertex::Int) = haskey(vertexdicts[field_idx], vertex)
# has_entity_dof(dh::AbstractDofHandler, field_idx::Int, edge::Tuple{Int,Int}) = haskey(edgedicts[field_idx], edge)
# has_entity_dof(dh::AbstractDofHandler, field_idx::Int, face::NTuple{dim,Int}) where {dim} = haskey(facedicts[field_idx], face)

has_cell_dofs(dh::AbstractDofHandler, field_idx::Int, cell::Int) = ncelldofs(getfieldinterpolation(dh, field_idx)) > 0
has_vertex_dofs(dh::AbstractDofHandler, field_idx::Int, vertex::VertexIndex) = nvertexdofs(getfieldinterpolation(dh, field_idx)) > 0
has_edge_dofs(dh::AbstractDofHandler, field_idx::Int, edge::EdgeIndex) = nedgedofs(getfieldinterpolation(dh, field_idx)) > 0
has_face_dofs(dh::AbstractDofHandler, field_idx::Int, face::FaceIndex) = nfacedofs(getfieldinterpolation(dh, field_idx)) > 0

# entity_dofs(dh::AbstractDofHandler, field_idx::Int, vertex::Int) = vertexdicts[field_idx][vertex]
# entity_dofs(dh::AbstractDofHandler, field_idx::Int, edge::Tuple{Int,Int}) = edgedicts[field_idx][edge]
# entity_dofs(dh::AbstractDofHandler, field_idx::Int, face::NTuple{dim,Int}) where {dim} = facedicts[field_idx][face]

"""
Compute the dofs belonging to a given cell of a given field.
"""
function cell_dofs(dh::AbstractDofHandler, field_idx::Int, cell::Int)
    ip = getfieldinterpolation(dh, field_idx)
    fdim = getfielddim(dh, field_idx)
    nentitydofs = fdim*ncelldofs(ip)
    totaldofs = fdim*getnbasefunctions(ip)
    ldofs = dof_range(dh, field_idx)[(totaldofs-nentitydofs+1):totaldofs]
    return celldofs(dh, cell)[ldofs]
end

"""
Compute the dofs belonging to a given vertex of a given field.
"""
function vertex_dofs(dh::AbstractDofHandler, field_idx::Int, vertex::VertexIndex)
    ip = getfieldinterpolation(dh, field_idx)
    nvdofs = Ferrite.nvertexdofs(ip)
    nvdofs == 0 && return Int[]
    fdim = getfielddim(dh, field_idx)
    cell,local_vertex_index = vertex
    cell_geo = getcells(getgrid(dh), cell)
    nvertices = length(Ferrite.vertices(cell_geo))
    nentitydofs = fdim*nvdofs*nvertices
    ldofr = Ferrite.dof_range(dh, field_idx)[1:nentitydofs]
    vdofs = Ferrite.celldofs(dh, cell)[ldofr]
    return reshape(vdofs, (fdim,nvertices))[:, local_vertex_index]
end

"""
Compute the dofs belonging to a given edge of a given field.
"""
function edge_dofs(dh::AbstractDofHandler, field_idx::Int, edge::EdgeIndex)
    ip = getfieldinterpolation(dh, field_idx)
    nedofs = Ferrite.nedgedofs(ip)
    nedofs == 0 && return Int[]
    nvdofs = Ferrite.nvertexdofs(ip)
    fdim = getfielddim(dh, field_idx)
    cell,local_edge_index = edge
    cell_geo = getcells(getgrid(dh), cell)
    nedges_on_cell = length(Ferrite.edges(cell_geo))
    nvertices_on_cell = length(Ferrite.vertices(cell_geo))
    nentitydofs = fdim*nedofs*nedges_on_cell
    offset = fdim*nvdofs*nvertices_on_cell
    edge_dofrange = Ferrite.dof_range(dh, field_idx)[(offset+1):(offset+nentitydofs)]
    lodal_edgedofs = Ferrite.celldofs(dh, cell)[edge_dofrange]
    return reshape(lodal_edgedofs, (fdim,nedges_on_cell))[:, local_edge_index]
end

"""
Compute the dofs belonging to a given face of a given field.
"""
function face_dofs(dh::AbstractDofHandler, field_idx::Int, face::FaceIndex)
    ip = Ferrite.getfieldinterpolation(dh, field_idx)
    dim = getdim(dh)
    nfdofs = Ferrite.nfacedofs(ip)
    nfdofs == 0 && return Int[]
    nvdofs = Ferrite.nvertexdofs(ip)
    fdim = getfielddim(dh, field_idx)
    cell,local_face_index = face
    cell_geo = getcells(getgrid(dh), cell)
    nedges_on_cell = length(Ferrite.edges(cell_geo))
    nfaces_on_cell = length(Ferrite.faces(cell_geo))
    nvertices_on_cell = length(Ferrite.vertices(cell_geo))
    nentitydofs = fdim*Ferrite.nfacedofs(ip)*nfaces_on_cell
    offset = fdim*nvdofs*nvertices_on_cell
    if dim > 2
        nedofs = Ferrite.nedgedofs(ip)
        offset += fdim*nedofs*nedges_on_cell
    end
    face_dofrange = Ferrite.dof_range(dh, field_idx)[(offset+1):(offset+nentitydofs)]
    local_facedofs = Ferrite.celldofs(dh, cell)[face_dofrange]
    return reshape(local_facedofs, (fdim,nfaces_on_cell))[:, local_face_index]
end

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
    j = findfirst(i->i == field_name, dh.field_names)
    j === nothing && error("could not find field :$field_name in DofHandler (existing fields: $(getfieldnames(dh)))")
    return j
end

# Calculate the offset to the first local dof of a field
function field_offset(dh::AbstractDofHandler, field_idx::Int)
    offset = 0
    for i in 1:field_idx-1
        offset += getnbasefunctions(getfieldinterpolation(dh,i))::Int * getfielddim(dh, i)
    end
    return offset
end

function field_offset(dh::AbstractDofHandler, field_name::Symbol)
    field_idx = findfirst(i->i == field_name, getfieldnames(dh))
    field_idx === nothing && error("did not find field $field_name")
    return field_offset(dh,field_idx)
end


"""
"""
function dof_range(dh::AbstractDofHandler, field_idx::Int)
    offset = field_offset(dh, field_idx)
    n_field_dofs = getnbasefunctions(getfieldinterpolation(dh, field_idx))::Int * getfielddim(dh, field_idx)
    return (offset+1):(offset+n_field_dofs)
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
function dof_range(dh::AbstractDofHandler, field_name::Symbol)
    field_idx = findfirst(i->i == field_name, getfieldnames(dh))
    field_idx === nothing && error("did not find field $field_name")
    return dof_range(dh, field_idx)
end

"""
    add!(dh::AbstractDofHandler, name::Symbol, dim::Int[, ip::Interpolation])

Add a `dim`-dimensional `Field` called `name` which is approximated by `ip` to `dh`.

The field is added to all cells of the underlying grid. In case no interpolation `ip` is given,
the default interpolation of the grid's celltype is used. 
If the grid uses several celltypes, [`add!(dh::MixedDofHandler, fh::FieldHandler)`](@ref) must be used instead.
"""
function add!(dh::AbstractDofHandler, name::Symbol, dim::Int, ip::Interpolation=default_interpolation(getcelltype(getgrid(dh))))
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
    __close!(dh)
    return dh
end

# close the DofHandler and distribute all the dofs
function __close!(dh::AbstractDofHandler)
    @assert !isclosed(dh)
    
    dim = getdim(dh)

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
    for (ci, cell) in enumerate(getcells(getgrid(dh)))
        @debug println("cell #$ci")
        for fi in 1:num_fields(dh)
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

function celldofs(dh::DofHandler, i::Int)
    @assert isclosed(dh)
    n = ndofs_per_cell(dh, i)
    global_dofs = zeros(Int, n)
    unsafe_copyto!(global_dofs, 1, dh.cell_dofs, dh.cell_dofs_offset[i], n)
    return global_dofs
end

# Compute a coupling matrix of size (ndofs_per_cell × ndofs_per_cell) based on the input
# coupling which can be of size i) (nfields × nfields) specifying coupling between fields,
# ii) (ncomponents × ncomponents) specifying coupling between components, or iii)
# (ndofs_per_cell × ndofs_per_cell) specifying coupling between all local dofs, i.e. a
# "template" local matrix.
function _coupling_to_local_dof_coupling(dh::DofHandler, coupling::AbstractMatrix{Bool}, sym::Bool)
    out = zeros(Bool, ndofs_per_cell(dh), ndofs_per_cell(dh))
    sz = size(coupling, 1)
    sz == size(coupling, 2) || error("coupling not square")
    sym && (issymmetric(coupling) || error("coupling not symmetric"))
    dof_ranges = [dof_range(dh, f) for f in dh.field_names]
    if sz == length(dh.field_names) # Coupling given by fields
        for (j, jrange) in pairs(dof_ranges), (i, irange) in pairs(dof_ranges)
            out[irange, jrange] .= coupling[i, j]
        end
    elseif sz == sum(dh.field_dims) # Coupling given by components
        component_offsets = pushfirst!(cumsum(dh.field_dims), 0)
        for (jf, jrange) in pairs(dof_ranges), (j, J) in pairs(jrange)
            jc = mod1(j, dh.field_dims[jf]) + component_offsets[jf]
            for (i_f, irange) in pairs(dof_ranges), (i, I) in pairs(irange)
                ic = mod1(i, dh.field_dims[i_f]) + component_offsets[i_f]
                out[I, J] = coupling[ic, jc]
            end
        end
    elseif sz == ndofs_per_cell(dh) # Coupling given by template local matrix
        out .= coupling
    else
        error("could not create coupling")
    end
    return out
end

# Creates a sparsity pattern from the dofs in a DofHandler.
# Returns a sparse matrix with the correct storage pattern.
"""
    create_sparsity_pattern(dh::AbstractDofHandler; coupling)

Create the sparsity pattern corresponding to the degree of freedom
numbering in the [`AbstractDofHandler`](@ref). Return a `SparseMatrixCSC`
with stored values in the correct places.

The keyword argument `coupling` can be used to specify how fields (or components) in the dof
handler couple to each other. `coupling` should be a square matrix of booleans with
`nfields` (or `ncomponents`) rows/columns with `true` if fields are coupled and `false` if
not. By default full coupling is assumed.

See the [Sparsity Pattern](@ref) section of the manual.
"""
function create_sparsity_pattern(dh::AbstractDofHandler; coupling=nothing)
    return _create_sparsity_pattern(dh, nothing, false, true, coupling)
end

"""
    create_symmetric_sparsity_pattern(dh::DofHandler; coupling)

Create the symmetric sparsity pattern corresponding to the degree of freedom
numbering in the [`AbstractDofHandler`](@ref) by only considering the upper
triangle of the matrix. Return a `Symmetric{SparseMatrixCSC}`.

See the [Sparsity Pattern](@ref) section of the manual.
"""
function create_symmetric_sparsity_pattern(dh::AbstractDofHandler; coupling=nothing)
    return Symmetric(_create_sparsity_pattern(dh, nothing, true, true, coupling), :U)
end

function _create_sparsity_pattern(dh::AbstractDofHandler, ch#=::Union{ConstraintHandler, Nothing}=#, sym::Bool, keep_constrained::Bool, coupling::Union{AbstractMatrix{Bool},Nothing})
    @assert isclosed(dh)
    if !keep_constrained
        @assert ch !== nothing && isclosed(ch)
    end
    ncells = getncells(getgrid(dh))
    if coupling !== nothing
        # Extend coupling to be of size (ndofs_per_cell × ndofs_per_cell)
        coupling = _coupling_to_local_dof_coupling(dh, coupling, sym)
    end
    # Compute approximate size for the buffers using the dofs in the first element
    n = ndofs_per_cell(dh)
    N = (coupling === nothing ? (sym ? div(n*(n+1), 2) : n^2) : count(coupling)) * ncells
    N += ndofs(dh) # always add the diagonal elements
    I = Int[]; resize!(I, N)
    J = Int[]; resize!(J, N)
    global_dofs = zeros(Int, n)
    cnt = 0
    for element_id in 1:ncells
        # MixedDofHandler might have varying number of dofs per element
        resize!(global_dofs, ndofs_per_cell(dh, element_id))
        celldofs!(global_dofs, dh, element_id)
        @inbounds for j in eachindex(global_dofs), i in eachindex(global_dofs)
            coupling === nothing || coupling[i, j] || continue
            dofi = global_dofs[i]
            dofj = global_dofs[j]
            sym && (dofi > dofj && continue)
            !keep_constrained && (haskey(ch.dofmapping, dofi) || haskey(ch.dofmapping, dofj)) && continue
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

    K = spzeros!!(Float64, I, J, ndofs(dh), ndofs(dh))

    # If ConstraintHandler is given, create the condensation pattern due to affine constraints
    if ch !== nothing
        @assert isclosed(ch)
        fill!(K.nzval, 1)
        _condense_sparsity_pattern!(K, ch.dofcoefficients, ch.dofmapping, keep_constrained)
        fillzero!(K)
    end

    return K
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
function reshape_to_nodes(dh::AbstractDofHandler, u::Vector{T}, fieldname::Symbol) where T
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

    for cell in CellIterator(dh, cellset, UpdateFlags(; nodes=true, coords=false, dofs=true))
        _celldofs = celldofs(cell)
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
