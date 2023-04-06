abstract type AbstractDofHandler end

"""
    Field(name::Symbol, interpolation::Interpolation, dim::Int)

Construct `dim`-dimensional `Field` called `name` which is approximated by `interpolation`.

The interpolation is used for distributing the degrees of freedom.
"""
struct Field
    name::Symbol
    interpolation::Interpolation
    dim::Int
end

"""
    FieldHandler(fields::Vector{Field}, cellset::Set{Int})

Construct a `FieldHandler` based on an array of [`Field`](@ref)s and assigns it a set of cells.

A `FieldHandler` must fulfill the following requirements:
- All [`Cell`](@ref)s in `cellset` are of the same type.
- Each field only uses a single interpolation on the `cellset`.
- Each cell belongs only to a single `FieldHandler`, i.e. all fields on a cell must be added within the same `FieldHandler`.

Notice that a `FieldHandler` can hold several fields.
"""
mutable struct FieldHandler
    fields::Vector{Field} # Should not be used, kept for compatibility for now
    field_names::Vector{Symbol}
    field_dims::Vector{Int}
    field_interpolations::Vector{Interpolation}
    cellset::Set{Int}
    ndofs_per_cell::Int # set in close(::DofHandler)
    function FieldHandler(fields, cellset)
        fh = new(fields, Symbol[], Int[], Interpolation[], cellset, -1)
        for f in fields
            push!(fh.field_names, f.name)
            push!(fh.field_dims, f.dim)
            push!(fh.field_interpolations, f.interpolation)
        end
        return fh
    end
end

"""
    DofHandler(grid::Grid)

Construct a `DofHandler` based on `grid`. Supports:
- `Grid`s with or without concrete element type (E.g. "mixed" grids with several different element types.)
- One or several fields, which can live on the whole domain or on subsets of the `Grid`.
"""
struct DofHandler{dim,G<:AbstractGrid{dim}} <: AbstractDofHandler
    fieldhandlers::Vector{FieldHandler}
    field_names::Vector{Symbol}
    # Dofs for cell i are stored in cell_dofs[cell_dofs_offset[i]:(cell_dofs_offset[i]+length[i]-1)].
    # Note that explicitly keeping track of ndofs_per_cell is necessary since dofs are *not*
    # distributed in cell order like for the DofHandler (where the length can be determined
    # by cell_dofs_offset[i+1]-cell_dofs_offset[i]).
    # TODO: ndofs_per_cell should probably be replaced by ndofs_per_fieldhandler, since all
    #       cells in a FieldHandler have the same number of dofs.
    cell_dofs::Vector{Int}
    cell_dofs_offset::Vector{Int}
    cell_to_fieldhandler::Vector{Int} # maps cell id -> fieldhandler id
    closed::ScalarWrapper{Bool}
    grid::G
    ndofs::ScalarWrapper{Int}
end

function DofHandler(grid::AbstractGrid{dim}) where dim
    ncells = getncells(grid)
    DofHandler{dim,typeof(grid)}(FieldHandler[], Symbol[], Int[], zeros(Int, ncells), zeros(Int, ncells), ScalarWrapper(false), grid, ScalarWrapper(-1))
end

function MixedDofHandler(::AbstractGrid)
    error("MixedDofHandler is the standard DofHandler in Ferrite now and has been renamed to DofHandler.
Use DofHandler even for mixed grids and fields on subdomains.")
end

function Base.show(io::IO, ::MIME"text/plain", dh::DofHandler)
    println(io, typeof(dh))
    println(io, "  Fields:")
    for fieldname in getfieldnames(dh)
        println(io, "    ", repr(fieldname), ", dim: ", getfielddim(dh, fieldname))
    end
    if !isclosed(dh)
        print(io, "  Not closed!")
    else
        print(io, "  Total dofs: ", ndofs(dh))
    end
end

isclosed(dh::AbstractDofHandler) = dh.closed[]

"""
    ndofs(dh::AbstractDofHandler)

Return the number of degrees of freedom in `dh`
"""
ndofs(dh::AbstractDofHandler) = dh.ndofs[]

"""
    ndofs_per_cell(dh::AbstractDofHandler[, cell::Int=1])

Return the number of degrees of freedom for the cell with index `cell`.

See also [`ndofs`](@ref).
"""
function ndofs_per_cell(dh::DofHandler, cell::Int=1)
    @boundscheck 1 <= cell <= getncells(dh.grid)
    return @inbounds dh.fieldhandlers[dh.cell_to_fieldhandler[cell]].ndofs_per_cell
end
nnodes_per_cell(dh::DofHandler, cell::Int=1) = nnodes_per_cell(dh.grid, cell) # TODO: deprecate, shouldn't belong to DofHandler any longer

"""
    celldofs!(global_dofs::Vector{Int}, dh::AbstractDofHandler, i::Int)

Store the degrees of freedom that belong to cell `i` in `global_dofs`.

See also [`celldofs`](@ref).
"""
function celldofs!(global_dofs::Vector{Int}, dh::DofHandler, i::Int)
    @assert isclosed(dh)
    @assert length(global_dofs) == ndofs_per_cell(dh, i)
    unsafe_copyto!(global_dofs, 1, dh.cell_dofs, dh.cell_dofs_offset[i], length(global_dofs))
    return global_dofs
end

"""
    celldofs(dh::AbstractDofHandler, i::Int)

Return a vector with the degrees of freedom that belong to cell `i`.

See also [`celldofs!`](@ref).
"""
function celldofs(dh::AbstractDofHandler, i::Int)
    return celldofs!(zeros(Int, ndofs_per_cell(dh, i)), dh, i)
end

function get_cell_coordinates!(global_coords::Vector{Vec{dim,T}}, dh::DofHandler, i::Union{Int, <:AbstractCell})
    get_cell_coordinates!(global_coords, dh.grid, i)
end

function cellnodes!(global_nodes::Vector{Int}, dh::DofHandler, i::Union{Int, <:AbstractCell})
    cellnodes!(global_nodes, dh.grid, i)
end

"""
    getfieldnames(dh::DofHandler)
    getfieldnames(fh::FieldHandler)

Return a vector with the names of all fields. Can be used as an iterable over all the fields
in the problem.
"""
getfieldnames(dh::DofHandler) = dh.field_names
getfieldnames(fh::FieldHandler) = fh.field_names

getfielddim(fh::FieldHandler, field_idx::Int) = fh.field_dims[field_idx]
getfielddim(fh::FieldHandler, field_name::Symbol) = getfielddim(fh, find_field(fh, field_name))

"""
    getfielddim(dh::DofHandler, field_idxs::NTuple{2,Int})
    getfielddim(dh::DofHandler, field_name::Symbol)
    getfielddim(dh::FieldHandler, field_idx::Int)
    getfielddim(dh::FieldHandler, field_name::Symbol)

Return the dimension of a given field. The field can be specified by its index (see
[`find_field`](@ref)) or its name.
"""
function getfielddim(dh::DofHandler, field_idxs::NTuple{2, Int})
    fh_idx, field_idx = field_idxs
    fielddim = getfielddim(dh.fieldhandlers[fh_idx], field_idx)
    return fielddim
end
getfielddim(dh::DofHandler, name::Symbol) = getfielddim(dh, find_field(dh, name))

"""
    add!(dh::DofHandler, fh::FieldHandler)

Add all fields of the [`FieldHandler`](@ref) `fh` to `dh`.
"""
function add!(dh::DofHandler, fh::FieldHandler)
    # TODO: perhaps check that a field with the same name is the same field?
    @assert !isclosed(dh)
    _check_same_celltype(dh.grid, collect(fh.cellset))
    _check_cellset_intersections(dh, fh)
    # the field interpolations should have the same refshape as the cells they are applied to
    # extract the celltype from the first cell as the celltypes are all equal
    cell_type = typeof(dh.grid.cells[first(fh.cellset)])
    refshape_cellset = getrefshape(default_interpolation(cell_type))
    for interpolation in fh.field_interpolations
        refshape = getrefshape(interpolation)
        refshape_cellset == refshape || error("The RefShapes of the fieldhandlers interpolations must correspond to the RefShape of the cells it is applied to.")
    end

    push!(dh.fieldhandlers, fh)
    return dh
end

function _check_cellset_intersections(dh::DofHandler, fh::FieldHandler)
    for _fh in dh.fieldhandlers
        isdisjoint(_fh.cellset, fh.cellset) || error("Each cell can only belong to a single FieldHandler.")
    end
end

function add!(dh::DofHandler, name::Symbol, dim::Int)
    celltype = getcelltype(dh.grid)
    isconcretetype(celltype) || error("If you have more than one celltype in Grid, you must use add!(dh::DofHandler, fh::FieldHandler)")
    add!(dh, name, dim, default_interpolation(celltype))
end

"""
    add!(dh::AbstractDofHandler, name::Symbol, dim::Int[, ip::Interpolation])
Add a `dim`-dimensional `Field` called `name` which is approximated by `ip` to `dh`.
The field is added to all cells of the underlying grid. In case no interpolation `ip` is
given, the default interpolation of the grid's celltype is used. If the grid uses several
celltypes, [`add!(dh::DofHandler, fh::FieldHandler)`](@ref) must be used instead.
"""
function add!(dh::DofHandler, name::Symbol, dim::Int, ip::Interpolation)
    @assert !isclosed(dh)

    celltype = getcelltype(dh.grid)
    @assert isconcretetype(celltype)

    if length(dh.fieldhandlers) == 0
        cellset = Set(1:getncells(dh.grid))
        push!(dh.fieldhandlers, FieldHandler(Field[], cellset))
    elseif length(dh.fieldhandlers) > 1
        error("If you have more than one FieldHandler, you must specify field")
    end
    fh = first(dh.fieldhandlers)

    push!(fh.field_names, name)
    push!(fh.field_dims, dim)
    push!(fh.field_interpolations, ip)

    field = Field(name,ip,dim)
    push!(fh.fields, field)

    return dh
end

"""
    close!(dh::AbstractDofHandler)

Closes `dh` and creates degrees of freedom for each cell.

If there are several fields, the dofs are added in the following order:
For a `DofHandler`, go through each `FieldHandler` in the order they were added.
For each field in the `FieldHandler` or in the `DofHandler` (again, in the order the fields were added),
create dofs for the cell.
This means that dofs on a particular cell, the dofs will be numbered according to the fields;
first dofs for field 1, then field 2, etc.
"""
function close!(dh::DofHandler)
    dh, _, _, _ = __close!(dh)
    return dh
end

function __close!(dh::DofHandler{dim}) where {dim}
    @assert !isclosed(dh)

    # Collect the global field names
    empty!(dh.field_names)
    for fh in dh.fieldhandlers, name in fh.field_names
        name in dh.field_names || push!(dh.field_names, name)
    end
    numfields = length(dh.field_names)

    # `vertexdict` keeps track of the visited vertices. The first dof added to vertex v is
    # stored in vertexdict[v]
    # TODO: No need to allocate this vector for fields that don't have vertex dofs
    vertexdicts = [zeros(Int, getnnodes(dh.grid)) for _ in 1:numfields]

    # `edgedict` keeps track of the visited edges, this will only be used for a 3D problem.
    # An edge is uniquely determined by two vertices, but we also need to store the
    # direction of the first edge we encounter and add dofs too. When we encounter the same
    # edge the next time we check if the direction is the same, otherwise we reuse the dofs
    # in the reverse order.
    edgedicts = [Dict{Tuple{Int,Int},Tuple{Int,Bool}}() for _ in 1:numfields]

    # `facedict` keeps track of the visited faces. We only need to store the first dof we
    # add to the face; if we encounter the same face again we *always* reverse the order. In
    # 2D a face (i.e. a line) is uniquely determined by 2 vertices, and in 3D a face (i.e. a
    # surface) is uniquely determined by 3 vertices.
    facedicts = [Dict{NTuple{dim,Int},Int}() for _ in 1:numfields]

    # Set initial values
    nextdof = 1  # next free dof to distribute

    @debug "\n\nCreating dofs\n"
    for (fhi, fh) in pairs(dh.fieldhandlers)
        nextdof = _close!(
            dh,
            fh,
            fhi, # TODO: Store in the FieldHandler?
            nextdof,
            vertexdicts,
            edgedicts,
            facedicts,
        )
    end
    dh.ndofs[] = maximum(dh.cell_dofs; init=0)
    dh.closed[] = true

    return dh, vertexdicts, edgedicts, facedicts

end

function _close!(dh::DofHandler{dim}, fh::FieldHandler, fh_index::Int, nextdof, vertexdicts, edgedicts, facedicts) where {dim}
    ip_infos = InterpolationInfo[]
    for interpolation in fh.field_interpolations
        ip_info = InterpolationInfo(interpolation)
        push!(ip_infos, ip_info)
        # TODO: More than one face dof per face in 3D are not implemented yet. This requires
        #       keeping track of the relative rotation between the faces, not just the
        #       direction as for faces (i.e. edges) in 2D.
        dim == 3 && @assert !any(x -> x > 1, ip_info.nfacedofs)
    end

    # TODO: Given the InterpolationInfo it should be possible to compute ndofs_per_cell, but
    # doesn't quite work for embedded elements right now (they don't distribute all dofs
    # "promised" by InterpolationInfo). Instead we compute it based on the number of dofs
    # added for the first cell in the set.
    first_cell = true
    ndofs_per_cell = -1

    # Mapping between the local field index and the global field index
    global_fidxs = Int[findfirst(gname -> gname === lname, dh.field_names) for lname in fh.field_names]

    # loop over all the cells, and distribute dofs for all the fields
    # TODO: Remove BitSet construction when SubDofHandler ensures sorted collections
    for ci in BitSet(fh.cellset)
        @debug "Creating dofs for cell #$ci"

        # TODO: _check_cellset_intersections can be removed in favor of this assertion
        @assert dh.cell_to_fieldhandler[ci] == 0
        dh.cell_to_fieldhandler[ci] = fh_index

        cell = getcells(dh.grid, ci)
        len_cell_dofs_start = length(dh.cell_dofs)
        dh.cell_dofs_offset[ci] = len_cell_dofs_start + 1

        for (lidx, gidx) in pairs(global_fidxs)
            @debug "\tfield: $(fh.field_names[lidx])"

            fdim = fh.field_dims[lidx]
            ip_info = ip_infos[lidx]

            # Distribute dofs for vertices
            nextdof = add_vertex_dofs(
                dh.cell_dofs, cell, vertexdicts[gidx], fdim,
                ip_info.nvertexdofs, nextdof
            )

            # Distribute dofs for edges (only applicable when dim is 3)
            if dim == 3 && (ip_info.dim == 3 || ip_info.dim == 2)
                # Regular 3D element or 2D interpolation embedded in 3D space
                nentitydofs = ip_info.dim == 3 ? ip_info.nedgedofs : ip_info.nfacedofs
                nextdof = add_edge_dofs(
                    dh.cell_dofs, cell, edgedicts[gidx], fdim,
                    nentitydofs, nextdof
                )
            end

            # Distribute dofs for faces. Filter out 2D interpolations in 3D space, since
            # they are added above as edge dofs.
            if ip_info.dim == dim
                nextdof = add_face_dofs(
                    dh.cell_dofs, cell, facedicts[gidx], fdim,
                    ip_info.nfacedofs, nextdof
                )
            end

            # Distribute internal dofs for cells
            nextdof = add_cell_dofs(
                dh.cell_dofs, fdim, ip_info.ncelldofs, nextdof
            )
        end

        if first_cell
            ndofs_per_cell = length(dh.cell_dofs) - len_cell_dofs_start
            fh.ndofs_per_cell = ndofs_per_cell
            first_cell = false
        else
            @assert ndofs_per_cell == length(dh.cell_dofs) - len_cell_dofs_start
        end

        # @debug "Dofs for cell #$ci:\n\t$cell_dofs"
    end # cell loop
    return nextdof
end

function add_vertex_dofs(cell_dofs, cell, vertexdict, field_dim, nvertexdofs, nextdof)
    for (vi, vertex) in pairs(vertices(cell))
        nvertexdofs[vi] > 0 || continue # skip if no dof on this vertex
        @assert nvertexdofs[vi] == 1
        first_dof = vertexdict[vertex]
        if first_dof > 0 # reuse dof
            for d in 1:field_dim
                reuse_dof = first_dof + (d-1)
                push!(cell_dofs, reuse_dof)
            end
        else # create dofs
            vertexdict[vertex] = nextdof
            for _ in 1:field_dim
                push!(cell_dofs, nextdof)
                nextdof += 1
            end
        end
    end
    return nextdof
end

function add_face_dofs(cell_dofs, cell, facedict, field_dim, nfacedofs, nextdof)
    @debug @assert all(nfacedofs .<= 1) "Currently only supports interpolations with less that 2 dofs per face"
    for (fi, face) in pairs(faces(cell))
        nfacedofs[fi] > 0 || continue # skip if no dof on this face
        sface = sortface(face)
        token = Base.ht_keyindex2!(facedict, sface)
        if token > 0 # haskey(facedict, sface) -> reuse dofs
            first_dof = facedict.vals[token] # facedict[sface]
            for dof_loc in nfacedofs[fi]:-1:1 # always reverse
                for d in 1:field_dim
                    reuse_dof = first_dof + (d-1) + (dof_loc-1)*field_dim
                    push!(cell_dofs, reuse_dof)
                end
            end
        else # !haskey(facedict, sface) -> create dofs
            Base._setindex!(facedict, nextdof, sface, -token) # facedict[sface] = nextdof
            for _ in 1:nfacedofs[fi], _ in 1:field_dim
                push!(cell_dofs, nextdof)
                nextdof += 1
            end
        end
    end
    return nextdof
end

function add_edge_dofs(cell_dofs, cell, edgedict, field_dim, nedgedofs, nextdof)
    for (ei, edge) in pairs(edges(cell))
        nedgedofs[ei] > 0 || continue # skip if no dof on this edge
        sedge, this_dir = sortedge(edge)
        token = Base.ht_keyindex2!(edgedict, sedge)
        if token > 0 # haskey(edgedict, sedge) -> reuse dofs
            first_dof, prev_dir = edgedict.vals[token] # edgedict[sedge]
            # For an edge between vertices v1 and v2 with "dof locations" l1 and l2 and
            # already distributed dofs d1-d6: v1 --- l1(d1,d2,d3) --- l2(d4,d5,d6) --- v2:
            #  - this_dir == prev_dir: first_dof is d1 and we loop as usual over dof
            #    locations then over field dims
            #  - this_dir != prev_dir: first_dof is d4 and we need to reverse the loop over
            #    the dof locations but *not* the field dims
            for dof_loc in (this_dir == prev_dir ? (1:nedgedofs[ei]) : (nedgedofs[ei]:-1:1))
                for d in 1:field_dim
                    reuse_dof = first_dof + (d-1) + (dof_loc-1)*field_dim
                    push!(cell_dofs, reuse_dof)
                end
            end
        else # !haskey(edgedict, sedge) -> create dofs
            Base._setindex!(edgedict, (nextdof, this_dir), sedge, -token) # edgedict[sedge] = (nextdof, this_dir)
            for _ in 1:nedgedofs[ei], _ in 1:field_dim
                push!(cell_dofs, nextdof)
                nextdof += 1
            end
        end
    end
    return nextdof
end

function add_cell_dofs(cell_dofs, field_dim, ncelldofs, nextdof)
    for _ in 1:ncelldofs, _ in 1:field_dim
        push!(cell_dofs, nextdof)
        nextdof += 1
    end
    return nextdof
end

"""
    sortedge(edge::Tuple{Int,Int})

Returns the unique representation of an edge and its orientation.
Here the unique representation is the sorted node index tuple. The
orientation is `true` if the edge is not flipped, where it is `false`
if the edge is flipped.
"""
function sortedge(edge::Tuple{Int,Int})
    a, b = edge
    a < b ? (return (edge, true)) : (return ((b, a), false))
end

"""
    sortface(face::Tuple{Int,Int}) 
    sortface(face::Tuple{Int,Int,Int})
    sortface(face::Tuple{Int,Int,Int,Int})

Returns the unique representation of a face.
Here the unique representation is the sorted node index tuple.
Note that in 3D we only need indices to uniquely identify a face,
so the unique representation is always a tuple length 3.
"""
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


"""
    find_field(dh::DofHandler, field_name::Symbol)::NTuple{2,Int}

Return the index of the field with name `field_name` in a `DofHandler`. The index is a
`NTuple{2,Int}`, where the 1st entry is the index of the `FieldHandler` within which the
field was found and the 2nd entry is the index of the field within the `FieldHandler`.

!!! note
    Always finds the 1st occurence of a field within `DofHandler`.

See also: [`find_field(fh::FieldHandler, field_name::Symbol)`](@ref),
[`_find_field(fh::FieldHandler, field_name::Symbol)`](@ref).
"""
function find_field(dh::DofHandler, field_name::Symbol)
    for (fh_idx, fh) in pairs(dh.fieldhandlers)
        field_idx = _find_field(fh, field_name)
        !isnothing(field_idx) && return (fh_idx, field_idx)
    end
    error("Did not find field :$field_name in DofHandler (existing fields: $(getfieldnames(dh))).")
end

"""
    find_field(fh::FieldHandler, field_name::Symbol)::Int

Return the index of the field with name `field_name` in a `FieldHandler`. Throw an
error if the field is not found.

See also: [`find_field(dh::DofHandler, field_name::Symbol)`](@ref), [`_find_field(fh::FieldHandler, field_name::Symbol)`](@ref).
"""
function find_field(fh::FieldHandler, field_name::Symbol)
    field_idx = _find_field(fh, field_name)
    if field_idx === nothing
        error("Did not find field :$field_name in FieldHandler (existing fields: $(fh.field_names))")
    end
    return field_idx
end

# No error if field not found
"""
    _find_field(fh::FieldHandler, field_name::Symbol)::Int

Return the index of the field with name `field_name` in the `FieldHandler` `fh`. Return 
`nothing` if the field is not found.

See also: [`find_field(dh::DofHandler, field_name::Symbol)`](@ref), [`find_field(fh::FieldHandler, field_name::Symbol)`](@ref).
"""
function _find_field(fh::FieldHandler, field_name::Symbol)
    return findfirst(x -> x === field_name, fh.field_names)
end

# Calculate the offset to the first local dof of a field
function field_offset(fh::FieldHandler, field_idx::Int)
    offset = 0
    for i in 1:(field_idx-1)
        offset += getnbasefunctions(fh.field_interpolations[i])::Int * fh.field_dims[i]
    end
    return offset
end
field_offset(fh::FieldHandler, field_name::Symbol) = field_offset(fh, find_field(fh, field_name))

field_offset(dh::DofHandler, field_name::Symbol) = field_offset(dh, find_field(dh, field_name))
function field_offset(dh::DofHandler, field_idxs::Tuple{Int, Int})
    fh_idx, field_idx = field_idxs
    field_offset(dh.fieldhandlers[fh_idx], field_idx)
end

"""
    dof_range(fh::FieldHandler, field_idx::Int)
    dof_range(fh::FieldHandler, field_name::Symbol)
    dof_range(dh:DofHandler, field_name::Symbol)

Return the local dof range for a given field. The field can be specified by its name or
index, where `field_idx` represents the index of a field within a `FieldHandler` and
`field_idxs` is a tuple of the `FieldHandler`-index within the `DofHandler` and the
`field_idx`.

!!! note
    The `dof_range` of a field can vary between different `FieldHandler`s. Therefore, it is
    advised to use the `field_idxs` or refer to a given `FieldHandler` directly in case
    several `FieldHandler`s exist. Using the `field_name` will always refer to the first
    occurence of `field` within the `DofHandler`.

Example:
```jldoctest
julia> grid = generate_grid(Triangle, (3, 3))
Grid{2, Triangle, Float64} with 18 Triangle cells and 16 nodes

julia> dh = DofHandler(grid); add!(dh, :u, 3); add!(dh, :p, 1); close!(dh);

julia> dof_range(dh, :u)
1:9

julia> dof_range(dh, :p)
10:12

julia> dof_range(dh, (1,1)) # field :u
1:9

julia> dof_range(dh.fieldhandlers[1], 2) # field :p
10:12
```
"""
function dof_range(fh::FieldHandler, field_idx::Int)
    offset = field_offset(fh, field_idx)
    field_interpolation = fh.field_interpolations[field_idx]
    field_dim = fh.field_dims[field_idx]
    n_field_dofs = getnbasefunctions(field_interpolation)::Int * field_dim
    return (offset+1):(offset+n_field_dofs)
end
dof_range(fh::FieldHandler, field_name::Symbol) = dof_range(fh, find_field(fh, field_name))

function dof_range(dh::DofHandler, field_name::Symbol)
    if length(dh.fieldhandlers) > 1
        error("The given DofHandler has $(length(dh.fieldhandlers)) FieldHandlers.
              Extracting the dof range based on the fieldname might not be a unique problem
              in this case. Use `dof_range(fh::FieldHandler, field_name)` instead.")
    end
    fh_idx, field_idx = find_field(dh, field_name)
    return dof_range(dh.fieldhandlers[fh_idx], field_idx)
end

"""
    getfieldinterpolation(dh::DofHandler, field_idxs::NTuple{2,Int})
    getfieldinterpolation(dh::FieldHandler, field_idx::Int)
    getfieldinterpolation(dh::FieldHandler, field_name::Symbol)

Return the interpolation of a given field. The field can be specified by its index (see
[`find_field`](@ref) or its name.
"""
function getfieldinterpolation(dh::DofHandler, field_idxs::NTuple{2,Int})
    fh_idx, field_idx = field_idxs
    ip = dh.fieldhandlers[fh_idx].field_interpolations[field_idx]
    return ip
end
getfieldinterpolation(fh::FieldHandler, field_idx::Int) = fh.field_interpolations[field_idx]
getfieldinterpolation(fh::FieldHandler, field_name::Symbol) = getfieldinterpolation(fh, find_field(fh, field_name))

"""
    reshape_to_nodes(dh::AbstractDofHandler, u::Vector{T}, fieldname::Symbol) where T

Reshape the entries of the dof-vector `u` which correspond to the field `fieldname` in nodal order.
Return a matrix with a column for every node and a row for every dimension of the field.
For superparametric fields only the entries corresponding to nodes of the grid will be returned. Do not use this function for subparametric approximations.
"""
function reshape_to_nodes(dh::DofHandler, u::Vector{T}, fieldname::Symbol) where T
    # make sure the field exists
    fieldname ∈ getfieldnames(dh) || error("Field $fieldname not found.")

    field_dim = getfielddim(dh, fieldname)
    space_dim = field_dim == 2 ? 3 : field_dim
    data = fill(T(NaN), space_dim, getnnodes(dh.grid))  # set default value

    for fh in dh.fieldhandlers
        # check if this fh contains this field, otherwise continue to the next
        field_idx = _find_field(fh, fieldname)
        field_idx === nothing && continue
        offset = field_offset(fh, field_idx)

        reshape_field_data!(data, dh, u, offset, field_dim, BitSet(fh.cellset))
    end
    return data
end

function reshape_field_data!(data::Matrix{T}, dh::AbstractDofHandler, u::Vector{T}, field_offset::Int, field_dim::Int, cellset=1:getncells(dh.grid)) where T

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
