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

A `FieldHandler` must fullfill the following requirements:
- All [`Cell`](@ref)s in `cellset` are of the same type.
- Each field only uses a single interpolation on the `cellset`.
- Each cell belongs only to a single `FieldHandler`, i.e. all fields on a cell must be added within the same `FieldHandler`.

Notice that a `FieldHandler` can hold several fields.
"""
struct FieldHandler{TCS}
    fields::Vector{Field}
    cellset::TCS
end

struct CellVector{T}
    values::Vector{T}
    offset::Vector{Int}
    length::Vector{Int}
end

function Base.getindex(elvec::CellVector, el::Int)
    offset = elvec.offset[el]
    return elvec.values[offset:offset + elvec.length[el]-1]
 end

"""
    MixedDofHandler(grid::Grid)

Construct a `MixedDofHandler` based on `grid`. Supports:
- `Grid`s with or without concrete element type (E.g. "mixed" grids with several different element types.)
- One or several fields, which can live on the whole domain or on subsets of the `Grid`.
"""
struct MixedDofHandler{dim,T,G<:AbstractGrid{dim}} <: AbstractDofHandler
    fieldhandlers::Vector{FieldHandler}
    cell_dofs::CellVector{Int}
    closed::ScalarWrapper{Bool}
    grid::G
    ndofs::ScalarWrapper{Int}
end

function MixedDofHandler(grid::Grid{dim,C,T}) where {dim,C,T}
    ncells = getncells(grid)
    MixedDofHandler{dim,T,typeof(grid)}(FieldHandler[], CellVector(Int[],zeros(Int,ncells),zeros(Int,ncells)), ScalarWrapper(false), grid, ScalarWrapper(-1))
end

getfieldnames(fh::FieldHandler) = [field.name for field in fh.fields]
getfielddims(fh::FieldHandler) = [field.dim for field in fh.fields]
getfieldinterpolations(fh::FieldHandler) = [field.interpolation for field in fh.fields]

"""
    ndofs_per_cell(dh::AbstractDofHandler[, cell::Int=1])

Return the number of degrees of freedom for the cell with index `cell`.

See also [`ndofs`](@ref).
"""
ndofs_per_cell(dh::MixedDofHandler, cell::Int=1) = dh.cell_dofs.length[cell]
nnodes_per_cell(dh::MixedDofHandler, cell::Int=1) = nnodes_per_cell(dh.grid, cell) # TODO: deprecate, shouldn't belong to MixedDofHandler any longer

"""
    celldofs!(global_dofs::Vector{Int}, dh::AbstractDofHandler, i::Int)

Store the degrees of freedom that belong to cell `i` in `global_dofs`.

See also [`celldofs`](@ref).
"""
function celldofs!(global_dofs::Vector{Int}, dh::MixedDofHandler, i::Int)
    @assert isclosed(dh)
    @assert length(global_dofs) == ndofs_per_cell(dh, i)
    unsafe_copyto!(global_dofs, 1, dh.cell_dofs.values, dh.cell_dofs.offset[i], length(global_dofs))
    return global_dofs
end

"""
    celldofs(dh::AbstractDofHandler, i::Int)

Return a vector with the degrees of freedom that belong to cell `i`.

See also [`celldofs!`](@ref).
"""
function celldofs(dh::MixedDofHandler, i::Int)
    @assert isclosed(dh)
    return dh.cell_dofs[i]
end

#TODO: perspectively remove in favor of `getcoordinates!(global_coords, grid, i)`?
function cellcoords!(global_coords::Vector{Vec{dim,T}}, dh::MixedDofHandler, i::Union{Int, <:AbstractCell}) where {dim,T}
    cellcoords!(global_coords, dh.grid, i)
end

function cellnodes!(global_nodes::Vector{Int}, dh::MixedDofHandler, i::Union{Int, <:AbstractCell})
    cellnodes!(global_nodes, dh.grid, i)
end

"""
    getfieldnames(dh::MixedDofHandler)
    getfieldnames(fh::FieldHandler)

Return a vector with the names of all fields. Can be used as an iterable over all the fields
in the problem.
"""
function getfieldnames(dh::MixedDofHandler)
    fieldnames = Vector{Symbol}()
    for fh in dh.fieldhandlers
        append!(fieldnames, getfieldnames(fh))
    end
    return unique!(fieldnames)
end

getfielddim(fh::FieldHandler, field_idx::Int) = fh.fields[field_idx].dim
getfielddim(fh::FieldHandler, field_name::Symbol) = getfielddim(fh, find_field(fh, field_name))

"""
    getfielddim(dh::MixedDofHandler, field_idxs::NTuple{2,Int})
    getfielddim(dh::MixedDofHandler, field_name::Symbol)
    getfielddim(dh::FieldHandler, field_idx::Int)
    getfielddim(dh::FieldHandler, field_name::Symbol)

Return the dimension of a given field. The field can be specified by its index (see
[`find_field`](@ref)) or its name.
"""
function getfielddim(dh::MixedDofHandler, field_idxs::NTuple{2, Int})
    fh_idx, field_idx = field_idxs
    fielddim = getfielddim(dh.fieldhandlers[fh_idx], field_idx)
    return fielddim
end
getfielddim(dh::MixedDofHandler, name::Symbol) = getfielddim(dh, find_field(dh, name))

"""
    nfields(dh::MixedDofHandler)

Returns the number of unique fields defined.
"""
nfields(dh::MixedDofHandler) = length(getfieldnames(dh))

"""
    add!(dh::MixedDofHandler, fh::FieldHandler)

Add all fields of the [`FieldHandler`](@ref) `fh` to `dh`.
"""
function add!(dh::MixedDofHandler, fh::FieldHandler)
    # TODO: perhaps check that a field with the same name is the same field?
    @assert !isclosed(dh)
    _check_same_celltype(dh.grid, fh.cellset)
    _check_cellset_intersections(dh, fh)
    # the field interpolations should have the same refshape as the cells they are applied to
    refshapes_fh = getrefshape.(getfieldinterpolations(fh))
    # extract the celltype from the first cell as the celltypes are all equal
    cell_type = typeof(dh.grid.cells[first(fh.cellset)])
    refshape_cellset = getrefshape(default_interpolation(cell_type))
    for refshape in refshapes_fh
        refshape_cellset == refshape || error("The RefShapes of the fieldhandlers interpolations must correspond to the RefShape of the cells it is applied to.")
    end

    push!(dh.fieldhandlers, fh)
    return dh
end

function _check_cellset_intersections(dh::MixedDofHandler, fh::FieldHandler)
    for _fh in dh.fieldhandlers
        isdisjoint(_fh.cellset, fh.cellset) || error("Each cell can only belong to a single FieldHandler.")
    end
end

function add!(dh::MixedDofHandler, name::Symbol, dim::Int, ip::Interpolation=default_interpolation(celltype))
    @assert !isclosed(dh)
    @assert isconcretetype(getcelltype(dh.grid)) "If you have more than one celltype in Grid, you must use add!(dh::MixedDofHandler, fh::FieldHandler)"

    if length(dh.fieldhandlers) == 0
        push!(dh.fieldhandlers, FieldHandler{UnitRange{Int}}(Field[Field(name,ip,dim)], 1:getncells(dh.grid)))
    elseif length(dh.fieldhandlers) == 1
        fh = first(dh.fieldhandlers)
        @assert name ∉ [field.name for field ∈ fh.fields]
        push!(fh.fields, Field(name,ip,dim))
    else
        error("If you have more than one FieldHandler, you must specify field")
    end

    return dh
end

"""
    close!(dh::AbstractDofHandler)

Closes `dh` and creates degrees of freedom for each cell.

If there are several fields, the dofs are added in the following order:
For a `MixedDofHandler`, go through each `FieldHandler` in the order they were added.
For each field in the `FieldHandler` or in the `DofHandler` (again, in the order the fields were added),
create dofs for the cell.
This means that dofs on a particular cell, the dofs will be numbered according to the fields;
first dofs for field 1, then field 2, etc.
"""
function close!(dh::MixedDofHandler)
    dh, _, _, _ = __close!(dh)
    return dh
end

function __close!(dh::MixedDofHandler{dim}) where {dim}
    @assert !isclosed(dh)
    field_names = getfieldnames(dh)  # all the fields in the problem
    numfields =  length(field_names)

    # Create dicts that store created dofs
    # Each key should uniquely identify the given type
    vertexdicts = [Dict{Int, UnitRange{Int}}() for _ in 1:numfields]
    edgedicts = [Dict{Tuple{Int,Int}, UnitRange{Int}}() for _ in 1:numfields]
    facedicts = [Dict{NTuple{dim,Int}, UnitRange{Int}}() for _ in 1:numfields]
    celldicts = [Dict{Int, UnitRange{Int}}() for _ in 1:numfields]

    # Set initial values
    nextdof = 1  # next free dof to distribute

    @debug "\n\nCreating dofs\n"
    for fh in dh.fieldhandlers
        nextdof = _close!(
            dh,
            fh.cellset,
            field_names,
            getfieldnames(fh),
            getfielddims(fh),
            getfieldinterpolations(fh),
            nextdof,
            vertexdicts,
            edgedicts,
            facedicts,
            celldicts)
    end
    dh.ndofs[] = maximum(dh.cell_dofs.values)
    dh.closed[] = true

    return dh, vertexdicts, edgedicts, facedicts

end

"""
Slow path if the FieldHandler has Sets to describe the subdomain. 
Here we convert the Set to a Vector to make it sortable because we want different collections to give the exact same dof distribution.
Note that a Set is in general not sortable.
"""
function _close!(dh::MixedDofHandler{dim}, cellnumbers::AbstractSet{Int}, global_field_names, field_names, field_dims, field_interpolations, nextdof, vertexdicts, edgedicts, facedicts, celldicts) where {dim}
    return _close!(
            dh,
            collect(cellnumbers),
            global_field_names,
            field_names,
            field_dims,
            field_interpolations,
            nextdof,
            vertexdicts,
            edgedicts,
            facedicts,
            celldicts)
end

function _close!(dh::MixedDofHandler{dim}, cellnumbers, global_field_names, field_names, field_dims, field_interpolations, nextdof, vertexdicts, edgedicts, facedicts, celldicts) where {dim}
    ip_infos = InterpolationInfo[]
    for interpolation in field_interpolations
        ip_info = InterpolationInfo(interpolation)
        # these are not implemented yet (or have not been tested)
        @assert(all(ip_info.nedgedofs .<= 1))
        @assert(all(ip_info.nfacedofs .<= 1))
        push!(ip_infos, ip_info)
    end
    
    # We have to ensure that the cell numbering is sorted in ascending order to allow the same dof distribution across different collections.
    sort!(cellnumbers)
    
    # loop over all the cells, and distribute dofs for all the fields
    cell_dofs = Int[]  # list of global dofs for each cell
    for ci in cellnumbers
        dh.cell_dofs.offset[ci] = length(dh.cell_dofs.values)+1

        cell = dh.grid.cells[ci]
        empty!(cell_dofs)
        @debug "Creating dofs for cell #$ci"

        for (local_num, field_name) in enumerate(field_names)
            fi = findfirst(i->i == field_name, global_field_names)
            @debug "\tfield: $(field_name)"
            ip_info = ip_infos[local_num]

            # We first distribute the vertex dofs
            nextdof = add_vertex_dofs(cell_dofs, cell, vertexdicts[fi], field_dims[local_num], ip_info.nvertexdofs, nextdof)

            # Then the edge dofs
            if dim == 3
                if ip_info.dim == 3 # Regular 3D element
                    nextdof = add_edge_dofs(cell_dofs, cell, edgedicts[fi], field_dims[local_num], ip_info.nedgedofs, nextdof)
                elseif ip_info.dim == 2 # 2D embedded element in 3D
                    nextdof = add_edge_dofs(cell_dofs, cell, edgedicts[fi], field_dims[local_num], ip_info.nfacedofs, nextdof)
                end
            end

            # Then the face dofs
            if ip_info.dim == dim # Regular 3D element
                nextdof = add_face_dofs(cell_dofs, cell, facedicts[fi], field_dims[local_num], ip_info.nfacedofs, nextdof)
            end

            # And finally the celldofs
            nextdof = add_cell_dofs(cell_dofs, ci, celldicts[fi], field_dims[local_num], ip_info.ncelldofs, nextdof)
        end
        # after done creating dofs for the cell, push them to the global list
        append!(dh.cell_dofs.values, cell_dofs)
        dh.cell_dofs.length[ci] = length(cell_dofs)

        @debug "Dofs for cell #$ci:\n\t$cell_dofs"
    end # cell loop
    return nextdof
end

"""
Returns the next global dof number and an array of dofs.
If dofs have already been created for the object (vertex, face) then simply return those, otherwise create new dofs.
"""
function get_or_create_dofs!(nextdof, field_dim; dict, key)
    token = Base.ht_keyindex2!(dict, key)
    if token > 0  # vertex, face etc. visited before
        # reuse stored dofs (TODO unless field is discontinuous)
        @debug "\t\tkey: $key dofs: $(dict[key])  (reused dofs)"
        return nextdof, dict[key]
    else  # create new dofs
        dofs = nextdof : (nextdof + field_dim-1)
        @debug "\t\tkey: $key dofs: $dofs"
        Base._setindex!(dict, dofs, key, -token) #
        nextdof += field_dim
        return nextdof, dofs
    end
end

function add_vertex_dofs(cell_dofs, cell, vertexdict, field_dim, nvertexdofs, nextdof)
    for (vi, vertex) in enumerate(vertices(cell))
        if nvertexdofs[vi] > 0
            @debug "\tvertex #$vertex"
            nextdof, dofs = get_or_create_dofs!(nextdof, field_dim, dict=vertexdict, key=vertex)
            append!(cell_dofs, dofs)
        end
    end
    return nextdof
end

function add_face_dofs(cell_dofs, cell, facedict, field_dim, nfacedofs, nextdof)
    @debug @assert all(nfacedofs .<= 1) "Currently only supports interpolations with less that 2 dofs per face"

    for (fi,face) in enumerate(faces(cell))
        if nfacedofs[fi] > 0
            sface = sortface(face)
            @debug "\tface #$sface"
            nextdof, dofs = get_or_create_dofs!(nextdof, field_dim, dict=facedict, key=sface)
            # TODO permutate dofs according to face orientation
            append!(cell_dofs, dofs)
        end
    end
    return nextdof
end

function add_edge_dofs(cell_dofs, cell, edgedict, field_dim, nedgedofs, nextdof)
    for (ei,edge) in enumerate(edges(cell))
        if nedgedofs[ei] > 0
            sedge, dir = sortedge(edge)
            @debug "\tedge #$sedge"
            nextdof, dofs = get_or_create_dofs!(nextdof, field_dim, dict=edgedict, key=sedge)
            append!(cell_dofs, dofs)
        end
    end
    return nextdof
end

function add_cell_dofs(cell_dofs, cell, celldict, field_dim, ncelldofs, nextdof)
    for celldof in 1:ncelldofs
        @debug "\tcell #$cell"
        nextdof, dofs = get_or_create_dofs!(nextdof, field_dim, dict=celldict, key=cell)
        append!(cell_dofs, dofs)
    end
    return nextdof
end

"""
    find_field(dh::MixedDofHandler, field_name::Symbol)::NTuple{2,Int}

Return the index of the field with name `field_name` in a `MixedDofHandler`. The index is a
`NTuple{2,Int}`, where the 1st entry is the index of the `FieldHandler` within which the
field was found and the 2nd entry is the index of the field within the `FieldHandler`.

!!! note
    Always finds the 1st occurence of a field within `MixedDofHandler`.

See also: [`find_field(fh::FieldHandler, field_name::Symbol)`](@ref),
[`_find_field(fh::FieldHandler, field_name::Symbol)`](@ref).
"""
function find_field(dh::MixedDofHandler, field_name::Symbol)
    for (fh_idx, fh) in pairs(dh.fieldhandlers)
        field_idx = _find_field(fh, field_name)
        !isnothing(field_idx) && return (fh_idx, field_idx)
    end
    error("Did not find field :$field_name (existing fields: $(getfieldnames(dh))).")
end

"""
    find_field(fh::FieldHandler, field_name::Symbol)::Int

Return the index of the field with name `field_name` in a `FieldHandler`. Throw an
error if the field is not found.

See also: [`find_field(dh::MixedDofHandler, field_name::Symbol)`](@ref), [`_find_field(fh::FieldHandler, field_name::Symbol)`](@ref).
"""
function find_field(fh::FieldHandler, field_name::Symbol)
    field_idx = _find_field(fh, field_name)
    if field_idx === nothing
        error("Did not find field :$field_name in FieldHandler (existing fields: $(getfieldnames(fh)))")
    end
    return field_idx
end

# No error if field not found
"""
    _find_field(fh::FieldHandler, field_name::Symbol)::Int

Return the index of the field with name `field_name` in the `FieldHandler` `fh`. Return 
`nothing` if the field is not found.

See also: [`find_field(dh::MixedDofHandler, field_name::Symbol)`](@ref), [`find_field(fh::FieldHandler, field_name::Symbol)`](@ref).
"""
function _find_field(fh::FieldHandler, field_name::Symbol)
    for (field_idx, field) in pairs(fh.fields)
        if field.name == field_name
            return field_idx
        end
    end
    return nothing
end

# Calculate the offset to the first local dof of a field
function field_offset(fh::FieldHandler, field_idx::Int)
    offset = 0
    for i in 1:(field_idx-1)
        offset += getnbasefunctions(fh.fields[i].interpolation)::Int * fh.fields[i].dim
    end
    return offset
end
field_offset(fh::FieldHandler, field_name::Symbol) = field_offset(fh, find_field(fh, field_name))

field_offset(dh::MixedDofHandler, field_name::Symbol) = field_offset(dh, find_field(dh, field_name))
function field_offset(dh::MixedDofHandler, field_idxs::Tuple{Int, Int})
    fh_idx, field_idx = field_idxs
    field_offset(dh.fieldhandlers[fh_idx], field_idx)
end

"""
    dof_range(fh::FieldHandler, field_idx::Int)
    dof_range(fh::FieldHandler, field_name::Symbol)
    dof_range(dh:MixedDofHandler, field_name::Symbol)

Return the local dof range for a given field. The field can be specified by its name or
index, where `field_idx` represents the index of a field within a `FieldHandler` and
`field_idxs` is a tuple of the `FieldHandler`-index within the `MixedDofHandler` and the
`field_idx`.

!!! note
    The `dof_range` of a field can vary between different `FieldHandler`s. Therefore, it is
    advised to use the `field_idxs` or refer to a given `FieldHandler` directly in case
    several `FieldHandler`s exist. Using the `field_name` will always refer to the first
    occurence of `field` within the `MixedDofHandler`.

Example:
```jldoctest
julia> grid = generate_grid(Triangle, (3, 3))
Grid{2, Triangle, Float64} with 18 Triangle cells and 16 nodes

julia> dh = MixedDofHandler(grid); add!(dh, :u, 3); add!(dh, :p, 1); close!(dh);

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
    field_interpolation = fh.fields[field_idx].interpolation
    field_dim = fh.fields[field_idx].dim
    n_field_dofs = getnbasefunctions(field_interpolation)::Int * field_dim
    return (offset+1):(offset+n_field_dofs)
end
dof_range(fh::FieldHandler, field_name::Symbol) = dof_range(fh, find_field(fh, field_name))

function dof_range(dh::MixedDofHandler, field_name::Symbol)
    if length(dh.fieldhandlers) > 1
        error("The given MixedDofHandler has $(length(dh.fieldhandlers)) FieldHandlers.
              Extracting the dof range based on the fieldname might not be a unique problem
              in this case. Use `dof_range(fh::FieldHandler, field_name)` instead.")
    end
    fh_idx, field_idx = find_field(dh, field_name)
    return dof_range(dh.fieldhandlers[fh_idx], field_idx)
end

"""
    getfieldinterpolation(dh::MixedDofHandler, field_idxs::NTuple{2,Int})
    getfieldinterpolation(dh::FieldHandler, field_idx::Int)
    getfieldinterpolation(dh::FieldHandler, field_name::Symbol)

Return the interpolation of a given field. The field can be specified by its index (see
[`find_field`](@ref) or its name.
"""
function getfieldinterpolation(dh::MixedDofHandler, field_idxs::NTuple{2,Int})
    fh_idx, field_idx = field_idxs
    ip = dh.fieldhandlers[fh_idx].fields[field_idx].interpolation
    return ip
end
getfieldinterpolation(fh::FieldHandler, field_idx::Int) = fh.fields[field_idx].interpolation
getfieldinterpolation(fh::FieldHandler, field_name::Symbol) = getfieldinterpolation(fh, find_field(fh, field_name))

function reshape_to_nodes(dh::MixedDofHandler, u::Vector{T}, fieldname::Symbol) where T
    # make sure the field exists
    fieldname ∈ getfieldnames(dh) || error("Field $fieldname not found.")

    field_dim = getfielddim(dh, fieldname)
    space_dim = field_dim == 2 ? 3 : field_dim
    data = fill(T(NaN), space_dim, getnnodes(dh.grid))  # set default value

    for fh in dh.fieldhandlers
        # check if this fh contains this field, otherwise continue to the next
        field_pos = findfirst(i->i == fieldname, getfieldnames(fh))
        field_pos === nothing && continue
        offset = field_offset(fh, fieldname)

        reshape_field_data!(data, dh, u, offset, field_dim, fh.cellset)
    end
    return data
end
