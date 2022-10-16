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
mutable struct FieldHandler
    fields::Vector{Field}
    cellset::Set{Int}
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
    cell_nodes::CellVector{Int}
    cell_coords::CellVector{Vec{dim,T}}
    closed::ScalarWrapper{Bool}
    grid::G
    ndofs::ScalarWrapper{Int}
end

function MixedDofHandler(grid::Grid{dim,C,T}) where {dim,C,T}
    MixedDofHandler{dim,T,typeof(grid)}(FieldHandler[], CellVector(Int[],Int[],Int[]), CellVector(Int[],Int[],Int[]), CellVector(Vec{dim,T}[],Int[],Int[]), Ferrite.ScalarWrapper(false), grid, Ferrite.ScalarWrapper(-1))
end

getfieldnames(fh::FieldHandler) = [field.name for field in fh.fields]
getfielddims(fh::FieldHandler) = [field.dim for field in fh.fields]
getfieldinterpolations(fh::FieldHandler) = [field.interpolation for field in fh.fields]
ndofs_per_cell(dh::MixedDofHandler, cell::Int=1) = dh.cell_dofs.length[cell]
nnodes_per_cell(dh::MixedDofHandler, cell::Int=1) = dh.cell_nodes.length[cell]


function celldofs!(global_dofs::Vector{Int}, dh::MixedDofHandler, i::Int)
    @assert isclosed(dh)
    @assert length(global_dofs) == ndofs_per_cell(dh, i)
    unsafe_copyto!(global_dofs, 1, dh.cell_dofs.values, dh.cell_dofs.offset[i], length(global_dofs))
    return global_dofs
end

function celldofs(dh::MixedDofHandler, i::Int)
    @assert isclosed(dh)
    return dh.cell_dofs[i]
end

function cellcoords!(global_coords::Vector{Vec{dim,T}}, dh::MixedDofHandler, i::Int) where {dim,T}
    @assert isclosed(dh)
    @assert length(global_coords) == nnodes_per_cell(dh, i)
    unsafe_copyto!(global_coords, 1, dh.cell_coords.values, dh.cell_coords.offset[i], length(global_coords))
    return global_coords
end

function cellnodes!(global_nodes::Vector{Int}, dh::MixedDofHandler, i::Int)
    @assert isclosed(dh)
    @assert length(global_nodes) == nnodes_per_cell(dh, i)
    unsafe_copyto!(global_nodes, 1, dh.cell_nodes.values, dh.cell_nodes.offset[i], length(global_nodes))
    return global_nodes
end


"""
    getfieldnames(dh::MixedDofHandler)

Returns the union of all the fields. Can be used as an iterable over all the fields in the problem.
"""
function getfieldnames(dh::MixedDofHandler)
    fieldnames = Vector{Symbol}()
    for fh in dh.fieldhandlers
        append!(fieldnames, getfieldnames(fh))
    end
    return unique!(fieldnames)
end

"""
    getfielddim(dh::MixedDofHandler, name::Symbol)

Returns the dimension of a specific field, given by name. Note that it will return the dimension of the first field found among the `FieldHandler`s.
"""
function getfielddim(dh::MixedDofHandler, name::Symbol)

    for fh in dh.fieldhandlers
        field_pos = findfirst(i->i == name, getfieldnames(fh))
        if field_pos !== nothing
            return fh.fields[field_pos].dim
        end
    end
    error("did not find field $name")
end


"""
    nfields(dh::MixedDofHandler)

Returns the number of unique fields defined.
"""
nfields(dh::MixedDofHandler) = length(getfieldnames(dh))

"""
    push!(dh::MixedDofHandler, fh::FieldHandler)

Add all fields of the [`FieldHandler`](@ref) `fh` to `dh`.
"""
function Base.push!(dh::MixedDofHandler, fh::FieldHandler)
    @assert !isclosed(dh)
    _check_same_celltype(dh.grid, collect(fh.cellset))
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

function Base.push!(dh::MixedDofHandler, name::Symbol, dim::Int)
    celltype = getcelltype(dh.grid)
    isconcretetype(celltype) || error("If you have more than one celltype in Grid, you must use push!(dh::MixedDofHandler, fh::FieldHandler)")
    push!(dh, name, dim, default_interpolation(celltype))
end

function Base.push!(dh::MixedDofHandler, name::Symbol, dim::Int, ip::Interpolation)
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

    field = Field(name,ip,dim)

    push!(fh.fields, field)

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

    @assert !Ferrite.isclosed(dh)
    field_names = Ferrite.getfieldnames(dh)  # all the fields in the problem
    numfields =  length(field_names)

    # Create dicts that store created dofs
    # Each key should uniquely identify the given type
    vertexdicts = [Dict{Int, Array{Int}}() for _ in 1:numfields]
    edgedicts = [Dict{Tuple{Int,Int}, Array{Int}}() for _ in 1:numfields]
    facedicts = [Dict{Tuple{Int,Int}, Array{Int}}() for _ in 1:numfields]
    celldicts = [Dict{Int, Array{Int}}() for _ in 1:numfields]

    # Set initial values
    nextdof = 1  # next free dof to distribute

    append!(dh.cell_dofs.offset, zeros(getncells(dh.grid)))
    append!(dh.cell_dofs.length, zeros(getncells(dh.grid)))

    @debug "\n\nCreating dofs\n"
    for fh in dh.fieldhandlers
        # sort the cellset since we want to loop through the cells in a fixed order
        cellnumbers = sort(collect(fh.cellset))
        nextdof = _close!(
            dh,
            cellnumbers,
            field_names,
            Ferrite.getfieldnames(fh),
            Ferrite.getfielddims(fh),
            Ferrite.getfieldinterpolations(fh),
            nextdof,
            vertexdicts,
            edgedicts,
            facedicts,
            celldicts)
    end
    dh.ndofs[] = maximum(dh.cell_dofs.values)
    dh.closed[] = true

    #Create cell_nodes and cell_coords (similar to cell_dofs)
    push!(dh.cell_nodes.offset, 1)
    push!(dh.cell_coords.offset, 1)
    for cell in dh.grid.cells
        for nodeid in cell.nodes
            push!(dh.cell_nodes.values, nodeid)
            push!(dh.cell_coords.values, dh.grid.nodes[nodeid].x)
        end
        push!(dh.cell_nodes.offset, length(dh.cell_nodes.values)+1)
        push!(dh.cell_coords.offset, length(dh.cell_coords.values)+1)
        push!(dh.cell_nodes.length, length(cell.nodes))
        push!(dh.cell_coords.length, length(cell.nodes))
    end

    return dh, vertexdicts, edgedicts, facedicts

end

function _close!(dh::MixedDofHandler{dim}, cellnumbers, global_field_names, field_names, field_dims, field_interpolations, nextdof, vertexdicts, edgedicts, facedicts, celldicts) where {dim}

    ip_infos = Ferrite.InterpolationInfo[]
    for interpolation in field_interpolations
        ip_info = Ferrite.InterpolationInfo(interpolation)
        # these are not implemented yet (or have not been tested)
        @assert(ip_info.nvertexdofs <= 1)
        @assert(ip_info.nedgedofs <= 1)
        @assert(ip_info.nfacedofs <= 1)
        @assert(ip_info.ncelldofs <= 1)  # not tested but probably works
        push!(ip_infos, ip_info)
    end

    # loop over all the cells, and distribute dofs for all the fields
    for ci in cellnumbers
        dh.cell_dofs.offset[ci] = length(dh.cell_dofs.values)+1

        cell = dh.grid.cells[ci]
        cell_dofs = Int[]  # list of global dofs for each cell
        @debug "Creating dofs for cell #$ci"

        for (local_num, field_name) in enumerate(field_names)
            fi = findfirst(i->i == field_name, global_field_names)
            @debug "\tfield: $(field_name)"
            ip_info = ip_infos[local_num]

            if ip_info.nvertexdofs > 0
                nextdof = add_vertex_dofs(cell_dofs, cell, vertexdicts[fi], field_dims[local_num], ip_info.nvertexdofs, nextdof)
            end

            if ip_info.nedgedofs > 0 && dim == 3 #Edges only in 3d
                nextdof = add_edge_dofs(cell_dofs, cell, edgedicts[fi], field_dims[local_num], ip_info.nedgedofs, nextdof)
            end

            if ip_info.nfacedofs > 0 && (ip_info.dim == dim)
                nextdof = add_face_dofs(cell_dofs, cell, facedicts[fi], field_dims[local_num], ip_info.nfacedofs, nextdof)
            end

            if ip_info.ncelldofs > 0
                nextdof = add_cell_dofs(cell_dofs, ci, celldicts[fi], field_dims[local_num], ip_info.ncelldofs, nextdof)
            end

        end
        # after done creating dofs for the cell, push them to the global list
        push!(dh.cell_dofs.values, cell_dofs...)
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
        dofs = collect(nextdof : nextdof + field_dim-1)
        @debug "\t\tkey: $key dofs: $dofs"
        Base._setindex!(dict, dofs, key, -token) #
        nextdof += field_dim
        return nextdof, dofs
    end
end

function add_vertex_dofs(cell_dofs, cell, vertexdict, field_dim, nvertexdofs, nextdof)
    for vertex in Ferrite.vertices(cell)
        @debug "\tvertex #$vertex"
        nextdof, dofs = get_or_create_dofs!(nextdof, field_dim, dict=vertexdict, key=vertex)
        push!(cell_dofs, dofs...)
    end
    return nextdof
end

function add_face_dofs(cell_dofs, cell, facedict, field_dim, nfacedofs, nextdof)
    @assert nfacedofs == 1 "Currently only supports interpolations with nfacedofs = 1"

    for face in Ferrite.faces(cell)
        sface = Ferrite.sortface(face)
        @debug "\tface #$sface"
        nextdof, dofs = get_or_create_dofs!(nextdof, field_dim, dict=facedict, key=sface)
        push!(cell_dofs, dofs...)
    end
    return nextdof
end

function add_edge_dofs(cell_dofs, cell, edgedict, field_dim, nedgedofs, nextdof)
    @assert nedgedofs == 1 "Currently only supports interpolations with nedgedofs = 1"
    for edge in Ferrite.edges(cell)
        sedge, dir = Ferrite.sortedge(edge)
        @debug "\tedge #$sedge"
        nextdof, dofs = get_or_create_dofs!(nextdof, field_dim, dict=edgedict, key=sedge)
        push!(cell_dofs, dofs...)
    end
    return nextdof
end

function add_cell_dofs(cell_dofs, cell, celldict, field_dim, ncelldofs, nextdof)
    for celldof in 1:ncelldofs
        @debug "\tcell #$cell"
        nextdof, dofs = get_or_create_dofs!(nextdof, field_dim, dict=celldict, key=cell)
        push!(cell_dofs, dofs...)
    end
    return nextdof
end


# TODO if not too slow it can replace the "Grid-version"
function _create_sparsity_pattern(dh::MixedDofHandler, ch#=::Union{ConstraintHandler,Nothing}=#, sym::Bool)
    @assert isclosed(dh)

    ncells = getncells(dh.grid)
    N::Int = 0
    for element_id = 1:ncells  # TODO check for correctness
        n = ndofs_per_cell(dh, element_id)
        N += sym ? div(n*(n+1), 2) : n^2
    end
    N += ndofs(dh) # always add the diagonal elements
    I = Int[]; resize!(I, N)
    J = Int[]; resize!(J, N)

    cnt = 0
    for element_id in 1:ncells
        n = ndofs_per_cell(dh, element_id)
        global_dofs = zeros(Int, n)
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

    # If ConstraintHandler is given, create the condensation pattern due to affine constraints
    if ch !== nothing
        @assert isclosed(ch)

        V = ones(length(I))
        K = sparse(I, J, V, ndofs(dh), ndofs(dh))
        _condense_sparsity_pattern!(K, ch.dofcoefficients, ch.dofmapping)
        fill!(K.nzval, 0.0)
    else
        V = zeros(length(I))
        K = sparse(I, J, V, ndofs(dh), ndofs(dh))
    end
    return K
end

create_sparsity_pattern(dh::MixedDofHandler) = _create_sparsity_pattern(dh, nothing, false)
create_symmetric_sparsity_pattern(dh::MixedDofHandler) = Symmetric(_create_sparsity_pattern(dh, nothing, true), :U)

function find_field(fh::FieldHandler, field_name::Symbol)
    j = findfirst(i->i == field_name, getfieldnames(fh))
    j === nothing && error("could not find field :$field_name in FieldHandler (existing fields: $(getfieldnames(fh)))")
    return j
end

# Calculate the offset to the first local dof of a field
function field_offset(fh::FieldHandler, field_name::Symbol)
    offset = 0
    for i in 1:find_field(fh, field_name)-1
        offset += getnbasefunctions(getfieldinterpolations(fh)[i])::Int * getfielddims(fh)[i]
    end
    return offset
end

function Ferrite.dof_range(fh::FieldHandler, field_name::Symbol)
    f = Ferrite.find_field(fh, field_name)
    offset = Ferrite.field_offset(fh, field_name)
    field_interpolation = fh.fields[f].interpolation
    field_dim = fh.fields[f].dim
    n_field_dofs = getnbasefunctions(field_interpolation)::Int * field_dim
    return (offset+1):(offset+n_field_dofs)
end

find_field(dh::MixedDofHandler, field_name::Symbol) = find_field(first(dh.fieldhandlers), field_name)
field_offset(dh::MixedDofHandler, field_name::Symbol) = field_offset(first(dh.fieldhandlers), field_name)
getfieldinterpolation(dh::MixedDofHandler, field_idx::Int) = dh.fieldhandlers[1].fields[field_idx].interpolation
getfielddim(dh::MixedDofHandler, field_idx::Int) = dh.fieldhandlers[1].fields[field_idx].dim

function reshape_to_nodes(dh::MixedDofHandler, u::Vector{T}, fieldname::Symbol) where T

    # make sure the field exists
    fieldname ∈ Ferrite.getfieldnames(dh) || error("Field $fieldname not found.")

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
