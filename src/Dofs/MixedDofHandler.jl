
mutable struct FieldHandler
    fields::Vector{Field}
    cellset::Set{Int}
    bc_values::Vector{BCValues} # for boundary conditions
end
"""
    FieldHandler(fields::Vector{Field}, cellset)

Construct a `FieldHandler` based on an array of `Field`s and assigns it a set of cells.
"""
function FieldHandler(fields::Vector{Field}, cellset)
    # TODO for now, only accept isoparamtric mapping
    bc_values = [BCValues(field.interpolation, field.interpolation) for field in fields]
    FieldHandler(fields, cellset, bc_values)
end


struct CellDofs
    dofs::Vector{Int}
    offset::Vector{Int}
    length::Vector{Int}
end

import Base.getindex
function getindex(eldofs::CellDofs, el::Int64)
     return eldofs.dofs[eldofs.offset[el]:eldofs.offset[el]+eldofs.length[el]-1]
 end


struct MixedDofHandler{dim,C,T} <: JuAFEM.AbstractDofHandler
    fieldhandlers::Vector{FieldHandler}
    cell_dofs::CellDofs
    closed::ScalarWrapper{Bool}
    grid::MixedGrid{dim,C,T}
end

function MixedDofHandler(grid::MixedGrid)
    MixedDofHandler(FieldHandler[], CellDofs([],[],[]), JuAFEM.ScalarWrapper(false), grid)
end

getfieldnames(fh::FieldHandler) = [field.name for field in fh.fields]
getfielddims(fh::FieldHandler) = [field.dim for field in fh.fields]
getfieldinterpolations(fh::FieldHandler) = [field.interpolation for field in fh.fields]
import JuAFEM.ndofs_per_cell, JuAFEM.celldofs, JuAFEM.isclosed, JuAFEM.ndofs
ndofs_per_cell(dh::MixedDofHandler, cell::Int64) = dh.cell_dofs.length[cell]
ndofs(dh::MixedDofHandler) = maximum(dh.cell_dofs.dofs)

function celldofs!(global_dofs::Vector{Int}, dh::MixedDofHandler, i::Int)
    @assert isclosed(dh)
    @assert length(global_dofs) == ndofs_per_cell(dh, i)
    unsafe_copyto!(global_dofs, 1, dh.cell_dofs.dofs, dh.cell_dofs.offset[i], length(global_dofs))
    return global_dofs
end

function celldofs(dh::MixedDofHandler, i::Int)
    @assert isclosed(dh)
    return dh.cell_dofs[i]
end

"""
    getfieldnames(dh::MixedDofHandler)

Returns the union of all the fields. Can be used as an iterable over all the fields in the problem.
"""
function getfieldnames(dh::MixedDofHandler)
    fieldnames = Vector{Symbol}()
    for fh in dh.fieldhandlers
        for name in getfieldnames(fh)
            push!(fieldnames, name)
        end
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
        if field_pos > 0
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

function Base.push!(dh::MixedDofHandler, fh::FieldHandler)
    @assert !isclosed(dh)
    push!(dh.fieldhandlers, fh)
    return dh
end

"""
    close!(dh::MixedDofHandler)

Closes the dofhandler and creates degrees of freedom for each cell.
Dofs are created in the following order: Go through each FieldHandler in the order they were added. For each field in the FieldHandler, create dofs for the cell. This means that dofs on a particular cell will be numbered according to the fields; first dofs for field 1, then field 2, etc.
"""
function close!(dh::MixedDofHandler{dim}) where {dim}

    @assert !JuAFEM.isclosed(dh)
    field_names = JuAFEM.getfieldnames(dh)  # all the fields in the problem
    numfields =  length(field_names)

    # Create dicts that stores created dofs
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
            JuAFEM.getfieldnames(fh),
            JuAFEM.getfielddims(fh),
            JuAFEM.getfieldinterpolations(fh),
            nextdof,
            vertexdicts,
            edgedicts,
            facedicts,
            celldicts)
    end
    dh.closed[] = true
    return dh
end

function _close!(dh::MixedDofHandler{dim}, cellnumbers, field_names, field_dims, field_interpolations, nextdof, vertexdicts, edgedicts, facedicts, celldicts) where {dim}

    ip_infos = JuAFEM.InterpolationInfo[]
    for interpolation in field_interpolations
        ip_info = JuAFEM.InterpolationInfo(interpolation)
        # these are implemented yet (or have not been tested)
        @assert(ip_info.nvertexdofs <= 1)
        @assert(ip_info.nedgedofs <= 1)
        @assert(ip_info.nfacedofs <= 1)
        @assert(ip_info.ncelldofs <= 1)  # not tested but probably works
        push!(ip_infos, ip_info)
    end

    # loop over all the cells, and distribute dofs for all the fields
    for ci in cellnumbers
        #dh.cell_dofs_offset[ci] = nextdof
        dh.cell_dofs.offset[ci] = length(dh.cell_dofs.dofs)+1

        cell = dh.grid.cells[ci]
        cell_dofs = Int[]  # list of global dofs for each cell
        @debug "Creating dofs for cell #$ci"

        for fi in 1:length(field_names)
            @debug "\tfield: $(field_names[fi])"
            ip_info = ip_infos[fi]

            if ip_info.nvertexdofs > 0
                nextdof = add_vertex_dofs(cell_dofs, cell, vertexdicts[fi], field_dims[fi], ip_info.nvertexdofs, nextdof)
            end

            if ip_info.nfacedofs > 0
                nextdof = add_face_dofs(cell_dofs, cell, facedicts[fi], field_dims[fi], ip_info.nfacedofs, nextdof)
            end

            if ip_info.nedgedofs > 0
                nextdof = add_edge_dofs(cell_dofs, cell, edgedicts[fi], field_dims[fi], ip_info.nedgedofs, nextdof)
            end

            if ip_info.ncelldofs > 0
                nextdof = add_cell_dofs(cell_dofs, ci, celldicts[fi], field_dims[fi], ip_info.ncelldofs, nextdof)
            end

        end
        # after done creating dofs for the cell, push them to the global list
        push!(dh.cell_dofs.dofs, cell_dofs...)
        dh.cell_dofs.length[ci] = length(cell_dofs)
        #push!(dh.cell_dofs, cell_dofs...)
        # push! the first index of the next cell to the offset vector
        #dh.cell_dofs_offset[ci] = nextdof
        #dh.cell_dofs_offset[ci+1] = length(dh.cell_dofs)+1
        #push!(dh.cell_dofs_offset, length(dh.cell_dofs)+1)


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
    for vertex in JuAFEM.vertices(cell)
        @debug "\tvertex #$vertex"
        nextdof, dofs = get_or_create_dofs!(nextdof, field_dim, dict=vertexdict, key=vertex)
        push!(cell_dofs, dofs...)
    end
    return nextdof
end

function add_face_dofs(cell_dofs, cell, facedict, field_dim, nfacedofs, nextdof)
    @assert nfacedofs == 1 "Currently only supports interpolations with nfacedofs = 1"

    for face in JuAFEM.faces(cell)
        sface = JuAFEM.sortface(face)
        @debug "\tface #$sface"
        nextdof, dofs = get_or_create_dofs!(nextdof, field_dim, dict=facedict, key=sface)
        push!(cell_dofs, dofs...)
    end
    return nextdof
end

function add_edge_dofs(cell_dofs, cell, edgedict, field_dim, nedgedofs, nextdof)
    @assert nedgedofs == 1 "Currently only supports interpolations with nedgedofs = 1"
    for edge in JuAFEM.edges(cell)
        sedge, dir = JuAFEM.sortedge(edge)
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
function _create_sparsity_pattern(dh::MixedDofHandler, sym::Bool)
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
    V = zeros(length(I))
    K = sparse(I, J, V)
    return K

end

@inline create_sparsity_pattern(dh::MixedDofHandler) = _create_sparsity_pattern(dh, false)


function find_field(fh::FieldHandler, field_name::Symbol)
    j = findfirst(i->i == field_name, getfieldnames(fh))
    j == 0 && error("did not find field $field_name")
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
