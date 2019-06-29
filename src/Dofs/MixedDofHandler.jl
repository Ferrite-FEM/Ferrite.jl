
mutable struct FieldHandler
    fields::Vector{Field}
    cellset::Set{Int}
    bc_values::Vector{BCValues} # for boundary conditions
end

function FieldHandler(fields::Vector{Field}, cellset)
    # TODO for now, only accept isoparamtric mapping
    bc_values = [BCValues(field.interpolation, field.interpolation) for field in fields]
    FieldHandler(fields, cellset, bc_values)
end

struct MixedDofHandler{dim,C,T} <: JuAFEM.AbstractDofHandler
    fieldhandlers::Vector{FieldHandler}
    cell_dofs::Vector{Int}
    cell_dofs_offset::Vector{Int}
    closed::ScalarWrapper{Bool}
    grid::MixedGrid{dim,C,T}
end

function MixedDofHandler(grid::MixedGrid)
    MixedDofHandler(FieldHandler[], Int[], Int[], JuAFEM.ScalarWrapper(false), grid)
end

getfieldnames(fh::FieldHandler) = [field.name for field in fh.fields]
getfielddims(fh::FieldHandler) = [field.dim for field in fh.fields]
getfieldinterpolations(fh::FieldHandler) = [field.interpolation for field in fh.fields]

"""
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

function getfielddim(dh::MixedDofHandler, name::Symbol)

    for fh in dh.fieldhandlers
        field_pos = findfirst(i->i == name, getfieldnames(fh))
        if field_pos > 0
            return fh.fields[field_pos].dim
        end
    end
    error("did not find field $name")
end

nfields(dh::MixedDofHandler) = length(getfieldnames(dh))

function Base.push!(dh::MixedDofHandler, fh::FieldHandler)
    @assert !isclosed(dh)
    push!(dh.fieldhandlers, fh)
    return dh
end


function close!(dh::MixedDofHandler{dim}) where {dim}

    @assert !isclosed(dh)
    field_names = getfieldnames(dh)  # all the fields in the problem

    numfields =  length(field_names)

    vertexdicts = [Dict{Int,Int}() for _ in 1:numfields]
    edgedicts = [Dict{Tuple{Int,Int}, Tuple{Int,Bool}}() for _ in 1:numfields]
    facedicts = [Dict{NTuple{dim,Int}, Int}() for _ in 1:numfields]

    nextdof = 1 # next free dof to distribute
    for fh in dh.fieldhandlers
        # sort the cellset since we want to loop through the elements in a fixed order
        cellnumbers = sort(collect(fh.cellset))
        nextdof = JuAFEM._close!(
            dh,
            cellnumbers,
            getfieldnames(fh),
            getfielddims(fh),
            getfieldinterpolations(fh),
            nextdof,
            vertexdicts,
            edgedicts,
            facedicts)
    end
end

function _close!(dh::MixedDofHandler{dim}, cellnumbers, field_names, field_dims, field_interpolations, nextdof, vertexdicts, edgedicts, facedicts) where {dim}

        interpolation_infos = JuAFEM.InterpolationInfo[]
        for interpolation in field_interpolations
            push!(interpolation_infos, JuAFEM.InterpolationInfo(interpolation))
        end
        if length(dh.cell_dofs_offset) == 0
            push!(dh.cell_dofs_offset, nextdof) # if this is the first dof
        end

        # not implemented yet: more than one facedof per face in 3D
        dim == 3 && @assert(!any(x->x.nfacedofs > 1, interpolation_infos))

        # loop over all the cells, and distribute dofs for all the fields
        for ci in cellnumbers
            cell = dh.grid.cells[ci]
            @debug println("cell #$ci")
            for fi in 1:length(field_names)
                interpolation_info = interpolation_infos[fi]
                @debug println("  field: $(fh.field_names[fi])")
                if interpolation_info.nvertexdofs > 0
                    for vertex in vertices(cell)
                        @debug println("    vertex#$vertex")
                        token = Base.ht_keyindex2!(vertexdicts[fi], vertex)
                        if token > 0 # haskey(vertexdicts[fi], vertex) # reuse dofs
                            reuse_dof = vertexdicts[fi].vals[token] # vertexdicts[fi][vertex]
                            for d in 1:field_dims[fi]
                                @debug println("      reusing dof #$(reuse_dof + (d-1))")
                                push!(dh.cell_dofs, reuse_dof + (d-1))
                            end
                        else # token <= 0, distribute new dofs
                            for vertexdof in 1:interpolation_info.nvertexdofs
                                Base._setindex!(vertexdicts[fi], nextdof, vertex, -token) # vertexdicts[fi][vertex] = nextdof
                                for d in 1:field_dims[fi]
                                    @debug println("      adding dof#$nextdof")
                                    push!(dh.cell_dofs, nextdof)
                                    nextdof += 1
                                end
                            end
                        end
                    end # vertex loop
                end
                # TODO Add the following
                # if dim == 3 # edges only in 3D
                #     if interpolation_info.nedgedofs > 0
                #         for edge in edges(cell)
                #             sedge, dir = sortedge(edge)
                #             @debug println("    edge#$sedge dir: $(dir)")
                #             token = Base.ht_keyindex2!(edgedicts[fi], sedge)
                #             if token > 0 # haskey(edgedicts[fi], sedge), reuse dofs
                #                 startdof, olddir = edgedicts[fi].vals[token] # edgedicts[fi][sedge] # first dof for this edge (if dir == true)
                #                 for edgedof in (dir == olddir ? (1:interpolation_info.nedgedofs) : (interpolation_info.nedgedofs:-1:1))
                #                     for d in 1:field_dims[fi]
                #                         reuse_dof = startdof + (d-1) + (edgedof-1)*field_dims[fi]
                #                         @debug println("      reusing dof#$(reuse_dof)")
                #                         push!(dh.cell_dofs, reuse_dof)
                #                     end
                #                 end
                #             else # token <= 0, distribute new dofs
                #                 Base._setindex!(edgedicts[fi], (nextdof, dir), sedge, -token) # edgedicts[fi][sedge] = (nextdof, dir),  store only the first dof for the edge
                #                 for edgedof in 1:interpolation_info.nedgedofs
                #                     for d in 1:field_dims[fi]
                #                         @debug println("      adding dof#$nextdof")
                #                         push!(dh.cell_dofs, nextdof)
                #                         nextdof += 1
                #                     end
                #                 end
                #             end
                #         end # edge loop
                #     end
                # end
                # if interpolation_info.nfacedofs > 0 # nfacedofs(interpolation) > 0
                #     for face in faces(cell)
                #         sface = sortface(face) # TODO: faces(cell) may as well just return the sorted list
                #         @debug println("    face#$sface")
                #         token = Base.ht_keyindex2!(facedicts[fi], sface)
                #         if token > 0 # haskey(facedicts[fi], sface), reuse dofs
                #             startdof = facedicts[fi].vals[token] # facedicts[fi][sface]
                #             for facedof in interpolation_info.nfacedofs:-1:1 # always reverse (YOLO)
                #                 for d in 1:field_dims[fi]
                #                     reuse_dof = startdof + (d-1) + (facedof-1)*fh.field_dims[fi]
                #                     @debug println("      reusing dof#$(reuse_dof)")
                #                     push!(dh.cell_dofs, reuse_dof)
                #                 end
                #             end
                #         else # distribute new dofs
                #             Base._setindex!(facedicts[fi], nextdof, sface, -token)# facedicts[fi][sface] = nextdof,  store the first dof for this face
                #             for facedof in 1:interpolation_info.nfacedofs
                #                 for d in 1:field_dims[fi]
                #                     @debug println("      adding dof#$nextdof")
                #                     push!(dh.cell_dofs, nextdof)
                #                     nextdof += 1
                #                 end
                #             end
                #         end
                #     end # face loop
                # end
                # if interpolation_info.ncelldofs > 0 # always distribute new dofs for cell
                #     @debug println("    cell#$ci")
                #     for celldof in 1:interpolation_info.ncelldofs
                #         for d in 1:field_dims[fi]
                #             @debug println("      adding dof#$nextdof")
                #             push!(dh.cell_dofs, nextdof)
                #             nextdof += 1
                #         end
                #     end # cell loop
                # end

            # push! the first index of the next cell to the offset vector
            push!(dh.cell_dofs_offset, length(dh.cell_dofs)+1)
            end
        end # cell loop
        dh.closed[] = true
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
