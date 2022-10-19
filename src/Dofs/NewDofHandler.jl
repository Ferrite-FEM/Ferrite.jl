abstract type AbstractDofHandler end

"""
    DofHandler(mesh::Mesh)

Construct a `DofHandler` based on `mesh`.

Operates slightly faster than [`MixedDofHandler`](@docs). Supports:
- `Mesh`s with a single concrete element type.
- One or several fields on the whole domaine.
"""
struct NewDofHandler{G<:AbstractMesh} <: AbstractDofHandler
    field_names::Vector{Symbol}
    field_dims::Vector{Int}
    # TODO: field_interpolations can probably be better typed: We should at least require
    #       all the interpolations to have the same dimension and reference shape
    field_interpolations::Vector{Interpolation}
    element_dofs::Vector{Int}
    element_dofs_offset::Vector{Int}
    closed::ScalarWrapper{Bool}
    mesh::G
    ndofs::ScalarWrapper{Int}
end

getmesh(dh::NewDofHandler) = dh.mesh

# Compat helper
@inline getgrid(dh::NewDofHandler) = getmesh(dh)
@inline ndofs_per_cell(dh::NewDofHandler, i::Int=1) = ndofs_per_element(dh, i)
@inline celldofs!(v::Vector, dh::NewDofHandler, i::Int) = elementdofs!(v, dh, i)
@inline cellnodes!(v::Vector, dh::NewDofHandler, i::Int) = elementnodes!(v, dh, i)
@inline cellcoords!(v, dh::NewDofHandler, i::Int) = elementcoords!(v, dh, i)

@inline ndofs_per_element(dh::NewDofHandler, element_idx::Int=1) = dh.element_dofs_offset[element_idx+1] - dh.element_dofs_offset[element_idx]

function NewDofHandler(mesh::AbstractMesh)
    # isconcretetype(getelementtype(mesh)) || error("Mesh includes different elementtypes. Use MixedNewDofHandler instead of NewDofHandler")
    NewDofHandler(Symbol[], Int[], Interpolation[], Int[], Int[], ScalarWrapper(false), mesh, Ferrite.ScalarWrapper(-1))
end

function Base.show(io::IO, ::MIME"text/plain", dh::NewDofHandler)
    println(io, "NewDofHandler")
    println(io, "  Fields:")
    for i in 1:nfields(dh)
        println(io, "    ", repr(dh.field_names[i]), ", interpolation: ", dh.field_interpolations[i],", dim: ", dh.field_dims[i])
    end
    if !isclosed(dh)
        print(io, "  Not closed!")
    else
        println(io, "  Dofs per element: ", ndofs_per_element(dh))
        print(io, "  Total dofs: ", ndofs(dh))
    end
end

function find_field(dh::NewDofHandler, field_name::Symbol)
    j = findfirst(i->i == field_name, dh.field_names)
    j === nothing && error("could not find field :$field_name in DofHandler (existing fields: $(getfieldnames(dh)))")
    return j
end

# Calculate the offset to the first local dof of a field
function field_offset(dh::NewDofHandler, field_name::Symbol)
    offset = 0
    for i in 1:find_field(dh, field_name)-1
        offset += getnbasefunctions(dh.field_interpolations[i])::Int * dh.field_dims[i]
    end
    return offset
end

getfieldinterpolation(dh::NewDofHandler, field_idx::Int) = dh.field_interpolations[field_idx]
getfielddim(dh::NewDofHandler, field_idx::Int) = dh.field_dims[field_idx]

function getfielddim(dh::NewDofHandler, name::Symbol)
    field_pos = findfirst(i->i == name, getfieldnames(dh))
    field_pos === nothing && error("did not find field $name")
    return dh.field_dims[field_pos]
end

"""
    dof_range(dh:DofHandler, field_name)

Return the local dof range for `field_name`. Example:

```jldoctest
julia> mesh = generate_mesh(Triangle, (3, 3))
Mesh{2, Triangle, Float64} with 18 Triangle elements and 16 nodes

julia> dh = DofHandler(mesh); push!(dh, :u, 3); push!(dh, :p, 1); close!(dh);

julia> dof_range(dh, :u)
1:9

julia> dof_range(dh, :p)
10:12
```
"""
function dof_range(dh::NewDofHandler, field_name::Symbol)
    f = find_field(dh, field_name)
    offset = field_offset(dh, field_name)
    n_field_dofs = getnbasefunctions(dh.field_interpolations[f])::Int * dh.field_dims[f]
    return (offset+1):(offset+n_field_dofs)
end

"""
    add_field!(dh::AbstractDofHandler, name::Symbol, dim::Int[, ip::Interpolation])

Add a `dim`-dimensional `Field` called `name` which is approximated by `ip` to `dh`.

The field is added to all elements of the underlying mesh. In case no interpolation `ip` is given,
the default interpolation of the mesh's elementtype is used.
If the mesh uses several elementtypes, [`push!(dh::MixedDofHandler, fh::FieldHandler)`](@ref) must be used instead.
"""
function add_field!(dh::NewDofHandler, name::Symbol, dim::Int, ip::Interpolation=default_interpolation(getelementtype(dh.mesh)))
    @assert !isclosed(dh)
    @assert !in(name, dh.field_names)
    push!(dh.field_names, name)
    push!(dh.field_dims, dim)
    push!(dh.field_interpolations, ip)
    return dh
end

function close!(dh::NewDofHandler)
    __close!(dh)
    return dh
end

# Helper from https://stackoverflow.com/questions/53503128/julia-match-any-type-belonging-to-a-union
gettypes(u::Union) = [u.a; gettypes(u.b)]
gettypes(u) = [u]

# TODO coalesce insert into list and index extraction...
# TODO look into how to handle more general elements (as in e.g. virtual element methods)
# Cubes
function add_entities!(face_list::Dict{NTuple{Dim, Int}, Int}, local_connectivity::Dict{Int,Vector{Tuple{Int, Int}}},  element_idx::Int, e::Element{Dim, RefCube, N}) where {Dim, N}
    for (face_idx, face) ∈ enumerate(faces(e))
        face_rep = sortface(face)
        if !haskey(face_list, face_rep)
            face_list[face_rep] = length(face_list)+1
        end
        if !haskey(local_connectivity, face_list[face_rep])
            local_connectivity[face_list[face_rep]] = Tuple{Int, Int}[]
        end
        push!(local_connectivity[face_list[face_rep]], (element_idx, face_idx))
    end
end

function add_entities!(edge_list::Dict{NTuple{2, Int}, Int}, local_connectivity::Dict{Int,Vector{Tuple{Int, Int}}},  element_idx::Int, e::Element{3, RefCube, N}) where {Dim, N}
    for (edge_idx, edge) ∈ enumerate(edges(e))
        edge_rep = sortedge(edge)[1]
        if !haskey(edge_list, edge_rep)
            edge_list[edge_rep] = length(edge_list)+1
        end
        if !haskey(local_connectivity, edge_list[edge_rep])
            local_connectivity[edge_list[edge]] = Tuple{Int, Int}[]
        end
        push!(local_connectivity[edge_list[edge_rep]], (element_idx, edge_idx))
    end
end

function add_entities!(vertex_list::Dict{Int, Int}, local_connectivity::Dict{Int,Vector{Tuple{Int, Int}}},  element_idx::Int, e::Element{Dim, RefCube, N}) where {Dim, N}
    for (vertex_idx, vertex) ∈ enumerate(vertices(e))
        if !haskey(vertex_list, vertex)
            vertex_list[vertex] = length(vertex_list)+1
        end
        if !haskey(local_connectivity, vertex_list[vertex])
            local_connectivity[vertex_list[vertex]] = Tuple{Int, Int}[]
        end
        push!(local_connectivity[vertex_list[vertex]], (element_idx, vertex_idx))
    end
end

# Simplices
function add_entities!(face_list::Dict{NTuple{Dim, Int}, Int}, local_connectivity::Dict{Int,Vector{Tuple{Int, Int}}},  element_idx::Int, e::Element{Dim, RefSimplex, N}) where {Dim, N}
    for (face_idx, face) ∈ enumerate(faces(e))
        face_rep = sortface(face)
        if !haskey(face_list, face_rep)
            face_list[face_rep] = length(face_list)+1
        end
        if !haskey(local_connectivity, face_list[face_rep])
            local_connectivity[face_list[face_rep]] = Tuple{Int, Int}[]
        end
        push!(local_connectivity[face_list[face_rep]], (element_idx, face_idx))
    end
end

function add_entities!(edge_list::Dict{NTuple{2, Int}, Int}, local_connectivity::Dict{Int,Vector{Tuple{Int, Int}}},  element_idx::Int, e::Element{3, RefSimplex, N}) where {Dim, N}
    for (edge_idx, edge) ∈ enumerate(edges(e))
        edge_rep = sortedge(edge)[1]
        if !haskey(edge_list, edge_rep)
            edge_list[edge_rep] = length(edge_list)+1
        end
        if !haskey(local_connectivity, edge_list[edge_rep])
            local_connectivity[edge_list[edge]] = Tuple{Int, Int}[]
        end
        push!(local_connectivity[edge_list[edge_rep]], (element_idx, edge_idx))
    end
end

function add_entities!(vertex_list::Dict{Int, Int}, local_connectivity::Dict{Int,Vector{Tuple{Int, Int}}},  element_idx::Int, e::Element{Dim, RefSimplex, N}) where {Dim, N}
    for (vertex_idx, vertex) ∈ enumerate(vertices(e))
        if !haskey(vertex_list, vertex)
            vertex_list[vertex] = length(vertex_list)+1
        end
        if !haskey(local_connectivity, vertex_list[vertex])
            local_connectivity[vertex_list[vertex]] = Tuple{Int, Int}[]
        end
        push!(local_connectivity[vertex_list[vertex]], (element_idx, vertex_idx))
    end
end

# TODO data structure for this?
function num_dofs_on_codim(interpolation_info::InterpolationInfo, dim::Int, codim::Int)
    if dim == codim
        return interpolation_info.nvertexdofs
    elseif codim == 2
        return interpolation_info.nedgedofs
    elseif codim == 1
        return interpolation_info.nfacedofs
    elseif codim == 0
        return interpolation_info.ncelldofs
    else
        error("No dof information for dim=$dim, codim=$codim in $interpolation_info.")
    end
end

get_compatible_interpolation(e, ip) = error("Incompatible interpolations provided!")
get_compatible_interpolation(e::Ferrite.Element{Dim, RefGeo, N}, ip::Interpolation{Dim, IPRefGeo, O}) where {N, Dim, O, IPRefGeo, RefGeo <: IPRefGeo} = typeof(ip).name.wrapper{Dim, RefGeo, O}()
get_compatible_interpolation(e::Type{Ferrite.Element{Dim, RefGeo, N}}, ip::Interpolation{Dim, IPRefGeo, O}) where {N, Dim, O, IPRefGeo, RefGeo <: IPRefGeo} = typeof(ip).name.wrapper{Dim, RefGeo, O}()

# close the DofHandler and distribute all the dofs
function __close!(dh::NewDofHandler)
    @assert !isclosed(dh)

    #######################################################################################
    ###### Phase 1: Materialize possibly broken mesh (i.e. ignore all non-conformities) ###
    #######################################################################################
    #
    # What happens here?
    # We basically build bijections between continuous indices and unique representations
    # of subentities and how they connect back to elements to allow fast identification.
    #
    #NOTE Refactor into the mesh topology interface.
    #NOTE Subobtimal data structures ahead!
    #
    mesh = getmesh(dh)
    element_types = gettypes(getelementtypes(mesh))
    element_type_idx = Dict([element_type => i for (i, element_type) ∈ enumerate(element_types)])
    max_element_dim = maximum([getdim(et) for et ∈ element_types])
    interpolation_infos = ntuple(field_idx->[InterpolationInfo(get_compatible_interpolation(element, dh.field_interpolations[field_idx])) for element ∈ element_types], nfields(dh))
    # TODO relax this assumption later - possibly non-conforming interpolations not supported yet.
    for field_interpolation_info ∈ interpolation_infos
        for i ∈ 2:length(field_interpolation_info)
            @assert field_interpolation_info[i].nvertexdofs == first(field_interpolation_info).nvertexdofs
            @assert field_interpolation_info[i].nedgedofs == first(field_interpolation_info).nedgedofs
            @assert field_interpolation_info[i].nfacedofs == first(field_interpolation_info).nfacedofs
        end
    end

    # 4d not functional yet
    @assert(max_element_dim < 4)

    #TODO better data structures...
    # "subentities" is a helper to map the unique representation of an entity to its unique index.
    # Example: subentities[element_dimension][entity_codimension] -> Simple "materialized" representation of entity
    subentities  = ntuple(element_dim->ntuple(entity_codim->(element_dim-entity_codim+1 > 1) ? Dict{NTuple{element_dim-entity_codim+1,Int},Int}() : Dict{Int, Int}(), max(element_dim, 0)), max_element_dim)
    # "connectivity" is a helper to find which elements are attached to the unique representation of an entity
    # connectivity[element_dimension][entity_codimension][unique_entity_index] -> Tuple vector in format "vector of (element_idx, local_entity_idx)"
    connectivity = ntuple(element_dim->ntuple(entity_codim->Dict{Int,Vector{Tuple{Int, Int}}}(), max(element_dim, 0)), max_element_dim)

    #NOTE this above is an inefficient, simplified version of what the topology interface should become!
    # Primitive materialization of elements and induction of simple topology
    for (element_idx, element) ∈ enumerate(getelements(mesh))
        # Loop over all codimensional entities, i.e. faces, edges, but ignore
        # the full actual element (codim=0).
        edim = getdim(element)
        for codim ∈ 1:edim
            add_entities!(subentities[edim][codim], connectivity[edim][codim], element_idx, element)
        end
    end
    @debug @show subentities
    @debug @show connectivity

    # TODO data structures and better algorithm...
    # This is basically a reordered version of the data structure above...
    entities = ntuple(entity_dim->(entity_dim > 1) ? Dict{NTuple{entity_dim,Int},Int}() : Dict{Int, Int}(), max_element_dim) 
    entity_to_element = ntuple(entity_dim->Dict{Int,Vector{Tuple{Int,Int}}}(), max_element_dim)
    for edim ∈ 1:max_element_dim
        for codim ∈ 1:edim # TODO codim 0 for interior dofs
            subentity_dim = edim-codim
            for (subentity_identifier, subentity_index) ∈ subentities[edim][codim]
                if !haskey(entities[subentity_dim+1], subentity_identifier)
                    entities[subentity_dim+1][subentity_identifier] = length(entities[subentity_dim+1])+1
                end
                entity_id = entities[subentity_dim+1][subentity_identifier]
                if !haskey(entity_to_element[subentity_dim+1], entity_id)
                    entity_to_element[subentity_dim+1][entity_id] = Tuple{Int,Int}[]
                end
                # TODO this is currently an incomplete list and just for demonstration purposes....
                append!(entity_to_element[subentity_dim+1][entity_id], connectivity[edim][codim][subentity_index])
            end
        end
    end
    @debug @show entities
    @debug @show entity_to_element

    #######################################################################################
    ######                             Phase 2: Distribute dofs                       #####
    #######################################################################################
    # not implemented yet: more than one facedof per face in 3D
    # max_element_dim == 3 && @assert(!any(x->x.nfacedofs > 1, interpolation_infos))
 
    # --------------------------------------------------------------------------
    # ------------------- DoF Distribution Pseudo-Algorithm -------------------- 
    # --------------------------------------------------------------------------
    # For each field `f`
    #   For each entity of dim `d`
    #     For each entity with topology `t` of dim `d` in mesh
    #       Compute local dof index via: dof = vdim*entity_index*ndofs_per_entity
    #       Shift dof index
    #       Store dof per element `dof_list`
    #     End
    #     For each entity with topology `t` of dim `d` in mesh
    #       Push shifted dofs in per element dof list in correct order
    #       Permutate higher order nodes on elements connected to  
    #     End
    #   End
    # End
    # --------------------------------------------------------------------------

    # element_dofs = [Vector{Int}() for i ∈ 1:getnelements(mesh)]
    # dof_offset = 0
    # for fi in 1:nfields(dh)
    #     for entity_dim ∈ 0:(max_element_dim-1) #FIXME interior dofs not included yet...
    #         # Unique iterator over entities!
    #         for (entity_identifier, entity_index) ∈ entities[entity_dim+1]
    #             @info (entity_identifier, entity_index)
    #             adjacent_element_info = entity_to_element[entity_dim+1][entity_identifier]
    #             # this is a hotfix for demonstration purposes... does not work for "mixed interpolations" yet
    #             interpolation_info = interpolation_infos[fi][element_type_idx[typeof(getelements(mesh, adjacent_element_info[1][1]))]]
    #             #TODO if 0 < entity_dim < max_element_dim permutate dofs correctly...
    #             @info current_ndofs_on_entity
    #         end
    #     end
    # end

    # # Wrapper to not break code which relies on these two arrays.
    # push!(dh.element_dofs_offset, 1)
    # for dofs_on_element ∈ element_dofs
    #     append(dh.element_dofs, dof_on_element)
    #     push!(dh.element_dofs_offset, dh.element_dofs_offset+length(dofs_on_element)+1)
    # end
    # dh.ndofs[] = maximum(dh.element_dofs)
    # dh.closed[] = true
    # return

    # ----------------------------- ALTERNATIVE TO ABOVE ---------------------------------

    #######################################################################################
    ######                             Phase 2: Distribute dofs                       #####
    #######################################################################################
    # not implemented yet: more than one facedof per face in 3D
    # max_element_dim == 3 && @assert(!any(x->x.nfacedofs > 1, interpolation_infos))

    # We can simplify this quite a bit by rearranging the loops below.
    # TODO check importance of this reordering...
    codim_ordering_by_dim = [
        [1,0],      # vertex, cell
        [2,1,0],    # vertex, face, cell
        [3,1,2,0],  # vertex, edge, face, cell
        [4,1,2,3,0] # Vertex, face, edge, planar, cell
    ]
    # Running index of current element
    element_current_idx_by_dim = [
        1,1,1,1
    ]

    # TODO We should be able to condense this one.
    field_offsets = zeros(Int, nfields(dh)+1)
    for fi in 1:nfields(dh)
        interpolation_info = interpolation_infos[fi][1]
        field_dim = getfielddim(dh, fi)
        for edim ∈ 1:max_element_dim
            for codim ∈ 0:edim
                num_entities = codim == 0 ? getnelements(mesh) : length(subentities[edim][codim])
                ndofs_on_entity = num_dofs_on_codim(interpolation_info, edim, codim)
                field_offsets[fi+1] += field_dim*num_entities*ndofs_on_entity
            end
        end
        field_offsets[fi+1] += field_offsets[fi]
        @debug println("field: $(dh.field_names[fi]) with dof range $((field_offsets[fi]+1):field_offsets[fi+1])")
    end

    #TODO check type stability and performance
    push!(dh.element_dofs_offset, 1) # dofs for the first element start at 1
    for (gei, element) in enumerate(getelements(getmesh(dh)))
        current_element_dim = getdim(element)
        ei = element_current_idx_by_dim[current_element_dim]
        @debug println("global element #$gei of dim $current_element_dim with dimension-local index $ei")
        for fi in 1:nfields(dh)
            interpolation_info = interpolation_infos[fi][element_type_idx[typeof(element)]]
            field_dim = getfielddim(dh, fi)
            #@debug println("  field: $(dh.field_names[fi]) with dof offset $next_field_offset")
            @debug println("  field: $(dh.field_names[fi])")
            entity_based_offset = 0
            for current_entity_codim ∈ codim_ordering_by_dim[current_element_dim]
                @debug println("    entity codim $current_entity_codim | dim $(current_element_dim-current_entity_codim) with dof offset $entity_based_offset")
                current_ndofs_on_entity = num_dofs_on_codim(interpolation_info, current_element_dim, current_entity_codim)
                if current_ndofs_on_entity > 0
                    # Compute the number of entities with codim > mine
                    for local_entity_rep ∈ entities_with_codim(element, current_entity_codim)
                        # TODO handle this better...
                        local_entity_rep_unique = (current_element_dim > 1 && current_entity_codim == 1) ? sortface(local_entity_rep) : (current_element_dim > 2 && current_entity_codim == 2) ? sortedge(local_entity_rep)[1] : local_entity_rep;
                        current_entity_idx      = current_entity_codim == 0 ? local_entity_rep_unique : subentities[current_element_dim][current_entity_codim][local_entity_rep_unique]
                        startdof = field_offsets[fi] + entity_based_offset + field_dim*(current_entity_idx-1) + 1
                        for d in 1:field_dim
                            computed_dof = startdof + (d-1) + (current_ndofs_on_entity-1)*field_dim
                            push!(dh.element_dofs, computed_dof)
                            @debug println("      added dof $computed_dof")
                        end
                    end
                    # TODO permutate the dofs according to a reference orientation given by the entity on the adjacent element with the lowest index!
                    num_entities = current_entity_codim == 0 ? 1 : length(subentities[current_element_dim][current_entity_codim])
                    entity_based_offset += current_ndofs_on_entity*num_entities*field_dim
                end
            end
        end

        element_current_idx_by_dim[current_element_dim] += 1
        push!(dh.element_dofs_offset, length(dh.element_dofs)+1)
    end
    dh.ndofs[] = maximum(dh.element_dofs)
    dh.closed[] = true

    # return dh, vertexdicts, edgedicts, facedicts
end

function elementdofs!(global_dofs::Vector{Int}, dh::NewDofHandler, i::Int)
    @assert isclosed(dh)
    @assert length(global_dofs) == ndofs_per_element(dh, i)
    unsafe_copyto!(global_dofs, 1, dh.element_dofs, dh.element_dofs_offset[i], length(global_dofs))
    return global_dofs
end

function elementnodes!(global_nodes::Vector{Int}, mesh::AbstractMesh{dim}, i::Int) where {dim}
    nodes = getelements(mesh,i).nodes
    N = length(nodes)
    @assert length(global_nodes) == N
    for j in 1:N
        global_nodes[j] = nodes[j]
    end
    return global_nodes
end

function elementcoords!(global_coords::Vector{Vec{dim,T}}, mesh::AbstractMesh{dim}, i::Int) where {dim,T}
    nodes = getelements(mesh,i).nodes
    N = length(nodes)
    @assert length(global_coords) == N
    for j in 1:N
        global_coords[j] = getcoordinates(getnodes(mesh,nodes[j]))
    end
    return global_coords
end

elementcoords!(global_coords::Vector{<:Vec}, dh::NewDofHandler, i::Int) = elementcoords!(global_coords, dh.mesh, i)

function elementdofs(dh::NewDofHandler, i::Int)
    @assert isclosed(dh)
    n = ndofs_per_element(dh, i)
    global_dofs = zeros(Int, n)
    unsafe_copyto!(global_dofs, 1, dh.element_dofs, dh.element_dofs_offset[i], n)
    return global_dofs
end

# Creates a sparsity pattern from the dofs in a DofHandler.
# Returns a sparse matrix with the correct storage pattern.
"""
    create_sparsity_pattern(dh::NewDofHandler)

Create the sparsity pattern corresponding to the degree of freedom
numbering in the [`DofHandler`](@ref). Return a `SparseMatrixCSC`
with stored values in the correct places.

See the [Sparsity Pattern](@ref) section of the manual.
"""
create_sparsity_pattern(dh::NewDofHandler) = _create_sparsity_pattern(dh, nothing, false)

"""
    create_symmetric_sparsity_pattern(dh::NewDofHandler)

Create the symmetric sparsity pattern corresponding to the degree of freedom
numbering in the [`DofHandler`](@ref) by only considering the upper
triangle of the matrix. Return a `Symmetric{SparseMatrixCSC}`.

See the [Sparsity Pattern](@ref) section of the manual.
"""
create_symmetric_sparsity_pattern(dh::NewDofHandler) = Symmetric(_create_sparsity_pattern(dh, nothing, true), :U)

function _create_sparsity_pattern(dh::NewDofHandler, ch#=::Union{ConstraintHandler, Nothing}=#, sym::Bool)
    @assert isclosed(dh)
    
    
    ncells = getncells(getmesh(dh))
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
        _condense_sparsity_pattern!(K, ch.acs)
        fill!(K.nzval, 0.0)
    else
        V = zeros(length(I))
        K = sparse(I, J, V, ndofs(dh), ndofs(dh))
    end
    return K
end

"""
    reshape_to_nodes(dh::AbstractDofHandler, u::Vector{T}, fieldname::Symbol) where T

Reshape the entries of the dof-vector `u` which correspond to the field `fieldname` in nodal order.
Return a matrix with a column for every node and a row for every dimension of the field.
For superparametric fields only the entries corresponding to nodes of the mesh will be returned. Do not use this function for subparametric approximations.
"""
function reshape_to_nodes(dh::NewDofHandler, u::Vector{T}, fieldname::Symbol) where T
    # make sure the field exists
    fieldname ∈ Ferrite.getfieldnames(dh) || error("Field $fieldname not found.")

    field_idx = findfirst(i->i==fieldname, getfieldnames(dh))
    offset = field_offset(dh, fieldname)
    field_dim = getfielddim(dh, field_idx)

    space_dim = field_dim == 2 ? 3 : field_dim
    data = fill(zero(T), space_dim, getnnodes(dh.mesh))

    reshape_field_data!(data, dh, u, offset, field_dim)

    return data
end
