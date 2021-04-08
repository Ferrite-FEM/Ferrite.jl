# abstract type Constraint end
"""
    Dirichlet(u, ∂Ω, f)
    Dirichlet(u, ∂Ω, f, component)

Create a Dirichlet boundary condition on `u` on the `∂Ω` part of
the boundary. `f` is a function that takes two arguments, `x` and `t`
where `x` is the spatial coordinate and `t` is the current time,
and returns the prescribed value. For example, here we create a
Dirichlet condition for the `:u` field, on the faceset called
`∂Ω` and the value given by the `sin` function:

```julia
dbc = Dirichlet(:u, ∂Ω, (x, t) -> sin(t))
```

If `:u` is a vector field we can specify which component the condition
should be applied to by specifying `component`. `component` can be given
either as an integer, or as a vector, for example:

```julia
dbc = Dirichlet(:u, ∂Ω, (x, t) -> sin(t), 1)      # applied to component 1
dbc = Dirichlet(:u, ∂Ω, (x, t) -> sin(t), [1, 3]) # applied to component 1 and 3
```

`Dirichlet` boundary conditions are added to a [`ConstraintHandler`](@ref)
which applies the condition via `apply!`.
"""
struct Dirichlet # <: Constraint
    f::Function # f(x,t) -> value
    faces::Union{Set{Int},Set{Tuple{Int,Int}}}
    field_name::Symbol
    components::Vector{Int} # components of the field
    local_face_dofs::Vector{Int}
    local_face_dofs_offset::Vector{Int}
end
function Dirichlet(field_name::Symbol, faces::Union{Set{Int},Set{Tuple{Int,Int}}}, f::Function, component::Int=1)
    Dirichlet(field_name, copy(faces), f, [component])
end
function Dirichlet(field_name::Symbol, faces::Union{Set{Int},Set{Tuple{Int,Int}}}, f::Function, components::AbstractVector{Int})
    unique(components) == components || error("components not unique: $components")
    # issorted(components) || error("components not sorted: $components")
    return Dirichlet(f, copy(faces), field_name, Vector(components), Int[], Int[])
end

"""
    ConstraintHandler

Collection of constraints.
"""
struct ConstraintHandler{DH<:AbstractDofHandler,T}
    dbcs::Vector{Dirichlet}
    prescribed_dofs::Vector{Int}
    free_dofs::Vector{Int}
    values::Vector{T}
    dofmapping::Dict{Int,Int} # global dof -> index into dofs and values
    bcvalues::Vector{BCValues{T}}
    dh::DH
    closed::ScalarWrapper{Bool}
end

"""
    RHSData

Stores the constrained columns and mean of the diagonal of stiffness matrix `A`.
"""
struct RHSData{T}
    m::T
    constrained_columns::SparseMatrixCSC{T, Int}
end

"""
    get_rhs_data(ch::ConstraintHandler, A::SparseMatrixCSC) -> RHSData

Returns the needed RHSData for apply_rhs!
"""
function get_rhs_data(ch::ConstraintHandler, A::SparseMatrixCSC)
    m = meandiag(A)
    constrained_columns = A[:, ch.prescribed_dofs]
    return RHSData(m, constrained_columns)
end

"""
    apply_rhs!(data::RHSData, f::AbstractVector, ch::ConstraintHandler, applyzero::Bool=false)

Applies the boundary condition to the right-hand-side vector without modifying the stiffness matrix
"""
function apply_rhs!(data::RHSData, f::AbstractVector, ch::ConstraintHandler, applyzero::Bool=false)
    K = data.constrained_columns
    @assert length(f) == 0 || length(f) == size(K, 1)
    @boundscheck length(f) == 0 || checkbounds(f, ch.prescribed_dofs)

    m = data.m
    @inbounds for i in 1:length(ch.values)
        d = ch.prescribed_dofs[i]
        v = ch.values[i]
        if !applyzero && v != 0
            for j in nzrange(K, i)
                f[K.rowval[j]] -= v * K.nzval[j]
            end
        end
        if length(f) != 0
            vz = applyzero ? zero(eltype(f)) : v
            f[d] = vz * m
        end
    end
end

function ConstraintHandler(dh::AbstractDofHandler)
    @assert isclosed(dh)
    ConstraintHandler(Dirichlet[], Int[], Int[], Float64[], Dict{Int,Int}(), BCValues{Float64}[], dh, ScalarWrapper(false))
end

function Base.show(io::IO, ::MIME"text/plain", ch::ConstraintHandler)
    println(io, "ConstraintHandler:")
    if !isclosed(ch)
        print(io, "  Not closed!")
    else
        print(io, "  BCs:")
        for dbc in ch.dbcs
            print(io, "\n    ", "Field: ", dbc.field_name, ", ", "Components: ", dbc.components)
        end
    end
end

isclosed(ch::ConstraintHandler) = ch.closed[]
free_dofs(ch::ConstraintHandler) = ch.free_dofs
prescribed_dofs(ch::ConstraintHandler) = ch.prescribed_dofs

"""
    close!(ch::ConstraintHandler)

Close and finalize the `ConstraintHandler`.
"""
function close!(ch::ConstraintHandler)
    fdofs = setdiff(1:ndofs(ch.dh), ch.prescribed_dofs)
    copy!!(ch.free_dofs, fdofs)
    copy!!(ch.prescribed_dofs, unique(ch.prescribed_dofs)) # for v0.7: unique!(ch.prescribed_dofs)
    sort!(ch.prescribed_dofs) # YOLO
    fill!(resize!(ch.values, length(ch.prescribed_dofs)), NaN)
    for i in 1:length(ch.prescribed_dofs)
        ch.dofmapping[ch.prescribed_dofs[i]] = i
    end
    ch.closed[] = true
    return ch
end

function dbc_check(ch::ConstraintHandler, dbc::Dirichlet)
    # check input
    dbc.field_name in getfieldnames(ch.dh) || throw(ArgumentError("field $(dbc.field_name) does not exist in DofHandler, existing fields are $(getfieldnames(ch.dh))"))
    #TODO FIX!!
    #for component in dbc.components
    #    0 < component <= ndim(ch.dh, dbc.field_name) || error("component $component is not within the range of field $field which has $(ndim(ch.dh, field)) dimensions")
    #end
    if length(dbc.faces) == 0
        @warn("added Dirichlet Boundary Condition to set containing 0 entities")
    end
end

"""
    add!(ch::ConstraintHandler, dbc::Dirichlet)

Add a `Dirichlet boundary` condition to the `ConstraintHandler`.
"""
function add!(ch::ConstraintHandler, dbc::Dirichlet)
    dbc_check(ch, dbc)
    field_idx = find_field(ch.dh, dbc.field_name)
    # Extract stuff for the field
    interpolation = getfieldinterpolation(ch.dh, field_idx)#ch.dh.field_interpolations[field_idx]
    field_dim = getfielddim(ch.dh, field_idx)#ch.dh.field_dims[field_idx] # TODO: I think we don't need to extract these here ...
    bcvalue = getbcvalue(ch.dh, field_idx)
    _add!(ch, dbc, dbc.faces, interpolation, field_dim, field_offset(ch.dh, dbc.field_name), bcvalue)
    return ch
end

function _add!(ch::ConstraintHandler, dbc::Dirichlet, bcfaces::Set{Tuple{Int,Int}}, interpolation::Interpolation, field_dim::Int, offset::Int, bcvalue::BCValues, cellset::Set{Int}=Set{Int}(1:getncells(ch.dh.grid)))
    # calculate which local dof index live on each face
    # face `i` have dofs `local_face_dofs[local_face_dofs_offset[i]:local_face_dofs_offset[i+1]-1]
    local_face_dofs = Int[]
    local_face_dofs_offset = Int[1]
    for (i, face) in enumerate(faces(interpolation))
        for fdof in face, d in 1:field_dim
            if d ∈ dbc.components # skip unless this component should be constrained
                push!(local_face_dofs, (fdof-1)*field_dim + d + offset)
            end
        end
        push!(local_face_dofs_offset, length(local_face_dofs) + 1)
    end
    copy!!(dbc.local_face_dofs, local_face_dofs)
    copy!!(dbc.local_face_dofs_offset, local_face_dofs_offset)

    # loop over all the faces in the set and add the global dofs to `constrained_dofs`
    constrained_dofs = Int[]
    redundant_faces = NTuple{2, Int}[]
    for (cellidx, faceidx) in bcfaces
        if cellidx ∉ cellset
            push!(redundant_faces, (cellidx, faceidx)) # will be removed from dbc
            continue # skip faces that are not part of the cellset
        end
        _celldofs = fill(0, ndofs_per_cell(ch.dh, cellidx))
        celldofs!(_celldofs, ch.dh, cellidx) # extract the dofs for this cell
        r = local_face_dofs_offset[faceidx]:(local_face_dofs_offset[faceidx+1]-1)
        append!(constrained_dofs, _celldofs[local_face_dofs[r]]) # TODO: for-loop over r and simply push! to ch.prescribed_dofs
        @debug println("adding dofs $(_celldofs[local_face_dofs[r]]) to dbc")
    end

    setdiff!(dbc.faces, redundant_faces)
    # save it to the ConstraintHandler
    push!(ch.dbcs, dbc)
    push!(ch.bcvalues, bcvalue)
    append!(ch.prescribed_dofs, constrained_dofs)
end

function _add!(ch::ConstraintHandler, dbc::Dirichlet, bcnodes::Set{Int}, interpolation::Interpolation, field_dim::Int, offset::Int, bcvalue::BCValues, cellset::Set{Int}=Set{Int}(1:getncells(ch.dh.grid)))
    if interpolation !== default_interpolation(typeof(ch.dh.grid.cells[first(cellset)]))
        @warn("adding constraint to nodeset is not recommended for sub/super-parametric approximations.")
    end

    ncomps = length(dbc.components)
    nnodes = getnnodes(ch.dh.grid)
    interpol_points = getnbasefunctions(interpolation)
    _celldofs = fill(0, ndofs_per_cell(ch.dh, first(cellset)))
    node_dofs = zeros(Int, ncomps, nnodes)
    visited = falses(nnodes)
    for cell in CellIterator(ch.dh, collect(cellset)) # only go over cells that belong to current FieldHandler
        celldofs!(_celldofs, cell) # update the dofs for this cell
        for idx in 1:min(interpol_points, length(cell.nodes))
            node = cell.nodes[idx]
            if !visited[node]
                noderange = (offset + (idx-1)*field_dim + 1):(offset + idx*field_dim) # the dofs in this node
                for (i,c) in enumerate(dbc.components)
                    node_dofs[i,node] = _celldofs[noderange[c]]
                    @debug println("adding dof $(_celldofs[noderange[c]]) to node_dofs")
                end
                visited[node] = true
            end
        end
    end

    constrained_dofs = Int[]
    sizehint!(constrained_dofs, ncomps*length(bcnodes))
    sizehint!(dbc.local_face_dofs, length(bcnodes))
    for node in bcnodes
        if !visited[node]
    # either the node belongs to another field handler or it does not have dofs in the constrained field
            continue
        end
        for i in 1:ncomps
            push!(constrained_dofs, node_dofs[i,node])
        end
        push!(dbc.local_face_dofs, node) # use this field to store the node idx for each node
    end

    # save it to the ConstraintHandler
    copy!!(dbc.local_face_dofs_offset, constrained_dofs) # use this field to store the global dofs
    push!(ch.dbcs, dbc)
    push!(ch.bcvalues, bcvalue)
    append!(ch.prescribed_dofs, constrained_dofs)
end

# Updates the DBC's to the current time `time`
function update!(ch::ConstraintHandler, time::Real=0.0)
    @assert ch.closed[]
    for (i,dbc) in enumerate(ch.dbcs)
        # Function barrier
        _update!(ch.values, dbc.f, dbc.faces, dbc.field_name, dbc.local_face_dofs, dbc.local_face_dofs_offset,
                 dbc.components, ch.dh, ch.bcvalues[i], ch.dofmapping, convert(Float64, time))
    end
end

# for faces
function _update!(values::Vector{Float64}, f::Function, faces::Set{Tuple{Int,Int}}, field::Symbol, local_face_dofs::Vector{Int}, local_face_dofs_offset::Vector{Int},
                  components::Vector{Int}, dh::AbstractDofHandler, facevalues::BCValues,
                  dofmapping::Dict{Int,Int}, time::T) where {T}

    dim = getdim(dh.grid)
    _tmp_cellid = first(faces)[1]

    N = nnodes_per_cell(dh.grid, _tmp_cellid)
    xh = zeros(Vec{dim, T}, N) # pre-allocate
    _celldofs = fill(0, ndofs_per_cell(dh, _tmp_cellid))

    for (cellidx, faceidx) in faces
        cellcoords!(xh, dh, cellidx)
        celldofs!(_celldofs, dh, cellidx) # update global dofs for this cell

        # no need to reinit!, enough to update current_face since we only need geometric shape functions M
        facevalues.current_face[] = faceidx

        # local dof-range for this face
        r = local_face_dofs_offset[faceidx]:(local_face_dofs_offset[faceidx+1]-1)
        counter = 1

        for location in 1:getnquadpoints(facevalues)
            x = spatial_coordinate(facevalues, location, xh)
            bc_value = f(x, time)
            @assert length(bc_value) == length(components)

            for i in 1:length(components)
                # find the global dof
                globaldof = _celldofs[local_face_dofs[r[counter]]]
                counter += 1

                dbc_index = dofmapping[globaldof]
                values[dbc_index] = bc_value[i]
                @debug println("prescribing value $(bc_value[i]) on global dof $(globaldof)")
            end
        end
    end
end

# for nodes
function _update!(values::Vector{Float64}, f::Function, nodes::Set{Int}, field::Symbol, nodeidxs::Vector{Int}, globaldofs::Vector{Int},
                  components::Vector{Int}, dh::AbstractDofHandler, facevalues::BCValues,
                  dofmapping::Dict{Int,Int}, time::Float64)
    counter = 1
    for (idx, nodenumber) in enumerate(nodeidxs)
        x = dh.grid.nodes[nodenumber].x
        bc_value = f(x, time)
        @assert length(bc_value) == length(components)
        for v in bc_value
            globaldof = globaldofs[counter]
            counter += 1
            dbc_index = dofmapping[globaldof]
            values[dbc_index] = v
            @debug println("prescribing value $(v) on global dof $(globaldof)")
        end
    end
end

# Saves the dirichlet boundary conditions to a vtkfile.
# Values will have a 1 where bcs are active and 0 otherwise
function WriteVTK.vtk_point_data(vtkfile, ch::ConstraintHandler)
    unique_fields = []
    for dbc in ch.dbcs
        push!(unique_fields, dbc.field_name)
    end
    unique_fields = unique(unique_fields) # TODO v0.7: unique!(unique_fields)

    for field in unique_fields
        nd = ndim(ch.dh, field)
        data = zeros(Float64, nd, getnnodes(ch.dh.grid))
        for dbc in ch.dbcs
            dbc.field_name != field && continue
            if eltype(dbc.faces) <: Tuple
                for (cellidx, faceidx) in dbc.faces
                    for facenode in faces(ch.dh.grid.cells[cellidx])[faceidx]
                        for component in dbc.components
                            data[component, facenode] = 1
                        end
                    end
                end
            else
                for nodeidx in dbc.faces
                    for component in dbc.components
                        data[component, nodeidx] = 1
                    end
                end
            end
        end
        vtk_point_data(vtkfile, data, string(field, "_bc"))
    end
    return vtkfile
end

function apply!(v::AbstractVector, ch::ConstraintHandler)
    @assert length(v) == ndofs(ch.dh)
    v[ch.prescribed_dofs] = ch.values
    return v
end

function apply_zero!(v::AbstractVector, ch::ConstraintHandler)
    @assert length(v) == ndofs(ch.dh)
    v[ch.prescribed_dofs] .= 0
    return v
end

function apply!(K::Union{SparseMatrixCSC,Symmetric}, ch::ConstraintHandler)
    apply!(K, eltype(K)[], ch, true)
end

function apply_zero!(K::Union{SparseMatrixCSC,Symmetric}, f::AbstractVector, ch::ConstraintHandler)
    apply!(K, f, ch, true)
end

@enum(ApplyStrategy, APPLY_TRANSPOSE, APPLY_INPLACE)

function apply!(KK::Union{SparseMatrixCSC,Symmetric}, f::AbstractVector, ch::ConstraintHandler, applyzero::Bool=false;
                strategy::ApplyStrategy=APPLY_TRANSPOSE)
    K = isa(KK, Symmetric) ? KK.data : KK
    @assert length(f) == 0 || length(f) == size(K, 1)
    @boundscheck checkbounds(K, ch.prescribed_dofs, ch.prescribed_dofs)
    @boundscheck length(f) == 0 || checkbounds(f, ch.prescribed_dofs)

    m = meandiag(K) # Use the mean of the diagonal here to not ruin things for iterative solver
    @inbounds for i in 1:length(ch.values)
        d = ch.prescribed_dofs[i]
        v = ch.values[i]

        if !applyzero && v != 0
            for j in nzrange(K, d)
                f[K.rowval[j]] -= v * K.nzval[j]
            end
        end
    end
    zero_out_columns!(K, ch.prescribed_dofs)
    if strategy == APPLY_TRANSPOSE
        K′ = copy(K)
        transpose!(K′, K)
        zero_out_columns!(K′, ch.prescribed_dofs)
        transpose!(K, K′)
    elseif strategy == APPLY_INPLACE
        K[ch.prescribed_dofs, :] .= 0
    else
        error("Unknown apply strategy")
    end
    @inbounds for i in 1:length(ch.values)
        d = ch.prescribed_dofs[i]
        v = ch.values[i]
        K[d, d] = m
        # We will only enter here with an empty f vector if we have assured that v == 0 for all dofs
        if length(f) != 0
            vz = applyzero ? zero(eltype(f)) : v
            f[d] = vz * m
        end
    end
end

# columns need to be stored entries, this is not checked
function zero_out_columns!(K, dofs::Vector{Int}) # can be removed in 0.7 with #24711 merged
    @debug @assert issorted(dofs)
    for col in dofs
        r = nzrange(K, col)
        K.nzval[r] .= 0.0
    end
end


function meandiag(K::AbstractMatrix)
    z = zero(eltype(K))
    for i in 1:size(K, 1)
        z += abs(K[i, i])
    end
    return z / size(K, 1)
end


#Function for adding constraint when using multiple celltypes
function add!(ch::ConstraintHandler, fh::FieldHandler, dbc::Dirichlet)
    _check_cellset_dirichlet(ch.dh.grid, fh.cellset, dbc.faces)

    field_idx = find_field(fh, dbc.field_name)
    # Extract stuff for the field
    interpolation = getfieldinterpolations(fh)[field_idx]
    field_dim = getfielddims(fh)[field_idx]
    bcvalue = fh.bc_values[field_idx]

    Ferrite._add!(ch, dbc, dbc.faces, interpolation, field_dim, field_offset(fh, dbc.field_name), bcvalue, fh.cellset)
    return ch
end

function _check_cellset_dirichlet(::AbstractGrid, cellset::Set{Int}, faceset::Set{Tuple{Int,Int}})
    for (cellid,faceid) in faceset
        if !(cellid in cellset)
            @warn("You are trying to add a constraint to a face that is not in the cellset of the fieldhandler. The face will be skipped.")
        end
    end
end

function _check_cellset_dirichlet(grid::AbstractGrid, cellset::Set{Int}, nodeset::Set{Int})
    nodes = Set{Int}()
    for cellid in cellset
        for nodeid in grid.cells[cellid].nodes
            nodeid ∈ nodes || push!(nodes, nodeid)
        end
    end

    for nodeid in nodeset
        if !(nodeid ∈ nodes)
            @warn("You are trying to add a constraint to a node that is not in the cellset of the fieldhandler. The node will be skipped.")
        end
    end
end
