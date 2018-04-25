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
struct FaceDirichlet # <: Constraint
    f::Function # f(x,t) -> value
    faces::Set{Tuple{Int,Int}}
    field_name::Symbol
    components::Vector{Int} # components of the field
    local_face_dofs::Vector{Int}
    local_face_dofs_offset::Vector{Int}
end
function Dirichlet(field_name::Symbol, faces::Set{Tuple{Int,Int}}, f::Function, component::Int=1)
    Dirichlet(field_name, faces, f, [component])
end
function Dirichlet(field_name::Symbol, faces::Set{Tuple{Int,Int}}, f::Function, components::Vector{Int})
    unique(components) == components || error("components not unique: $components")
    # issorted(components) || error("components not sorted: $components")
    return FaceDirichlet(f, faces, field_name, components, Int[], Int[])
end

"""
    NodeDirichlet

A Dirichlet boundary condition is a boundary where the solution is fixed to take a certain value.

`NodeDirichlet` boundary conditions are added to certain components of a field for a specific nodes of the grid.
A function is also given that should be of the form `(x,t) -> v` where `x` is the coordinate of the node, `t` is a
time parameter and `v` should be of the same length as the number of components the bc is applied to.

"""
struct NodeDirichlet
    f::Function
    nodes::Vector{Int}
    field_name::Symbol
    components::Vector{Int}
    dofs::Vector{Int}
end
function Dirichlet(field_name::Symbol, nodes::Union{Set{Int}, Vector{Int}}, f::Function, component::Int=1)
    Dirichlet(field_name, nodes, f, [component])
end
function Dirichlet(field_name::Symbol, nodes::Union{Set{Int}, Vector{Int}}, f::Function, components::Vector{Int})
    unique(components) == components || error("components not unique: $components")
    # issorted(components) || error("components not sorted: $components")
    return NodeDirichlet(f, sort!([node for node in nodes]), field_name, components, Int[])
end

"""
    ConstraintHandler

Collection of constraints.
"""
struct ConstraintHandler{DH<:DofHandler,T}
    face_dbcs::Vector{FaceDirichlet}
    node_dbcs::Vector{NodeDirichlet}
    prescribed_dofs::Vector{Int}
    free_dofs::Vector{Int}
    values::Vector{T}
    dofmapping::Dict{Int,Int} # global dof -> index into dofs and values
    dh::DH
    closed::ScalarWrapper{Bool}
end

function ConstraintHandler(dh::DofHandler)
    @assert isclosed(dh)
    ConstraintHandler(FaceDirichlet[], NodeDirichlet[], Int[], Int[], Float64[], Dict{Int,Int}(), dh, ScalarWrapper(false))
end

function Base.show(io::IO, ch::ConstraintHandler)
    println(io, "ConstraintHandler:")
    if !isclosed(ch)
        print(io, "  Not closed!")
    else
        print(io, "  Face BCs:")
        for dbc in ch.face_dbcs
            print(io, "\n    ", "Field: ", dbc.field_name, ", ", "Components: ", dbc.components)
        end
        print(io, "\n  Node BCs:")
        for dbc in ch.node_dbcs
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

function dbc_check(ch::ConstraintHandler, dbc::Union{NodeDirichlet, FaceDirichlet})
    # check input
    dbc.field_name in ch.dh.field_names || throw(ArgumentError("field $field does not exist in DofHandler, existing fields are $(dh.field_names)"))
    for component in dbc.components
        0 < component <= ndim(ch.dh, dbc.field_name) || error("component $component is not within the range of field $field which has $(ndim(ch.dh, field)) dimensions")
    end
    if dbc isa FaceDirichlet
        if length(dbc.faces) == 0
            warn("added Dirichlet Boundary Condition to face set containing 0 faces")
        end
    elseif dbc isa NodeDirichlet
        if length(dbc.nodes) == 0
            warn("added Dirichlet Boundary Condition to node set containing 0 nodes")
        end
    end                
end

"""
    add!(ch::ConstraintHandler, dbc::Union{NodeDirichlet, FaceDirichlet})

Add a `Dirichlet boundary` condition to the `ConstraintHandler`.
"""
function add!(ch::ConstraintHandler, dbc::Union{NodeDirichlet, FaceDirichlet})
    dbc_check(ch, dbc)
    field_idx = find_field(ch.dh, dbc.field_name)
    # Extract stuff for the field
    interpolation = ch.dh.field_interpolations[field_idx]
    field_dim = ch.dh.field_dims[field_idx] # TODO: I think we don't need to extract these here ...
    if dbc isa NodeDirichlet
        _add!(ch, dbc, field_dim, field_offset_nodes(ch.dh, dbc.field_name))
    else
        _add!(ch, dbc, interpolation, field_dim, field_offset(ch.dh, dbc.field_name))
    end
    return ch
end

function _add!(ch::ConstraintHandler, dbc::FaceDirichlet, interpolation::Interpolation, field_dim::Int, offset::Int)
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
    _celldofs = fill(0, ndofs_per_cell(ch.dh))
    for (cellidx, faceidx) in dbc.faces
        celldofs!(_celldofs, ch.dh, cellidx) # extract the dofs for this cell
        r = local_face_dofs_offset[faceidx]:(local_face_dofs_offset[faceidx+1]-1)
        append!(constrained_dofs, _celldofs[local_face_dofs[r]]) # TODO: for-loop over r and simply push! to ch.prescribed_dofs
        @debug println("adding dofs $(_celldofs[local_face_dofs[r]]) to dbc")
    end

    # save it to the ConstraintHandler
    push!(ch.face_dbcs, dbc)
    append!(ch.prescribed_dofs, constrained_dofs)
end

function _add!(ch::ConstraintHandler, dbc::NodeDirichlet, field_dim::Int, field_offset::Int)
    dofs = dbc.dofs
    resize!(dofs, length(dbc.nodes) * length(dbc.components))
    count = 1
    for node in dbc.nodes
        start = ch.dh.ndofs_per_node[] * (node-1) + field_offset 
        r = start : (start + field_dim)
        for c in dbc.components
            dofs[count] = ch.dh.node_dofs[r[c]]
            count += 1
        end
    end
    push!(ch.node_dbcs, dbc)
    append!(ch.prescribed_dofs, dofs)
end

# Updates the DBC's to the current time `time`
function update!(ch::ConstraintHandler, time::Float64=0.0)
    @assert ch.closed[]
    for dbc in ch.face_dbcs
        field_idx = find_field(ch.dh, dbc.field_name)
        # Function barrier
        _update_face_dbc!(ch.values, dbc.f, dbc.faces, dbc.field_name, dbc.local_face_dofs, dbc.local_face_dofs_offset,
                 dbc.components, ch.dh, ch.dh.bc_values[field_idx], ch.dofmapping, time)
    end
    for dbc in ch.node_dbcs
        # Function barrier
        _update_node_dbc!(ch.values, dbc.f, dbc.nodes, dbc.dofs,
                 dbc.components, ch.dh, ch.dofmapping, time)
    end
end

function _update_face_dbc!(values::Vector{Float64}, f::Function, faces::Set{Tuple{Int,Int}}, field::Symbol, local_face_dofs::Vector{Int}, local_face_dofs_offset::Vector{Int},
                  components::Vector{Int}, dh::DofHandler{dim,N,T,M}, facevalues::BCValues,
                  dofmapping::Dict{Int,Int}, time::Float64) where {dim,N,T,M}
    grid = dh.grid

    xh = zeros(Vec{dim, T}, N) # pre-allocate
    _celldofs = fill(0, ndofs_per_cell(dh))

    for (cellidx, faceidx) in faces
        getcoordinates!(xh, grid, cellidx)
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

function _update_node_dbc!(values::Vector{Float64}, f::Function, nodes::Vector{Int}, 
    dbc_dofs::Vector{Int}, components::Vector{Int}, dh::DofHandler{dim,N,T,M}, 
    dofmapping::Dict{Int,Int}, time::Float64) where {dim, N, T, M}

    dbc_node_dofs = reshape(dbc_dofs, length(components), length(nodes))
    grid = dh.grid
    for (j, nodeidx) in enumerate(nodes)
        bc_value = f(grid.nodes[nodeidx], time)
        @assert length(bc_value) == length(components)
        for i in 1:length(components)
            dof = dbc_node_dofs[i,j]
            dbc_index = dofmapping[dof]
            values[dbc_index] = bc_value[i]
        end
    end
end

# Saves the dirichlet boundary conditions to a vtkfile.
# Values will have a 1 where bcs are active and 0 otherwise
function WriteVTK.vtk_point_data(vtkfile, ch::ConstraintHandler)
    unique_fields = []
    for dbc in ch.face_dbcs
        push!(unique_fields, dbc.field_name)
    end
    for dbc in ch.node_dbcs
        push!(unique_fields, dbc.field_name)
    end
    unique_fields = unique(unique_fields) # TODO v0.7: unique!(unique_fields)

    for field in unique_fields
        nd = ndim(ch.dh, field)
        data = zeros(Float64, nd, getnnodes(ch.dh.grid))
        for face_dbc in ch.face_dbcs
            face_dbc.field_name != field && continue
            for (cellidx, faceidx) in face_dbc.faces
                for facenode in faces(ch.dh.grid.cells[cellidx])[faceidx]
                    for component in face_dbc.components
                        data[component, facenode] = 1
                    end
                end
            end
        end
        for node_dbc in ch.node_dbcs
            node_dbc.field_name != field && continue
            for nodeidx in node_dbc.nodes
                for component in node_dbc.components
                    data[component, nodeidx] = 1
                end
            end
        end
        vtk_point_data(vtkfile, data, string(field, "_bc"))
    end
    return vtkfile
end

function apply!(v::Vector, ch::ConstraintHandler)
    @assert length(v) == ndofs(ch.dh)
    v[ch.prescribed_dofs] = ch.values # .= ??
    return v
end

function apply_zero!(v::Vector, ch::ConstraintHandler)
    @assert length(v) == ndofs(ch.dh)
    v[ch.prescribed_dofs] = 0 # .= ?
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
        K[ch.prescribed_dofs, :] = 0
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
    @debug assert(issorted(dofs))
    for col in dofs
        r = nzrange(K, col)
        K.nzval[r] = 0.0
    end
end


function meandiag(K::AbstractMatrix)
    z = zero(eltype(K))
    for i in 1:size(K, 1)
        z += abs(K[i, i])
    end
    return z / size(K, 1)
end
