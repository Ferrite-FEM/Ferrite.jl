# abstract type Constraint end
"""
    DirichletBoundaryConditions

A Dirichlet boundary condition is a boundary where the solution is fixed to take a certain value.
The struct `DirichletBoundaryConditions` represents a collection of such boundary conditions.

It is created from a `DofHandler`

```jldoctest dbc
julia> dbc = DirichletBoundaryConditions(dh)
```

Dirichlet boundary conditions are added to certain components of a field for a specific nodes of the grid.
A function is also given that should be of the form `(x,t) -> v` where `x` is the coordinate of the node, `t` is a
time parameter and `v` should be of the same length as the number of components the bc is applied to:

```jldoctest
julia> addnodeset!(grid, "clamped", x -> norm(x[1]) ≈ 0.0);

julia> nodes = grid.nodesets["clamped"]

julia> push!(dbc, :temperature, nodes, (x,t) -> t * [x[2], 0.0, 0.0], [1, 2, 3])
```

Boundary conditions are now updates by specifying a time:

```jldoctest
julia> t = 1.0;

julia> update!(dbc, t)
```

The boundary conditions can be applied to a vector:

```jldoctest
julia> u = zeros(ndofs(dh))

julia> apply!(u, dbc)
```

"""
struct DirichletBoundaryCondition # <: Constraint
    f::Function # f(x,t) -> value
    faces::Set{Tuple{Int,Int}}
    field_name::Symbol
    components::Vector{Int} # components of the field
    local_face_dofs::Vector{Int}
    local_face_dofs_offset::Vector{Int}
end
function DirichletBoundaryCondition(field_name::Symbol, faces::Set{Tuple{Int,Int}}, f::Function, component::Int=1)
    DirichletBoundaryCondition(field_name, faces, f, [component])
end
function DirichletBoundaryCondition(field_name::Symbol, faces::Set{Tuple{Int,Int}}, f::Function, components::Vector{Int})
    unique(components) == components || error("components not unique: $components")
    # issorted(components) || error("components not sorted: $components")
    return DirichletBoundaryCondition(f, faces, field_name, components, Int[], Int[])
end

struct DirichletBoundaryConditions{DH<:DofHandler,T}
    dbcs::Vector{DirichletBoundaryCondition}
    prescribed_dofs::Vector{Int}
    free_dofs::Vector{Int}
    values::Vector{T}
    dofmapping::Dict{Int,Int} # global dof -> index into dofs and values
    dh::DH
    closed::ScalarWrapper{Bool}
end

function DirichletBoundaryConditions(dh::DofHandler)
    @assert isclosed(dh)
    DirichletBoundaryConditions(DirichletBoundaryCondition[], Int[], Int[], Float64[], Dict{Int,Int}(), dh, ScalarWrapper(false))
end

function Base.show(io::IO, dbcs::DirichletBoundaryConditions)
    println(io, "DirichletBoundaryConditions:")
    if !isclosed(dbcs)
        print(io, "  Not closed!")
    else
        println(io, "  BCs:")
        for dbc in dbcs.dbcs
            println(io, "    ", "Field: ", dbc.field_name, ", ", "Components: ", dbc.components)
        end
    end
end

isclosed(dbcs::DirichletBoundaryConditions) = dbcs.closed[]
free_dofs(dbcs::DirichletBoundaryConditions) = dbcs.free_dofs
prescribed_dofs(dbcs::DirichletBoundaryConditions) = dbcs.prescribed_dofs

function close!(dbcs::DirichletBoundaryConditions)
    fdofs = setdiff(1:ndofs(dbcs.dh), dbcs.prescribed_dofs)
    copy!!(dbcs.free_dofs, fdofs)
    copy!!(dbcs.prescribed_dofs, unique(dbcs.prescribed_dofs)) # for v0.7: unique!(dbcs.prescribed_dofs)
    sort!(dbcs.prescribed_dofs) # YOLO
    fill!(resize!(dbcs.values, length(dbcs.prescribed_dofs)), NaN)
    for i in 1:length(dbcs.prescribed_dofs)
        dbcs.dofmapping[dbcs.prescribed_dofs[i]] = i
    end
    dbcs.closed[] = true
    return dbcs
end

function dbc_check(dbcs::DirichletBoundaryConditions, dbc::DirichletBoundaryCondition)
    # check input
    dbc.field_name in dbcs.dh.field_names || throw(ArgumentError("field $field does not exist in DofHandler, existing fields are $(dh.field_names)"))
    for component in dbc.components
        0 < component <= ndim(dbcs.dh, dbc.field_name) || error("component $component is not within the range of field $field which has $(ndim(dbcs.dh, field)) dimensions")
    end
    if length(dbc.faces) == 0
        warn("added Dirichlet Boundary Condition to face set containing 0 faces")
    end
end

# Adds a boundary condition to the DirichletBoundaryConditions
function add!(dbcs::DirichletBoundaryConditions, dbc::DirichletBoundaryCondition)
    dbc_check(dbcs, dbc)
    field_idx = find_field(dbcs.dh, dbc.field_name)
    # Extract stuff for the field
    interpolation = dbcs.dh.field_interpolations[field_idx]
    field_dim = dbcs.dh.field_dims[field_idx] # TODO: I think we don't need to extract these here ...
    _add!(dbcs, dbc, interpolation, field_dim, field_offset(dbcs.dh, dbc.field_name))
    return dbcs
end

function _add!(dbcs::DirichletBoundaryConditions, dbc::DirichletBoundaryCondition, interpolation::Interpolation, field_dim::Int, offset::Int)
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
    _celldofs = fill(0, ndofs_per_cell(dbcs.dh))
    for (cellidx, faceidx) in dbc.faces
        celldofs!(_celldofs, dbcs.dh, cellidx) # extract the dofs for this cell
        r = local_face_dofs_offset[faceidx]:(local_face_dofs_offset[faceidx+1]-1)
        append!(constrained_dofs, _celldofs[local_face_dofs[r]]) # TODO: for-loop over r and simply push! to dbcs.prescribed_dofs
        @debug println("adding dofs $(_celldofs[local_face_dofs[r]]) to dbc")
    end

    # save it to the DirichletBoundaryConditions
    push!(dbcs.dbcs, dbc)
    append!(dbcs.prescribed_dofs, constrained_dofs)
end

# Updates the DBC's to the current time `time`
function update!(dbcs::DirichletBoundaryConditions, time::Float64=0.0)
    @assert dbcs.closed[]
    for dbc in dbcs.dbcs
        field_idx = find_field(dbcs.dh, dbc.field_name)
        # Function barrier
        _update!(dbcs.values, dbc.f, dbc.faces, dbc.field_name, dbc.local_face_dofs, dbc.local_face_dofs_offset,
                 dbc.components, dbcs.dh, dbcs.dh.bc_values[field_idx], dbcs.dofmapping, time)
    end
end

function _update!(values::Vector{Float64}, f::Function, faces::Set{Tuple{Int,Int}}, field::Symbol, local_face_dofs::Vector{Int}, local_face_dofs_offset::Vector{Int},
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

# Saves the dirichlet boundary conditions to a vtkfile.
# Values will have a 1 where bcs are active and 0 otherwise
function WriteVTK.vtk_point_data(vtkfile, dbcs::DirichletBoundaryConditions)
    unique_fields = []
    for dbc in dbcs.dbcs
        push!(unique_fields, dbc.field_name)
    end
    unique_fields = unique(unique_fields) # TODO v0.7: unique!(unique_fields)

    for field in unique_fields
        nd = ndim(dbcs.dh, field)
        data = zeros(Float64, nd, getnnodes(dbcs.dh.grid))
        for dbc in dbcs.dbcs
            dbc.field_name != field && continue
            for (cellidx, faceidx) in dbc.faces
                for facenode in faces(dbcs.dh.grid.cells[cellidx])[faceidx]
                    for component in dbc.components
                        data[component, facenode] = 1
                    end
                end
            end
        end
        vtk_point_data(vtkfile, data, string(field, "_bc"))
    end
    return vtkfile
end

function apply!(v::Vector, dbcs::DirichletBoundaryConditions)
    @assert length(v) == ndofs(dbcs.dh)
    v[dbcs.prescribed_dofs] = dbcs.values # .= ??
    return v
end

function apply_zero!(v::Vector, dbcs::DirichletBoundaryConditions)
    @assert length(v) == ndofs(dbcs.dh)
    v[dbcs.prescribed_dofs] = 0 # .= ?
    return v
end

function apply!(K::Union{SparseMatrixCSC,Symmetric}, dbcs::DirichletBoundaryConditions)
    apply!(K, eltype(K)[], dbcs, true)
end

function apply_zero!(K::Union{SparseMatrixCSC,Symmetric}, f::AbstractVector, dbcs::DirichletBoundaryConditions)
    apply!(K, f, dbcs, true)
end

@enum(ApplyStrategy, APPLY_TRANSPOSE, APPLY_INPLACE)

function apply!(KK::Union{SparseMatrixCSC,Symmetric}, f::AbstractVector, dbcs::DirichletBoundaryConditions, applyzero::Bool=false;
                strategy::ApplyStrategy=APPLY_TRANSPOSE)
    K = isa(KK, Symmetric) ? KK.data : KK
    @assert length(f) == 0 || length(f) == size(K, 1)
    @boundscheck checkbounds(K, dbcs.prescribed_dofs, dbcs.prescribed_dofs)
    @boundscheck length(f) == 0 || checkbounds(f, dbcs.prescribed_dofs)

    m = meandiag(K) # Use the mean of the diagonal here to not ruin things for iterative solver
    @inbounds for i in 1:length(dbcs.values)
        d = dbcs.prescribed_dofs[i]
        v = dbcs.values[i]

        if !applyzero && v != 0
            for j in nzrange(K, d)
                f[K.rowval[j]] -= v * K.nzval[j]
            end
        end
    end
    zero_out_columns!(K, dbcs.prescribed_dofs)
    if strategy == APPLY_TRANSPOSE
        K′ = copy(K)
        transpose!(K′, K)
        zero_out_columns!(K′, dbcs.prescribed_dofs)
        transpose!(K, K′)
    elseif strategy == APPLY_INPLACE
        K[dbcs.prescribed_dofs, :] = 0
    else
        error("Unknown apply strategy")
    end
    @inbounds for i in 1:length(dbcs.values)
        d = dbcs.prescribed_dofs[i]
        v = dbcs.values[i]
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
