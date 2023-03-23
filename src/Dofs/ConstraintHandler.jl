# abstract type Constraint end
"""
    Dirichlet(u::Symbol, ∂Ω::Set, f::Function, components=nothing)

Create a Dirichlet boundary condition on `u` on the `∂Ω` part of
the boundary. `f` is a function of the form `f(x)` or `f(x, t)`
where `x` is the spatial coordinate and `t` is the current time,
and returns the prescribed value. `components` specify the components
of `u` that are prescribed by this condition. By default all components
of `u` are prescribed.

For example, here we create a
Dirichlet condition for the `:u` field, on the faceset called
`∂Ω` and the value given by the `sin` function:

*Examples*
```jldoctest
# Obtain the faceset from the grid
∂Ω = getfaceset(grid, "boundary-1")

# Prescribe scalar field :s on ∂Ω to sin(t)
dbc = Dirichlet(:s, ∂Ω, (x, t) -> sin(t))

# Prescribe all components of vector field :v on ∂Ω to 0
dbc = Dirichlet(:v, ∂Ω, x -> 0 * x)

# Prescribe component 2 and 3 of vector field :v on ∂Ω to [sin(t), cos(t)]
dbc = Dirichlet(:v, ∂Ω, (x, t) -> [sin(t), cos(t)], [2, 3])
```

`Dirichlet` boundary conditions are added to a [`ConstraintHandler`](@ref)
which applies the condition via [`apply!`](@ref) and/or [`apply_zero!`](@ref).
"""
struct Dirichlet # <: Constraint
    f::Function # f(x) or f(x,t) -> value(s)
    faces::Union{Set{Int},Set{FaceIndex},Set{EdgeIndex},Set{VertexIndex}}
    field_name::Symbol
    components::Vector{Int} # components of the field
    local_face_dofs::Vector{Int}
    local_face_dofs_offset::Vector{Int}
end
function Dirichlet(field_name::Symbol, faces::Set, f::Function, components=nothing)
    return Dirichlet(f, copy(faces), field_name, __to_components(components), Int[], Int[])
end

# components=nothing is default and means that all components should be constrained
# but since number of components isn't known here it will be populated in add!
__to_components(::Nothing) = Int[]
function __to_components(c)
    components = convert(Vector{Int}, vec(collect(Int, c)))
    isempty(components) && error("components are empty: $c")
    issorted(components) || error("components not sorted: $c")
    allunique(components) || error("components not unique: $c")
    return components
end

const DofCoefficients{T} = Vector{Pair{Int,T}}
"""
    AffineConstraint(constrained_dof::Int, entires::Vector{Pair{Int,T}}, b::T) where T

Define an affine/linear constraint to constrain one degree of freedom, `u[i]`, 
such that `u[i] = ∑(u[j] * a[j]) + b`, 
where `i=constrained_dof` and each element in `entries` are `j => a[j]`
"""
struct AffineConstraint{T}
    constrained_dof::Int
    entries::DofCoefficients{T} # masterdofs and factors
    b::T # inhomogeneity
end

"""
    ConstraintHandler

Collection of constraints.
"""
struct ConstraintHandler{DH<:AbstractDofHandler,T}
    dbcs::Vector{Dirichlet}
    prescribed_dofs::Vector{Int}
    free_dofs::Vector{Int}
    inhomogeneities::Vector{T}
    # Store the original constant inhomogeneities for affine constraints used to compute
    # "effective" inhomogeneities in `update!` and then stored in .inhomogeneities.
    affine_inhomogeneities::Vector{Union{Nothing,T}}
    # `nothing` for pure DBC constraint, otherwise affine constraint
    dofcoefficients::Vector{Union{Nothing, DofCoefficients{T}}}
    # global dof -> index into dofs and inhomogeneities and dofcoefficients
    dofmapping::Dict{Int,Int}
    bcvalues::Vector{BCValues{T}}
    dh::DH
    closed::ScalarWrapper{Bool}
end

function ConstraintHandler(dh::AbstractDofHandler)
    @assert isclosed(dh)
    ConstraintHandler(
        Dirichlet[], Int[], Int[], Float64[], Union{Nothing,Float64}[], Union{Nothing,DofCoefficients{Float64}}[],
        Dict{Int,Int}(), BCValues{Float64}[], dh, ScalarWrapper(false),
    )
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

Returns the needed [`RHSData`](@ref) for [`apply_rhs!`](@ref).

This must be used when the same stiffness matrix is reused for multiple steps,
for example when timestepping, with different non-homogeneouos Dirichlet boundary
conditions.
"""
function get_rhs_data(ch::ConstraintHandler, A::SparseMatrixCSC)
    m = meandiag(A)
    constrained_columns = A[:, ch.prescribed_dofs]
    return RHSData(m, constrained_columns)
end

"""
    apply_rhs!(data::RHSData, f::AbstractVector, ch::ConstraintHandler, applyzero::Bool=false)

Applies the boundary condition to the right-hand-side vector without modifying the stiffness matrix.

See also: [`get_rhs_data`](@ref).
"""
function apply_rhs!(data::RHSData, f::AbstractVector, ch::ConstraintHandler, applyzero::Bool=false)
    K = data.constrained_columns
    @assert length(f) == size(K, 1)
    @boundscheck checkbounds(f, ch.prescribed_dofs)
    m = data.m

    # TODO: Can the loops be combined or does the order matter?
    @inbounds for i in 1:length(ch.inhomogeneities)
        v = ch.inhomogeneities[i]
        if !applyzero && v != 0
            for j in nzrange(K, i)
                f[K.rowval[j]] -= v * K.nzval[j]
            end
        end
    end
    @inbounds for (i, pdof) in pairs(ch.prescribed_dofs)
        dofcoef = ch.dofcoefficients[i]
        b = ch.inhomogeneities[i]
        if dofcoef !== nothing # if affine constraint
            for (d, v) in dofcoef
                f[d] += f[pdof] * v
            end
        end
        bz = applyzero ? zero(eltype(f)) : b
        f[pdof] = bz * m
    end
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
    @assert(!isclosed(ch))
    @assert( allunique(ch.prescribed_dofs) )

    I = sortperm(ch.prescribed_dofs)
    ch.prescribed_dofs .= ch.prescribed_dofs[I]
    ch.inhomogeneities .= ch.inhomogeneities[I]
    ch.affine_inhomogeneities .= ch.affine_inhomogeneities[I]
    ch.dofcoefficients .= ch.dofcoefficients[I]

    copy!(ch.free_dofs, setdiff(1:ndofs(ch.dh), ch.prescribed_dofs))

    for i in 1:length(ch.prescribed_dofs)
        ch.dofmapping[ch.prescribed_dofs[i]] = i
    end

    # TODO: Store index for each affine constraint?
    # affine_mapping = Dict{Int,Int}(pdof => i for (i, pdof) in pairs(cd.prescribed_dofs) if ch.dofcoefficients[i] !== nothing )

    # TODO:
    # Do a bunch of checks to see if the affine constraints are linearly indepented etc.
    # If they are not, it is possible to automatically reformulate the constraints
    # such that they become independent. However, at this point, it is left to
    # the user to assure this.

    # Basic verification of constraints:
    # - `add_prescribed_dof` make sure all prescribed dofs are unique by overwriting the old
    #   constraint when adding a new (TODO: Might change in the future, see comment in
    #   `add_prescribed_dof`.)
    # - We allow affine constraints to have prescribed dofs as master dofs iff those master
    #   dofs are constrained with just an inhomogeneity (i.e. DBC). The effective
    #   inhomogeneity is computed in `update!`.
    for coeffs in ch.dofcoefficients
        coeffs === nothing && continue
        for (d, _) in coeffs
            i = get(ch.dofmapping, d, 0)
            i == 0 && continue
            icoeffs = ch.dofcoefficients[i]
            if !(icoeffs === nothing || isempty(icoeffs))
                error("nested affine constraints currently not supported")
            end
        end
    end

    ch.closed[] = true

    # Compute the prescribed values by calling update!: This should be cheap, and for the
    # common case where constraints does not depend on time it is annoying and easy to
    # forget to call this on the outside.
    update!(ch)

    return ch
end

"""
    add!(ch::ConstraintHandler, dbc::Dirichlet)

Add a `Dirichlet` boundary condition to the `ConstraintHandler`.
"""
function add!(ch::ConstraintHandler, dbc::Dirichlet)
    if length(dbc.faces) == 0
        @warn("adding Dirichlet Boundary Condition to set containing 0 entities")
    end
    celltype = getcelltype(ch.dh.grid)
    @assert isconcretetype(celltype)

    # Extract stuff for the field
    field_idx = find_field(ch.dh, dbc.field_name) # throws if name not found
    interpolation = getfieldinterpolation(ch.dh, field_idx)
    field_dim = getfielddim(ch.dh, field_idx)

    if !all(c -> 0 < c <= field_dim, dbc.components)
        error("components $(dbc.components) not within range of field :$(dbc.field_name) ($(field_dim) dimension(s))")
    end

    # Empty components means constrain them all
    isempty(dbc.components) && append!(dbc.components, 1:field_dim)

    if eltype(dbc.faces)==Int #Special case when dbc.faces is a nodeset
        bcvalue = BCValues(interpolation, default_interpolation(celltype), FaceIndex) #Not used by node bcs, but still have to pass it as an argument
    else
        bcvalue = BCValues(interpolation, default_interpolation(celltype), eltype(dbc.faces))
    end
    _add!(ch, dbc, dbc.faces, interpolation, field_dim, field_offset(ch.dh, dbc.field_name), bcvalue)

    return ch
end

"""
    add!(ch::ConstraintHandler, ac::AffineConstraint)

Add the `AffineConstraint` to the `ConstraintHandler`.
"""
function add!(ch::ConstraintHandler, ac::AffineConstraint)
    # TODO: Would be nice to pass nothing if ac.entries is empty, but then we lose the fact
    #       that this constraint is an AffineConstraint which is currently needed in update!
    #       in order to not update inhomogeneities for affine constraints
    add_prescribed_dof!(ch, ac.constrained_dof, ac.b, #=isempty(ac.entries) ? nothing : =# ac.entries)
end

"""
    add_prescribed_dof!(ch, constrained_dof::Int, inhomogeneity, dofcoefficients=nothing)

Add a constrained dof directly to the `ConstraintHandler`.
This function checks if the `constrained_dof` is already constrained, and overrides the old
constraint if true.
"""
function add_prescribed_dof!(ch::ConstraintHandler, constrained_dof::Int, inhomogeneity, dofcoefficients=nothing)
    @assert(!isclosed(ch))
    i = get(ch.dofmapping, constrained_dof, 0)
    if i != 0
        @debug @warn "dof $i already prescribed, overriding the old constraint"
        ch.prescribed_dofs[i] = constrained_dof
        ch.inhomogeneities[i] = inhomogeneity
        ch.affine_inhomogeneities[i] = dofcoefficients === nothing ? nothing : inhomogeneity
        ch.dofcoefficients[i] = dofcoefficients
    else
        N = length(ch.dofmapping)
        push!(ch.prescribed_dofs, constrained_dof)
        push!(ch.inhomogeneities, inhomogeneity)
        push!(ch.affine_inhomogeneities, dofcoefficients === nothing ? nothing : inhomogeneity)
        push!(ch.dofcoefficients, dofcoefficients)
        ch.dofmapping[constrained_dof] = N + 1
    end
    return ch
end

function _add!(ch::ConstraintHandler, dbc::Dirichlet, bcfaces::Set{Index}, interpolation::Interpolation, field_dim::Int, offset::Int, bcvalue::BCValues, cellset::Set{Int}=Set{Int}(1:getncells(ch.dh.grid))) where {Index<:BoundaryIndex}
    local_face_dofs, local_face_dofs_offset =
        _local_face_dofs_for_bc(interpolation, field_dim, dbc.components, offset, boundarydof_indices(eltype(bcfaces)))
    copy!(dbc.local_face_dofs, local_face_dofs)
    copy!(dbc.local_face_dofs_offset, local_face_dofs_offset)

    # loop over all the faces in the set and add the global dofs to `constrained_dofs`
    constrained_dofs = Int[]
    cc = CellCache(ch.dh, UpdateFlags(; nodes=false, coords=false, dofs=true))
    for (cellidx, faceidx) in bcfaces
        if cellidx ∉ cellset
            delete!(dbc.faces, Index(cellidx, faceidx))
            continue # skip faces that are not part of the cellset
        end
        reinit!(cc, cellidx)
        r = local_face_dofs_offset[faceidx]:(local_face_dofs_offset[faceidx+1]-1)
        append!(constrained_dofs, cc.dofs[local_face_dofs[r]]) # TODO: for-loop over r and simply push! to ch.prescribed_dofs
        @debug println("adding dofs $(cc.dofs[local_face_dofs[r]]) to dbc")
    end

    # save it to the ConstraintHandler
    push!(ch.dbcs, dbc)
    push!(ch.bcvalues, bcvalue)
    for d in constrained_dofs
        add_prescribed_dof!(ch, d, NaN, nothing)
    end
    return ch
end

# Calculate which local dof index live on each face:
# face `i` have dofs `local_face_dofs[local_face_dofs_offset[i]:local_face_dofs_offset[i+1]-1]
function _local_face_dofs_for_bc(interpolation, field_dim, components, offset, boundaryfunc::F=facedof_indices) where F
    @assert issorted(components)
    local_face_dofs = Int[]
    local_face_dofs_offset = Int[1]
    for (_, face) in enumerate(boundaryfunc(interpolation))
        for fdof in face, d in 1:field_dim
            if d in components
                push!(local_face_dofs, (fdof-1)*field_dim + d + offset)
            end
        end
        push!(local_face_dofs_offset, length(local_face_dofs) + 1)
    end
    return local_face_dofs, local_face_dofs_offset
end

function _add!(ch::ConstraintHandler, dbc::Dirichlet, bcnodes::Set{Int}, interpolation::Interpolation, field_dim::Int, offset::Int, bcvalue::BCValues, cellset::Set{Int}=Set{Int}(1:getncells(ch.dh.grid)))
    if interpolation !== default_interpolation(typeof(ch.dh.grid.cells[first(cellset)]))
        @warn("adding constraint to nodeset is not recommended for sub/super-parametric approximations.")
    end

    ncomps = length(dbc.components)
    nnodes = getnnodes(ch.dh.grid)
    interpol_points = getnbasefunctions(interpolation)
    node_dofs = zeros(Int, ncomps, nnodes)
    visited = falses(nnodes)
    for cell in CellIterator(ch.dh, cellset) # only go over cells that belong to current FieldHandler
        for idx in 1:min(interpol_points, length(cell.nodes))
            node = cell.nodes[idx]
            if !visited[node]
                noderange = (offset + (idx-1)*field_dim + 1):(offset + idx*field_dim) # the dofs in this node
                for (i,c) in enumerate(dbc.components)
                    node_dofs[i,node] = cell.dofs[noderange[c]]
                    @debug println("adding dof $(cell.dofs[noderange[c]]) to node_dofs")
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
    copy!(dbc.local_face_dofs_offset, constrained_dofs) # use this field to store the global dofs
    push!(ch.dbcs, dbc)
    push!(ch.bcvalues, bcvalue)
    for d in constrained_dofs
        add_prescribed_dof!(ch, d, NaN, nothing)
    end
    return ch
end

"""
    update!(ch::ConstraintHandler, time::Real=0.0)

Update time-dependent inhomogeneities for the new time. This calls `f(x)` or `f(x, t)` when
applicable, where `f` is the function(s) corresponding to the constraints in the handler, to
compute the inhomogeneities.

Note that this is called implicitly in `close!(::ConstraintHandler)`.
"""
function update!(ch::ConstraintHandler, time::Real=0.0)
    @assert ch.closed[]
    for (i, dbc) in pairs(ch.dbcs)
        # If the BC function only accept one argument, i.e. f(x), we create a wrapper
        # g(x, t) = f(x) that discards the second parameter so that _update! can always call
        # the function with two arguments internally.
        wrapper_f = hasmethod(dbc.f, Tuple{Any,Any}) ? dbc.f : (x, _) -> dbc.f(x)
        # Function barrier
        _update!(ch.inhomogeneities, wrapper_f, dbc.faces, dbc.field_name, dbc.local_face_dofs, dbc.local_face_dofs_offset,
                 dbc.components, ch.dh, ch.bcvalues[i], ch.dofmapping, ch.dofcoefficients, time)
    end
    # Compute effective inhomogeneity for affine constraints with prescribed dofs in the
    # RHS. For example, in u2 = w3 * u3 + w4 * u4 + b2 we allow e.g. u3 to be prescribed by
    # a trivial constraint with just an inhomogeneity (e.g. DBC), for example u3 = f(t).
    # This value have now been computed in _update! and we can compute the effective
    # inhomogeneity h2 for u2 which becomes h2 = w3 * u3 + b2 = w3 * f3(t) + b2.
    for i in eachindex(ch.prescribed_dofs, ch.dofcoefficients, ch.inhomogeneities)
        coeffs = ch.dofcoefficients[i]
        coeffs === nothing && continue
        h = ch.affine_inhomogeneities[i]
        @assert h !== nothing
        for (d, w) in coeffs
            j = get(ch.dofmapping, d, 0)
            j == 0 && continue
            # If this dof is prescribed it must only have an inhomogeneity (verified in close!)
            @assert (jcoeffs = ch.dofcoefficients[j]; jcoeffs === nothing || isempty(jcoeffs))
            h += ch.inhomogeneities[j] * w
        end
        ch.inhomogeneities[i] = h
    end
    return nothing
end

# for vertices, faces and edges
function _update!(inhomogeneities::Vector{Float64}, f::Function, boundary_entities::Set{<:BoundaryIndex}, field::Symbol, local_face_dofs::Vector{Int}, local_face_dofs_offset::Vector{Int},
                  components::Vector{Int}, dh::AbstractDofHandler, boundaryvalues::BCValues,
                  dofmapping::Dict{Int,Int}, dofcoefficients::Vector{Union{Nothing,DofCoefficients{T}}}, time::Real) where {T}

    cc = CellCache(dh, UpdateFlags(; nodes=false, coords=true, dofs=true))
    for (cellidx, entityidx) in boundary_entities
        reinit!(cc, cellidx)

        # no need to reinit!, enough to update current_entity since we only need geometric shape functions M
        boundaryvalues.current_entity[] = entityidx

        # local dof-range for this face
        r = local_face_dofs_offset[entityidx]:(local_face_dofs_offset[entityidx+1]-1)
        counter = 1
        for location in 1:getnquadpoints(boundaryvalues)
            x = spatial_coordinate(boundaryvalues, location, cc.coords)
            bc_value = f(x, time)
            @assert length(bc_value) == length(components)

            for i in 1:length(components)
                # find the global dof
                globaldof = cc.dofs[local_face_dofs[r[counter]]]
                counter += 1

                dbc_index = dofmapping[globaldof]
                # Only DBC dofs are currently update!-able so don't modify inhomogeneities
                # for affine constraints
                if dofcoefficients[dbc_index] === nothing
                    inhomogeneities[dbc_index] = bc_value[i]
                    @debug println("prescribing value $(bc_value[i]) on global dof $(globaldof)")
                end
            end
        end
    end
end

# for nodes
function _update!(inhomogeneities::Vector{Float64}, f::Function, nodes::Set{Int}, field::Symbol, nodeidxs::Vector{Int}, globaldofs::Vector{Int},
                  components::Vector{Int}, dh::AbstractDofHandler, facevalues::BCValues,
                  dofmapping::Dict{Int,Int}, dofcoefficients::Vector{Union{Nothing,DofCoefficients{T}}}, time::Real) where T
    counter = 1
    for (idx, nodenumber) in enumerate(nodeidxs)
        x = dh.grid.nodes[nodenumber].x
        bc_value = f(x, time)
        @assert length(bc_value) == length(components)
        for v in bc_value
            globaldof = globaldofs[counter]
            counter += 1
            dbc_index = dofmapping[globaldof]
            # Only DBC dofs are currently update!-able so don't modify inhomogeneities
            # for affine constraints
            if dofcoefficients[dbc_index] === nothing
                inhomogeneities[dbc_index] = v
                @debug println("prescribing value $(v) on global dof $(globaldof)")
            end
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
    unique!(unique_fields)

    for field in unique_fields
        nd = ndim(ch.dh, field)
        data = zeros(Float64, nd, getnnodes(ch.dh.grid))
        for dbc in ch.dbcs
            dbc.field_name != field && continue
            if eltype(dbc.faces) <: BoundaryIndex
                functype = boundaryfunction(eltype(dbc.faces))
                for (cellidx, faceidx) in dbc.faces
                    for facenode in functype(ch.dh.grid.cells[cellidx])[faceidx]
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

"""

    apply!(K::SparseMatrixCSC, rhs::AbstractVector, ch::ConstraintHandler)

Adjust the matrix `K` and right hand side `rhs` to account for the Dirichlet boundary
conditions specified in `ch` such that `K \\ rhs` gives the expected solution.

!!! note
    `apply!(K, rhs, ch)` essentially calculates

    `rhs[free_dofs] = rhs[free_dofs] - K[free_dofs, constrained_dofs] * a[constrained]`
    
    where `a[constrained]` are the inhomogeneities. 
    Consequently, the sign of `rhs` matters (in contrast to for `apply_zero!`).


    apply!(v::AbstractVector, ch::ConstraintHandler)

Apply Dirichlet boundary conditions and affine constraints, specified in `ch`, to the solution vector `v`.

# Examples
```julia
K, f = assemble_system(...) # Assemble system
apply!(K, f, ch)            # Adjust K and f to account for boundary conditions
u = K \\ f                   # Solve the system, u should be "approximately correct"
apply!(u, ch)               # Explicitly make sure bcs are correct
```

!!! note
    The last operation is not strictly necessary since the boundary conditions should
    already be fulfilled after `apply!(K, f, ch)`. However, solvers of linear systems are
    not exact, and thus `apply!(u, ch)` can be used to make sure the boundary conditions
    are fulfilled exactly.
"""
apply!

"""
    apply_zero!(K::SparseMatrixCSC, rhs::AbstractVector, ch::ConstraintHandler)

Adjust the matrix `K` and the right hand side `rhs` to account for prescribed Dirichlet
boundary conditions and affine constraints such that `du = K \\ rhs` gives the expected 
result (e.g. `du` zero for all prescribed degrees of freedom).

    apply_zero!(v::AbstractVector, ch::ConstraintHandler)

Zero-out values in `v` corresponding to prescribed degrees of freedom and update values 
prescribed by affine constraints, such that if `a` fullfills the constraints,
`a ± v` also will.

These methods are typically used in e.g. a Newton solver where the increment, `du`, should
be prescribed to zero even for non-homogeneouos boundary conditions.

See also: [`apply!`](@ref).

# Examples
```julia
u = un + Δu                 # Current guess
K, g = assemble_system(...) # Assemble residual and tangent for current guess
apply_zero!(K, g, ch)       # Adjust tangent and residual to take prescribed values into account
ΔΔu = K \\ g                # Compute the (negative) increment, prescribed values are "approximately" zero
apply_zero!(ΔΔu, ch)        # Make sure values are exactly zero
Δu .-= ΔΔu                  # Update current guess
```

!!! note
    The last call to `apply_zero!` is only strictly necessary for affine constraints. 
    However, even if the Dirichlet boundary conditions should be fulfilled after 
    `apply!(K, g, ch)`, solvers of linear systems are not exact. 
    `apply!(ΔΔu, ch)` can be used to make sure the values
    for the prescribed degrees of freedom are fulfilled exactly.
"""
apply_zero!

apply_zero!(v::AbstractVector, ch::ConstraintHandler) = _apply_v(v, ch, true)
apply!(     v::AbstractVector, ch::ConstraintHandler) = _apply_v(v, ch, false)

function _apply_v(v::AbstractVector, ch::ConstraintHandler, apply_zero::Bool)
    @assert length(v) >= ndofs(ch.dh)
    v[ch.prescribed_dofs] .= apply_zero ? 0.0 : ch.inhomogeneities
    # Apply affine constraints, e.g u2 = s6*u6 + s3*u3 + h2
    for (dof, dofcoef, h) in zip(ch.prescribed_dofs, ch.dofcoefficients, ch.affine_inhomogeneities)
        dofcoef === nothing && continue
        @assert h !== nothing
        v[dof] = apply_zero ? 0.0 : h
        for (d, s) in dofcoef
            v[dof] += s * v[d]
        end
    end
    return v
end

function apply!(K::Union{SparseMatrixCSC,Symmetric}, ch::ConstraintHandler)
    apply!(K, eltype(K)[], ch, true)
end

function apply_zero!(K::Union{SparseMatrixCSC,Symmetric}, f::AbstractVector, ch::ConstraintHandler)
    apply!(K, f, ch, true)
end

# For backwards compatibility, not used anymore
@enumx ApplyStrategy Transpose Inplace
const APPLY_TRANSPOSE = ApplyStrategy.Transpose
const APPLY_INPLACE = ApplyStrategy.Inplace

function apply!(KK::Union{SparseMatrixCSC,Symmetric}, f::AbstractVector, ch::ConstraintHandler, applyzero::Bool=false;
                strategy::ApplyStrategy.T=ApplyStrategy.Transpose)
    sym = isa(KK, Symmetric)
    K = sym ? KK.data : KK
    @assert length(f) == 0 || length(f) == size(K, 1)
    @boundscheck checkbounds(K, ch.prescribed_dofs, ch.prescribed_dofs)
    @boundscheck length(f) == 0 || checkbounds(f, ch.prescribed_dofs)

    m = meandiag(K) # Use the mean of the diagonal here to not ruin things for iterative solver

    # Add inhomogeneities to f: (f - K * ch.inhomogeneities)
    if !applyzero
        @inbounds for i in 1:length(ch.inhomogeneities)
            d = ch.prescribed_dofs[i]
            v = ch.inhomogeneities[i]
            if v != 0
                for j in nzrange(K, d)
                    r = K.rowval[j]
                    sym && r > d && break # don't look below diagonal
                    f[r] -= v * K.nzval[j]
                end
            end
        end
        if sym
            # In the symmetric case, for a constrained dof `d`, we handle the contribution
            # from `K[1:d, d]` in the loop above, but we are still missing the contribution
            # from `K[(d+1):size(K,1), d]`. These values are not stored, but since the
            # matrix is symmetric we can instead use `K[d, (d+1):size(K,1)]`. Looping over
            # rows is slow, so loop over all columns again, and check if the row is a
            # constrained row.
            @inbounds for col in 1:size(K, 2)
                for ri in nzrange(K, col)
                    row = K.rowval[ri]
                    row >= col && break
                    if (i = get(ch.dofmapping, row, 0); i != 0)
                        f[col] -= ch.inhomogeneities[i] * K.nzval[ri]
                    end
                end
            end
        end
    end

    # Condense K (C' * K * C) and f (C' * f)
    _condense!(K, f, ch.dofcoefficients, ch.dofmapping, sym)

    # Remove constrained dofs from the matrix
    zero_out_columns!(K, ch.prescribed_dofs)
    zero_out_rows!(K, ch.dofmapping)

    # Add meandiag to constraint dofs
    @inbounds for i in 1:length(ch.inhomogeneities)
        d = ch.prescribed_dofs[i]
        K[d, d] = m
        if length(f) != 0
            vz = applyzero ? zero(eltype(f)) : ch.inhomogeneities[i]
            f[d] = vz * m
        end
    end
end

# Fetch dof coefficients for a dof prescribed by an affine constraint. Return nothing if the
# dof is not prescribed, or prescribed by DBC.
@inline function coefficients_for_dof(dofmapping, dofcoeffs, dof)
    idx = get(dofmapping, dof, 0)
    idx == 0 && return nothing
    return dofcoeffs[idx]
end

# Condenses K and f: C'*K*C, C'*f, in-place assuming the sparsity pattern is correct
function _condense!(K::SparseMatrixCSC, f::AbstractVector, dofcoefficients::Vector{Union{Nothing, DofCoefficients{T}}}, dofmapping::Dict{Int,Int}, sym::Bool=false) where T

    ndofs = size(K, 1)
    condense_f = !(length(f) == 0)
    condense_f && @assert( length(f) == ndofs )

    # Return early if there are no non-trivial affine constraints
    any(i -> !(i === nothing || isempty(i)), dofcoefficients) || return

    # TODO: The rest of this method can't handle K::Symmetric
    if sym
        error("condensation of ::Symmetric matrix not supported")
    end

    for col in 1:ndofs
        col_coeffs = coefficients_for_dof(dofmapping, dofcoefficients, col)
        if col_coeffs === nothing
            for a in nzrange(K, col)
                Kval = K.nzval[a]
                iszero(Kval) && continue
                row = K.rowval[a]
                row_coeffs = coefficients_for_dof(dofmapping, dofcoefficients, row)
                row_coeffs === nothing && continue
                for (d, v) in row_coeffs
                    addindex!(K, v * Kval, d, col)
                end

                # Perform f - K*g. However, this has already been done in outside this functions so we skip this.
                # if condense_f
                #     f[col] -= K.nzval[a] * ac.b;
                # end
            end
        else
            for a in nzrange(K, col)
                Kval = K.nzval[a]
                iszero(Kval) && continue
                row = K.rowval[a]
                row_coeffs = coefficients_for_dof(dofmapping, dofcoefficients, row)
                if row_coeffs === nothing
                    for (d, v) in col_coeffs
                        addindex!(K, v * Kval, row, d)
                    end
                else
                    for (d1, v1) in col_coeffs, (d2, v2) in row_coeffs
                        addindex!(K, v1 * v2 * Kval, d1, d2)
                    end
                end
            end

            if condense_f
                for (d, v) in col_coeffs
                    f[d] += f[col] * v
                end
                f[col] = 0.0
            end
        end
    end
end

function _add_or_grow(cnt::Int, I::Vector{Int}, J::Vector{Int}, dofi::Int, dofj::Int)
    if cnt > length(J)
        resize!(I, trunc(Int, length(I) * 1.5))
        resize!(J, trunc(Int, length(J) * 1.5))
    end
    I[cnt] = dofi
    J[cnt] = dofj
end

"""
    create_constraint_matrix(ch::ConstraintHandler)

Create and return the constraint matrix, `C`, and the inhomogeneities, `g`, from the affine
(linear) and Dirichlet constraints in `ch`.

The constraint matrix relates constrained, `a_c`, and free, `a_f`, degrees of freedom via
`a_c = C * a_f + g`. The condensed system of linear equations is obtained as
`C' * K * C = C' *  (f - K * g)`.
"""
function create_constraint_matrix(ch::ConstraintHandler{dh,T}) where {dh,T}
    @assert(isclosed(ch))

    I = Int[]; J = Int[]; V = T[];
    g = zeros(T, ndofs(ch.dh)) # inhomogeneities

    for (j, d) in enumerate(ch.free_dofs)
       push!(I, d)
       push!(J, j)
       push!(V, 1.0)
    end

    for (i,pdof) in enumerate(ch.prescribed_dofs)
        dofcoef = ch.dofcoefficients[i]
        if dofcoef !== nothing #if affine constraint
            for (d, v) in dofcoef
                push!(I, pdof)
                j = searchsortedfirst(ch.free_dofs, d)
                push!(J, j)
                push!(V, v)
            end
        end
    end
    g[ch.prescribed_dofs] .= ch.inhomogeneities

    C = sparse(I, J,  V, ndofs(ch.dh), length(ch.free_dofs))

    return C, g
end

# columns need to be stored entries, this is not checked
function zero_out_columns!(K, dofs::Vector{Int}) # can be removed in 0.7 with #24711 merged
    @debug @assert issorted(dofs)
    for col in dofs
        r = nzrange(K, col)
        K.nzval[r] .= 0.0
    end
end

function zero_out_rows!(K, dofmapping)
    rowval = K.rowval
    nzval = K.nzval
    @inbounds for i in eachindex(rowval, nzval)
        if haskey(dofmapping, rowval[i])
            nzval[i] = 0
        end
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
function add!(ch::ConstraintHandler{<:MixedDofHandler}, dbc::Dirichlet)
    dbc_added = false
    for fh in ch.dh.fieldhandlers
        if dbc.field_name in getfieldnames(fh) && _in_cellset(ch.dh.grid, fh.cellset, dbc.faces; all=false)
            # Dofs in `dbc` not in `fh` will be removed, hence `dbc.faces` must be copied.
            # Recreating the `dbc` will create a copy of `dbc.faces`.
            # In this case, add! will warn, unless `warn_not_in_cellset=false`
            dbc_ = Dirichlet(dbc.field_name, dbc.faces, dbc.f, 
                isempty(dbc.components) ? nothing : dbc.components) 
                # Check for empty already done when user created `dbc`
            add!(ch, fh, dbc_, warn_not_in_cellset=false)
            dbc_added = true
        end
    end
    dbc_added || @warn("No overlap between dbc::Dirichlet and fields in the ConstraintHandler's MixedDofHandler")
    return ch
end

function add!(ch::ConstraintHandler, fh::FieldHandler, dbc::Dirichlet; warn_not_in_cellset=true)
    if warn_not_in_cellset && !(_in_cellset(ch.dh.grid, fh.cellset, dbc.faces; all=true))
        @warn("You are trying to add a constraint a face/edge/node that is not in the cellset of the fieldhandler. This location will be skipped")
    end

    celltype = getcelltype(ch.dh.grid, first(fh.cellset)) #Assume same celltype of all cells in fh.cellset

    # Extract stuff for the field
    field_idx = find_field(fh, dbc.field_name)
    interpolation = getfieldinterpolations(fh)[field_idx]
    field_dim = getfielddims(fh)[field_idx]

    if !all(c -> 0 < c <= field_dim, dbc.components)
        error("components $(dbc.components) not within range of field :$(dbc.field_name) ($(field_dim) dimension(s))")
    end

    # Empty components means constrain them all
    isempty(dbc.components) && append!(dbc.components, 1:field_dim)

    if eltype(dbc.faces)==Int #Special case when dbc.faces is a nodeset
        bcvalue = BCValues(interpolation, default_interpolation(celltype), FaceIndex) #Not used by node bcs, but still have to pass it as an argument
    else
        bcvalue = BCValues(interpolation, default_interpolation(celltype), eltype(dbc.faces))
    end

    Ferrite._add!(ch, dbc, dbc.faces, interpolation, field_dim, field_offset(fh, dbc.field_name), bcvalue, fh.cellset)
    return ch
end

# If all==true, return true only if all items in faceset/nodeset are in the cellset
# If all==false, return true if some items in faceset/nodeset are in the cellset
function _in_cellset(::AbstractGrid, cellset::Set{Int}, faceset::Set{<:BoundaryIndex}; all=true)
    for (cellid,faceid) in faceset
        if cellid in cellset
            all || return true
        else
            all && return false
        end
    end
    return all # if not returned by now and all==false, then no `cellid`s where in cellset
end

function _in_cellset(grid::AbstractGrid, cellset::Set{Int}, nodeset::Set{Int}; all=true)
    nodes = Set{Int}()
    for cellid in cellset
        for nodeid in grid.cells[cellid].nodes
            nodeid ∈ nodes || push!(nodes, nodeid)
        end
    end

    for nodeid in nodeset
        if nodeid ∈ nodes
            all || return true 
        else
            all && return false
        end
    end
    return all # if not returned by now and all==false, then no `cellid`s where in cellset
end

"""
    create_symmetric_sparsity_pattern(dh::AbstractDofHandler, ch::ConstraintHandler, coupling)

Create a symmetric sparsity pattern accounting for affine constraints in `ch`. See 
the Affine Constraints section of the manual for further details. 
"""
function create_symmetric_sparsity_pattern(dh::AbstractDofHandler, ch::ConstraintHandler;
        keep_constrained::Bool=true, coupling=nothing)
    return Symmetric(_create_sparsity_pattern(dh, ch, true, keep_constrained, coupling), :U)
end
"""
    create_sparsity_pattern(dh::AbstractDofHandler, ch::ConstraintHandler; coupling)

Create a sparsity pattern accounting for affine constraints in `ch`. See 
the Affine Constraints section of the manual for further details. 
"""
function create_sparsity_pattern(dh::AbstractDofHandler, ch::ConstraintHandler;
        keep_constrained::Bool=true, coupling=nothing)
    return _create_sparsity_pattern(dh, ch, false, keep_constrained, coupling)
end

struct PeriodicFacePair
    mirror::FaceIndex
    image::FaceIndex
    rotation::UInt8 # relative rotation of the mirror face counter-clockwise the *image* normal (only relevant in 3D)
    mirrored::Bool  # mirrored => opposite normal vectors
end

"""
    PeriodicDirichlet(u::Symbol, face_mapping, components=nothing)
    PeriodicDirichlet(u::Symbol, face_mapping, R::AbstractMatrix, components=nothing)
    PeriodicDirichlet(u::Symbol, face_mapping, f::Function, components=nothing)

Create a periodic Dirichlet boundary condition for the field `u` on the face-pairs given in
`face_mapping`. The mapping can be computed with [`collect_periodic_faces`](@ref). The
constraint ensures that degrees-of-freedom on the mirror face are constrained to the
corresponding degrees-of-freedom on the image face. `components` specify the components of
`u` that are prescribed by this condition. By default all components of `u` are prescribed.

If the mapping is not aligned with the coordinate axis (e.g. rotated) a rotation matrix `R`
should be passed to the constructor. This matrix rotates dofs on the mirror face to the
image face. Note that this is only applicable for vector-valued problems.

To construct an inhomogeneous periodic constraint it is possible to pass a function `f`.
Note that this is currently only supported when the periodicity is aligned with the
coordinate axes.

See the manual section on [Periodic boundary conditions](@ref) for more information.
"""
struct PeriodicDirichlet
    field_name::Symbol
    components::Vector{Int} # components of the field
    face_pairs::Vector{Pair{String,String}} # legacy that will populate face_map on add!
    face_map::Vector{PeriodicFacePair}
    func::Union{Function,Nothing}
    rotation_matrix::Union{Matrix{Float64},Nothing}
end

# Default to no inhomogeneity function/rotation
PeriodicDirichlet(fn::Symbol, fp::Union{Vector{<:Pair},Vector{PeriodicFacePair}}, c=nothing) =
    PeriodicDirichlet(fn, fp, nothing, c)

# Basic constructor for the simple case where face_map will be populated in
# add!(::ConstraintHandler, ...) instead
function PeriodicDirichlet(fn::Symbol, fp::Vector{<:Pair}, f::Union{Function,Nothing}, c=nothing)
    face_map = PeriodicFacePair[] # This will be populated in add!(::ConstraintHandler, ...) instead
    return PeriodicDirichlet(fn, __to_components(c), fp, face_map, f, nothing)
end

function PeriodicDirichlet(fn::Symbol, fm::Vector{PeriodicFacePair}, f_or_r::Union{AbstractMatrix,Function,Nothing}, c=nothing)
    f = f_or_r isa Function ? f_or_r : nothing
    rotation_matrix = f_or_r isa AbstractMatrix ? f_or_r : nothing
    components = __to_components(c)
    return PeriodicDirichlet(fn, components, Pair{String,String}[], fm, f, rotation_matrix)
end

function add!(ch::ConstraintHandler, pdbc::PeriodicDirichlet)
    # Legacy code: Might need to build the face_map
    is_legacy = !isempty(pdbc.face_pairs) && isempty(pdbc.face_map)
    if is_legacy
        for (mset, iset) in pdbc.face_pairs
            collect_periodic_faces!(pdbc.face_map, ch.dh.grid, mset, iset, identity) # TODO: Better transform
        end
    end
    field_idx = find_field(ch.dh, pdbc.field_name)
    interpolation = getfieldinterpolation(ch.dh, field_idx)
    field_dim = getfielddim(ch.dh, field_idx)

    if !all(c -> 0 < c <= field_dim, pdbc.components)
        error("components $(pdbc.components) not within range of field :$(pdbc.field_name) ($(field_dim) dimension(s))")
    end

    # Empty components means constrain them all
    isempty(pdbc.components) && append!(pdbc.components, 1:field_dim)

    if pdbc.rotation_matrix === nothing
        dof_map_t = Int
        iterator_f = identity
    else
        @assert pdbc.func === nothing # Verified in constructor
        if is_legacy
            error("legacy mode not supported with rotations")
        end
        nc = length(pdbc.components)
        if !(nc == size(pdbc.rotation_matrix, 1) == size(pdbc.rotation_matrix, 2))
            error("size of rotation matrix does not match the number of components")
        end
        if nc !== field_dim
            error("rotations currently only supported when all components are periodic")
        end
        dof_map_t = Vector{Int}
        iterator_f = x -> Iterators.partition(x, nc)
    end
    _add!(ch, pdbc, interpolation, field_dim, field_offset(ch.dh, pdbc.field_name), is_legacy, pdbc.rotation_matrix, dof_map_t, iterator_f)
    return ch
end

function _add!(ch::ConstraintHandler, pdbc::PeriodicDirichlet, interpolation::Interpolation,
               field_dim::Int, offset::Int, is_legacy::Bool, rotation_matrix::Union{Matrix{T},Nothing}, ::Type{dof_map_t}, iterator_f::F) where {T, dof_map_t, F <: Function}
    grid = ch.dh.grid
    face_map = pdbc.face_map
    Tx = typeof(first(ch.dh.grid.nodes).x) # Vec{D,T}

    # Indices of the local dofs for the faces
    local_face_dofs, local_face_dofs_offset =
        _local_face_dofs_for_bc(interpolation, field_dim, pdbc.components, offset)
    mirrored_indices =
        mirror_local_dofs(local_face_dofs, local_face_dofs_offset, interpolation, length(pdbc.components))
    rotated_indices = rotate_local_dofs(local_face_dofs, local_face_dofs_offset, interpolation, length(pdbc.components))

    # Dof map for mirror dof => image dof
    dof_map = Dict{dof_map_t,dof_map_t}()

    mirror_dofs = zeros(Int, ndofs_per_cell(ch.dh))
     image_dofs = zeros(Int, ndofs_per_cell(ch.dh))
    for face_pair in face_map
        m = face_pair.mirror
        i = face_pair.image
        celldofs!(mirror_dofs, ch.dh, m[1])
        celldofs!( image_dofs, ch.dh, i[1])

        mdof_range = local_face_dofs_offset[m[2]] : (local_face_dofs_offset[m[2] + 1] - 1)
        idof_range = local_face_dofs_offset[i[2]] : (local_face_dofs_offset[i[2] + 1] - 1)

        for (md, id) in zip(iterator_f(mdof_range), iterator_f(idof_range))
            mdof = image_dofs[local_face_dofs[id]]
            # Rotate the mirror index
            rotated_md = rotated_indices[md, face_pair.rotation + 1]
            # Mirror the mirror index (maybe) :)
            mirrored_md = face_pair.mirrored ? mirrored_indices[rotated_md] : rotated_md
            cdof = mirror_dofs[local_face_dofs[mirrored_md]]

            if haskey(dof_map, mdof)
                mdof′ = dof_map[mdof]
                # @info "$cdof => $mdof, but $mdof => $mdof′, remapping $cdof => $mdof′."
                # TODO: Is this needed now when untangling below?
                push!(dof_map, cdof => mdof′)
            # elseif haskey(dof_map, cdof) && dof_map[cdof] == mdof
                # @info "$cdof => $mdof already in the set, skipping."
            elseif haskey(dof_map, cdof)
                # @info "$cdof => $mdof, but $cdof => $(dof_map[cdof]) already, skipping."
            elseif cdof == mdof
                # @info "Skipping self-constraint $cdof => $mdof."
            else
                # @info "$cdof => $mdof."
                push!(dof_map, cdof => mdof)
            end
        end
    end

    # Need to untangle in case we have 1 => 2 and 2 => 3 into 1 => 3 and 2 => 3.
    # Note that a single pass is enough (no need to iterate) since all constraints are
    # between just one mirror dof and one image dof.
    remaps = Dict{dof_map_t, dof_map_t}()
    for (k, v) in dof_map
        if haskey(dof_map, v)
            remaps[k] = get(remaps, v, dof_map[v])
        end
    end
    for (k, v) in remaps
        # @info "Remapping $k => $(dof_map[k]) to $k => $v"
        dof_map[k] = v
    end
    @assert isempty(intersect(keys(dof_map), values(dof_map)))

    # For legacy code add Dirichlet conditions in the corners
    if is_legacy
        Base.depwarn("It looks like you are using legacy code for PeriodicDirichlet " *
                     "meaning that the solution is automatically locked in the \"corners\"." *
                     "This will not be done automatically in the future. Instead add a " *
                     "Dirichlet boundary condition on the relevant nodeset.",
                     :PeriodicDirichlet)
        all_node_idxs = Set{Int}()
        min_x = Tx(i -> typemax(eltype(Tx)))
        max_x = Tx(i -> typemin(eltype(Tx)))
        for facepair in face_map, faceidx in (facepair.mirror, facepair.image)
            cellidx, faceidx = faceidx
            nodes = faces(grid.cells[cellidx])[faceidx]
            union!(all_node_idxs, nodes)
            for n in nodes
                x = grid.nodes[n].x
                min_x = Tx(i -> min(min_x[i], x[i]))
                max_x = Tx(i -> max(max_x[i], x[i]))
            end
        end
        all_node_idxs_v = collect(all_node_idxs)
        points = construct_cornerish(min_x, max_x)
        tree = KDTree(Tx[grid.nodes[i].x for i in all_node_idxs_v])
        idxs, _ = NearestNeighbors.nn(tree, points)
        corner_set = Set{Int}(all_node_idxs_v[i] for i in idxs)

        dbc = Dirichlet(pdbc.field_name, corner_set,
            pdbc.func === nothing ? (x, _) -> pdbc.components * eltype(x)(0) : pdbc.func,
            pdbc.components
        )

        # Create a temp constraint handler just to find the dofs in the nodes...
        chtmp = ConstraintHandler(ch.dh)
        add!(chtmp, dbc)
        close!(chtmp)
        # No need to update!(chtmp, t) here since we only care about the dofs
        # TODO: Right? maybe if the user passed f we need to...
        foreach(x -> delete!(dof_map, x), chtmp.prescribed_dofs)

        # Need to reset the internal of this DBC in order to add! it again...
        resize!(dbc.local_face_dofs, 0)
        resize!(dbc.local_face_dofs_offset, 0)

        # Add the Dirichlet for the corners
        add!(ch, dbc)
    end

    inhomogeneity_map = nothing
    if pdbc.func !== nothing
        # Create another temp constraint handler if we need to compute inhomogeneities
        chtmp2 = ConstraintHandler(ch.dh)
        all_faces = Set{FaceIndex}()
        union!(all_faces, (x.mirror for x in face_map))
        union!(all_faces, (x.image for x in face_map))
        dbc_all = Dirichlet(pdbc.field_name, all_faces, pdbc.func, pdbc.components)
        add!(chtmp2, dbc_all); close!(chtmp2)
        # Call update! here since we need it to construct the affine constraints...
        # TODO: This doesn't allow for time dependent constraints...
        update!(chtmp2, 0.0)
        inhomogeneity_map = Dict{Int, Float64}()
        for (k, v) in dof_map
            g = chtmp2.inhomogeneities
            push!(inhomogeneity_map,
                  k => - g[chtmp2.dofmapping[v]] + g[chtmp2.dofmapping[k]]
            )
        end
    end

    # Any remaining mappings are added as homogeneous AffineConstraints
    for (k, v) in dof_map
        if dof_map_t === Int
            ac = AffineConstraint(k, [v => 1.0], inhomogeneity_map === nothing ? 0.0 : inhomogeneity_map[k])
            add!(ch, ac)
        else
            @assert inhomogeneity_map === nothing
            @assert rotation_matrix !== nothing
            for (i, ki) in pairs(k)
                # u_mirror = R ⋅ u_image
                vs = Pair{Int,eltype(T)}[v[j] => rotation_matrix[i, j] for j in 1:length(v)]
                ac = AffineConstraint(ki, vs, 0.0)
                add!(ch, ac)
            end
        end
    end

    return ch
end

function construct_cornerish(min_x::V, max_x::V) where {T, V <: Vec{1,T}}
    lx = max_x - min_x
    max_x += lx
    min_x -= lx
    return V[min_x, max_x]
end
function construct_cornerish(min_x::V, max_x::V) where {T, V <: Vec{2,T}}
    lx = max_x - min_x
    max_x += lx
    min_x -= lx
    return V[
       max_x,
       min_x,
       Vec{2,T}((max_x[1], min_x[2])),
       Vec{2,T}((min_x[1], max_x[2])),
    ]
end
function construct_cornerish(min_x::V, max_x::V) where {T, V <: Vec{3,T}}
    lx = max_x - min_x
    max_x += lx
    min_x -= lx
    return V[
        min_x,
        max_x,
        Vec{3,T}((max_x[1], min_x[2] , min_x[3])),
        Vec{3,T}((max_x[1], max_x[2] , min_x[3])),
        Vec{3,T}((min_x[1], max_x[2] , min_x[3])),
        Vec{3,T}((min_x[1], min_x[2] , max_x[3])),
        Vec{3,T}((max_x[1], min_x[2] , max_x[3])),
        Vec{3,T}((min_x[1], max_x[2] , max_x[3])),
    ]
end

function mirror_local_dofs(_, _, ::Lagrange{1}, ::Int)
    # For 1D there is nothing to do
end
function mirror_local_dofs(local_face_dofs, local_face_dofs_offset, ip::Lagrange{2,<:Union{RefCube,RefTetrahedron}}, n::Int)
    # For 2D we always permute since Ferrite defines dofs counter-clockwise
    ret = collect(1:length(local_face_dofs))
    for (i, f) in enumerate(facedof_indices(ip))
        this_offset = local_face_dofs_offset[i]
        other_offset = this_offset + n
        for d in 1:n
            idx1 = this_offset + (d - 1)
            idx2 = other_offset + (d - 1)
            tmp = ret[idx1]
            ret[idx1] = ret[idx2]
            ret[idx2] = tmp
        end
    end
    return ret
end

# TODO: Can probably be combined with the method above.
function mirror_local_dofs(local_face_dofs, local_face_dofs_offset, ip::Lagrange{3,<:Union{RefCube,RefTetrahedron},O}, n::Int) where O
    @assert 1 <= O <= 2
    N = ip isa Lagrange{3,RefCube} ? 4 : 3
    ret = collect(1:length(local_face_dofs))

    # Mirror by changing from counter-clockwise to clockwise
    for (i, f) in enumerate(facedof_indices(ip))
        r = local_face_dofs_offset[i]:(local_face_dofs_offset[i+1] - 1)
        # 1. Rotate the corners
        vertex_range = r[1:(N*n)]
        vlr = @view ret[vertex_range]
        for i in 1:N
            reverse!(vlr, (i - 1) * n + 1, i * n)
        end
        reverse!(vlr)
        circshift!(vlr, n)
        # 2. Rotate the edge dofs for quadratic interpolation
        if O > 1
            edge_range = r[(N*n+1):(2N*n)]
            elr = @view ret[edge_range]
            for i in 1:N
                reverse!(elr, (i - 1) * n + 1, i * n)
            end
            reverse!(elr)
            # circshift!(elr, n) # !!! Note: no shift here
        end
    end
    return ret
end

if VERSION < v"1.8.0"
    function circshift!(x::AbstractVector, shift::Integer)
        return circshift!(x, copy(x), shift)
    end
else
    # See JuliaLang/julia#46759
    const CIRCSHIFT_WRONG_DIRECTION = Base.circshift!([1, 2, 3], 1) != Base.circshift([1, 2, 3], 1)
    function circshift!(x::AbstractVector, shift::Integer)
        shift = CIRCSHIFT_WRONG_DIRECTION ? -shift : shift
        return Base.circshift!(x, shift)
    end
end

circshift!(args...) = Base.circshift!(args...)


function rotate_local_dofs(local_face_dofs, local_face_dofs_offset, ip::Lagrange{2}, ncomponents)
    return collect(1:length(local_face_dofs)) # TODO: Return range?
end
function rotate_local_dofs(local_face_dofs, local_face_dofs_offset, ip::Lagrange{3,<:Union{RefCube,RefTetrahedron}, O}, ncomponents) where O
    @assert 1 <= O <= 2
    N = ip isa Lagrange{3,RefCube} ? 4 : 3
    ret = similar(local_face_dofs, length(local_face_dofs), N)
    ret[:, :] .= 1:length(local_face_dofs)
    for f in 1:length(local_face_dofs_offset)-1
        face_range = local_face_dofs_offset[f]:(local_face_dofs_offset[f+1]-1)
        for i in 1:(N-1)
            # 1. Rotate the vertex dofs
            vertex_range = face_range[1:(N*ncomponents)]
            circshift!(@view(ret[vertex_range, i+1]), @view(ret[vertex_range, i]), -ncomponents)
            # 2. Rotate the edge dofs
            if O > 1
                edge_range = face_range[(N*ncomponents+1):(2N*ncomponents)]
                circshift!(@view(ret[edge_range, i+1]), @view(ret[edge_range, i]), -ncomponents)
            end
        end
    end
    return ret
end

"""
    collect_periodic_faces(grid::Grid, mset, iset, transform::Union{Function,Nothing}=nothing)

Match all mirror faces in `mset` with a corresponding image face in `iset`. Return a
dictionary which maps each mirror face to a image face. The result can then be passed to
[`PeriodicDirichlet`](@ref).

`mset` and `iset` can be given as a `String` (an existing face set in the grid) or as a
`Set{FaceIndex}` directly.

By default this function looks for a matching face in the directions of the coordinate
system. For other types of periodicities the `transform` function can be used. The
`transform` function is applied on the coordinates of the image face, and is expected to
transform the coordinates to the matching locations in the mirror set.

See also: [`collect_periodic_faces!`](@ref), [`PeriodicDirichlet`](@ref).
"""
function collect_periodic_faces(grid::Grid, mset::Union{Set{FaceIndex},String}, iset::Union{Set{FaceIndex},String}, transform::Union{Function,Nothing}=nothing)
    return collect_periodic_faces!(PeriodicFacePair[], grid, mset, iset, transform)
end

"""
    collect_periodic_faces(grid::Grid, all_faces::Union{Set{FaceIndex},String,Nothing}=nothing)

Split all faces in `all_faces` into image and mirror sets. For each matching pair, the face
located further along the vector `(1, 1, 1)` becomes the image face.

If no set is given, all faces on the outer boundary of the grid (i.e. all faces that do not
have a neighbor) is used.

See also: [`collect_periodic_faces!`](@ref), [`PeriodicDirichlet`](@ref).
"""
function collect_periodic_faces(grid::Grid, all_faces::Union{Set{FaceIndex},String,Nothing}=nothing)
    return collect_periodic_faces!(PeriodicFacePair[], grid, all_faces)
end


"""
    collect_periodic_faces!(face_map::Vector{PeriodicFacePair}, grid::Grid, mset, iset, transform::Union{Function,Nothing})

Same as [`collect_periodic_faces`](@ref) but adds all matches to the existing `face_map`.
"""
function collect_periodic_faces!(face_map::Vector{PeriodicFacePair}, grid::Grid, mset::Union{Set{FaceIndex},String}, iset::Union{Set{FaceIndex},String}, transform::Union{Function,Nothing}=nothing)
    mset = __to_faceset(grid, mset)
    iset = __to_faceset(grid, iset)
    if transform === nothing
        # This method is destructive, hence the copy
        __collect_periodic_faces_bruteforce!(face_map, grid, copy(mset), copy(iset), #=known_order=#true)
    else
        # This method relies on ordering, hence the collect
        __collect_periodic_faces_tree!(face_map, grid, collect(mset), collect(iset), transform)
    end
    return face_map
end

function collect_periodic_faces!(face_map::Vector{PeriodicFacePair}, grid::Grid, faceset::Union{Set{FaceIndex},String,Nothing})
    faceset = faceset === nothing ? __collect_boundary_faces(grid) : copy(__to_faceset(grid, faceset))
    if mod(length(faceset), 2) != 0
        error("uneven number of faces")
    end
    return __collect_periodic_faces_bruteforce!(face_map, grid, faceset, faceset, #=known_order=#false)
end

__to_faceset(_, set::Set{FaceIndex}) = set
__to_faceset(grid, set::String) = getfaceset(grid, set)
function __collect_boundary_faces(grid::Grid)
    candidates = Dict{Tuple, FaceIndex}()
    for (ci, c) in enumerate(grid.cells)
        for (fi, fn) in enumerate(faces(c))
            fn = sortface(fn)
            if haskey(candidates, fn)
                delete!(candidates, fn)
            else
                candidates[fn] = FaceIndex(ci, fi)
            end
        end
    end
    return Set{FaceIndex}(values(candidates))
end

function __collect_periodic_faces_tree!(face_map::Vector{PeriodicFacePair}, grid::Grid, mset::Vector{FaceIndex}, iset::Vector{FaceIndex}, transformation::F) where F <: Function
    if length(mset) != length(mset)
        error("different number of faces in mirror and image set")
    end
    Tx = typeof(first(grid.nodes).x)

    mirror_mean_x = Tx[]
    for (c, f) in mset
        fn = faces(grid.cells[c])[f]
        push!(mirror_mean_x, sum(grid.nodes[i].x for i in fn) / length(fn))
    end

    # Same dance for the image
    image_mean_x = Tx[]
    for (c, f) in iset
        fn = faces(grid.cells[c])[f]
        # Apply transformation to all coordinates
        push!(image_mean_x, sum(transformation(grid.nodes[i].x)::Tx for i in fn) / length(fn))
    end

    # Use KDTree to find closest face
    tree = KDTree(image_mean_x)
    idxs, _ = NearestNeighbors.nn(tree, mirror_mean_x)
    for (midx, iidx) in zip(eachindex(mset), idxs)
        r = __check_periodic_faces_f(grid, mset[midx], iset[iidx], mirror_mean_x[midx], image_mean_x[iidx], transformation)
        if r === nothing
            error("Could not find matching face for $(mset[midx])")
        end
        push!(face_map, r)
    end

    # Make sure the mapping is unique
    @assert all(x -> in(x, Set{FaceIndex}(p.mirror for p in face_map)), mset)
    @assert all(x -> in(x, Set{FaceIndex}(p.image for p in face_map)), iset)
    if !allunique(Set{FaceIndex}(p.image for p in face_map))
        error("did not find a unique mapping between faces")
    end

    return face_map
end

# This method empties mset and iset
function __collect_periodic_faces_bruteforce!(face_map::Vector{PeriodicFacePair}, grid::Grid, mset::Set{FaceIndex}, iset::Set{FaceIndex}, known_order::Bool)
    if length(mset) != length(iset)
        error("different faces in mirror and image")
    end
    while length(mset) > 0
        fi = first(mset)
        found = false
        for fj in iset
            fi == fj && continue
            r = __check_periodic_faces(grid, fi, fj, known_order)
            r === nothing && continue
            push!(face_map, r)
            delete!(mset, fi)
            delete!(iset, fj)
            found = true
            break
        end
        found || error("did not find a corresponding periodic face")
    end
    @assert isempty(mset) && isempty(iset)
    return face_map
end

function __periodic_options(::T) where T <: Vec{2}
    # (3^2 - 1) / 2 options
    return (
        Vec{2}((1.0,  0.0)),
        Vec{2}((0.0,  1.0)),
        Vec{2}((1.0,  1.0)) / sqrt(2),
        Vec{2}((1.0,  -1.0)) / sqrt(2),
    )
end
function __periodic_options(::T) where T <: Vec{3}
    # (3^3 - 1) / 2 options
    return (
        Vec{3}((1.0,  0.0, 0.0)),
        Vec{3}((0.0,  1.0, 0.0)),
        Vec{3}((0.0,  0.0, 1.0)),
        Vec{3}((1.0,  1.0, 0.0)) / sqrt(2),
        Vec{3}((0.0,  1.0, 1.0)) / sqrt(2),
        Vec{3}((1.0,  0.0, 1.0)) / sqrt(2),
        Vec{3}((1.0,  1.0, 1.0)) / sqrt(3),
        Vec{3}((1.0,  -1.0, 0.0)) / sqrt(2),
        Vec{3}((0.0,  1.0, -1.0)) / sqrt(2),
        Vec{3}((1.0,  0.0, -1.0)) / sqrt(2),
        Vec{3}((1.0,  1.0, -1.0)) / sqrt(3),
        Vec{3}((1.0,  -1.0, 1.0)) / sqrt(3),
        Vec{3}((1.0,  -1.0, -1.0)) / sqrt(3),
    )
end

function __outward_normal(grid::Grid{2}, nodes, transformation::F=identity) where F <: Function
    n1::Vec{2} = transformation(grid.nodes[nodes[1]].x)
    n2::Vec{2} = transformation(grid.nodes[nodes[2]].x)
    n = Vec{2}((n2[2] - n1[2], - n2[1] + n1[1]))
    return n / norm(n)
end

function __outward_normal(grid::Grid{3}, nodes, transformation::F=identity) where F <: Function
    n1::Vec{3} = transformation(grid.nodes[nodes[1]].x)
    n2::Vec{3} = transformation(grid.nodes[nodes[2]].x)
    n3::Vec{3} = transformation(grid.nodes[nodes[3]].x)
    n = (n3 - n2) × (n1 - n2)
    return n / norm(n)
end

function circshift_tuple(x::T, n) where T
    Tuple(circshift!(collect(x), n))::T
end

# Check if two faces are periodic. This method assumes that the faces are mirrored and thus
# have opposing normal vectors
function __check_periodic_faces(grid::Grid, fi::FaceIndex, fj::FaceIndex, known_order::Bool)
    cii, fii = fi
    nodes_i = faces(grid.cells[cii])[fii]
    cij, fij = fj
    nodes_j = faces(grid.cells[cij])[fij]

    # 1. Check that normals are opposite TODO: Should use FaceValues here
    ni = __outward_normal(grid, nodes_i)
    nj = __outward_normal(grid, nodes_j)
    TOL = 1e-12
    if norm(ni + nj) >= TOL
        return nothing
    end

    # 2. Find the periodic direction using the vector between the midpoint of the faces
    xmi = sum(grid.nodes[i].x for i in nodes_i) / length(nodes_i)
    xmj = sum(grid.nodes[i].x for i in nodes_j) / length(nodes_j)
    xmij = xmj - xmi
    h = 2 * norm(xmj - grid.nodes[nodes_j[1]].x) # Approximate element size
    TOLh = TOL * h
    found = false
    local len
    for o in __periodic_options(xmij)
        len = xmij ⋅ o
        if norm(xmij - len * o) < TOLh
            found = true
            break
        end
    end
    found || return nothing

    # 3. Check that the first node of fj have a corresponding node in fi
    #    In this method faces are mirrored (opposite normal vectors) so reverse the nodes
    nodes_i = circshift_tuple(reverse(nodes_i), 1)
    xj = grid.nodes[nodes_j[1]].x
    node_rot = 0
    found = false
    for i in eachindex(nodes_i)
        xi = grid.nodes[nodes_i[i]].x
        xij = xj - xi
        if norm(xij - xmij) < TOLh
            found = true
            break
        end
        node_rot += 1
    end
    found || return nothing

    # 4. Check the remaining nodes for the same criteria, now with known node_rot
    for j in 2:length(nodes_j)
        xi = grid.nodes[nodes_i[mod1(j + node_rot, end)]].x
        xj = grid.nodes[nodes_j[j]].x
        xij = xj - xi
        if norm(xij - xmij) >= TOLh
            return nothing
        end
    end

    # Rotation is only relevant for 3D
    if getdim(grid) == 3
        node_rot = mod(node_rot, length(nodes_i))
    else
        node_rot = 0
    end

    # 5. Faces match! Face below the diagonal become the mirror.
    if known_order || len > 0
        return PeriodicFacePair(fi, fj, node_rot, true)
    else
        return PeriodicFacePair(fj, fi, node_rot, true)
    end
end

# This method is quite similar to __check_periodic_faces, but is used when user have passed
# a transformation function and we have then used the KDTree to find the matching pair of
# faces. This function only need to i) check whether faces have aligned or opposite normal
# vectors, and ii) compute the relative rotation.
function __check_periodic_faces_f(grid::Grid, fi::FaceIndex, fj::FaceIndex, xmi, xmj, transformation::F) where F
    cii, fii = fi
    nodes_i = faces(grid.cells[cii])[fii]
    cij, fij = fj
    nodes_j = faces(grid.cells[cij])[fij]

    # 1. Check if normals are aligned or opposite TODO: Should use FaceValues here
    ni = __outward_normal(grid, nodes_i)
    nj = __outward_normal(grid, nodes_j, transformation)
    TOL = 1e-12
    if norm(ni + nj) < TOL
        mirror = true
    elseif norm(ni - nj) < TOL
        mirror = false
    else
        return nothing
    end

    # 2. Compute the relative rotation
    xmij = xmj - xmi
    h = 2 * norm(xmj - grid.nodes[nodes_j[1]].x) # Approximate element size
    TOLh = TOL * h
    nodes_i = mirror ? circshift_tuple(reverse(nodes_i), 1) : nodes_i # reverse if necessary
    xj = transformation(grid.nodes[nodes_j[1]].x)
    node_rot = 0
    found = false
    for i in eachindex(nodes_i)
        xi = grid.nodes[nodes_i[i]].x
        xij = xj - xi
        if norm(xij - xmij) < TOLh
            found = true
            break
        end
        node_rot += 1
    end
    found || return nothing

    # 3. Rotation is only relevant for 3D.
    if getdim(grid) == 3
        node_rot = mod(node_rot, length(nodes_i))
    else
        node_rot = 0
    end

    return PeriodicFacePair(fi, fj, node_rot, mirror)
end


######################################
## Local application of constraints ##
######################################

"""
    apply_local!(
        local_matrix::AbstractMatrix, local_vector::AbstractVector,
        global_dofs::AbstractVector, ch::ConstraintHandler;
        apply_zero::Bool = false
    )

Similar to [`apply!`](@ref) but perform condensation of constrained degrees-of-freedom
locally in `local_matrix` and `local_vector` *before* they are to be assembled into the
global system.

When the keyword argument `apply_zero` is `true` all inhomogeneities are set to `0` (cf.
[`apply!`](@ref) vs [`apply_zero!`](@ref)).

This method can only be used if all constraints are "local", i.e. no constraint couples with
dofs outside of the element dofs (`global_dofs`) since condensation of such constraints
requires writing to entries in the global matrix/vector. For such a case,
[`apply_assemble!`](@ref) can be used instead.

Note that this method is destructive since it, by definition, modifies `local_matrix` and
`local_vector`.
"""
function apply_local!(local_matrix::AbstractMatrix, local_vector::AbstractVector,
                      global_dofs::AbstractVector, ch::ConstraintHandler;
                      apply_zero::Bool = false)
    return _apply_local!(local_matrix, local_vector, global_dofs, ch, apply_zero,
                         #=global_matrix=# nothing, #=global_vector=# nothing)
end

# Element local application of boundary conditions. Global matrix and vectors are necessary
# if there are affine constraints that connect dofs from different elements.
function _apply_local!(local_matrix::AbstractMatrix, local_vector::AbstractVector,
                       global_dofs::AbstractVector, ch::ConstraintHandler, apply_zero::Bool,
                       global_matrix, global_vector)
    # TODO: With apply_zero it shouldn't be required to pass the vector.
    length(global_dofs) == size(local_matrix, 1) == size(local_matrix, 2) == length(local_vector) || error("?")
    # First pass over the dofs check whether there are any constrained dofs at all
    has_constraints = false
    has_nontrivial_affine_constraints = false
    # 1. Adjust local vector
    @inbounds for (local_dof, global_dof) in pairs(global_dofs)
        # Check if this dof is constrained
        pdofs_index = get(ch.dofmapping, global_dof, nothing)
        pdofs_index === nothing && continue # Not constrained, move on
        has_constraints = true
        # Add inhomogeneities to local_vector: local_vector - local_matrix * inhomogeneities
        v = ch.inhomogeneities[pdofs_index]
        if !apply_zero && v != 0
            for j in axes(local_matrix, 1)
                local_vector[j] -= v * local_matrix[j, local_dof]
            end
        end
        # Check if this is an affine constraint
        has_nontrivial_affine_constraints = has_nontrivial_affine_constraints || (
           coeffs = ch.dofcoefficients[pdofs_index];
           !(coeffs === nothing || isempty(coeffs))
        )
    end
    # 2. Compute mean of diagonal before modifying local matrix
    m = has_constraints ? meandiag(local_matrix) : zero(eltype(local_matrix))
    # 3. Condense any affine constraints
    if has_nontrivial_affine_constraints
        # Condense this constraint locally if possible, and otherwise modifies the global arrays.
        _condense_local!(local_matrix, local_vector, global_matrix, global_vector, global_dofs, ch.dofmapping, ch.dofcoefficients)
    end
    # 4. Zero out columns/rows of local matrix and replace diagonal entries with the mean
    if has_constraints
        @inbounds for (local_dof, global_dof) in pairs(global_dofs)
            pdofs_index = get(ch.dofmapping, global_dof, nothing)
            pdofs_index === nothing && continue # Not constrained, move on
            # Zero the column
            for local_row in axes(local_matrix, 1)
                local_matrix[local_row, local_dof] = 0
            end
            # Zero the row
            for local_col in axes(local_matrix, 2)
                local_matrix[local_dof, local_col] = 0
            end
            # Replace diagonal with mean
            local_matrix[local_dof, local_dof] = m
            v = ch.inhomogeneities[pdofs_index]
            local_vector[local_dof] = apply_zero ? zero(eltype(local_vector)) : (v * m)
        end
    end
    return
end

# Condensation of affine constraints on element level. If possible this function only
# modifies the local arrays.
@noinline missing_global() = error("can not condense constraint without the global matrix and vector")
function _condense_local!(local_matrix::AbstractMatrix, local_vector::AbstractVector,
                          global_matrix#=::SparseMatrixCSC=#, global_vector#=::Vector=#,
                          global_dofs::AbstractVector, dofmapping::Dict, dofcoefficients::Vector)
    @assert axes(local_matrix, 1) == axes(local_matrix, 2) ==
            axes(local_vector, 1) == axes(global_dofs, 1)
    has_global_arrays = global_matrix !== nothing && global_vector !== nothing
    for (local_col, global_col) in pairs(global_dofs)
        col_coeffs = coefficients_for_dof(dofmapping, dofcoefficients, global_col)
        if col_coeffs === nothing
            for (local_row, global_row) in pairs(global_dofs)
                m = local_matrix[local_row, local_col]
                iszero(m) && continue # Skip early when zero to avoid remaining lookups
                row_coeffs = coefficients_for_dof(dofmapping, dofcoefficients, global_row)
                row_coeffs === nothing && continue # Neither the column nor the row are constrained: Do nothing
                for (global_mrow, weight) in row_coeffs
                    mw = m * weight
                    local_mrow = findfirst(==(global_mrow), global_dofs)
                    if local_mrow === nothing
                        # Only modify the global array if this isn't prescribed since we
                        # can't zero it out later like with the local matrix.
                        if !haskey(dofmapping, global_col) && !haskey(dofmapping, global_mrow)
                            has_global_arrays || missing_global()
                            addindex!(global_matrix, mw, global_mrow, global_col)
                        end
                    else
                        local_matrix[local_mrow, local_col] += mw
                    end
                end
            end
        else
            for (local_row, global_row) in pairs(global_dofs)
                m = local_matrix[local_row, local_col]
                iszero(m) && continue # Skip early when zero to avoid remaining lookups
                row_coeffs = coefficients_for_dof(dofmapping, dofcoefficients, global_row)
                if row_coeffs === nothing
                    for (global_mcol, weight) in col_coeffs
                        local_mcol = findfirst(==(global_mcol), global_dofs)
                        mw = m * weight
                        if local_mcol === nothing
                            # Only modify the global array if this isn't prescribed since we
                            # can't zero it out later like with the local matrix.
                            if !haskey(dofmapping, global_row) && !haskey(dofmapping, global_mcol)
                                has_global_arrays || missing_global()
                                addindex!(global_matrix, mw, global_row, global_mcol)
                            end
                        else
                            local_matrix[local_row, local_mcol] += mw
                        end
                    end
                else
                    for (global_mcol, weight_col) in col_coeffs
                        local_mcol = findfirst(==(global_mcol), global_dofs)
                        for (global_mrow, weight_row) in row_coeffs
                            mww = m * weight_col * weight_row
                            local_mrow = findfirst(==(global_mrow), global_dofs)
                            if local_mcol === nothing || local_mrow === nothing
                                # Only modify the global array if this isn't prescribed since we
                                # can't zero it out later like with the local matrix.
                                if !haskey(dofmapping, global_mrow) && !haskey(dofmapping, global_mcol)
                                    has_global_arrays || missing_global()
                                    addindex!(global_matrix, mww, global_mrow, global_mcol)
                                end
                            else
                                local_matrix[local_mrow, local_mcol] += mww
                            end
                        end
                    end
                end
            end
            for (global_mcol, weight) in col_coeffs
                vw = local_vector[local_col] * weight
                local_mcol = findfirst(==(global_mcol), global_dofs)
                if local_mcol === nothing
                    has_global_arrays || missing_global()
                    addindex!(global_vector, vw, global_mcol)
                else
                    local_vector[local_mcol] += vw
                end
            end
            local_vector[local_col] = 0
        end
    end
end
