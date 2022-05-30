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
which applies the condition via [`apply!`](@ref) and/or [`apply_zero!`](@ref).
"""
struct Dirichlet # <: Constraint
    f::Function # f(x,t) -> value
    faces::Union{Set{Int},Set{FaceIndex},Set{EdgeIndex},Set{VertexIndex}}
    field_name::Symbol
    components::Vector{Int} # components of the field
    local_face_dofs::Vector{Int}
    local_face_dofs_offset::Vector{Int}
end
function Dirichlet(field_name::Symbol, faces::Union{T}, f::Function, component::Int=1) where T
    Dirichlet(field_name, copy(faces), f, [component])
end
function Dirichlet(field_name::Symbol, faces::Set{T}, f::Function, components::AbstractVector{Int}) where T
    unique(components) == components || error("components not unique: $components")
    # issorted(components) || error("components not sorted: $components")
    return Dirichlet(f, copy(faces), field_name, Vector(components), Int[], Int[])
end


"""
    AffineConstraint(constrained_dofs::Int, master_dofs::Vector{Int}, coeffs::Vector{T}, b::T) where T

Define an affine/linear constraint to constrain dofs of the form `u_i = ∑(u[j] * a[j]) + b`.
"""
struct AffineConstraint{T}
    constrained_dof::Int
    entries::Vector{Pair{Int, T}} # masterdofs and factors
    b::T # inhomogeneity
end

"""
    ConstraintHandler

Collection of constraints.
"""
struct ConstraintHandler{DH<:AbstractDofHandler,T}
    dbcs::Vector{Dirichlet}
    acs::Vector{AffineConstraint}
    prescribed_dofs::Vector{Int}
    free_dofs::Vector{Int}
    inhomogeneities::Vector{T}
    dofmapping::Dict{Int,Int} # global dof -> index into dofs and inhomogeneities
    bcvalues::Vector{BCValues{T}}
    dh::DH
    closed::ScalarWrapper{Bool}
end

function ConstraintHandler(dh::AbstractDofHandler)
    @assert isclosed(dh)
    ConstraintHandler(Dirichlet[], AffineConstraint[], Int[], Int[], Float64[], Dict{Int,Int}(), BCValues{Float64}[], dh, ScalarWrapper(false))
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
    @inbounds for ac in ch.acs
        global_dof = ac.constrained_dof
        for (d, v) in ac.entries
            f[d] += f[global_dof] * v
        end
        f[global_dof] = 0
    end
    @inbounds for i in 1:length(ch.inhomogeneities)
        d = ch.prescribed_dofs[i]
        v = ch.inhomogeneities[i]
        vz = applyzero ? zero(eltype(f)) : v
        f[d] = vz * m
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

    # Make ch.prescribed_dofs unique and sorted, and do the same operations for ch.inhomogeneities
    # TODO: This is probably quite slow atm, and the unique!() and sort() functions can be combined?
    dofs_vals = unique(first, zip(ch.prescribed_dofs, ch.inhomogeneities))
    copy!(ch.prescribed_dofs, getindex.(dofs_vals, 1))
    copy!(ch.inhomogeneities, getindex.(dofs_vals, 2))

    I = sortperm(ch.prescribed_dofs)
    ch.prescribed_dofs .= ch.prescribed_dofs[I]
    ch.inhomogeneities .= ch.inhomogeneities[I]

    copy!(ch.free_dofs, setdiff(1:ndofs(ch.dh), ch.prescribed_dofs))

    for i in 1:length(ch.prescribed_dofs)
        ch.dofmapping[ch.prescribed_dofs[i]] = i
    end

    # TODO:
    # Do a bunch of checks to see if the affine constraints are linearly indepented etc.
    # If they are not, it is possible to automatically reformulate the constraints
    # such that they become independent. However, at this point, it is left to
    # the user to assure this.
    sort!(ch.acs, by = ac -> ac.constrained_dof)

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

Add a `Dirichlet` boundary condition to the `ConstraintHandler`.
"""
function add!(ch::ConstraintHandler, dbc::Dirichlet)
    dbc_check(ch, dbc)
    celltype = getcelltype(ch.dh.grid)
    @assert isconcretetype(celltype)

    field_idx = find_field(ch.dh, dbc.field_name)
    # Extract stuff for the field
    interpolation = getfieldinterpolation(ch.dh, field_idx)#ch.dh.field_interpolations[field_idx]
    field_dim = getfielddim(ch.dh, field_idx)#ch.dh.field_dims[field_idx] # TODO: I think we don't need to extract these here ...


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
function add!(ch::ConstraintHandler, newac::AffineConstraint)
    # Basic error checking
    for ac in ch.acs
        (ac.constrained_dof == newac.constrained_dof) &&
            error("Constraint already exist for dof $(ac.constrained_dof)")
        any(x -> x.first == newac.constrained_dof, ac.entries) &&
            error("New constrained dof $(newac.constrained_dof) is already used as a master dof.")
    end
    push!(ch.acs, newac)
    push!(ch.prescribed_dofs, newac.constrained_dof)
    push!(ch.inhomogeneities, newac.b)
    return ch
end

function _add!(ch::ConstraintHandler, dbc::Dirichlet, bcfaces::Set{Index}, interpolation::Interpolation, field_dim::Int, offset::Int, bcvalue::BCValues, cellset::Set{Int}=Set{Int}(1:getncells(ch.dh.grid))) where {Index<:BoundaryIndex}
    local_face_dofs, local_face_dofs_offset =
        _local_face_dofs_for_bc(interpolation, field_dim, dbc.components, offset, boundaryfunction(eltype(bcfaces)))
    copy!(dbc.local_face_dofs, local_face_dofs)
    copy!(dbc.local_face_dofs_offset, local_face_dofs_offset)

    # loop over all the faces in the set and add the global dofs to `constrained_dofs`
    constrained_dofs = Int[]
    redundant_faces = Index[]
    for (cellidx, faceidx) in bcfaces
        if cellidx ∉ cellset
            push!(redundant_faces, Index(cellidx, faceidx)) # will be removed from dbc
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
    for _ in 1:length(constrained_dofs)
        push!(ch.inhomogeneities, NaN)
    end
    return ch
end

# Calculate which local dof index live on each face:
# face `i` have dofs `local_face_dofs[local_face_dofs_offset[i]:local_face_dofs_offset[i+1]-1]
function _local_face_dofs_for_bc(interpolation, field_dim, components, offset, boundaryfunc::F=faces) where F
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
    copy!(dbc.local_face_dofs_offset, constrained_dofs) # use this field to store the global dofs
    push!(ch.dbcs, dbc)
    push!(ch.bcvalues, bcvalue)
    append!(ch.prescribed_dofs, constrained_dofs)
    for _ in 1:length(constrained_dofs)
        push!(ch.inhomogeneities, NaN)
    end
    return ch
end

# Updates the DBC's to the current time `time`
function update!(ch::ConstraintHandler, time::Real=0.0)
    @assert ch.closed[]
    for (i,dbc) in enumerate(ch.dbcs)
        # Function barrier
        _update!(ch.inhomogeneities, dbc.f, dbc.faces, dbc.field_name, dbc.local_face_dofs, dbc.local_face_dofs_offset,
                 dbc.components, ch.dh, ch.bcvalues[i], ch.dofmapping, convert(Float64, time))
    end
end

# for faces
function _update!(inhomogeneities::Vector{Float64}, f::Function, faces::Set{<:BoundaryIndex}, field::Symbol, local_face_dofs::Vector{Int}, local_face_dofs_offset::Vector{Int},
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
                inhomogeneities[dbc_index] = bc_value[i]
                @debug println("prescribing value $(bc_value[i]) on global dof $(globaldof)")
            end
        end
    end
end

# for nodes
function _update!(inhomogeneities::Vector{Float64}, f::Function, nodes::Set{Int}, field::Symbol, nodeidxs::Vector{Int}, globaldofs::Vector{Int},
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
            inhomogeneities[dbc_index] = v
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

    apply!(v::AbstractVector, ch::ConstraintHandler)

Apply Dirichlet boundary conditions, specified in `ch`, to the solution vector `v`.

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
boundary conditions such that `du = K \\ rhs` give the expected result (e.g. with `du` zero
for all prescribed degrees of freedom).

    apply_zero!(v::AbstractVector, ch::ConstraintHandler)

Zero-out values in `v` corresponding to prescribed degrees of freedom.

These methods are typically used in e.g. a Newton solver where the increment, `du`, should
be prescribed to zero even for non-homogeneouos boundary conditions.

See also: [`apply!`](@ref).

# Examples
```julia
u = un + Δu                 # Current guess
K, g = assemble_system(...) # Assemble residual and tangent for current guess
apply_zero!(K, g, ch)       # Adjust tangent and residual to take prescribed values into account
ΔΔu = - K \\ g               # Compute the increment, prescribed values are "approximately" zero
apply_zero!(ΔΔu, ch)        # Make sure values are exactly zero
Δu .+= ΔΔu                  # Update current guess
```

!!! note
    The last call to `apply_zero!` is not strictly necessary since the boundary conditions
    should already be fulfilled after `apply!(K, g, ch)`. However, solvers of linear
    systems are not exact, and thus `apply!(ΔΔu, ch)` can be used to make sure the values
    for the prescribed degrees of freedom are fulfilled exactly.
"""
apply_zero!

apply_zero!(v::AbstractVector, ch::ConstraintHandler) = _apply_v(v, ch, true)
apply!(     v::AbstractVector, ch::ConstraintHandler) = _apply_v(v, ch, false)

function _apply_v(v::AbstractVector, ch::ConstraintHandler, apply_zero::Bool)
    @assert length(v) >= ndofs(ch.dh)
    v[ch.prescribed_dofs] .= apply_zero ? 0.0 : ch.inhomogeneities
    # Apply affine constraints, e.g u2 = u6 + b
    for ac in ch.acs
        for (d, s) in ac.entries
            v[ac.constrained_dof] += s * v[d]
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

@enumx ApplyStrategy Transpose Inplace
# For backwards compatibility
const APPLY_TRANSPOSE = ApplyStrategy.Transpose
const APPLY_INPLACE = ApplyStrategy.Inplace

function apply!(KK::Union{SparseMatrixCSC,Symmetric}, f::AbstractVector, ch::ConstraintHandler, applyzero::Bool=false;
                strategy::ApplyStrategy.T=ApplyStrategy.Transpose)
    K = isa(KK, Symmetric) ? KK.data : KK
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
                    f[K.rowval[j]] -= v * K.nzval[j]
                end
            end
        end
    end

    # Condense K (C' * K * C) and f (C' * f)
    _condense!(K, f, ch.acs)

    # Remove constrained dofs from the matrix
    zero_out_columns!(K, ch.prescribed_dofs)
    if strategy === ApplyStrategy.Transpose
        K′ = copy(K)
        transpose!(K′, K)
        zero_out_columns!(K′, ch.prescribed_dofs)
        transpose!(K, K′)
    elseif strategy === ApplyStrategy.Inplace
        K[ch.prescribed_dofs, :] .= 0
    else
        error("Unknown apply strategy")
    end

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

# Similar to Ferrite._condense!(K, ch), but only add the non-zero entries to K (that arises from the condensation process)
function _condense_sparsity_pattern!(K::SparseMatrixCSC, acs::Vector{AffineConstraint})
    ndofs = size(K, 1)

    # Store linear constraint index for each constrained dof
    distribute = Dict{Int,Int}(acs[c].constrained_dof => c for c in 1:length(acs))

    for col in 1:ndofs
        # Since we will possibly be pushing new entries to K, the field K.rowval will grow.
        # Therefor we must extract this before iterating over K
        range = nzrange(K, col)
        _rows = K.rowval[range]
        dcol = get(distribute, col, 0)
        if dcol == 0
            for row in _rows
                drow = get(distribute, row, 0)
                if drow != 0
                    ac = acs[drow]
                    for (d, _) in ac.entries
                        add_entry!(K, d, col)
                    end
                end
            end
        else
            for row in _rows
                drow = get(distribute, row, 0)
                if drow == 0
                    ac = acs[dcol]
                    for (d, _) in ac.entries
                        add_entry!(K, row, d)
                    end
                else
                    ac1 = acs[dcol]
                    for (d1, _) in ac1.entries
                        ac2 = acs[distribute[row]]
                        for (d2, _) in ac2.entries
                            add_entry!(K, d1, d2)
                        end
                    end
                end
            end
        end
    end
end

# Condenses K and f: C'*K*C, C'*f, in-place assuming the sparsity pattern is correct
function _condense!(K::SparseMatrixCSC, f::AbstractVector, acs::Vector{AffineConstraint})

    ndofs = size(K, 1)
    condense_f = !(length(f) == 0)
    condense_f && @assert( length(f) == ndofs )

    # Store linear constraint index for each constrained dof
    distribute = Dict{Int,Int}(acs[c].constrained_dof => c for c in 1:length(acs))

    for col in 1:ndofs
        dcol = get(distribute, col, 0)
        if dcol == 0
            for a in nzrange(K, col)
                row = K.rowval[a]
                drow = get(distribute, row, 0)
                if drow != 0
                    ac = acs[drow]
                    for (d, v) in ac.entries
                        Kval = K.nzval[a]
                        _addindex_sparsematrix!(K, v * Kval, d, col)
                    end

                    # Perform f - K*g. However, this has already been done in outside this functions so we skip this.
                    #if condense_f
                    #    f[col] -= K.nzval[a] * ac.b;
                    #end
                end
            end
        else
            for a in nzrange(K, col)
                row = K.rowval[a]
                drow = get(distribute, row, 0)
                if drow == 0
                    ac = acs[dcol]
                    for (d,v) in ac.entries
                        Kval = K.nzval[a]
                        _addindex_sparsematrix!(K, v * Kval, row, d)
                    end
                else
                    ac1 = acs[dcol]
                    for (d1,v1) in ac1.entries
                        ac2 = acs[drow]
                        for (d2,v2) in ac2.entries
                            Kval = K.nzval[a]
                            _addindex_sparsematrix!(K, v1 * v2 * Kval, d1, d2)
                        end
                    end
                end
            end

            if condense_f
                ac = acs[dcol]
                for (d,v) in ac.entries
                    f[d] += f[col] * v
                end
                @assert ac.constrained_dof == col
                f[ac.constrained_dof] = 0.0
            end
        end
    end
end

# Copied from SparseArrays._setindex_scalar!(...)
# Custom SparseArrays._setindex_scalar!() that throws error if entry K(_i,_j) does not exist
function _addindex_sparsematrix!(A::SparseMatrixCSC{Tv,Ti}, v::Tv, i::Ti, j::Ti) where {Tv, Ti}
    if !((1 <= i <= size(A, 1)) & (1 <= j <= size(A, 2)))
        throw(BoundsError(A, (i,j)))
    end
    coljfirstk = Int(SparseArrays.getcolptr(A)[j])
    coljlastk = Int(SparseArrays.getcolptr(A)[j+1] - 1)
    searchk = searchsortedfirst(rowvals(A), i, coljfirstk, coljlastk, Base.Order.Forward)
    if searchk <= coljlastk && rowvals(A)[searchk] == i
        # Column j contains entry A[i,j]. Update and return
        nonzeros(A)[searchk] += v
        return A
    end
    error("Sparsity pattern missing entries for the condensation pattern. Make sure to call `create_sparsity_pattern(dh::DofHandler, ch::ConstraintHandler) when using linear constraints.`")
end

# A[i,j] += 0.0 does not add entries to sparse matrices, so we need to first add 1.0, and then remove it
# TODO: Maybe this can be done for vectors i and j instead of doing it individually?
function add_entry!(A::SparseMatrixCSC, i::Int, j::Int)
    if iszero(A[i,j]) # Check first if zero to not remove already non-zero entries
        A[i,j] = 1
        A[i,j] = 0
    end
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

    for ac in ch.acs
        for (d, v) in ac.entries
            push!(I, ac.constrained_dof)
            j = searchsortedfirst(ch.free_dofs, d)
            push!(J, j)
            push!(V, v)
        end
        g[ac.constrained_dof] = ac.b
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

    celltype = getcelltype(ch.dh.grid, first(fh.cellset)) #Assume same celltype of all cells in fh.cellset

    field_idx = find_field(fh, dbc.field_name)
    # Extract stuff for the field
    interpolation = getfieldinterpolations(fh)[field_idx]
    field_dim = getfielddims(fh)[field_idx]

    if eltype(dbc.faces)==Int #Special case when dbc.faces is a nodeset
        bcvalue = BCValues(interpolation, default_interpolation(celltype), FaceIndex) #Not used by node bcs, but still have to pass it as an argument
    else
        bcvalue = BCValues(interpolation, default_interpolation(celltype), eltype(dbc.faces))
    end

    Ferrite._add!(ch, dbc, dbc.faces, interpolation, field_dim, field_offset(fh, dbc.field_name), bcvalue, fh.cellset)
    return ch
end

function _check_cellset_dirichlet(::AbstractGrid, cellset::Set{Int}, faceset::Set{<:BoundaryIndex})
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

create_symmetric_sparsity_pattern(dh::MixedDofHandler, ch::ConstraintHandler) = Symmetric(_create_sparsity_pattern(dh, ch, true), :U)
create_symmetric_sparsity_pattern(dh::DofHandler,      ch::ConstraintHandler) = Symmetric(_create_sparsity_pattern(dh, ch, true), :U)

create_sparsity_pattern(dh::MixedDofHandler, ch::ConstraintHandler) = _create_sparsity_pattern(dh, ch, false)
create_sparsity_pattern(dh::DofHandler,      ch::ConstraintHandler) = _create_sparsity_pattern(dh, ch, false)


"""
    PeriodicDirichlet(u, Γ⁻ => Γ⁺, component=1)
    PeriodicDirichlet(u, Γ⁻ => Γ⁺, f, component=1)

Create a periodic Dirichlet boundary condition for the field `u`, with a mirror boundary,
`Γ⁻` and an image boundary, `Γ⁺`. The condition is imposed in a strong sense, and requires
(i) a periodic domain (usually a cube) and (ii) a periodic mesh.

See also manual section on [Periodic boundary conditions](@ref).
"""
struct PeriodicDirichlet
    field_name::Symbol
    components::Vector{Int} # components of the field
    face_pairs::Vector{Pair{String,String}}
    func::Union{Function,Nothing}
end

PeriodicDirichlet(fn::Symbol, fp::Vector{<:Pair}, c::Union{Int,Vector{Int}}=1) =
    PeriodicDirichlet(fn, fp, nothing, c)

PeriodicDirichlet(fn::Symbol, fp::Vector{<:Pair}, f::Union{Function,Nothing}, c::Union{Int,Vector{Int}}=1) =
    PeriodicDirichlet(fn, sort!(vec(collect(c))), fp, f)

function add!(ch::ConstraintHandler, pdbc::PeriodicDirichlet)
    field_idx = find_field(ch.dh, pdbc.field_name)
    interpolation = getfieldinterpolation(ch.dh, field_idx)
    field_dim = getfielddim(ch.dh, field_idx)
    _add!(ch, pdbc, interpolation, field_dim, field_offset(ch.dh, pdbc.field_name))
    return ch
end

function _add!(ch::ConstraintHandler, pdbc::PeriodicDirichlet, interpolation::Interpolation,
               field_dim::Int, offset::Int)
    grid = ch.dh.grid
    Tx = typeof(first(ch.dh.grid.nodes).x) # Vec{D,T}
    # TODO: This requires full periodicity for now. (Why? I don't remember,
    #       but something in the code below assumes this... I think...)
    @assert length(pdbc.face_pairs) == length(Tx)

    # Indices of the local dofs for the faces
    local_face_dofs, local_face_dofs_offset =
        _local_face_dofs_for_bc(interpolation, field_dim, pdbc.components, offset)
    mirrored_local_face_dofs =
        mirror_local_dofs(local_face_dofs, local_face_dofs_offset, interpolation, length(pdbc.components))

    # First, create the facemap which maps mirror faces to image faces
    face_map = Dict{FaceIndex,FaceIndex}() # mirror face => image face
    dof_map = Dict{Int,Int}() # mirror dof => image => dof

    # Some stuff for finding the corner points
    all_node_idxs = Set{Int}()
    max_x = Tx(i -> typemin(eltype(Tx)))
    min_x = Tx(i -> typemax(eltype(Tx)))

    for (xyz, (mirror, image)) in enumerate(pdbc.face_pairs)
        # Need the order here, so collect
        mirror_faceset = collect(getfaceset(grid, mirror))
        mirror_mean_x = Tx[] # mean face coordinates
        for (c, f) in mirror_faceset
            fn = faces(grid.cells[c])[f]
            push!(mirror_mean_x, sum(grid.nodes[i].x for i in fn) / length(fn))
            # Also keep track of all nodes for corner finding
            union!(all_node_idxs, grid.cells[c].nodes)
        end
        # Same dance for the image
        image_faceset = collect(getfaceset(grid,  image))
        image_mean_x = Tx[]
        for (c, f) in image_faceset
            fn = faces(grid.cells[c])[f]
            push!(image_mean_x, sum(grid.nodes[i].x for i in fn) / length(fn))
            # Also keep track of all nodes for corner finding
            union!(all_node_idxs, grid.cells[c].nodes)
        end
        function extract_single_x_on_face(fs) # :)
            c, f = first(fs)
            fn = faces(grid.cells[c])[f]
            return grid.nodes[fn[1]].x
        end
        # Add a shift to help the tree search
        xmirror = extract_single_x_on_face(mirror_faceset)
        ximage  = extract_single_x_on_face(image_faceset)
        xdist = ximage[xyz] - xmirror[xyz]
        @assert xdist > 0
        shift = Tx(i -> i == xyz ? xdist : 0)
        for i in eachindex(mirror_mean_x)
            mirror_mean_x[i] += shift
        end
        # Use KDTree to find closest face
        tree = KDTree(image_mean_x)
        idxs, _ = NearestNeighbors.nn(tree, mirror_mean_x)
        for (m, idx) in zip(mirror_faceset, idxs)
            face_map[m] = image_faceset[idx]
        end
        # Update maximum/minimum coords from the bounding box of the tree
        max_x = Tx(i -> i == xyz ? max(max_x[i], tree.hyper_rec.maxes[i]) : max_x[i])
        min_x = Tx(i -> i == xyz ? min(min_x[i], tree.hyper_rec.maxes[i] - xdist) : min_x[i])

        # Create the constraints
        mirror_dofs = zeros(Int, ndofs_per_cell(ch.dh))
         image_dofs = zeros(Int, ndofs_per_cell(ch.dh))
        for (m, i) in face_map
            celldofs!(mirror_dofs, ch.dh, m[1])
            celldofs!( image_dofs, ch.dh, i[1])

            mdof_range = local_face_dofs_offset[m[2]] : (local_face_dofs_offset[m[2] + 1] - 1)
            idof_range = local_face_dofs_offset[i[2]] : (local_face_dofs_offset[i[2] + 1] - 1)

            for (md, id) in zip(mdof_range, idof_range)
                cdof = mirror_dofs[mirrored_local_face_dofs[md]]
                mdof = image_dofs[local_face_dofs[id]]

                if haskey(dof_map, mdof)
                    mdof′ = dof_map[mdof]
                    # @info "$cdof => $mdof, but $mdof => $mdof′, remapping $cdof => $mdof′."
                    push!(dof_map, cdof => mdof′)
                # elseif haskey(dof_map, cdof) && dof_map[cdof] == mdof
                    # @info "$cdof => $mdof already in the set, skipping."
                elseif haskey(dof_map, cdof)
                    # @info "$cdof => $mdof, but $cdof => $(dof_map[cdof]) already, skipping."
                else
                    # @info "$cdof => $mdof."
                    push!(dof_map, cdof => mdof)
                end
            end
        end
    end

    # Find the corners and put homogeneous Dirichlet there instead
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

    # Create another temp constraint handler if we need to compute inhomogeneities
    inhomogeneity_map = nothing
    if pdbc.func !== nothing
        chtmp2 = ConstraintHandler(ch.dh)
        all_faces = union!(Set{FaceIndex}(),
            (getfaceset(grid, x) for x in Iterators.flatten(pdbc.face_pairs))...
        )
        dbc_all = Dirichlet(pdbc.field_name, all_faces, pdbc.func, pdbc.components)
        add!(chtmp2, dbc_all); close!(chtmp2)
        # Call update! here since we need it to construct the affine constraints...
        # TODO: This doesn't allow for time dependent constraints...
        update!(chtmp2, 0.0)
        inhomogeneity_map = Dict{Int, Float64}()
        for (k, v) in dof_map
            g = chtmp2.inhomogeneities
            push!(inhomogeneity_map,
                  v => - g[chtmp2.dofmapping[v]] + g[chtmp2.dofmapping[k]]
            )
        end
    end

    # Any remaining mappings are added as homogeneous AffineConstraints
    for (k, v) in dof_map
        ac = AffineConstraint(k, [v => 1.0], inhomogeneity_map === nothing ? 0.0 : inhomogeneity_map[v])
        add!(ch, ac)
    end
    # Add the Dirichlet for the corners
    add!(ch, dbc)

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
    error("not implemented yet, please contribute :)")
end

function mirror_local_dofs(_, _, ::Lagrange{1}, ::Int)
    # For 1D there is nothing to do
end
function mirror_local_dofs(local_face_dofs, local_face_dofs_offset, ip::Lagrange{2,<:Union{RefCube,RefTetrahedron}}, n::Int)
    # For 2D we always permute since Ferrite defines dofs counter-clockwise
    ret = copy(local_face_dofs)
    for (i, f) in enumerate(faces(ip))
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
function mirror_local_dofs(_, _, ::Lagrange{3})
    error("not implemented yet, please contribute :)")
end
