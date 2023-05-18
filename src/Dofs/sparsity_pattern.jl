"""
    create_sparsity_pattern(dh::DofHandler; coupling, topology::Union{Nothing, AbstractTopology} = nothing)

Create the sparsity pattern corresponding to the degree of freedom
numbering in the [`DofHandler`](@ref). Return a `SparseMatrixCSC`
with stored values in the correct places.

The keyword argument `coupling` can be used to specify how fields (or components) in the dof
handler couple to each other. `coupling` should be a square matrix of booleans with
number of rows/columns equal to the total number of fields, or total number of components,
in the DofHandler with `true` if fields are coupled and `false` if
not. By default full coupling is assumed.

See the [Sparsity Pattern](@ref) section of the manual.

```julia-repl
julia> using Ferrite

julia> grid = generate_grid(Line, (3,));

julia> topology = ExclusiveTopology(grid);

julia> ip = DiscontinuousLagrange{1, RefCube, 2}();

julia> ipc = Lagrange{1, RefCube, 1}();

julia> dh = DofHandler(grid);

julia> add!(dh, :u, 1,ip);

julia> add!(dh, :v, 1,ipc);

julia> close!(dh);;

julia> K = create_sparsity_pattern(dh, topology = topology)
13×13 SparseMatrixCSC{Float64, Int64} with 109 stored entries:
 0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0   ⋅    ⋅    ⋅    ⋅    ⋅
 0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0   ⋅    ⋅    ⋅    ⋅    ⋅
 0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0   ⋅    ⋅    ⋅    ⋅    ⋅
 0.0  0.0  0.0  0.0  0.0   ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅ 
 0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0   ⋅    ⋅    ⋅    ⋅
 0.0  0.0  0.0   ⋅   0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0   ⋅
 0.0  0.0  0.0   ⋅   0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0   ⋅
 0.0  0.0  0.0   ⋅   0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0   ⋅
  ⋅    ⋅    ⋅    ⋅   0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
  ⋅    ⋅    ⋅    ⋅    ⋅   0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
  ⋅    ⋅    ⋅    ⋅    ⋅   0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
  ⋅    ⋅    ⋅    ⋅    ⋅   0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
  ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅   0.0  0.0  0.0  0.0  0.0
```
"""
function create_sparsity_pattern(dh::AbstractDofHandler; coupling=nothing,
    topology::Union{Nothing, AbstractTopology} = nothing)
    return _create_sparsity_pattern(dh, nothing, false, true, coupling; topology)
end

"""
    create_symmetric_sparsity_pattern(dh::DofHandler; coupling, topology::Union{Nothing, AbstractTopology} = nothing)

Create the symmetric sparsity pattern corresponding to the degree of freedom
numbering in the [`DofHandler`](@ref) by only considering the upper
triangle of the matrix. Return a `Symmetric{SparseMatrixCSC}`.

See the [Sparsity Pattern](@ref) section of the manual.
"""
function create_symmetric_sparsity_pattern(dh::AbstractDofHandler; coupling=nothing,
    topology::Union{Nothing, AbstractTopology} = nothing)
    return Symmetric(_create_sparsity_pattern(dh, nothing, true, true, coupling; topology), :U)
end

"""
    create_symmetric_sparsity_pattern(dh::AbstractDofHandler, ch::ConstraintHandler, coupling, topology::Union{Nothing, AbstractTopology} = nothing)

Create a symmetric sparsity pattern accounting for affine constraints in `ch`. See
the Affine Constraints section of the manual for further details.
"""
function create_symmetric_sparsity_pattern(dh::AbstractDofHandler, ch::ConstraintHandler;
        keep_constrained::Bool=true, coupling=nothing, topology::Union{Nothing, AbstractTopology} = nothing)
    return Symmetric(_create_sparsity_pattern(dh, ch, true, keep_constrained, coupling; topology), :U)
end

"""
    create_sparsity_pattern(dh::AbstractDofHandler, ch::ConstraintHandler; coupling, topology::Union{Nothing, AbstractTopology} = nothing)

Create a sparsity pattern accounting for affine constraints in `ch`. See
the Affine Constraints section of the manual for further details.
"""
function create_sparsity_pattern(dh::AbstractDofHandler, ch::ConstraintHandler;
        keep_constrained::Bool=true, coupling=nothing, topology::Union{Nothing, AbstractTopology} = nothing)
    return _create_sparsity_pattern(dh, ch, false, keep_constrained, coupling; topology)
end

# Compute a coupling matrix of size (ndofs_per_cell × ndofs_per_cell) based on the input
# coupling which can be of size i) (nfields × nfields) specifying coupling between fields,
# ii) (ncomponents × ncomponents) specifying coupling between components, or iii)
# (ndofs_per_cell × ndofs_per_cell) specifying coupling between all local dofs, i.e. a
# "template" local matrix.
function _coupling_to_local_dof_coupling(dh::DofHandler, coupling::AbstractMatrix{Bool}, sym::Bool)
    sz = size(coupling, 1)
    sz == size(coupling, 2) || error("coupling not square")
    sym && (issymmetric(coupling) || error("coupling not symmetric"))

    # Return one matrix per (potential) sub-domain
    outs = Matrix{Bool}[]
    field_dims = map(fieldname -> getfielddim(dh, fieldname), dh.field_names)

    for fh in dh.fieldhandlers
        out = zeros(Bool, ndofs_per_cell(fh), ndofs_per_cell(fh))
        push!(outs, out)

        dof_ranges = [dof_range(fh, f) for f in fh.field_names]
        global_idxs = [findfirst(x -> x === f, dh.field_names) for f in fh.field_names]

        if sz == length(dh.field_names) # Coupling given by fields
            for (j, jrange) in pairs(dof_ranges), (i, irange) in pairs(dof_ranges)
                out[irange, jrange] .= coupling[global_idxs[i], global_idxs[j]]
            end
        elseif sz == sum(field_dims) # Coupling given by components
            component_offsets = pushfirst!(cumsum(field_dims), 0)
            for (jf, jrange) in pairs(dof_ranges), (j, J) in pairs(jrange)
                jc = mod1(j, field_dims[global_idxs[jf]]) + component_offsets[global_idxs[jf]]
                for (i_f, irange) in pairs(dof_ranges), (i, I) in pairs(irange)
                    ic = mod1(i, field_dims[global_idxs[i_f]]) + component_offsets[global_idxs[i_f]]
                    out[I, J] = coupling[ic, jc]
                end
            end
        elseif sz == ndofs_per_cell(fh) # Coupling given by template local matrix
            # TODO: coupling[fieldhandler_idx] if different template per subddomain
            out .= coupling
        else
            error("could not create coupling")
        end
    end
    return outs
end

function _create_sparsity_pattern(dh::AbstractDofHandler, ch#=::Union{ConstraintHandler, Nothing}=#, sym::Bool, keep_constrained::Bool, coupling::Union{AbstractMatrix{Bool},Nothing};
    topology::Union{Nothing, AbstractTopology} = nothing)
    @assert isclosed(dh)
    if !keep_constrained
        @assert ch !== nothing && isclosed(ch)
    end

    couplings = isnothing(coupling) ? nothing : _coupling_to_local_dof_coupling(dh, coupling, sym)

    # Allocate buffers. Compute an upper bound for the buffer length and allocate it all up
    # front since they will become large and expensive to re-allocate. The bound is exact
    # when keeping constrained dofs (default) and if not it only over-estimates with number
    # of entries eliminated by constraints.
    max_buffer_length = ndofs(dh) # diagonal elements
    uses_dg = false
    for (fhi, fh) in pairs(dh.fieldhandlers)
        set = fh.cellset
        n = ndofs_per_cell(fh)
        entries_per_cell = if coupling === nothing
            sym ? div(n * (n + 1), 2) : n^2
        else
            coupling_fh = couplings[fhi]
            count(coupling_fh[i, j] for i in 1:n for j in (sym ? i : 1):n)
        end
        if any(ip -> IsDiscontinuous(typeof(ip)<: VectorizedInterpolation ? typeof(ip.ip) : typeof(ip)),fh.field_interpolations)
            uses_dg = true
        end
        max_buffer_length += entries_per_cell * length(set)
    end
    dg_cnt = 0
    if uses_dg
        isnothing(topology) && (topology = ExclusiveTopology(dh.grid))
        dg_cnt = cross_element_coupling_count(dh,topology, sym, keep_constrained, couplings)        
    end
    max_buffer_length += dg_cnt
    I = Vector{Int}(undef, max_buffer_length)
    J = Vector{Int}(undef, max_buffer_length)
    global_dofs = Int[]
    cnt = 0

    for (fhi, fh) in pairs(dh.fieldhandlers)
        coupling === nothing || (coupling_fh = couplings[fhi])
        # TODO: Remove BitSet construction when SubDofHandler ensures sorted collections
        set = BitSet(fh.cellset)
        n = ndofs_per_cell(fh)
        resize!(global_dofs, n)
        @inbounds for element_id in set
            celldofs!(global_dofs, dh, element_id)
            for j in eachindex(global_dofs), i in eachindex(global_dofs)
                coupling === nothing || coupling_fh[i, j] || continue
                dofi = global_dofs[i]
                dofj = global_dofs[j]
                sym && (dofi > dofj && continue)
                !keep_constrained && (haskey(ch.dofmapping, dofi) || haskey(ch.dofmapping, dofj)) && continue
                cnt += 1
                I[cnt] = dofi
                J[cnt] = dofj
            end
        end
    end
    if uses_dg
        I[cnt+1:cnt+dg_cnt],J[cnt+1:cnt+dg_cnt] = cross_element_coupling(dh,topology,sym, keep_constrained, couplings, max_buffer_length = dg_cnt)
        cnt += dg_cnt
    end
    # Always add diagonal entries
    resize!(I, cnt + ndofs(dh))
    resize!(J, cnt + ndofs(dh))
    @inbounds for d in 1:ndofs(dh)
        cnt += 1
        I[cnt] = d
        J[cnt] = d
    end
    @assert length(I) == length(J) == cnt

    K = spzeros!!(Float64, I, J, ndofs(dh), ndofs(dh))

    # If ConstraintHandler is given, create the condensation pattern due to affine constraints
    if ch !== nothing
        @assert isclosed(ch)
        fill!(K.nzval, 1)
        _condense_sparsity_pattern!(K, ch.dofcoefficients, ch.dofmapping, keep_constrained)
        fillzero!(K)
    end

    return K
end

# Similar to Ferrite._condense!(K, ch), but only add the non-zero entries to K (that arises from the condensation process)
function _condense_sparsity_pattern!(K::SparseMatrixCSC{T}, dofcoefficients::Vector{Union{Nothing,DofCoefficients{T}}}, dofmapping::Dict{Int,Int}, keep_constrained::Bool) where T
    ndofs = size(K, 1)

    # Return early if there are no non-trivial affine constraints
    any(i -> !(i === nothing || isempty(i)), dofcoefficients) || return

    # Adding new entries to K is extremely slow, so create a new sparsity triplet for the
    # condensed sparsity pattern
    N = 2 * length(dofcoefficients) # TODO: Better size estimate for additional condensed sparsity pattern.
    I = Int[]; resize!(I, N)
    J = Int[]; resize!(J, N)

    cnt = 0
    for col in 1:ndofs
        col_coeffs = coefficients_for_dof(dofmapping, dofcoefficients, col)
        if col_coeffs === nothing
            !keep_constrained && haskey(dofmapping, col) && continue
            for ri in nzrange(K, col)
                row = K.rowval[ri]
                row_coeffs = coefficients_for_dof(dofmapping, dofcoefficients, row)
                row_coeffs === nothing && continue
                for (d, _) in row_coeffs
                    cnt += 1
                    _add_or_grow(cnt, I, J, d, col)
                end
            end
        else
            for ri in nzrange(K, col)
                row = K.rowval[ri]
                row_coeffs = coefficients_for_dof(dofmapping, dofcoefficients, row)
                if row_coeffs === nothing
                    !keep_constrained && haskey(dofmapping, row) && continue
                    for (d, _) in col_coeffs
                        cnt += 1
                        _add_or_grow(cnt, I, J, row, d)
                    end
                else
                    for (d1, _) in col_coeffs
                        !keep_constrained && haskey(dofmapping, d1) && continue
                        for (d2, _) in row_coeffs
                            !keep_constrained && haskey(dofmapping, d2) && continue
                            cnt += 1
                            _add_or_grow(cnt, I, J, d1, d2)
                        end
                    end
                end
            end
        end
    end

    resize!(I, cnt)
    resize!(J, cnt)

    # Fill the sparse matrix with a non-zero value so that :+ operation does not remove entries with value zero.
    K2 = spzeros!!(Float64, I, J, ndofs, ndofs)
    fill!(K2.nzval, 1)

    K .+= K2

    return nothing
end

for (func,                              pre_f,                                                                                      inner_f,                                return_values) in (
    (:cross_element_coupling_count,     :(nothing),                                                                                 :(nothing),                             :(cnt)),
    (:cross_element_coupling,           :(I = Vector{Int}(undef, max_buffer_length); J = Vector{Int}(undef, max_buffer_length)),    :(I[cnt] = dofi; J[cnt] = dofj;),       :(I, J)),
)
    @eval begin
        function $(func)(dh::AbstractDofHandler, topology::ExclusiveTopology, sym::Bool, keep_constrained::Bool, couplings::Union{AbstractVector{<:AbstractMatrix{Bool}},Nothing} ; max_buffer_length::Int = 0)
            $(pre_f)
            element_dof_start = 0
            cnt = 0
            for (fhi, fh) in pairs(dh.fieldhandlers)
                isnothing(couplings) || (coupling_fh = couplings[fhi])
                for fi in fh.field_interpolations
                    if(!IsDiscontinuous(typeof(fi)<: VectorizedInterpolation ? typeof(fi.ip) : typeof(fi)))
                        element_dof_start += getnbasefunctions(fi)
                        continue
                    end
                    cont_fi =  get_continuous_interpolation(fi)
                    for cell_idx in BitSet(fh.cellset)
                        current_face_neighborhood = getdim(dh.grid.cells[cell_idx]) >1 ? topology.face_neighbor[cell_idx,:] : topology.vertex_neighbor[cell_idx,:]
                        shared_faces_idx = findall(!isempty,current_face_neighborhood)
                        for face_idx in shared_faces_idx
                            for neighbor_face in current_face_neighborhood[face_idx]
                                cell_dofs = celldofs(dh,cell_idx)[element_dof_start + 1 : element_dof_start + getnbasefunctions(fi)]
                                neighbour_dof_start = 0
                                for fi2 in fh.field_interpolations
                                    neighbour_dofs = celldofs(dh,neighbor_face[1])[neighbour_dof_start + 1 : neighbour_dof_start + getnbasefunctions(fi2)]
                                    neighbour_unique_dofs = neighbour_dofs[.!(neighbour_dofs .∈ Ref(celldofs(dh,cell_idx)))]
                                    for j in eachindex(neighbour_unique_dofs), i in eachindex(cell_dofs)
                                        isnothing(couplings) || coupling_fh[i+element_dof_start,j+neighbour_dof_start] || continue
                                        dofi = cell_dofs[i]
                                        dofj = neighbour_unique_dofs[j]
                                        sym && (dofi > dofj && continue)
                                        !keep_constrained && (haskey(ch.dofmapping, dofi) || haskey(ch.dofmapping, dofj)) && continue
                                        cnt += 1
                                        $(inner_f)
                                    end
                                    neighbour_dof_start += getnbasefunctions(fi2)
                                end
                            end
                        end
                    end
                    element_dof_start += getnbasefunctions(fi)
                end
            end
            return $(return_values)
        end
    end
end