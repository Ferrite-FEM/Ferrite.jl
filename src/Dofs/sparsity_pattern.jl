"""
    create_sparsity_pattern(dh::DofHandler; coupling, [topology::Union{Nothing, AbstractTopology}], [elements_coupling])

Create the sparsity pattern corresponding to the degree of freedom
numbering in the [`DofHandler`](@ref). Return a `SparseMatrixCSC`
with stored values in the correct places.

The keyword arguments `coupling` and `elements_coupling` can be used to specify how fields (or components) in the dof
handler couple to each other. `coupling` and `elements_coupling` should be square matrices of booleans with
number of rows/columns equal to the total number of fields, or total number of components,
in the DofHandler with `true` if fields are coupled and `false` if
not. By default full coupling is assumed inside the element with no coupling between elements.

If `topology` and `elements_coupling` are passed, dof of fields with discontinuous interpolations are coupled between elements according to `elements_coupling`.

See the [Sparsity Pattern](@ref) section of the manual.
"""
function create_sparsity_pattern(dh::AbstractDofHandler; coupling=nothing,
    topology::Union{Nothing, AbstractTopology} = nothing, elements_coupling = nothing)
    return _create_sparsity_pattern(dh, nothing, false, true, coupling, topology, elements_coupling)
end

"""
    create_symmetric_sparsity_pattern(dh::DofHandler; coupling, topology::Union{Nothing, AbstractTopology}, elements_coupling)

Create the symmetric sparsity pattern corresponding to the degree of freedom
numbering in the [`DofHandler`](@ref) by only considering the upper
triangle of the matrix. Return a `Symmetric{SparseMatrixCSC}`.

See the [Sparsity Pattern](@ref) section of the manual.
"""
function create_symmetric_sparsity_pattern(dh::AbstractDofHandler; coupling=nothing,
    topology::Union{Nothing, AbstractTopology} = nothing, elements_coupling = nothing)
    return Symmetric(_create_sparsity_pattern(dh, nothing, true, true, coupling, topology, elements_coupling), :U)
end

"""
    create_symmetric_sparsity_pattern(dh::AbstractDofHandler, ch::ConstraintHandler; coupling, topology::Union{Nothing, AbstractTopology}, elements_coupling)

Create a symmetric sparsity pattern accounting for affine constraints in `ch`. See
the Affine Constraints section of the manual for further details.
"""
function create_symmetric_sparsity_pattern(dh::AbstractDofHandler, ch::ConstraintHandler;
        keep_constrained::Bool=true, coupling=nothing, topology::Union{Nothing, AbstractTopology} = nothing,
        elements_coupling = nothing)
    return Symmetric(_create_sparsity_pattern(dh, ch, true, keep_constrained, coupling, topology, elements_coupling), :U)
end

"""
    create_sparsity_pattern(dh::AbstractDofHandler, ch::ConstraintHandler; coupling, topology::Union{Nothing, AbstractTopology} = nothing)

Create a sparsity pattern accounting for affine constraints in `ch`. See
the Affine Constraints section of the manual for further details.
"""
function create_sparsity_pattern(dh::AbstractDofHandler, ch::ConstraintHandler;
        keep_constrained::Bool=true, coupling=nothing, topology::Union{Nothing, AbstractTopology} = nothing,
        elements_coupling = nothing)
    return _create_sparsity_pattern(dh, ch, false, keep_constrained, coupling, topology, elements_coupling)
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

    for sdh in dh.subdofhandlers
        out = zeros(Bool, ndofs_per_cell(sdh), ndofs_per_cell(sdh))
        push!(outs, out)

        dof_ranges = [dof_range(sdh, f) for f in sdh.field_names]
        global_idxs = [findfirst(x -> x === f, dh.field_names) for f in sdh.field_names]

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
        elseif sz == ndofs_per_cell(sdh) # Coupling given by template local matrix
            # TODO: coupling[fieldhandler_idx] if different template per subddomain
            out .= coupling
        else
            error("could not create coupling")
        end
    end
    return outs
end

"""
    _add_elements_coupling(coupling_sdh, dof_i, dof_j, cell_field_dofs, neighbor_field_dofs, i, j, sym, keep_constrained, ch, cnt, I, J)

Helper function used to mutate `I` and `J` to add cross-element coupling.
"""
function _add_elements_coupling(coupling_sdh::Matrix{Bool}, dof_i::Int, dof_j::Int,
        cell_field_dofs::Union{Vector{Int}, SubArray}, neighbor_field_dofs::Union{Vector{Int}, SubArray},
        i::Int, j::Int, sym::Bool, keep_constrained::Bool, ch::Union{ConstraintHandler, Nothing}, cnt::Int, I::Vector{Int}, J::Vector{Int})

    coupling_sdh[dof_i, dof_j] || return cnt
    dofi = cell_field_dofs[i]
    dofj = neighbor_field_dofs[j]
    sym && (dofj > dofi && return cnt)
    !keep_constrained && (haskey(ch.dofmapping, dofi) || haskey(ch.dofmapping, dofj)) && return cnt
    cnt += 1
    _add_or_grow(cnt, I, J, dofi, dofj) 
    return cnt
end

"""
    cross_element_coupling!(dh::DofHandler, topology::ExclusiveTopology, sym::Bool, keep_constrained::Bool, couplings::Union{AbstractVector{<:AbstractMatrix{Bool}},Nothing}, cnt::Int, I::Vector{Int}, J::Vector{Int})

Mutates `I, J` to account for cross-element coupling in fields with discontinuous interpolations by calling [`_add_elements_coupling`](@ref).
Returns the updated value of `cnt`.

Used internally for sparsity patterns with discontinuous interpolations.
"""
function cross_element_coupling!(dh::DofHandler, ch::Union{ConstraintHandler, Nothing}, topology::ExclusiveTopology, sym::Bool, keep_constrained::Bool, couplings::AbstractVector{<:AbstractMatrix{Bool}}, cnt::Int, I::Vector{Int}, J::Vector{Int})
    cc1 = CellCache(dh, UpdateFlags(false, false, true))
    cc2 = CellCache(dh, UpdateFlags(false, false, true))
    neighborhood = getdim(dh.grid.cells[1]) > 1 ? topology.cell_face_neighbor : topology.cell_neighbor
    for (sdh_idx, sdh) in pairs(dh.subdofhandlers)
        # Buffering interpolation information for type stability/better allocations
        ip_infos = [InterpolationInfo(ip) for ip in sdh.field_interpolations]
        coupling_sdh = couplings[sdh_idx]
        for (cell_field_i, cell_field) in pairs(sdh.field_names)
            fii = ip_infos[cell_field_i]
            # Couple element for discontinuous interpolations only
            fii.is_discontinuous || continue
            for cell_idx in BitSet(sdh.cellset)
                # Making sure current field is defined for current cell
                dofrange1 = dof_range(sdh, cell_field)
                reinit!(cc1, cell_idx)
                cell_field_dofs = @view cc1.dofs[dofrange1]
                # For 1D case, cells must share faces to be neighbors. Otherwise use cell_face_neighbor for higher dimensions
                for neighbor_cell in neighborhood[cell_idx]
                    sdh2 = dh.subdofhandlers[dh.cell_to_subdofhandler[neighbor_cell.idx]]
                    for (neighbor_field_i, neighbor_field) in enumerate(sdh.field_names)
                        fii2 = ip_infos[neighbor_field_i]
                        neighbor_field ∈ sdh2.field_names && fii2.is_discontinuous || continue
                        dofrange2 = dof_range(sdh2, neighbor_field)
                        reinit!(cc2, neighbor_cell.idx)
                        neighbor_field_dofs = @view cc2.dofs[dofrange2]
                        # Typical coupling procedure
                        for (j, dof_j) in pairs(dofrange2), (i, dof_i) in pairs(dofrange1)
                            cnt = _add_elements_coupling(coupling_sdh, dof_i, dof_j, cell_field_dofs, neighbor_field_dofs, i, j, sym, keep_constrained, ch, cnt, I, J)
                            cnt = _add_elements_coupling(coupling_sdh, dof_j, dof_i, neighbor_field_dofs, cell_field_dofs, j, i, sym, keep_constrained, ch, cnt, I, J)
                        end
                    end
                end
            end
        end
    end
    return cnt
end

function _create_sparsity_pattern(dh::AbstractDofHandler, ch#=::Union{ConstraintHandler, Nothing}=#, sym::Bool, keep_constrained::Bool, coupling::Union{AbstractMatrix{Bool},Nothing},
    topology::Union{Nothing, AbstractTopology}, elements_coupling::Union{AbstractMatrix{Bool},Nothing})
    @assert isclosed(dh)
    if !keep_constrained
        @assert ch !== nothing && isclosed(ch)
    end

    couplings = isnothing(coupling) ? nothing : _coupling_to_local_dof_coupling(dh, coupling, sym)
    elements_couplings = isnothing(elements_coupling) ? nothing : _coupling_to_local_dof_coupling(dh, elements_coupling, sym)

    # Allocate buffers. Compute an upper bound for the buffer length and allocate it all up
    # front since they will become large and expensive to re-allocate. The bound is exact
    # when keeping constrained dofs (default) and if not it only over-estimates with number
    # of entries eliminated by constraints.
    max_buffer_length = ndofs(dh) # diagonal elements
    has_discontinuous_ip = false
    for (sdh_idx, sdh) in pairs(dh.subdofhandlers)
        set = sdh.cellset
        n = ndofs_per_cell(sdh)
        entries_per_cell = if coupling === nothing
            sym ? div(n * (n + 1), 2) : n^2
        else
            coupling_sdh = couplings[sdh_idx]
            count(coupling_sdh[i, j] for i in 1:n for j in (sym ? i : 1):n)
        end
        has_discontinuous_ip = has_discontinuous_ip || any(ip -> is_discontinuous(ip),sdh.field_interpolations)
        max_buffer_length += entries_per_cell * length(set)
    end
    I = Vector{Int}(undef, max_buffer_length)
    J = Vector{Int}(undef, max_buffer_length)
    global_dofs = Int[]
    cnt = 0

    for (sdh_idx, sdh) in pairs(dh.subdofhandlers)
        coupling === nothing || (coupling_sdh = couplings[sdh_idx])
        # TODO: Remove BitSet construction when SubDofHandler ensures sorted collections
        set = BitSet(sdh.cellset)
        n = ndofs_per_cell(sdh)
        resize!(global_dofs, n)
        @inbounds for element_id in set
            celldofs!(global_dofs, dh, element_id)
            for j in eachindex(global_dofs), i in eachindex(global_dofs)
                coupling === nothing || coupling_sdh[i, j] || continue
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
    if has_discontinuous_ip && !isnothing(topology) && !isnothing(elements_coupling) && any(elements_coupling)
       cnt = cross_element_coupling!(dh, ch, topology, sym, keep_constrained, elements_couplings, cnt, I, J)
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
