struct IJAssembler <: Ferrite.AbstractSparseAssembler
    I::Vector{Int}
    J::Vector{Int}
end
function assemble!(a::IJAssembler, dofs::AbstractVector{Int}, local_matrix::AbstractMatrix{Bool})
    for (j, J) in pairs(dofs), (i, I) in pairs(dofs)
        if local_matrix[i, j]
            push!(a.I, I)
            push!(a.J, J)
        end
    end
    return a
end
function addindex!(A::IJAssembler, _, i::Integer, j::Integer)
    push!(A.I, i)
    push!(A.J, j)
    return A
end

struct BlackHole end
addindex!(::BlackHole, _, ::Int) = nothing

function create_sparsity_pattern2(dh::AbstractDofHandler, ch::ConstraintHandler; coupling=nothing, keep_constrained::Bool=true)
    @assert isclosed(dh)
    @assert isclosed(ch)
    @assert ch.dh === dh
    # Set up a template local matrix
    n = ndofs_per_cell(dh)
    if coupling === nothing
        template = ones(Bool, n, n)
    else
        template = _coupling_to_local_dof_coupling(dh, coupling, #=sym=# false)
    end
    # Create a cache of the matrices for MixedDofHandler where the element sizes might differ
    matrix_cache = Dict{Int,NTuple{2,Matrix{Bool}}}(n => (copy(template), template))
    local_vector = zeros(Float64, n) # dummy vector
    assembler = Ferrite.IJAssembler(Int[], Int[])
    for cc in CellIterator(dh)
        n = ndofs_per_cell(dh, cellid(cc))
        local_matrix, template_local_matrix = get!(matrix_cache, n) do
            @assert dh isa MixedDofHandler
            (ones(Bool, n, n), ones(Bool, n, n))
        end
        #
        # Reset local matrix to the template matrix
        copy!(local_matrix, template_local_matrix)
        # Assemble the local matrix *before* constraints if constrained entries are kept
        if keep_constrained
            assemble!(assembler, celldofs(cc), local_matrix)
        end
        # Use the inner part of apply_assemble! which modifies the local
        # matrix/vector and only adds remote elements to the assembler
        _apply_local!(
            local_matrix, local_vector, celldofs(cc), ch,
            #=apply_zero=# false, assembler, BlackHole(),
        )
        # Assemble the local matrix *after* constraints if constrained entries are not kept
        if !keep_constrained
            assemble!(assembler, celldofs(cc), local_matrix)
        end
    end
    return spzeros!!(Float64, assembler.I, assembler.J, ndofs(dh), ndofs(dh))
end

export create_sparsity_pattern2
