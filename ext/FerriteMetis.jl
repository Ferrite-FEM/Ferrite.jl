module FerriteMetis

using Ferrite
using Ferrite: AbstractDofHandler
using Metis.LibMetis: idx_t
using Metis: Metis
using SparseArrays: sparse

struct MetisOrder <: DofOrder.Ext{Metis}
    coupling::Union{Matrix{Bool},Nothing}
end

"""
    DofOrder.Ext{Metis}(; coupling)

Fill-reducing permutation order from [Metis.jl](https://github.com/JuliaSparse/Metis.jl).

Since computing the permutation involves constructing the structural couplings between all
DoFs the field/component coupling can be provided; see [`create_sparsity_pattern`](@ref) for
details.
"""
function DofOrder.Ext{Metis}(;
    coupling::Union{AbstractMatrix{Bool},Nothing}=nothing,
)
    return MetisOrder(coupling)
end

function Ferrite.compute_renumber_permutation(
    dh::AbstractDofHandler,
    ch::Union{ConstraintHandler,Nothing},
    order::DofOrder.Ext{Metis}
)

    # Expand the coupling matrix to size ndofs_per_cell × ndofs_per_cell
    coupling = order.coupling
    if coupling === nothing
        n = ndofs_per_cell(dh)
        entries_per_cell = n * (n - 1)
    else # coupling !== nothing
        # Set sym = true since Metis.permutation requires a symmetric graph.
        # TODO: Perhaps just symmetrize it: coupling = coupling' .| coupling
        coupling = Ferrite._coupling_to_local_dof_coupling(dh, coupling, #= sym =# true)
        # Compute entries per cell, subtract diagonal elements
        entries_per_cell =
            count(coupling[i, j] for i in axes(coupling, 1), j in axes(coupling, 2) if i != j)
    end

    # Create the CSR (CSC, but pattern is symmetric so equivalent) using
    # Metis.idx_t as the integer type
    L = entries_per_cell * getncells(dh.grid)
    I = Vector{idx_t}(undef, L)
    J = Vector{idx_t}(undef, L)
    idx = 0
    @inbounds for cc in CellIterator(dh)
        dofs = celldofs(cc)
        for (i, dofi) in pairs(dofs), (j, dofj) in pairs(dofs)
            dofi == dofj && continue # Metis doesn't want the diagonal
            coupling === nothing || coupling[i, j] || continue
            idx += 1
            I[idx] = dofi
            J[idx] = dofj
        end
    end
    @assert idx == L
    N = ndofs(dh)
    # TODO: Use spzeros! in Julia 1.10.
    S = sparse(I, J, zeros(Float32, length(I)), N, N)

    # Add entries from affine constraints
    if ch !== nothing
        error("TODO: Use constraints.")
    end

    # Construct a Metis.Graph
    G = Metis.Graph(idx_t(N), S.colptr, S.rowval)

    # Compute the permutation
    _, perm = Metis.permutation(G)

    return perm
end

end # module FerriteMetis
