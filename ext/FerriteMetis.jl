module FerriteMetis

# This extension requires modules as type parameters
# https://github.com/JuliaLang/julia/pull/47749
if VERSION >= v"1.10.0-DEV.90"

using Ferrite
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
    dh::DofHandler,
    ch::Union{ConstraintHandler,Nothing},
    order::DofOrder.Ext{Metis}
)

    # Expand the coupling matrix to size ndofs_per_cell × ndofs_per_cell
    coupling = order.coupling
    if coupling !== nothing
        # Set sym = true since Metis.permutation requires a symmetric graph.
        # TODO: Perhaps just symmetrize it: coupling = coupling' .| coupling
        couplings = Ferrite._coupling_to_local_dof_coupling(dh, coupling, #= sym =# true)
    end

    # Create the CSR (CSC, but pattern is symmetric so equivalent) using
    # Metis.idx_t as the integer type
    buffer_length = 0
    for (sdhi, sdh) in pairs(dh.subdofhandlers)
        n = ndofs_per_cell(sdh)
        entries_per_cell = if coupling === nothing
            n * (n - 1)
        else
            count(couplings[sdhi][i, j] for i in 1:n, j in 1:n if i != j)
        end
        buffer_length += entries_per_cell * length(sdh.cellset)
    end
    I = Vector{idx_t}(undef, buffer_length)
    J = Vector{idx_t}(undef, buffer_length)
    idx = 0

    for (sdhi, sdh) in pairs(dh.subdofhandlers)
        coupling === nothing || (coupling_fh = couplings[sdhi])
        for cc in CellIterator(dh, sdh.cellset)
            dofs = celldofs(cc)
            for (j, dofj) in pairs(dofs), (i, dofi) in pairs(dofs)
                dofi == dofj && continue # Metis doesn't want the diagonal
                coupling === nothing || coupling_fh[i, j] || continue
                idx += 1
                I[idx] = dofi
                J[idx] = dofj
            end
        end
    end
    @assert length(I) == length(J) == idx
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

end # VERSION check

end # module FerriteMetis
