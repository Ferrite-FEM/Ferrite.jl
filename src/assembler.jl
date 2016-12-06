immutable Assembler{T}
    I::Vector{Int}
    J::Vector{Int}
    V::Vector{T}
end


function Assembler(N)
    I = Int[]
    J = Int[]
    V = Float64[]
    sizehint!(I, N)
    sizehint!(J, N)
    sizehint!(V, N)

    Assembler(I, J, V)
end

"""
    start_assemble([N=0]) -> Assembler

Call before starting an assembly.

Returns an `Assembler` type that is used to hold the intermediate
data before an assembly is finished.
"""
function start_assemble(N::Int=0)
    return Assembler(N)
end

"""
    assemble!(a, Ke, edof)

Assembles the element matrix `Ke` into `a`.
"""
function assemble!{T}(a::Assembler{T}, Ke::AbstractMatrix{T}, edof::AbstractVector{Int})
    n_dofs = length(edof)
    append!(a.V, Ke[:])
    for dof1 in 1:n_dofs, dof2 in 1:n_dofs
        push!(a.J, edof[dof1])
        push!(a.I, edof[dof2])
    end
end

"""
    end_assemble(a::Assembler) -> K

Finalizes an assembly. Returns a sparse matrix with the
assembled values.
"""
function end_assemble(a::Assembler)
    return sparse(a.I, a.J, a.V)
end
