type Assembler
    I::Vector{Int}
    J::Vector{Int}
    V::Vector{Float64}
    n::Int
end
Assembler(N) = Assembler(Int[], Int[], Float64[], N)

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
    assemble(edof, a, Ke)

Assembles the element matrix `Ke` into `a`.
"""
function assemble(edof, a::Assembler, Ke::Matrix)
    for ele in size(edof, 1)
        append!(a.V, Ke[:])
        for dof1 in edof[2:end], dof2 in edof[2:end]
            push!(a.I, dof1)
            push!(a.J, dof2)
        end
    end
end

"""
    end_assemble(a::Assembler) -> K

Finalizes an assembly. Returns a sparse matrix with the
assembled values.
"""
function end_assemble(a::Assembler)
    if a.n == 0
        return sparse(a.I, a.J, a.V)
    else
        return sparse(a.I, a.J, a.V, a.n, a.n)
    end
end
