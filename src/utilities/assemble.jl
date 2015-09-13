"""
Assembles the the element stiffness matrix `Ke`
to the global stiffness matrix `K`.
"""
function assemble(edof::Array, K::AbstractMatrix, Ke::Matrix)
    (nr, nc) = size(Ke)
    if nr != nc
        throw(DimensionMismatch("Stiffness matrix is not square (#rows=$nr #cols=$nc)"))
    elseif length(edof) != nr + 1
        len_edof = length(edof)
        throw(DimensionMismatch("Mismatch between sizes in edof and Ke (edof($len_edof) Ke($nr,$nc)"))
    end
    _Ke = convert(Matrix{Float64}, Ke)
    _edof = convert(Array{Int}, edof)
    dofs = vec(_edof[2:end])
    K[dofs, dofs] += Ke
    return K
end

const assem = assemble # for CALFEM API compatability


