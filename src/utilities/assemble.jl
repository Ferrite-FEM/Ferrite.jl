"""
Assembles the the element stiffness matrix Ke to the global stiffness matrix K.
"""
function assemble(edof::Array, K::AbstractMatrix, Ke::Matrix)
    _Ke = convert(Matrix{Float64}, Ke)
    _edof = convert(Array{Int}, edof)
    dofs = vec(_edof[2:end])
    K[dofs, dofs] += Ke
    return K
end

const assem = assemble # for CALFEM API compatability


