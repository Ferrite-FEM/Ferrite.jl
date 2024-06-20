struct GPUAssemblerSparsityPattern{Tv,Ti} <: AbstractSparseAssembler
    K::GPUSparseMatrixCSC{Tv,Ti}
    f::AbstractVector{Tv}
end


function start_assemble(K::GPUSparseMatrixCSC{Tv}, f::AbstractVector{Tv}) where {Tv}
    return GPUAssemblerSparsityPattern(K, f)
end

@propagate_inbounds function assemble!(A::GPUAssemblerSparsityPattern, dofs::AbstractVector{Int}, Ke::AbstractMatrix, fe::AbstractVector)
    _assemble!(A, dofs, Ke, fe)
end

function _assemble!(A::GPUAssemblerSparsityPattern, dofs::AbstractVector{Int}, Ke::AbstractMatrix, fe::AbstractVector)
    # Brute force assembly
    K = A.K
    f = A.f
    for i in eachindex(dofs)
        ig = cell_dofs[i]
        f[ig] += fe[i]
        for j in eachindex(dofs)
            jg = cell_dofs[j]
            K[ig, jg] += ke[i,j]
        end
    end
end