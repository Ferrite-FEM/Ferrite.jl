struct GPUAssemblerSparsityPattern{Tv,Ti,VEC_FLOAT<:AbstractVector{Tv},SPARSE_MAT<:GPUSparseMatrixCSC{Tv,Ti}} <: AbstractSparseAssembler
    K::SPARSE_MAT
    f::VEC_FLOAT
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
    for i= 1:length(dofs)
        ig = dofs[i]
        f[ig] += fe[i]
        for j = 1:length(dofs)
            jg = dofs[j]
            K[ig, jg] += Ke[i,j]
        end
    end
end