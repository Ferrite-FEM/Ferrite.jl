struct GPUAssemblerSparsityPattern{Tv,Ti,VEC_FLOAT<:AbstractVector{Tv},SPARSE_MAT<:AbstractSparseArray{Tv,Ti}} <: AbstractSparseAssembler
    K::SPARSE_MAT
    f::VEC_FLOAT
end


function start_assemble(K::AbstractSparseArray{Tv}, f::AbstractVector{Tv}) where {Tv}
    return GPUAssemblerSparsityPattern(K, f)
end

@propagate_inbounds function assemble!(A::GPUAssemblerSparsityPattern, dofs::AbstractVector{Int32}, Ke::AbstractMatrix, fe::AbstractVector)
    _assemble!(A, dofs, Ke, fe)
end

function _assemble!(A::GPUAssemblerSparsityPattern, dofs::AbstractVector{Int32}, Ke::AbstractMatrix, fe::AbstractVector)
    # Brute force assembly
    K = A.K
    f = A.f
    for i = 1:length(dofs)
        ig = dofs[i]
        f[ig] += fe[i]
        for j = 1:length(dofs)
            jg = dofs[j]
            # set the value of the global matrix
           _add_to_index!(K, Ke[i,j], ig, jg)
        end
    end
end

@inline function _add_to_index!(K::AbstractSparseArray{Tv,Ti}, v::Tv, i::Int32, j::Int32) where {Tv,Ti}
    col_start = K.colPtr[j]
    col_end = K.colPtr[j + Int32(1)] - Int32(1)

    for k in col_start:col_end
        if K.rowVal[k] == i
            # Update the existing element
                K.nzVal[k] += v
            return
        end
    end
end
