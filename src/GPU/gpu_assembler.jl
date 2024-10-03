## GPU Assembler ###
### First abstract types and interfaces ###

abstract type GPUAbstractSparseAssembler{Tv,Ti} end

function assemble!(A::GPUAbstractSparseAssembler, dofs::AbstractVector{Int32}, Ke::MATRIX, fe::VECTOR) where {MATRIX, VECTOR}
    throw(NotImplementedError("A concrete implementation of assemble! is required"))
end


struct GPUAssemblerSparsityPattern{Tv,Ti,VEC_FLOAT<:AbstractVector{Tv},SPARSE_MAT<:AbstractSparseArray{Tv,Ti}} <: GPUAbstractSparseAssembler{Tv,Ti}
    K::SPARSE_MAT
    f::VEC_FLOAT
end

function start_assemble(K::AbstractSparseArray{Tv,Ti}, f::AbstractVector{Tv}) where {Tv,Ti}
    return GPUAssemblerSparsityPattern(K, f)
end


"""
    assemble!(A::GPUAssemblerSparsityPattern, dofs::AbstractVector{Int32}, Ke::MATRIX, fe::VECTOR)

Assembles the global stiffness matrix `Ke` and the global force vector `fe` into the the global stiffness matrix `K` and the global force vector `f` of the `GPUAssemblerSparsityPattern` object `A`.

"""
@propagate_inbounds function assemble!(A::GPUAssemblerSparsityPattern, dofs::AbstractVector{Int32}, Ke::MATRIX, fe::VECTOR) where {MATRIX, VECTOR}
    # Note: MATRIX and VECTOR are cuda dynamic shared memory
    _assemble!(A, dofs, Ke, fe)
end


function _assemble!(A::GPUAssemblerSparsityPattern, dofs::AbstractVector{Int32}, Ke::MATRIX, fe::VECTOR) where {MATRIX, VECTOR}
    # # Brute force assembly
    K = A.K
    f = A.f
    for i = 1:length(dofs)
        ig = dofs[i]
        CUDA.@atomic f[ig] += fe[i]
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
              CUDA.@atomic K.nzVal[k] += v
            return
        end
    end
end
