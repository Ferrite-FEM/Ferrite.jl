## GPU Assembler ###
### First abstract types and interfaces ###

abstract type AbstractGPUSparseAssembler{Tv, Ti} end

function Ferrite.assemble!(A::AbstractGPUSparseAssembler, dofs::AbstractVector{Int32}, Ke::MATRIX, fe::VECTOR) where {MATRIX, VECTOR}
    throw(NotImplementedError("A concrete implementation of assemble! is required"))
end


struct GPUAssemblerSparsityPattern{Tv, Ti, VEC_FLOAT <: AbstractVector{Tv}, SPARSE_MAT <: AbstractSparseArray{Tv, Ti}} <: AbstractGPUSparseAssembler{Tv, Ti}
    K::SPARSE_MAT
    f::VEC_FLOAT
end

function Ferrite.start_assemble(K::CUSPARSE.CuSparseDeviceMatrixCSC{Tv, Ti}, f::CuDeviceVector{Tv}; fillzero = false) where {Tv, Ti}
    ##fillzero && (fillzero!(K); fillzero!(f))
    return GPUAssemblerSparsityPattern(K, f)
end


"""
    assemble!(A::GPUAssemblerSparsityPattern, dofs::AbstractVector{Int32}, Ke::MATRIX, fe::VECTOR)

Assembles the global stiffness matrix `Ke` and the global force vector `fe` into the the global stiffness matrix `K` and the global force vector `f` of the `GPUAssemblerSparsityPattern` object `A`.

"""
@propagate_inbounds function Ferrite.assemble!(A::GPUAssemblerSparsityPattern, dofs::AbstractVector{Int32}, Ke::MATRIX, fe::VECTOR) where {MATRIX, VECTOR}
    # Note: MATRIX and VECTOR are cuda dynamic shared memory
    return _assemble!(A, dofs, Ke, fe)
end


function _assemble!(A::GPUAssemblerSparsityPattern, dofs::AbstractVector{Int32}, Ke::MATRIX, fe::VECTOR) where {MATRIX, VECTOR}
    # # Brute force assembly
    K = A.K
    f = A.f
    for i in 1:length(dofs)
        ig = dofs[i]
        CUDA.@atomic f[ig] += fe[i]
        for j in 1:length(dofs)
            jg = dofs[j]
            # set the value of the global matrix
            _add_to_index!(K, Ke[i, j], ig, jg)
        end
    end
    return
end

@inline function _add_to_index!(K::AbstractSparseArray{Tv, Ti}, v::Tv, i::Int32, j::Int32) where {Tv, Ti}
    col_start = K.colPtr[j]
    col_end = K.colPtr[j + Int32(1)] - Int32(1)

    for k in col_start:col_end
        if K.rowVal[k] == i
            # Update the existing element
            CUDA.@atomic K.nzVal[k] += v
            return
        end
    end
    return
end
