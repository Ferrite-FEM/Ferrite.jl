# This is a lightweight implementation of a sparse matrix in the Compressed Sparse Column (CSC)
# to be portable tothe GPU and to be driver(backend) independent (i.e. to be used by KernelAbstractions.jl)



# function GPUSparseMatrixCSC{Tv}(m::Int32, n::Int32, colptr::AbstractVector{Int32},
#                         rowval:: AbstractVector{Int32}, nzval::AbstractVector{Tv}) where {Tv}
#     new(m, n, colptr, rowval, nzval)
# end

#abstract type AbstractGPUSparseMatrixCSC{Tv,Ti} end

struct GPUSparseMatrixCSC{Tv,Ti, VEC_INT<:AbstractArray{Ti,1}, VEC_FLOAT<:AbstractArray{Tv,1}} <: AbstractSparseArray{Tv,Ti,2}
    m::Ti
    n::Ti
    colPtr::VEC_INT
    rowVal::VEC_INT
    nzVal::VEC_FLOAT
end

function allocate_gpu_matrix(backend::Backend,A::SparseMatrixCSC{Tv,Ti}) where {Tv,Ti}
    # create a GPU version of the sparse matrix in the specified backend, with the same data
    c_ptr = KernelAbstractions.zeros(backend,Int32,length(A.colptr))
    r_ptr = KernelAbstractions.zeros(backend,Int32,length(A.rowval))
    n_ptr = KernelAbstractions.zeros(backend,Tv,length(A.nzval))
    KernelAbstractions.copyto!(backend,c_ptr,A.colptr)
    KernelAbstractions.copyto!(backend,r_ptr,A.rowval)
    KernelAbstractions.copyto!(backend,n_ptr,A.nzval)
    m = convert(Ti,A.m)
    n = convert(Ti,A.n)
    GPUSparseMatrixCSC(m,n, c_ptr, r_ptr, n_ptr)
end


@inbounds function Base.getindex(A::GPUSparseMatrixCSC{Tv}, i::Int, j::Int) where Tv

    col_start = A.colPtr[j]
    col_end = A.colPtr[j + 1] - 1

    for k in col_start:col_end
        if A.rowVal[k] == i
            return A.nzVal[k]
        end
    end

    return zero(Tv)
end

@inbounds function Base.setindex!(A::GPUSparseMatrixCSC{Tv}, v::Tv, i::Int, j::Int) where Tv
    col_start = A.colPtr[j]
    col_end = A.colPtr[j + 1] - 1

    for k in col_start:col_end
        if A.rowVal[k] == i
            # Update the existing element
                A.nzVal[k] = v
            return
        end
    end
end


# function custom_atomic_add!(A::GPUSparseMatrixCSC{T}, v::Float32, i::Int32, j::Int32) where T
#     col_start = A.colptr[j]
#     col_end = A.colptr[j + 1] - 1

#     for k in col_start:col_end
#         if A.rowval[k] == i
#             # Update the existing element
#             CUDA.@atomic A.nzval[k] += v
#             return
#         end
#     end

# end

function gpu_sparse_norm(A::GPUSparseMatrixCSC{T}, p::Real=2) where T
    if p == 2  # Frobenius norm
        return sqrt(sum(abs2, A.nzVal))
    elseif p == 1  # L1 norm
        col_sums = zeros(T, A.n)
        for j in 1:A.n
            for k in A.colPtr[j]:(A.colPtr[j + 1] - 1)
                col_sums[j] += abs(A.nzVal[k])
            end
        end
        return maximum(col_sums)
    elseif p == Inf  # L∞ norm
        row_sums = zeros(T, A.m)
        for j in 1:A.n
            for k in A.colPtr[j]:(A.colPtr[j + 1] - 1)
                i = A.rowVal[k]
                row_sums[i] += abs(A.nzVal[k])
            end
        end
        return maximum(row_sums)
    else
        return -one(T)
    end
end
