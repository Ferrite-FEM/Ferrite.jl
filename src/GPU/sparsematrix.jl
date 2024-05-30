using SparseArrays
using CUDA
struct GPUSparseMatrixCSC{Tv,VEC <: AbstractVector{Int32} , NZVEC <: AbstractVector{Tv}} 
    m::Int32                  # Number of rows
    n::Int32                  # Number of columns
    colptr::VEC     # Column i is in colptr[i]:(colptr[i+1]-1)
    rowval::VEC     # Row indices of stored values
    nzval::NZVEC       # Stored values, typically nonzeros
end

function GPUSparseMatrixCSC{Tv}(m::Int32, n::Int32, colptr::AbstractVector{Int32},
                        rowval:: AbstractVector{Int32}, nzval::AbstractVector{Tv}) where {Tv}
    new(m, n, colptr, rowval, nzval)
end

function GPUSparseMatrixCSC{Tv}(A::SparseMatrixCSC{Tv}) where {Tv}
    GPUSparseMatrixCSC(A.m, A.n, A.colptr, A.rowval, A.nzval)
end


function Base.getindex(A::GPUSparseMatrixCSC{Tv}, i::Int32, j::Int32) where Tv 
    # TODO: Add bounds checking

    col_start = A.colptr[j] 
    col_end = A.colptr[j + 1] - 1

    for k in col_start:col_end
        if A.rowval[k] == i
            return A.nzval[k]
        end
    end

    return zero(Tv)
end

function Base.setindex!(A::GPUSparseMatrixCSC{T}, v::Float32, i::Int32, j::Int32) where T
    col_start = A.colptr[j]
    col_end = A.colptr[j + 1] - 1

    for k in col_start:col_end
        if A.rowval[k] == i
            # Update the existing element
                A.nzval[k] = v
            return
        end
    end
end


function custom_atomic_add!(A::GPUSparseMatrixCSC{T}, v::Float32, i::Int32, j::Int32) where T
    col_start = A.colptr[j]
    col_end = A.colptr[j + 1] - 1

    for k in col_start:col_end
        if A.rowval[k] == i
            # Update the existing element
            CUDA.@atomic A.nzval[k] += v
            return
        end
    end

end

function gpu_sparse_norm(A::GPUSparseMatrixCSC{T}, p::Real=2) where T
    if p == 2  # Frobenius norm
        return sqrt(sum(abs2, A.nzval))
    elseif p == 1  # L1 norm
        col_sums = zeros(T, A.n)
        for j in 1:A.n
            for k in A.colptr[j]:(A.colptr[j + 1] - 1)
                col_sums[j] += abs(A.nzval[k])
            end
        end
        return maximum(col_sums)
    elseif p == Inf  # L∞ norm
        row_sums = zeros(T, A.m)
        for j in 1:A.n
            for k in A.colptr[j]:(A.colptr[j + 1] - 1)
                i = A.rowval[k]
                row_sums[i] += abs(A.nzval[k])
            end
        end
        return maximum(row_sums)
    else
        return -1.0f0
    end
end

