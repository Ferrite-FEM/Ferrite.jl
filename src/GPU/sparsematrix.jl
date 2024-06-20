struct GPUSparseMatrixCSC{Tv,Ti, VEC_INT<:AbstractArray{Ti,1}, VEC_FLOAT<:AbstractArray{Tv,1}}
    m::Ti   
    n::Ti 
    colptr::VEC_INT  
    rowval::VEC_INT  
    nzval::VEC_FLOAT      
end

# function GPUSparseMatrixCSC{Tv}(m::Int32, n::Int32, colptr::AbstractVector{Int32},
#                         rowval:: AbstractVector{Int32}, nzval::AbstractVector{Tv}) where {Tv}
#     new(m, n, colptr, rowval, nzval)
# end

function GPUSparseMatrixCSC{Tv}(A::SparseMatrixCSC{Tv}) where {Tv}
    GPUSparseMatrixCSC(A.m, A.n, A.colptr |> cu, A.rowval |> cu, A.nzval |> cu)
end


function Base.getindex(A::GPUSparseMatrixCSC{Tv}, i::Int, j::Int) where Tv 
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

function Base.setindex!(A::GPUSparseMatrixCSC{Tv}, v::Tv, i::Int, j::Int) where Tv
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
        return -one(T)
    end
end
