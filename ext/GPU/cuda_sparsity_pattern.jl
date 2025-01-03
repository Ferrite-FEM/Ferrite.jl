function Ferrite.allocate_matrix(::Type{S}, dh::DofHandler, args...; kwargs...) where {Tv, Ti <: Integer, S <: CUSPARSE.CuSparseMatrixCSC{Tv, Ti}}
    ## TODO: decide whether create the matrix from the very beginning or just create cpu version then copy
    K = allocate_matrix(SparseMatrixCSC{Tv, Ti}, dh, args...; kwargs...)
    Kgpu = CUSPARSE.CuSparseMatrixCSC(K)
    return Kgpu
end
