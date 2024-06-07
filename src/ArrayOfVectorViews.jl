struct ArrayOfVectorViews{T, N} <: AbstractArray{SubArray{T, 1, Vector{T}, Tuple{UnitRange{Int64}}, true}, N}
    indices::Vector{Int}
    data::Vector{T}
    lin_idx::LinearIndices{N, NTuple{N, Base.OneTo{Int}}}
end
# AbstractArray interface (https://docs.julialang.org/en/v1/manual/interfaces/#man-interface-array)
Base.size(cv::ArrayOfVectorViews) = size(cv.lin_idx)
@inline function Base.getindex(cv::ArrayOfVectorViews, linear_index::Int)
    @boundscheck checkbounds(cv.lin_idx, linear_index)
    return @inbounds view(cv.data, cv.indices[linear_index]:(cv.indices[linear_index+1]-1))
end
@inline function Base.getindex(cv::ArrayOfVectorViews, idx...)
    linear_index = getindex(cv.lin_idx, idx...)
    return @inbounds getindex(cv, linear_index)
end
Base.IndexStyle(::Type{ArrayOfVectorViews{<:Any, N}}) where N = Base.IndexStyle(Array{Int, N})

# Structure for building this efficiently
struct AdaptiveRange
    start::Int
    ncurrent::Int
    nmax::Int
end

struct ConstructionBuffer{T, N}
    indices::Array{AdaptiveRange, N}
    data::Vector{T}
    sizehint::Int
end

function ConstructionBuffer(data::Vector, dims::NTuple{<:Any, Int}, sizehint::Int)
    indices = fill(AdaptiveRange(0, 0, 0), dims)
    return ConstructionBuffer(indices, data, sizehint)
end

function add!(b::ConstructionBuffer, val, indices...)
    r = getindex(b.indices, indices...)
    n = length(b.data)
    if r.start == 0
        # `indices...` not previously added, allocate new space for it at the end of `b.data`
        resize!(b.data, n + b.sizehint)
        b.data[n+1] = val
        setindex!(b.indices, AdaptiveRange(n + 1, 1, b.sizehint), indices...)
    elseif r.ncurrent == r.nmax
        # We have used up our space, move data associated with `indices...` to the end of `b.data`
        resize!(b.data, n + r.nmax + b.sizehint)
        for i in 1:r.ncurrent
            b.data[n + i] = b.data[r.start + i - 1]
        end
        b.data[n + r.ncurrent + 1] = val
        setindex!(b.indices, AdaptiveRange(n + 1, r.ncurrent + 1, r.nmax + b.sizehint), indices...)
    else # We have space in an already allocated section
        b.data[r.start + r.ncurrent] = val
        setindex!(b.indices, AdaptiveRange(r.start, r.ncurrent + 1, r.nmax), indices...)
    end
    return b
end

function ArrayOfVectorViews(f!::F, data::Vector, dims::Tuple; sizehint = nothing) where {F <: Function}
    sizehint === nothing && error("Providing sizehint is mandatory")
    b = ConstructionBuffer(data, dims, sizehint)
    f!(b)
    return ArrayOfVectorViews(b)
end

function ArrayOfVectorViews(b::ConstructionBuffer{T}) where T
    indices = Vector{Int}(undef, length(b.indices) + 1)
    lin_idx = LinearIndices(b.indices)
    data_length = sum(ar.ncurrent for ar in b.indices)
    data = Vector{T}(undef, data_length)
    data_index = 1
    for (idx, ar) in pairs(b.indices)
        copyto!(data, data_index, b.data, ar.start, ar.ncurrent)
        indices[lin_idx[idx]] = data_index
        data_index += ar.ncurrent
    end
    indices[length(indices)] = data_index
    resize!(b.data, 0); sizehint!(b.data, 0)
    isa(b.indices, Vector) && (resize!(b.indices, 0); sizehint!(b.indices, 0))
    return ArrayOfVectorViews(indices, data, lin_idx)
end
