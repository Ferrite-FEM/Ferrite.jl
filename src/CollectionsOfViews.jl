module CollectionsOfViews

export ArrayOfVectorViews, push_at_index!, ConstructionBuffer

# `AdaptiveRange` and `ConstructionBuffer` are used to efficiently build up an `ArrayOfVectorViews`
# when the size of each view is unknown.
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

"""
    ConstructionBuffer(data::Vector, dims::NTuple{N, Int}, sizehint)

Create a buffer for creating an [`ArrayOfVectorViews`](@ref), representing an array with `N` axes.
`sizehint` sets the number of elements in `data` allocated when a new index is added via `push_at_index!`,
or when the current storage for the index is full, how much many additional elements are reserved for that index.
Any content in `data` is overwritten, but performance is improved by pre-allocating it to a reasonable size or
by `sizehint!`ing it.
"""
function ConstructionBuffer(data::Vector, dims::NTuple{<:Any, Int}, sizehint::Int)
    indices = fill(AdaptiveRange(0, 0, 0), dims)
    return ConstructionBuffer(indices, empty!(data), sizehint)
end

"""
    push_at_index!(b::ConstructionBuffer, val, indices::Int...)

`push!` the value `val` to the `Vector` view at the index given by `indices`, typically called
inside the [`ArrayOfVectorViews`](@ref) constructor do-block. But can also be used when manually
creating a `ConstructionBuffer`.
"""
function push_at_index!(b::ConstructionBuffer, val, indices...)
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

struct ArrayOfVectorViews{T, N} <: AbstractArray{SubArray{T, 1, Vector{T}, Tuple{UnitRange{Int64}}, true}, N}
    indices::Vector{Int}
    data::Vector{T}
    lin_idx::LinearIndices{N, NTuple{N, Base.OneTo{Int}}}
    function ArrayOfVectorViews{T, N}(indices::Vector{Int}, data::Vector{T}, lin_idx::LinearIndices{N}) where {T, N}
        return new{T, N}(indices, data, lin_idx)
    end
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
Base.IndexStyle(::Type{<:ArrayOfVectorViews{<:Any, N}}) where N = Base.IndexStyle(Array{Int, N})

# Constructors
"""
    ArrayOfVectorViews(f!::Function, data::Vector{T}, dims::NTuple{N, Int}; sizehint)

Create an `ArrayOfVectorViews` to store many vector views of potentially different sizes,
emulating an `Array{Vector{T}, N}` with size `dims`. However, it avoids allocating each vector individually
by storing all data in `data`, and instead of `Vector{T}`, the each element is a `typeof(view(data, 1:2))`.

When the length of each vector is unknown, the `ArrayOfVectorViews` can be created reasonably efficient
with the following do-block, which creates an intermediate `buffer::ConstructionBuffer` supporting the
[`push_at_index!`](@ref) function.
```
vector_views = ArrayOfVectorViews(data, dims; sizehint) do buffer
    for (ind, val) in some_data
        push_at_index!(buffer, val, ind)
    end
end
```
`sizehint` tells how much space to allocate for the index `ind` if no `val` has been added to that index before,
or how much more space to allocate in case all previously allocated space for `ind` has been used up.
"""
function ArrayOfVectorViews(f!::F, data::Vector, dims::Tuple; sizehint = nothing) where {F <: Function}
    sizehint === nothing && error("Providing sizehint is mandatory")
    b = ConstructionBuffer(data, dims, sizehint)
    f!(b)
    return ArrayOfVectorViews(b)
end

"""
    ArrayOfVectorViews(b::CollectionsOfViews.ConstructionBuffer)

Creates the `ArrayOfVectorViews` directly from the `ConstructionBuffer` that was manually created and filled.
"""
function ArrayOfVectorViews(b::ConstructionBuffer{T}) where T
    indices = Vector{Int}(undef, length(b.indices) + 1)
    lin_idx = LinearIndices(b.indices)
    data_length = sum(ar.ncurrent for ar in b.indices; init=0)
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

"""
    ArrayOfVectorViews(indices::Vector{Int}, data::Vector{T}, lin_idx::LinearIndices{N}; checkargs = true)

Creates the `ArrayOfVectorViews` directly where the user is responsible for having the correct input data.
Checking of the argument dimensions can be elided by setting `checkargs = false`, but incorrect dimensions
may lead to illegal out of bounds access later.

`data` is indexed by `indices[i]:indices[i+1]`, where `i = lin_idx[idx...]` and `idx...` are the user-provided
indices to the `ArrayOfVectorViews`.
"""
function ArrayOfVectorViews(indices::Vector{Int}, data::Vector{T}, lin_idx::LinearIndices{N}; checkargs = true) where {T, N}
    if checkargs
        checkbounds(data, 1:(last(indices) - 1))
        checkbounds(indices, last(lin_idx) + 1)
        issorted(indices) || throw(ArgumentError("indices must be weakly increasing"))
    end
    return ArrayOfVectorViews{T, N}(indices, data, lin_idx)
end

end
