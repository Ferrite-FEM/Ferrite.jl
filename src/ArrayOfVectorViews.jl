struct ArrayOfVectorViews{T, N} <: AbstractArray{SubArray{T, 1, Vector{T}, Tuple{UnitRange{Int64}}, true}, N}
    indices::Array{UnitRange{Int}, N}
    data::Vector{T}
end
# AbstractArray interface (https://docs.julialang.org/en/v1/manual/interfaces/#man-interface-array)
Base.size(cv::ArrayOfVectorViews) = size(cv.indices)
function Base.getindex(cv::ArrayOfVectorViews, idx...)
    return view(cv.data, getindex(cv.indices, idx...))
end
Base.IndexStyle(::Type{ArrayOfVectorViews{<:Any, N}}) where N = Base.IndexStyle(Array{Int, N})


# Structure for building this efficiently
struct AdaptiveRange
    start::Int
    ncurrent::Int   # These two could be UInt8 or similar in
    nmax::Int       # many applications, but probably not worth.
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
    if r.start == 0 # Not previously added
        resize!(b.data, n + b.sizehint)
        b.data[n+1] = val
        setindex!(b.indices, AdaptiveRange(n + 1, 1, b.sizehint), indices...)
    elseif r.ncurrent == r.nmax # We have used up our space, move data to the end of the vector.
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

function compress_data!(data, indices, sorted_iterator, buffer_indices)
    n = 0
    for (index, r) in sorted_iterator
        nstop = r.start + r.ncurrent - 1
        for (iold, inew) in zip(r.start:nstop, (n + 1):(n + r.ncurrent))
            @assert inew ≤ iold # To not overwrite
            data[inew] = data[iold]
        end
        indices[index] = (n + 1):(n + r.ncurrent)
        n += r.ncurrent
    end
    resize!(data, n)
    sizehint!(data, n)      # Free memory
    if buffer_indices isa Vector # Higher-dim Array's don't support empty!/resize!
        empty!(buffer_indices)
        sizehint!(buffer_indices, 0) # Free memory
    end
    return data, indices
end

function ArrayOfVectorViews(f!::F, data::Vector, dims::Tuple; sizehint = nothing) where {F <: Function}
    sizehint === nothing && error("Providing sizehint is mandatory")
    b = ConstructionBuffer(data, dims, sizehint)
    f!(b)
    return ArrayOfVectorViews(b)
end

function ArrayOfVectorViews(b::ConstructionBuffer)
    I = sortperm(reshape(b.indices, :); by = x -> x.start)
    sorted_iterator = ((idx, b.indices[idx]) for idx in I if b.indices[idx].start != 0)
    indices = fill(1:0, size(b.indices))
    compress_data!(b.data, indices, sorted_iterator, b.indices)
    return ArrayOfVectorViews(indices, b.data)
end
