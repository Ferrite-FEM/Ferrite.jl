struct CollectionOfVectors{IT<:Union{AbstractArray{UnitRange{Int}}, AbstractDict{<:Any, UnitRange{Int}}}, DT}
    indices::IT
    data::Vector{DT}
end

Base.getindex(cv::CollectionOfVectors, index) = view(cv.data, cv.indices[index])
Base.keys(cv::CollectionOfVectors) = keys(cv.indcies)
Base.values(cv::CollectionOfVectors) = (view(cv.data, idx) for idx in values(cv.indices))
nonempty_values(cv::CollectionOfVectors) = (view(cv.data, idx) for idx in values(cv.indices) if !isempty(idx))

# Structure for building this efficiently
struct AdaptiveRange
    start::Int
    ncurrent::Int   # These two could be UInt8 or similar in
    nmax::Int       # many applications, but probably not worth.
end

struct ConstructionBuffer{IT, DT}
    indices::IT # eltype = AdaptiveRange
    data::Vector{DT}
    sizehint::Int
end

function ConstructionBuffer(IT::Type{<:Array}, data::Vector; dims = nothing, sizehint)
    dims === nothing && error("dims must be given when indexed by an $IT")
    ndims(IT) == length(dims) || error("The number of dims must match IT's number of dimensions")

    indices = fill(AdaptiveRange(0, 0, 0), dims)
    return ConstructionBuffer(indices, data, sizehint)
end
function ConstructionBuffer(::Type{<:OrderedDict{K}}, data::Vector; sizehint::Int) where K
    return ConstructionBuffer(OrderedDict{K, AdaptiveRange}(), data, sizehint)
end

function Base.setindex!(b::ConstructionBuffer{<:Array}, val, indices...)
    r = getindex(b.indices, indices...)
    n = length(b.data)
    if r.start == 0 # Not previously added
        resize!(b.data, n + b.sizehint)
        b.data[n+1] = val
        setindex!(b.indices, AdaptiveRange(n + 1, 1, b.sizehint), indices...)
    elseif r.ncurrent == r.nmax # We have used up our space, move data to the end of the vector.
        resize!(b.data, n + r.nmax + sizehint)
        for i in 1:r.ncurrent
            b.data[n + i] = b.data[r.start + i - 1]
        end
        b.data[n + r.ncurrent + 1] = val
        setindex!(b.indices, AdaptiveRange(n + 1, r.ncurrent + 1, r.nmax + sizehint), indices...)
    else # We have space in an already allocated section
        b.data[r.start + r.ncurrent] = val
        setindex!(b.indices, AdaptiveRange(r.start, r.ncurrent + 1, r.nmax), indices...)
    end
    return b
end

function Base.setindex!(b::ConstructionBuffer{<:AbstractDict}, val, key)
    n = length(b.data)
    added_range = AdaptiveRange(n + 1, 1, b.sizehint)
    r = get!(b.indices, key) do
        # Enters only if key is not in b.indices
        resize!(b.data, n + b.sizehint)
        b.data[n + 1] = val
        added_range
    end
    r === added_range && return b # We added `added_range` and can exit

    # Otherwise, `added_range` already exists in `b.indices` and we
    # need to add more elements to this index.

    if r.ncurrent == r.nmax # Need to move to the end of the vector
        b.indices[key] = AdaptiveRange(n + 1, r.ncurrent + 1, r.nmax + b.sizehint)
        resize!(b.data, n + r.nmax + sizehint)
        for i in 1:r.ncurrent
            b.data[n + i] = b.data[r.start + i - 1]
        end
        b.data[n + r.ncurrent + 1] = val
    else
        b.indices[key] = AdaptiveRange(r.start, r.ncurrent + 1, r.nmax)
        b.data[r.start + r.ncurrent] = val
    end
    return b
end

function compress_data!(data, indices, sorted_iterator, buffer_indices)
    n = 0
    for (index, r) in sorted_iterator
        nstop = r.start + r.ncurrent - 1
        for (iold, inew) in zip(nstop:-1:r.start, n .+ (r.ncurrent:-1:1))
            @assert inew ≤ iold # To not overwrite
            data[inew] = data[iold]
        end
        indices[index] = (n + 1):(n + r.ncurrent)
        n += r.ncurrent
    end
    resize!(data, n)
    sizehint!(data, n)      # Free memory
    empty!(buffer_indices)
    sizehint!(buffer_indices, 0) # Free memory
    return data, indices
end

function CollectionOfVectors(f!::F, IT::Type, DT::Type; sizehint = nothing, kwargs...) where {F <: Function}
    sizehint === nothing && error("Providing sizehint is mandatory")
    b = ConstructionBuffer(IT, DT[]; sizehint, kwargs...)
    f!(b)
    return CollectionOfVectors(b)
end

function CollectionOfVectors(b::ConstructionBuffer{<:Array})
    I = sortperm(reshape(b.indices, :); by = x -> x.start)
    sorted_iterator = enumerate(b.indices[idx] for idx in I if b.indices[idx].start != 0)
    indices = fill(1:0, size(b.indices))
    compress_data!(b.data, indices, sorted_iterator, b.indices)
    return CollectionOfVectors(indices, b.data)
end


# Efficient creation of a new OrderedDict with new types of values
function _withnewvalues(d::OrderedDict{K}, vals::Vector{V}) where {K, V}
    if d.ndel > 0
        OrderedCollections.rehash!(d)
    end
    @assert d.ndel == 0
    length(d.vals) == length(vals) || error("Length of new vals must match old")
    OrderedDict{K,V}(copy(d.slots), copy(d.keys), vals, 0, d.maxprobe, false)
end

function CollectionOfVectors(b::ConstructionBuffer{<:OrderedDict})
    sort!(b.indices; byvalue=true, by = r -> r.start)
    indices = _withnewvalues(b.indices, fill(1:0, length(b.indices)))
    compress_data!(b.data, indices, b.indices, b.indices)
    return CollectionOfVectors(indices, b.data)
end
