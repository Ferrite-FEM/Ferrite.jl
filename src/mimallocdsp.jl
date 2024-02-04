using mimalloc_jll: libmimalloc

struct MallocVector{T} <: AbstractVector{T}
    ptr::Ptr{T}
    allocated_length::Int # malloc'd bytes: allocated_length * sizeof(T)
    length::Int
end

Base.size(mv::MallocVector) = (mv.length, )
Base.IndexStyle(::Type{MallocVector}) = IndexLinear()
Base.@propagate_inbounds function Base.getindex(mv::MallocVector, i::Int)
    @boundscheck checkbounds(mv, i)
    return unsafe_load(mv.ptr, i)
end
Base.@propagate_inbounds function Base.setindex!(mv::MallocVector{T}, v::T, i::Int) where T
    @boundscheck checkbounds(mv, i)
    return unsafe_store!(mv.ptr, v, i)
end

function mi_malloc(size)
    return @ccall libmimalloc.mi_malloc(size::Csize_t)::Ptr{Cvoid}
    # return Libc.malloc(size)
end
function mi_realloc(p, newsize)
    return @ccall libmimalloc.mi_realloc(p::Ptr{Cvoid}, newsize::Csize_t)::Ptr{Cvoid}
    # return Libc.realloc(p, newsize)
end
function mi_free(p)
    return @ccall libmimalloc.mi_free(p::Ptr{Cvoid})::Cvoid
    # return Libc.free(p)
end

function mi_heap_new()
    return @ccall libmimalloc.mi_heap_new()::Ptr{Cvoid}
end
function mi_heap_destroy(heap)
    return @ccall libmimalloc.mi_heap_destroy(heap::Ptr{Cvoid})::Cvoid
end
function mi_heap_malloc(heap, size)
    # return mi_malloc(size)
    return @ccall libmimalloc.mi_heap_malloc(heap::Ptr{Cvoid}, size::Csize_t)::Ptr{Cvoid}
end
function mi_heap_realloc(heap, p, newsize)
    # return mi_realloc(p, newsize)
    return @ccall libmimalloc.mi_heap_realloc(heap::Ptr{Cvoid}, p::Ptr{Cvoid}, newsize::Csize_t)::Ptr{Cvoid}
end

mutable struct MiMallocDSP <: AbstractSparsityPattern
    const nrows::Int
    const ncols::Int
    const growth_factor::Float64
    const rows::Vector{MallocVector{Int}}
    const heap::Ptr{Cvoid}
    function MiMallocDSP(
            nrows::Int, ncols::Int;
            nnz_per_row::Int = 8,
            growth_factor::Number = 2,
        )
        heap = mi_heap_new()
        rows = Vector{MallocVector{Int}}(undef, nrows)
        for i in 1:nrows
            ptr = convert(Ptr{Int}, mi_heap_malloc(heap, nnz_per_row * sizeof(Int)))
            rows[i] = MallocVector(ptr, nnz_per_row, 0)
        end
        dsp = new(nrows, ncols, growth_factor, rows, heap)
        finalizer(d -> mi_heap_destroy(d.heap), dsp)
        return dsp
    end
end

function Base.show(io::IO, ::MIME"text/plain", dsp::MiMallocDSP)
    iob = IOBuffer()
    println(iob, "$(n_rows(dsp))×$(n_cols(dsp)) $(sprint(show, typeof(dsp))):")
    min_entries = typemax(Int)
    max_entries = typemin(Int)
    stored_entries = 0
    allocated_entries = 0
    for r in dsp.rows
        l = length(r)
        min_entries = min(min_entries, l)
        max_entries = max(max_entries, l)
        stored_entries += l
        allocated_entries += r.allocated_length
    end
    sparsity = round(
        (n_rows(dsp) * n_cols(dsp) - stored_entries) / (n_rows(dsp) * n_cols(dsp)) * 100 * 100
    ) / 100
    println(iob, " - Sparsity: $(sparsity)% ($(stored_entries) stored entries)")
    avg_entries = round(stored_entries / n_rows(dsp) * 10) / 10
    println(iob, " - Entries per row (min, max, avg): $(min_entries), $(max_entries), $(avg_entries)")
    bytes_used = Base.format_bytes(stored_entries * sizeof(Int))
    bytes_allocated = Base.format_bytes(allocated_entries * sizeof(Int))
    print(iob,   " - Memory: $(bytes_used) used, $(bytes_allocated) allocated")
    write(io, seekstart(iob))
    return
end

n_rows(dsp::MiMallocDSP) = dsp.nrows
n_cols(dsp::MiMallocDSP) = dsp.ncols

function add_entry!(dsp::MiMallocDSP, row::Int, col::Int)
    @boundscheck (1 <= row <= n_rows(dsp) && 1 <= col <= n_cols(dsp)) || throw(BoundsError())
    r = @inbounds dsp.rows[row]
    k = searchsortedfirst(r, col)
    len = length(r)
    if k > len || @inbounds(r[k]) != col
        # TODO: This assumes we only grow by single entry every time
        ptr = r.ptr
        allocated_length = r.allocated_length
        if len == allocated_length
            # c = ceil(Int, allocated_length * dsp.growth_factor)
            # o = allocated_length + 1
            allocated_length = max(
                ceil(Int, allocated_length * dsp.growth_factor),
                allocated_length + 1,
            ) # nextpow2(allocated_length + 1)
            # println("Increasing size from $len to either $c or $o, picking $allocated_length")
            ptr = convert(Ptr{Int}, mi_heap_realloc(dsp.heap, r.ptr, allocated_length * sizeof(Int)))
        end
        r = MallocVector(ptr, allocated_length, len + 1)
        # Shift elements after the insertion point to the back
        @inbounds for i in len:-1:k
            r[i+1] = r[i]
        end
        # Insert the new element
        @inbounds r[k] = col
        @inbounds dsp.rows[row] = r
    end
    return
end

eachrow(dsp::MiMallocDSP) = dsp.rows
function eachrow(dsp::MiMallocDSP, row::Int)
    return dsp.rows[row]
end
