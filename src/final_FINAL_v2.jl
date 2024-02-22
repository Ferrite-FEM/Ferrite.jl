module Final

using ..Ferrite: Ferrite
using ..Ferrite: add_entry!, n_rows, n_cols, AbstractSparsityPattern
import ..Ferrite: add_entry!, n_rows, n_cols, eachrow

include("HeapAllocator.jl")
using .HeapAllocator: HeapAllocator

struct SparsityPattern{T} <: AbstractSparsityPattern
    nrows::Int
    ncols::Int
    heap::HeapAllocator.Heap
    rows::Vector{HeapAllocator.HeapVector{T}}
    function SparsityPattern{T}(nrows::Int, ncols::Int; nnz_per_row::Int = 8) where T <: Integer
        heap = HeapAllocator.Heap()
        rows = Vector{HeapAllocator.HeapVector{T}}(undef, nrows)
        for i in 1:nrows
            rows[i] = HeapAllocator.resize(HeapAllocator.alloc_array(heap, Int, nnz_per_row), 0)
        end
        return new{T}(nrows, ncols, heap, rows)
    end
end

function SparsityPattern(nrows::Int, ncols::Int; kwargs...)
    return SparsityPattern{Int}(nrows, ncols; kwargs...)
end

function Base.show(io::IO, ::MIME"text/plain", sp::SparsityPattern{T}) where T
    iob = IOBuffer()
    println(iob, "$(n_rows(sp))×$(n_cols(sp)) $(sprint(show, typeof(sp))):")
    min_entries = typemax(Int)
    max_entries = typemin(Int)
    stored_entries = 0
    allocated_entries = 0
    for r in sp.rows
        l = length(r)
        stored_entries += l
        min_entries = min(min_entries, l)
        max_entries = max(max_entries, l)
        allocated_entries += HeapAllocator.allocated_length(r)
    end
    ##
    bytes_estimate = 0
    bytes_estimate_used = 0
    bytes_estimate      += n_rows(sp) * sizeof(eltype(sp.rows))
    bytes_estimate_used += n_rows(sp) * sizeof(eltype(sp.rows))
    bytes_estimate_used += stored_entries * sizeof(T)
    ##
    bytes_malloced = 0
    for heapidx in 1:length(sp.heap.size_heaps)
        isassigned(sp.heap.size_heaps, heapidx) || continue
        bytes_malloced += length(sp.heap.size_heaps[heapidx].pages) * Int(HeapAllocator.MALLOC_PAGE_SIZE)
        bytes_estimate += length(sp.heap.size_heaps[heapidx].pages) * Int(HeapAllocator.MALLOC_PAGE_SIZE)
    end
    sparsity = round(
        (n_rows(sp) * n_cols(sp) - stored_entries) / (n_rows(sp) * n_cols(sp)) * 100 * 100
    ) / 100
    println(iob, " - Sparsity: $(sparsity)% ($(stored_entries) stored entries)")
    avg_entries = round(stored_entries / n_rows(sp) * 10) / 10
    println(iob, " - Entries per row (min, max, avg): $(min_entries), $(max_entries), $(avg_entries)")
    bytes_used = Base.format_bytes(stored_entries * sizeof(T))
    bytes_allocated = Base.format_bytes(allocated_entries * sizeof(T))
    bytes_mallocated = Base.format_bytes(bytes_malloced)
    bytes_est = Base.format_bytes(bytes_estimate)
    bytes_est_used = Base.format_bytes(bytes_estimate_used)
    println(iob,   " - Memory: $(bytes_used) used, $(bytes_allocated) allocated, $(bytes_mallocated) malloc'd")
    print(iob,   " - Memory estimate: $(bytes_est_used) used, $(bytes_est) allocated")
    write(io, seekstart(iob))
    return
end

n_rows(sp::SparsityPattern) = sp.nrows
n_cols(sp::SparsityPattern) = sp.ncols

@inline function add_entry!(sp::SparsityPattern, row::Int, col::Int)
    @boundscheck 1 <= row <= n_rows(sp) && 1 <= col <= n_cols(sp)
    r = @inbounds sp.rows[row]
    r = insert_sorted(r, col)
    @inbounds sp.rows[row] = r
    return
end

@inline function insert_sorted(x::HeapAllocator.HeapVector{Int}, item::Int)
    k = searchsortedfirst(x, item)
    if k == length(x) + 1 || @inbounds(x[k]) != item
        x = HeapAllocator.insert(x, k, item)
    end
    return x
end

eachrow(sp::SparsityPattern) = sp.rows
function eachrow(sp::SparsityPattern, row::Int)
    return sp.rows[row]
end

end # module Final
