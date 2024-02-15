module Final

using ..Ferrite: Ferrite
using ..Ferrite: add_entry!, n_rows, n_cols, AbstractSparsityPattern
import ..Ferrite: add_entry!, n_rows, n_cols, eachrow

const MALLOC_PAGE_SIZE = 4 * 1024 * 1024 % UInt # 4 MiB

# Like Ptr{T} but also stores the number of bytes allocated
struct SizedPtr{T}
    ptr::Ptr{T}
    size::UInt
end
Ptr{T}(ptr::SizedPtr) where {T} = Ptr{T}(ptr.ptr)


# Fixed buffer size (1MiB)
# Rowsize  | size | # Rows |
#       4  | 1MiB |  32768 |
#       8  | 1MiB |  16384 |
#      16  | 1MiB |   8192 |
#      32  | 1MiB |   4096 |
#      64  | 1MiB |   2048 |
#     128  | 1MiB |   1024 |
#     256  | 1MiB |    512 |

# A page corresponds to a larger Libc.malloc call (MALLOC_PAGE_SIZE). Each page
# is split into smaller blocks to minimize the number of Libc.malloc/Libc.free
# calls.
mutable struct MemoryPage{T}
    const ptr::SizedPtr{T} # malloc'd pointer
    const blocksize::UInt  # blocksize for this page
    free::SizedPtr{T}      # head of the free-list
    n_free::UInt           # number of free blocks
end

function MemoryPage(ptr::SizedPtr{T}, blocksize::UInt) where T
    n_blocks, r = divrem(ptr.size, blocksize)
    @assert r == 0
    # The first free pointer is the first block
    free = SizedPtr(ptr.ptr, blocksize)
    # Set up the free list
    for i in 0:(n_blocks - 1)
        ptr_this = ptr.ptr + i * blocksize
        ptr_next = i == n_blocks - 1 ? Ptr{T}() : ptr_this + blocksize
        unsafe_store!(Ptr{Ptr{T}}(ptr_this), ptr_next)
    end
    return MemoryPage{T}(ptr, blocksize, free, n_blocks)
end

function malloc(page::MemoryPage{T}, size::UInt) where T
    if size != page.blocksize
        error("malloc: requested size does not match the blocksize")
    end
    # Return early with null if the page is full
    if page.n_free == 0
        @assert page.free.ptr == C_NULL
        return SizedPtr{T}(Ptr{T}(), size)
    end
    # Read the pointer to be returned
    ret = page.free
    @assert ret.ptr != C_NULL
    # Look up and store the next free pointer
    page.free = SizedPtr{T}(unsafe_load(Ptr{Ptr{T}}(ret)), size)
    page.n_free -= 1
    return ret
end

function free(page::MemoryPage{T}, ptr::SizedPtr{T}) where T
    if !(ptr.size == page.blocksize && page.ptr.ptr <= ptr.ptr < page.ptr.ptr + page.ptr.size)
        error("free: not allocated in this page")
    end
    # Write the current free pointer to the pointer to be freed
    unsafe_store!(Ptr{Ptr{T}}(ptr), Ptr{T}(page.free))
    # Store the just-freed pointer and increment the availability counter
    page.free = ptr
    page.n_free += 1
    return
end

# Collection of pages for a specific size
struct MemoryPool{T}
    # nslots::Int
    blocksize::UInt
    pages::Vector{MemoryPage{T}}
end

function malloc(mempool::MemoryPool{T}, size::UInt) where T
    @assert mempool.blocksize == size
    # Try all existing pages
    # TODO: backwards is probably better since it is more likely there is room in the back?
    for page in mempool.pages
        ptr = malloc(page, size)
        ptr.ptr == C_NULL || return ptr
    end
    # Allocate a new page
    # TODO: Replace Libc.malloc with Memory in recent Julias?
    mptr = SizedPtr{T}(Ptr{T}(Libc.malloc(MALLOC_PAGE_SIZE)), MALLOC_PAGE_SIZE)
    page = MemoryPage(mptr, mempool.blocksize)
    push!(mempool.pages, page)
    # Allocate block in the new page
    ptr = malloc(page, size)
    @assert ptr.ptr != C_NULL
    return ptr
end

mutable struct MemoryHeap{T}
    const pools::Vector{MemoryPool{T}} # 2, 4, 6, 8, ...
    function MemoryHeap{T}() where T
        heap = new{T}(MemoryPool{T}[])
        finalizer(heap) do h
            for i in 1:length(h.pools)
                isassigned(h.pools, i) || continue
                for page in h.pools[i].pages
                    Libc.free(page.ptr.ptr)
                end
            end
            return
        end
        return heap
    end
end

function poolindex_from_blocksize(blocksize::UInt)
    return (8 * sizeof(UInt) - leading_zeros(blocksize)) % Int
end

function malloc(heap::MemoryHeap{T}, size::Integer) where T
    blocksize = nextpow(2, size % UInt)
    poolidx = poolindex_from_blocksize(blocksize)
    if length(heap.pools) < poolidx
        resize!(heap.pools, poolidx)
    end
    if !isassigned(heap.pools, poolidx)
        pool = MemoryPool{T}(blocksize, MemoryPage{T}[])
        heap.pools[poolidx] = pool
    else
        pool = heap.pools[poolidx]
    end
    return malloc(pool, blocksize)
end

@inline function find_page(heap::MemoryHeap{T}, ptr::SizedPtr{T}) where T
    poolidx = poolindex_from_blocksize(ptr.size)
    if !isassigned(heap.pools, poolidx)
        error("pointer not malloc'd in this heap")
    end
    pool = heap.pools[poolidx]
    # Search for the page containing the pointer
    # TODO: Insert pages in sorted order and use searchsortedfirst?
    # TODO: Align the pages and store the base pointer -> page index mapping in a dict?
    pageidx = findfirst(p -> p.ptr.ptr <= ptr.ptr < (p.ptr.ptr + p.ptr.size), pool.pages)
    if pageidx === nothing
        error("pointer not malloc'd in this heap")
    end
    return pool.pages[pageidx]
end

function free(heap::MemoryHeap{T}, ptr::SizedPtr{T}) where T
    free(find_page(heap, ptr), ptr)
    return
end

function realloc(heap::MemoryHeap{T}, ptr::SizedPtr{T}, newsize::UInt) where T
    @assert newsize > ptr.size # TODO: Allow shrinkage?
    # Find the page for the pointer to make sure it was allocated in this heap
    page = find_page(heap, ptr)
    # Allocate the new pointer
    newptr = malloc(heap, newsize)
    # Copy the data
    Libc.memcpy(newptr.ptr, ptr.ptr, ptr.size)
    # Free the old pointer and return
    free(page, ptr)
    return newptr
end

# Minimal AbstractVector implementation on top of a raw pointer + a length
struct PtrVector{T} <: AbstractVector{T}
    ptr::SizedPtr{T}
    length::Int
    heap::MemoryHeap{T}
end
Base.size(mv::PtrVector) = (mv.length, )
allocated_length(mv::PtrVector{T}) where T = mv.ptr.size ÷ sizeof(T)
Base.IndexStyle(::Type{PtrVector}) = IndexLinear()
Base.@propagate_inbounds function Base.getindex(mv::PtrVector, i::Int)
    @boundscheck checkbounds(mv, i)
    return unsafe_load(mv.ptr.ptr, i)
end
Base.@propagate_inbounds function Base.setindex!(mv::PtrVector{T}, v::T, i::Int) where T
    @boundscheck checkbounds(mv, i)
    return unsafe_store!(mv.ptr.ptr, v, i)
end

struct SparsityPattern{T} <: AbstractSparsityPattern
    nrows::Int
    ncols::Int
    heap::MemoryHeap{T}
    rows::Vector{PtrVector{T}}
    function SparsityPattern(
            nrows::Int, ncols::Int;
            nnz_per_row::Int = 8,
        )
        T = Int
        heap = MemoryHeap{T}()
        rows = Vector{PtrVector{T}}(undef, nrows)
        for i in 1:nrows
            ptr = malloc(heap, nnz_per_row * sizeof(T))
            rows[i] = PtrVector{T}(ptr, 0, heap)
        end
        return new{T}(nrows, ncols, heap, rows)
    end
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
        allocated_entries += allocated_length(r)
    end
    ##
    bytes_estimate = 0
    bytes_estimate_used = 0
    bytes_estimate      += n_rows(sp) * sizeof(eltype(sp.rows))
    bytes_estimate_used += n_rows(sp) * sizeof(eltype(sp.rows))
    bytes_estimate_used += stored_entries * sizeof(T)
    ##
    bytes_malloced = 0
    for poolidx in 1:length(sp.heap.pools)
        isassigned(sp.heap.pools, poolidx) || continue
        bytes_malloced += length(sp.heap.pools[poolidx].pages) * Int(MALLOC_PAGE_SIZE)
        bytes_estimate += length(sp.heap.pools[poolidx].pages) * Int(MALLOC_PAGE_SIZE)
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

function add_entry!(sp::SparsityPattern{T}, row::Int, col::Int) where T
    @boundscheck (1 <= row <= n_rows(sp) && 1 <= col <= n_cols(sp)) || throw(BoundsError())
    # println("adding entry: $row, $col")
    r = @inbounds sp.rows[row]
    rptr = r.ptr
    rlen = r.length
    # rx = r.x
    k = searchsortedfirst(r, col)
    if k == rlen + 1 || @inbounds(r[k]) != col
        # TODO: This assumes we only grow by single entry every time
        if rlen == allocated_length(r) # % Int XXX
            @assert ispow2(rptr.size)
            rptr = realloc(sp.heap, rptr, 2 * rptr.size)
        end
        r = PtrVector{T}(rptr, rlen + 1, sp.heap)
        @inbounds sp.rows[row] = r
        # Shift elements after the insertion point to the back
        @inbounds for i in rlen:-1:k
            r[i+1] = r[i]
        end
        # Insert the new element
        @inbounds r[k] = col
    end
    return
end

# Auxiliary type that also wraps the heap to make sure the heap isn't freed
# while the vector is in use
# struct RootedPtrVector{T} <: AbstractVector{T}
#     x::PtrVector{T}
#     heap::MemoryHeap{T}
# end
# Base.size(mv::RootedPtrVector) = size(mv.x)
# Base.IndexStyle(::Type{RootedPtrVector}) = IndexLinear()
# Base.@propagate_inbounds function Base.getindex(mv::RootedPtrVector, i::Int)
#     return getindex(mv.x, i)
# end

# struct EachRow{T}
#     sp::SparsityPattern{T}
# end
# function Base.iterate(rows::EachRow, row::Int = 1)
#     row > length(rows) && return nothing
#     return eachrow(rows.sp, row), row + 1
# end
# Base.eltype(::Type{EachRow{T}}) where {T} = RootedPtrVector{T}
# Base.length(rows::EachRow) = n_rows(rows.sp)
# Base.keys(rows::EachRow) = 1:n_rows(rows.sp)

# eachrow(sp::SparsityPattern) = EachRow(sp)
# function eachrow(sp::SparsityPattern, row::Int)
#     return RootedPtrVector(sp.rows[row], sp.heap)
# end
eachrow(sp::SparsityPattern) = sp.rows
function eachrow(sp::SparsityPattern, row::Int)
    return sp.rows[row]
end

end # module Final
