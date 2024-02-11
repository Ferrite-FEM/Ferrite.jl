module Final

const ENABLE_ASSERT = false

macro assert(cond)
    if ENABLE_ASSERT
        return :($(esc(cond)) || throw(AssertionError()))
    else
        return nothing
    end
end

using ..Ferrite: Ferrite
using ..Ferrite: add_entry!, n_rows, n_cols, AbstractSparsityPattern
import ..Ferrite: add_entry!, n_rows, n_cols, eachrow

const MALLOC_PAGE_SIZE = 4 * 1024 * 1024 % UInt

# Like Ptr{T} but also stores the number of bytes allocated
struct SizedPtr{T}
    ptr::Ptr{T}
    size::UInt
end
Ptr{T}(ptr::SizedPtr) where {T} = Ptr{T}(ptr.ptr)

struct NullPtr end
const nullptr = NullPtr()

function Base.:(==)(ptr::SizedPtr, ::NullPtr)
    return ptr.ptr == C_NULL
end
function Base.:(==)(::NullPtr, ptr::SizedPtr)
    return ptr == nullptr
end

# Minimal AbstractVector implementation on top of a raw pointer + a length
struct PtrVector{T} <: AbstractVector{T}
    ptr::SizedPtr{T}
    length::Int
end
Base.size(mv::PtrVector) = (mv.length, )
Base.IndexStyle(::Type{PtrVector}) = IndexLinear()
Base.@propagate_inbounds function Base.getindex(mv::PtrVector, i::Int)
    @boundscheck checkbounds(mv, i)
    return unsafe_load(mv.ptr.ptr, i)
end
Base.@propagate_inbounds function Base.setindex!(mv::PtrVector{T}, v::T, i::Int) where T
    @boundscheck checkbounds(mv, i)
    return unsafe_store!(mv.ptr.ptr, v, i)
end

# Fixed buffer size (1MiB)
# Rowsize  | size | # Rows | # pages x86_64 | # pages aarch64
#       4  | 1MiB |  32768 |            256 |              64
#       8  | 1MiB |  16384 |            256 |              64
#      16  | 1MiB |   8192 |            256 |              64
#      32  | 1MiB |   4096 |            256 |              64
#      64  | 1MiB |   2048 |            256 |              64
#     128  | 1MiB |   1024 |            256 |              64
#     256  | 1MiB |    512 |            256 |              64

# Fixed number of rows (1024)
# Rowsize  |   size | # Rows | # pages x86_64 | # pages aarch64
#       4  |  32KiB |   1024 |              8 |               2
#       8  |  64KiB |   1024 |             16 |               4
#      16  | 128KiB |   1024 |             32 |               8
#      32  | 256KiB |   1024 |             64 |              16
#      64  | 512KiB |   1024 |            128 |              32
#     128  |   1MiB |   1024 |            256 |              64
#     256  |   2MiB |   1024 |            512 |             128


# A bit faster than nextpow(2, v) from Base (#premature-optimization),
# https://graphics.stanford.edu/~seander/bithacks.html#RoundUpPowerOf2
function nextpow2(x::Union{Int64, UInt64})
    v = x
    v -= 1
    v |= v >>> 1
    v |= v >>> 2
    v |= v >>> 4
    v |= v >>> 8
    v |= v >>> 16
    v |= v >>> 32
    v += 1
    @assert v >= x && rem(v, 2) == 0
    return v
end

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
        @assert page.free == nullptr
        return SizedPtr{T}(Ptr{T}(), size)
    end
    # Read the pointer to be returned
    ret = page.free
    @assert ret != nullptr
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
        ptr == nullptr || return ptr
    end
    # Allocate a new page
    mptr = SizedPtr{T}(Ptr{T}(Libc.malloc(MALLOC_PAGE_SIZE)), MALLOC_PAGE_SIZE)
    page = MemoryPage(mptr, mempool.blocksize)
    push!(mempool.pages, page)
    # Allocate block in the new page
    ptr = malloc(page, size)
    @assert ptr != nullptr
    return ptr
end

mutable struct ArenaAllocator{T}
    const pools::Vector{MemoryPool{T}} # 2, 4, 6, 8, ...
    function ArenaAllocator{T}() where T
        arena = new{T}(MemoryPool{T}[])
        finalizer(arena) do a
            for i in 1:length(a.pools)
                isassigned(a.pools, i) || continue
                for page in a.pools[i].pages
                    Libc.free(page.ptr.ptr)
                end
            end
        end
        return arena
    end
end

function poolindex_from_blocksize(blocksize::UInt)
    return (8 * sizeof(UInt) - leading_zeros(blocksize)) % Int
end

function malloc(arena::ArenaAllocator{T}, size::Integer) where T
    blocksize = nextpow2(size % UInt)
    poolidx = poolindex_from_blocksize(blocksize)
    if length(arena.pools) < poolidx
        resize!(arena.pools, poolidx)
    end
    if !isassigned(arena.pools, poolidx)
        pool = MemoryPool{T}(blocksize, MemoryPage{T}[])
        arena.pools[poolidx] = pool
    else
        pool = arena.pools[poolidx]
    end
    return malloc(pool, blocksize)
end

@inline function find_page(arena::ArenaAllocator{T}, ptr::SizedPtr{T}) where T
    poolidx = poolindex_from_blocksize(ptr.size)
    if !isassigned(arena.pools, poolidx)
        error("pointer not malloc'd in this arena")
    end
    pool = arena.pools[poolidx]
    # Search for the page containing the pointer
    # TODO: Insert pages in sorted order and use searchsortedfirst?
    # TODO: Align the pages and store the base pointer -> page index mapping in a dict?
    pageidx = findfirst(p -> p.ptr.ptr <= ptr.ptr < (p.ptr.ptr + p.ptr.size), pool.pages)
    if pageidx === nothing
        error("pointer not malloc'd in this arena")
    end
    return pool.pages[pageidx]
end

function free(arena::ArenaAllocator{T}, ptr::SizedPtr{T}) where T
    free(find_page(arena, ptr), ptr)
    return
end

function realloc(arena::ArenaAllocator{T}, ptr::SizedPtr{T}, newsize::UInt) where T
    @assert newsize > ptr.size # TODO: Allow shrinkage?
    # Find the page for the pointer to make sure it was allocated in this arena
    page = find_page(arena, ptr)
    # Allocate the new pointer
    newptr = malloc(arena, newsize)
    # Copy the data
    Libc.memcpy(newptr.ptr, ptr.ptr, ptr.size)
    # Free the old pointer and return
    free(page, ptr)
    return newptr
end

struct MallocDSP{T} <: AbstractSparsityPattern
    nrows::Int
    ncols::Int
    arena::ArenaAllocator{T}
    rows::Vector{PtrVector{T}}
    # rows_per_chunk::Int
    # growth_factor::Float64
    # rowptr::Vector{Vector{Int}}
    # rowlength::Vector{Vector{Int}}
    # colval::Vector{Vector{Int}}
    function MallocDSP(
            nrows::Int, ncols::Int;
            nnz_per_row::Int = 8,
            # rows_per_chunk::Int = 1024,
            # growth_factor::Float64 = 1.5,
        )
        T = Int
        arena = ArenaAllocator{T}()
        rows = Vector{PtrVector{T}}(undef, nrows)
        for i in 1:nrows
            ptr = malloc(arena, nnz_per_row * sizeof(T))
            rows[i] = PtrVector{T}(ptr, 0)
        end
        return new{T}(nrows, ncols, arena, rows)
    end
end


n_rows(dsp::MallocDSP) = dsp.nrows
n_cols(dsp::MallocDSP) = dsp.ncols

# hits::Int = 0
# misses::Int = 0
# buffer_growths::Int = 0



# Like mod1
# function divrem1(x::Int, y::Int)
#     a, b = divrem(x - 1, y)
#     return a + 1, b + 1
# end

function add_entry!(dsp::MallocDSP{T}, row::Int, col::Int) where T
    @boundscheck (1 <= row <= n_rows(dsp) && 1 <= col <= n_cols(dsp)) || throw(BoundsError())
    # println("adding entry: $row, $col")
    r = @inbounds dsp.rows[row]
    rptr = r.ptr
    rlen = r.length
    # rx = r.x
    k = searchsortedfirst(r, col)
    if k == rlen + 1 || @inbounds(r[k]) != col
        # TODO: This assumes we only grow by single entry every time
        if rlen == rptr.size ÷ sizeof(T) # % Int XXX
            @assert ispow2(rptr.size)
            rptr = realloc(dsp.arena, rptr, 2 * rptr.size)
        else
            # @assert length(rx) < rs
            # ptr = rx.pointer
        end
        r = PtrVector{T}(rptr, rlen + 1)
        @inbounds dsp.rows[row] = r
        # Shift elements after the insertion point to the back
        @inbounds for i in rlen:-1:k
            r[i+1] = r[i]
        end
        # Insert the new element
        @inbounds r[k] = col
    end
    return
end

# struct RowIterator
#     colval::Vector{Int}
#     rowptr::Int
#     rowlength::Int
#     function RowIterator(dsp::MallocDSP, row::Int)
#         @assert 1 <= row <= n_rows(dsp)
#         chunkidx, idx = divrem(row-1, dsp.rows_per_chunk)
#         chunkidx += 1
#         idx      += 1
#         rowptr    = dsp.rowptr[chunkidx]
#         rowlength = dsp.rowlength[chunkidx]
#         colval    = dsp.colval[chunkidx]
#         # # Construct the colvalview for this row
#         # colvalview = view(colval, rowptr[idx]:rowptr[idx] + rowlength[idx] - 1)
#         return new(colval, rowptr[idx], rowlength[idx])
#     end
# end

# function Base.iterate(it::RowIterator, i = 1)
#     if i > it.rowlength || it.rowlength == 0
#         return nothing
#     else
#         return it.colval[it.rowptr + i - 1], i + 1
#     end
# end

# eachrow(dsp::MallocDSP) = (RowIterator(dsp, row) for row in 1:n_rows(dsp))
# eachrow(dsp::MallocDSP, row::Int) = RowIterator(dsp, row)

# View version
eachrow(dsp::MallocDSP) = (eachrow(dsp, row) for row in 1:n_rows(dsp))
function eachrow(dsp::MallocDSP, row::Int)
    return dsp.rows[row]
end
# @inline function rowview(dsp::MallocDSP, row::Int)
#     chunkidx, idx = divrem1(row, dsp.rows_per_chunk)
#     rowptr    = dsp.rowptr[chunkidx]
#     rowlength = dsp.rowlength[chunkidx]
#     colval    = dsp.colval[chunkidx]
#     nzrange = rowptr[idx]:rowptr[idx] + rowlength[idx] - 1
#     return view(colval, nzrange)
# end


end # module Final
