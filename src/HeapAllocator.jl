module HeapAllocator

# SizedPtr{UInt8}:
#  - alloc(heap, n)
#  - realloc(heap, ptr, n)
#  - free(heap, ptr)
#
# SizedPtr{T}:
#  - alloc(heap, T, n)
#  - realloc(heap, ptr, n)
#  - free(heap, ptr)
#
# HeapArray{T, N}:
#  - alloc_array(heap, T, n...)
#  - realloc(heap, x, n...)
#  - free(heap, x)

const MALLOC_PAGE_SIZE = 4 * 1024 * 1024 % UInt # 4 MiB

# Like Ptr{T} but also stores the number of bytes allocated
struct SizedPtr{T}
    ptr::Ptr{T}
    size::UInt
end
Ptr{T}(ptr::SizedPtr) where {T} = Ptr{T}(ptr.ptr)

# SizedPtr{T}(ptr::SizedPtr{T}) where {T} = ptr
SizedPtr{T}(ptr::SizedPtr) where {T} = SizedPtr{T}(Ptr{T}(ptr.ptr), ptr.size)


# A page corresponds to a larger Libc.malloc call (MALLOC_PAGE_SIZE). Each page
# is split into smaller blocks to minimize the number of Libc.malloc/Libc.free
# calls.
mutable struct Page
    const ptr::SizedPtr{UInt8} # malloc'd pointer
    const blocksize::UInt  # blocksize for this page
    free::SizedPtr{UInt8}      # head of the free-list
    n_free::UInt           # number of free blocks
end

function Page(ptr::SizedPtr{UInt8}, blocksize::UInt)
    n_blocks, r = divrem(ptr.size, blocksize)
    @assert r == 0
    # The first free pointer is the first block
    free = SizedPtr(ptr.ptr, blocksize)
    # Set up the free list
    for i in 0:(n_blocks - 1)
        ptr_this = ptr.ptr + i * blocksize
        ptr_next = i == n_blocks - 1 ? Ptr{UInt8}() : ptr_this + blocksize
        unsafe_store!(Ptr{Ptr{UInt8}}(ptr_this), ptr_next)
    end
    return Page(ptr, blocksize, free, n_blocks)
end

function _malloc(page::Page, size::UInt)
    if size != page.blocksize
        error("malloc: requested size does not match the blocksize")
    end
    # Return early with null if the page is full
    if page.n_free == 0
        @assert page.free.ptr == C_NULL
        return SizedPtr{UInt8}(Ptr{UInt8}(), size)
    end
    # Read the pointer to be returned
    ret = page.free
    @assert ret.ptr != C_NULL
    # Look up and store the next free pointer
    page.free = SizedPtr{UInt8}(unsafe_load(Ptr{Ptr{UInt8}}(ret)), size)
    page.n_free -= 1
    return ret
end

function _free(page::Page, ptr::SizedPtr{UInt8})
    if !(ptr.size == page.blocksize && page.ptr.ptr <= ptr.ptr < page.ptr.ptr + page.ptr.size)
        error("free: not allocated in this page")
    end
    # Write the current free pointer to the pointer to be freed
    unsafe_store!(Ptr{Ptr{UInt8}}(ptr), Ptr{UInt8}(page.free))
    # Store the just-freed pointer and increment the availability counter
    page.free = ptr
    page.n_free += 1
    return
end

# Collection of pages for a specific size
struct FixedSizeHeap
    # nslots::Int
    blocksize::UInt
    pages::Vector{Page}
end

function _malloc(fheap::FixedSizeHeap, size::UInt)
    @assert fheap.blocksize == size
    # Try all existing pages
    # TODO: backwards is probably better since it is more likely there is room in the back?
    for page in fheap.pages
        ptr = _malloc(page, size)
        ptr.ptr == C_NULL || return ptr
    end
    # Allocate a new page
    # TODO: Replace Libc.malloc with Memory in recent Julias?
    mptr = SizedPtr{UInt8}(Libc.malloc(MALLOC_PAGE_SIZE), MALLOC_PAGE_SIZE)
    page = Page(mptr, fheap.blocksize)
    push!(fheap.pages, page)
    # Allocate block in the new page
    ptr = _malloc(page, size)
    @assert ptr.ptr != C_NULL
    return ptr
end

mutable struct Heap
    const size_heaps::Vector{FixedSizeHeap} # 2, 4, 6, 8, ...
    function Heap()
        heap = new(FixedSizeHeap[])
        finalizer(heap) do h
            for i in 1:length(h.size_heaps)
                isassigned(h.size_heaps, i) || continue
                for page in h.size_heaps[i].pages
                    Libc.free(page.ptr.ptr)
                end
            end
            return
        end
        return heap
    end
end

function heapindex_from_blocksize(blocksize::UInt)
    return (8 * sizeof(UInt) - leading_zeros(blocksize)) % Int
end

function malloc(heap::Heap, size::UInt)
    blocksize = nextpow(2, size)
    fheapidx = heapindex_from_blocksize(blocksize)
    if length(heap.size_heaps) < fheapidx
        resize!(heap.size_heaps, fheapidx)
    end
    if !isassigned(heap.size_heaps, fheapidx)
        fheap = FixedSizeHeap(blocksize, Page[])
        heap.size_heaps[fheapidx] = fheap
    else
        fheap = heap.size_heaps[fheapidx]
    end
    return _malloc(fheap, blocksize)
end

malloc(heap::Heap, size::Integer) = malloc(heap, size % UInt)

function malloc(heap::Heap, ::Type{T}, size::Integer) where T
    @assert isbitstype(T)
    ptr = malloc(heap, (sizeof(T) * size) % UInt)
    return SizedPtr{T}(ptr)
end


@inline function find_page(heap::Heap, ptr::SizedPtr{UInt8})
    heapidx = heapindex_from_blocksize(ptr.size)
    if !isassigned(heap.size_heaps, heapidx)
        error("pointer not malloc'd in this heap")
    end
    size_heap = heap.size_heaps[heapidx]
    # Search for the page containing the pointer
    # TODO: Insert pages in sorted order and use searchsortedfirst?
    # TODO: Align the pages and store the base pointer -> page index mapping in a dict?
    pageidx = findfirst(p -> p.ptr.ptr <= ptr.ptr < (p.ptr.ptr + p.ptr.size), size_heap.pages)
    if pageidx === nothing
        error("pointer not malloc'd in this heap")
    end
    return size_heap.pages[pageidx]
end

function free(heap::Heap, ptr::SizedPtr{UInt8})
    error("TODO")
    free(find_page(heap, ptr), ptr)
    return
end

function free(heap::Heap, ptr::SizedPtr)
    error("TODO")
    return free(heap, SizedPtr{UInt8}(ptr))
end

function realloc(heap::Heap, ptr::SizedPtr{UInt8}, newsize::UInt)
    @assert newsize > ptr.size # TODO: Allow shrinkage?
    # Find the page for the pointer to make sure it was allocated in this heap
    page = find_page(heap, ptr)
    # Allocate the new pointer
    ptr′ = malloc(heap, newsize)
    # Copy the data
    Libc.memcpy(ptr′.ptr, ptr.ptr, ptr.size)
    # Free the old pointer and return
    _free(page, ptr)
    return ptr′
end

# Rewrap to the same pointer type
function realloc(heap::Heap, ptr::SizedPtr{T}, newsize::Integer) where T
    ptr′ = realloc(heap, SizedPtr{UInt8}(ptr), (newsize * sizeof(T)) % UInt)
    return SizedPtr{T}(ptr′)
end

# Minimal AbstractVector implementation on top of a raw pointer + a length
struct HeapArray{T, N} <: AbstractArray{T, N}
    ptr::SizedPtr{T}
    size::NTuple{N, Int}
    heap::Heap
end

const HeapVector{T} = HeapArray{T, 1}

Base.size(mv::HeapArray) = mv.size
allocated_length(mv::HeapArray{T}) where T = mv.ptr.size ÷ sizeof(T)
Base.IndexStyle(::Type{<:HeapArray}) = IndexLinear()
Base.@propagate_inbounds function Base.getindex(mv::HeapArray, i::Int)
    @boundscheck checkbounds(mv, i)
    return unsafe_load(mv.ptr.ptr, i)
end
Base.@propagate_inbounds function Base.setindex!(mv::HeapArray{T}, v::T, i::Int) where T
    @boundscheck checkbounds(mv, i)
    return unsafe_store!(mv.ptr.ptr, v, i)
end

function alloc_array(heap::Heap, ::Type{T}, size::NTuple{N, Int}) where {T, N}
    ptr = malloc(heap, T, prod(size))
    return HeapArray{T, N}(ptr, size, heap)
end

# Allow vararg dims
function alloc_array(heap::Heap, ::Type{T}, size::Int) where T
    return alloc_array(heap, T, (size, ))
end
function alloc_array(heap::Heap, ::Type{T}, size1::Int, size2::Int, sizes::Int...) where T
    size = (size1, size2, map(Int, sizes)..., )
    return alloc_array(heap, T, size)
end

function realloc(heap::Heap, x::HeapVector{T}, size::Int) where T
    @assert heap === x.heap
    ptr = realloc(heap, x.ptr, size)
    return HeapVector{T}(ptr, (size, ), heap)
end

realloc(x::HeapVector, n::Int) = realloc(x.heap, x, n)

function resize(x::HeapVector{T}, n::Int) where T
    if n > allocated_length(x)
        return realloc(x, n)
    else
        return HeapVector{T}(x.ptr, (n, ), x.heap)
    end
end

end # module HeapAllocator
