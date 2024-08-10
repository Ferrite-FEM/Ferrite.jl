module PoolAllocator

# Checkmate LanguageServer.jl
const var"@propagate_inbounds" = Base.var"@propagate_inbounds"

@eval macro $(Symbol("const"))(field)
    if VERSION >= v"1.8.0-DEV.1148"
        Expr(:const, esc(field))
    else
        return esc(field)
    end
end

const PAGE_SIZE = 4 * 1024 * 1024 # 4 MiB

# A page corresponds to a memory block of size `PAGE_SIZE` bytes.
# Allocations of arrays are views into this block.
mutable struct Page{T}
    @const buf::Vector{T}      # data buffer (TODO: Memory in recent Julias?)
    @const blocksize::Int      # blocksize for this page
    @const freelist::BitVector # block is free/used
    n_free::Int               # number of free blocks
    function Page{T}(blocksize::Int) where T
        @assert isbitstype(T)
        buf = Vector{T}(undef, PAGE_SIZE ÷ sizeof(T))
        n_blocks, r = divrem(length(buf), blocksize)
        @assert r == 0
        return new{T}(buf, blocksize, trues(n_blocks), n_blocks)
    end
end

# Find a free block and mark it as used
function _malloc(page::Page, size::Int)
    if size != page.blocksize
        error("malloc: requested size does not match the blocksize of this page")
    end
    # Return early if the page is already full
    if page.n_free == 0
        return nothing
    end
    # Find the first free block
    blockindex = findfirst(page.freelist)::Int
    if !@inbounds(page.freelist[blockindex])
        error("malloc: block already in use")
    end
    @inbounds page.freelist[blockindex] = false
    offset = (blockindex - 1) * page.blocksize
    page.n_free -= 1
    return offset
end

# Mark a block as free
function _free(page::Page, offset::Int)
    blockindex = offset ÷ page.blocksize + 1
    if @inbounds page.freelist[blockindex]
        error("free: block already free'd")
    end
    @inbounds page.freelist[blockindex] = true
    page.n_free += 1
    # TODO: If this page is completely unused it can be collected and reused.
    return
end

# A book is a collection of pages with a specific blocksize
struct Book{T}
    blocksize::Int
    pages::Vector{Page{T}}
end

# Find a page with a free block of the requested size
function _malloc(book::Book{T}, size::Int) where {T}
    @assert book.blocksize == size
    # Check existing pages
    for page in book.pages
        offset = _malloc(page, size)
        if offset !== nothing
            return (page, offset)
        end
    end
    # Allocate a new page
    page = Page{T}(book.blocksize)
    push!(book.pages, page)
    # Allocate block in the new page
    offset = _malloc(page, size)
    @assert offset !== nothing
    return (page, offset)
end

struct MemoryPool{T}
    books::Vector{Book{T}} # blocksizes 2, 4, 6, 8, ...
    function MemoryPool{T}() where T
        mempool = new(Book{T}[])
        return mempool
    end
end

# Free all pages by resizing all page containers to 0
function free(mempool::MemoryPool)
    for i in 1:length(mempool.books)
        isassigned(mempool.books, i) || continue
        resize!(mempool.books[i].pages, 0)
    end
    resize!(mempool.books, 0)
    return
end

function mempool_stats(mempool::MemoryPool{T}) where T
    bytes_used = 0
    bytes_allocated = 0
    for bookidx in 1:length(mempool.books)
        isassigned(mempool.books, bookidx) || continue
        book = mempool.books[bookidx]
        bytes_allocated += length(book.pages) * PAGE_SIZE
        for page in book.pages
            bytes_used += count(!, page.freelist) * page.blocksize * sizeof(T)
        end
    end
    return bytes_used, bytes_allocated
end

function Base.show(io::IO, ::MIME"text/plain", mempool::MemoryPool{T}) where T
    n_books = count(i -> isassigned(mempool.books, i), 1:length(mempool.books))
    print(io, "PoolAllocator.MemoryPool{$(T)} with $(n_books) fixed size pools")
    n_books == 0 && return
    println(io, ":")
    for idx in 1:length(mempool.books)
        isassigned(mempool.books, idx) || continue
        h = mempool.books[idx]
        blocksize = h.blocksize
        # @assert blocksize == 2^idx
        npages = length(h.pages)
        n_free = mapreduce(p -> p.n_free, +, h.pages; init=0)
        n_tot = npages * PAGE_SIZE ÷ blocksize ÷ sizeof(T)
        println(io, " - blocksize: $(blocksize), npages: $(npages), usage: $(n_tot - n_free) / $(n_tot)")
    end
    return
end

function bookindex_from_blocksize(blocksize::Int)
    return (8 * sizeof(Int) - leading_zeros(blocksize)) % Int
end

function malloc(mempool::MemoryPool{T}, dims::NTuple{N, Int}) where {T, N}
    @assert prod(dims) > 0
    blocksize = nextpow(2, prod(dims))
    bookidx = bookindex_from_blocksize(blocksize)
    if length(mempool.books) < bookidx
        resize!(mempool.books, bookidx)
    end
    if !isassigned(mempool.books, bookidx)
        book = Book(blocksize, Page{T}[])
        mempool.books[bookidx] = book
    else
        book = mempool.books[bookidx]
    end
    page, offset = _malloc(book, blocksize)

    return PoolArray{T, N}(mempool, page, offset, dims)
end


# PoolArray is a view into a page that also has a reference to the MemoryPool so that it can
# be resized/reallocated.
struct PoolArray{T, N} <: AbstractArray{T, N}
    mempool::MemoryPool{T}
    page::Page{T}
    offset::Int
    size::NTuple{N, Int}
end

const PoolVector{T} = PoolArray{T, 1}

# Constructors
function malloc(mempool::MemoryPool, dim1::Int)
    return malloc(mempool, (dim1, ))
end
function malloc(mempool::MemoryPool, dim1::Int, dim2::Int, dimx::Int...)
    dims = (dim1, dim2, map(Int, dimx)..., )
    return malloc(mempool, dims)
end

function free(x::PoolArray)
    _free(x.page, x.offset)
    return
end

function realloc(x::PoolArray{T}, newsize::Int) where T
    @assert newsize > length(x) # TODO: Allow shrinkage?
    @assert newsize <= PAGE_SIZE ÷ sizeof(T) # TODO: Might be required
    # Find the page for the block to make sure it was allocated in this mempool
    # page = find_page(x.mempool, )
    # Allocate the new block
    x′ = malloc(x.mempool, newsize)
    # Copy the data
    copyto!(x′, x)
    # Free the old block and return
    _free(x.page, x.offset)
    return x′
end

# AbstractArray interface
Base.size(mv::PoolArray) = mv.size
allocated_length(mv::PoolArray) = mv.page.blocksize
Base.IndexStyle(::Type{<:PoolArray}) = IndexLinear()
@propagate_inbounds function Base.getindex(mv::PoolArray, i::Int)
    @boundscheck checkbounds(mv, i)
    return @inbounds mv.page.buf[mv.offset + i]
end
@propagate_inbounds function Base.setindex!(mv::PoolArray{T}, v::T, i::Int) where T
    @boundscheck checkbounds(mv, i)
    @inbounds mv.page.buf[mv.offset + i] = v
    return mv
end

# Utilities needed for the sparsity pattern
@inline function resize(x::PoolVector{T}, n::Int) where T
    if n > allocated_length(x)
        return realloc(x, n)
    else
        return PoolVector{T}(x.mempool, x.page, x.offset, (n, ))
    end
end

@inline function insert(x::PoolVector{T}, k::Int, item::T) where T
    lx = length(x)
    # Make room
    x = resize(x, lx + 1)
    # Shift elements after the insertion point to the back
    @inbounds for i in lx:-1:k
        x[i + 1] = x[i]
    end
    # Insert the new element
    @inbounds x[k] = item
    return x
end

end # module PoolAllocator
