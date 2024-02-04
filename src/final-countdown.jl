module Europe

using ..Ferrite: Ferrite, n_rows, n_cols, add_entry!

const KiB =        1024 % UInt
const MiB = 1024 * 1024 % UInt

const OS_MALLOC_SIZE_BYTES  = (4 * 1024 * 1024) % UInt #  4 MiB
const CLASS_PAGE_SIZE_BYTES = (      64 * 1024) % UInt # 64 KiB

struct SizedPtr{T}
    ptr::Ptr{T}
    size::UInt
end

# # Conversion SizedPtr{Cvoid} <-> SizedPtr{T}
# function SizedPtr{T}(ptr::SizedPtr{Cvoid}) where T
#     return SizedPtr{T}(convert(Ptr{T}, ptr.ptr), ptr.size ÷ sizeof(T))
# end
# function SizedPtr{Cvoid}(ptr::SizedPtr{T}) where T
#     return SizedPtr{Cvoid}(convert(Ptr{Cvoid}, ptr.ptr), ptr.size * sizeof(T))
# end

struct MallocVector3{T} <: AbstractVector{T}
    ptr::SizedPtr{T}
    length::Int
end
Base.size(mv::MallocVector3) = (mv.length, )
Base.IndexStyle(::Type{MallocVector3}) = IndexLinear()
Base.@propagate_inbounds function Base.getindex(mv::MallocVector3, i::Int)
    @boundscheck checkbounds(mv, i)
    return unsafe_load(mv.ptr.ptr, i)
end
Base.@propagate_inbounds function Base.setindex!(mv::MallocVector3{T}, v::T, i::Int) where T
    @boundscheck checkbounds(mv, i)
    return unsafe_store!(mv.ptr.ptr, v, i)
end

mutable struct MemoryPage{T}
    const ptr::SizedPtr{T}
    const blocksize::UInt
    const freelist::BitVector
    n_avail::UInt
end

# malloc_calls_small = 0
# malloc_calls_big = 0

# function MemoryPage{T}(blocksize::Int) where T
#     error()
#     blocksizebytes = blocksize * sizeof(T)
#     if blocksizebytes <= CLASS_PAGE_SIZE_BYTES
#         # global malloc_calls_small += 1
#         ptr = convert(Ptr{T}, Libc.malloc(CLASS_PAGE_SIZE_BYTES))
#         nblocks = CLASS_PAGE_SIZE_BYTES ÷ blocksizebytes
#         @assert rem(CLASS_PAGE_SIZE_BYTES, blocksizebytes) == 0
#     else
#         error()
#         # global malloc_calls_big += 1
#         ptr = convert(Ptr{T}, Libc.malloc(blocksizebytes))
#         nblocks = 1
#     end
#     freelist = trues(nblocks)
#     return MemoryPage{T}(blocksize, ptr, freelist, nblocks)
# end

function malloc(page::MemoryPage{T}, sz::Int) where T
    error()
    # sz === nextpow2(sz) === page.blocksize || error("malloc: size mismatch")
    if page.n_avail == 0
        return Ptr{T}()
    end
    idx = findfirst(page.freelist)
    if idx === nothing
        error("n_avail > 0 but no free idx")
    end
    page.freelist[idx] = false
    page.n_avail -= 1
    ptr = page.ptr + (idx - 1) * sz * sizeof(T)
    return ptr
end

# Collection of pages for a fixed size
struct MemPool{T}
    blocksize::UInt
    pages::Vector{MemoryPage{T}}
end

const USE_OS_PAGES = true


# All allocations up to 64KiB end up in pools
# A pool is aligned at 64KiB to find it faster
# A pool gets memory from an 4MiB (aligned) OS allocation
# Larger allocs use malloc directly.

# pools[i] contains slabs for size 2^(i - 3)
mutable struct ArenaAllocator4{T}
    const pools::Vector{MemPool{T}} # 8, 16, 32, ...
    const pages::Vector{MemoryPage{T}} # Libc.malloc'd chunks

    function ArenaAllocator4{T}() where T
        pools = MemPool{T}[]
        pages = MemoryPage{T}[]
        arena = new{T}(pools, pages)
        finalizer(arena) do a
            ccall(:jl_safe_printf, Cvoid, (Cstring, ), "Finalizing arena...")
            # for i in 1:length(a.pools)
            #     isassigned(a.pools, i) || continue
            #     for page in a.pools[i].pages
            #         Libc.free(page.ptr)
            #         # count += 1
            #     end
            # end
            for page in arena.pages
                Libc.free(page.ptr.ptr)
            end
        end

        return arena
    end
end

function poolindex_from_blocksize(blocksize::UInt)
    idx = Int(8 * sizeof(UInt) - leading_zeros(blocksize) - 3)
    idx > 0 || error()
    return idx
end

aligned_allocs = 0

function aligned_alloc(alignment::UInt, size::UInt)
    return @ccall aligned_alloc(alignment::Csize_t, size::Csize_t)::Ptr{Cvoid}
end

using ..Ferrite: nextpow2

# entrypoint
function malloc(arena::ArenaAllocator4{T}, sz::Int) where T
    @assert sz >= 8
    blocksize = nextpow2(sz % UInt)
    @assert blocksize * sizeof(T) <= 64KiB
    # Find the correct pool for this size
    idx = poolindex_from_blocksize(blocksize)
    # Make sure we have allocated the pool
    if length(arena.pools) < idx
        resize!(arena.pools, idx)
    end
    # Get or create the pool
    if isassigned(arena.pools, idx)
        pool = arena.pools[idx]
    else
        pool = MemPool{T}(blocksize, MemoryPage{T}[])
        arena.pools[idx] = pool
    end
    # Allocate in the pool
    ptr = malloc(arena, pool, blocksize)
    return ptr

    # # OLD
    # ptr = malloc(pool, blocksize)
    # if ptr == C_NULL
    #     # Pool needs a new page from the OS
    #     pageidx = findfirst(p -> p.n_avail > 0, arena.pages)
    #     if pageidx === nothing
    #         # osptr = Libc.malloc(OS_MALLOC_SIZE_BYTES)
    #         global aligned_allocs += 1
    #         osptr = aligned_alloc(OS_MALLOC_SIZE_BYTES, OS_MALLOC_SIZE_BYTES)
    #         n_avail = OS_MALLOC_SIZE_BYTES ÷ CLASS_PAGE_SIZE_BYTES
    #         freelist = trues(n_avail)
    #         page = MemoryPage{T}(CLASS_PAGE_SIZE_BYTES, osptr, freelist, n_avail)
    #         push!(arena.pages, page)
    #     else
    #         page = arena.pages[pageidx]
    #     end
    #     # also need to create the subpage for the mempool
    #     idx = findfirst(page.freelist)
    #     idx === nothing && error("nope")
    #     page.freelist[idx] = false
    #     page.n_avail -= 1
    #     pptr = page.ptr + (idx - 1) * CLASS_PAGE_SIZE_BYTES
    #     let n_avail = CLASS_PAGE_SIZE_BYTES ÷ (blocksize * sizeof(T)),
    #         freelist = trues(n_avail)
    #         poolpage = MemoryPage{T}(blocksize, pptr, freelist, n_avail)
    #         k = searchsortedfirst(pool.pages, poolpage; by = x -> x.ptr)
    #         insert!(pool.pages, k, poolpage)
    #     end
    #     # Try again lol
    #     ptr = malloc(pool, blocksize)
    # end
    # return SizedPtr{T}(ptr, sz)
end

@inline function assert_aligned(ptr::SizedPtr, alignment::Csize_t)
    Csize_t(ptr.ptr) == Csize_t(ptr.ptr) & ~(alignment - 1) || error("bad alignment")
end

function malloc(arena::ArenaAllocator4, mempool::MemPool{T}, size::UInt) where T
    size === mempool.blocksize || error("malloc: size mismatch")
    # Find a page with a free slot
    pageidx = findlast(x -> x.n_avail > 0, mempool.pages)
    if pageidx === nothing
        # Need to request a new page from the arena
        ospageidx = findlast(x -> x.n_avail > 0, arena.pages)
        if ospageidx === nothing
            # Need to request a new page from the OS
            osptr = SizedPtr{T}(aligned_alloc(4MiB, 4MiB), 4MiB ÷ sizeof(T))
            # @info "Allocating 4MiB" osptr
            assert_aligned(osptr, 4MiB)
            os_n_avail = 4MiB ÷ 64KiB
            os_freelist = trues(os_n_avail)
            ospage = MemoryPage(osptr, 64MiB, os_freelist, os_n_avail)
            push!(arena.pages, ospage)
        else
            # Use existing page
            ospage = arena.pages[ospageidx]
        end
        @assert ospage.n_avail > 0
        blockidx = findfirst(ospage.freelist)
        ospage.freelist[blockidx] = false
        ospage.n_avail -= 1
        blockptr = SizedPtr{T}(ospage.ptr.ptr + (blockidx - 1) * 64KiB, 64KiB ÷ sizeof(T))
        # @info "Allocating 64KiB" ospage.ptr.ptr blockidx blockptr
        assert_aligned(blockptr, 64KiB)
        n_avail = 64KiB ÷ (mempool.blocksize * sizeof(T))
        freelist = trues(n_avail)
        page = MemoryPage(blockptr, mempool.blocksize, freelist, n_avail)
        push!(mempool.pages, page)
        pageidx = lastindex(mempool.pages) # FOR LOGGING
    else
        # Use existing page
        page = mempool.pages[pageidx]
    end
    # Find available slot in the page
    block_idx = findfirst(page.freelist)
    # if poolindex_from_blocksize(mempool.blocksize) == 4 &&
    #     pageidx == 1 && block_idx == 8
    #     @info "Allocating block: poolidx = $(poolindex_from_blocksize(mempool.blocksize)) pageidx = $(pageidx) blockidx = $(block_idx)"
    # end
    # @info "Allocating block for ints" pageidx block_idx Int(size)
    page.freelist[block_idx] = false
    page.n_avail -= 1
    ptr = SizedPtr{T}(page.ptr.ptr + (block_idx - 1) * mempool.blocksize * sizeof(T), mempool.blocksize)
    return ptr
end


function free(arena::ArenaAllocator4{T}, ptr::Ptr{T}) where T
    error()
    for poolidx in 1:length(arena.pools)
        if isassigned(arena.pools, poolidx)
            pool = arena.pools[poolidx]
            nslots = pool.nslots
            slotsize = pool.slotsize
            for block in pool.blocks
                block.ptr <= ptr <= (block.ptr + nslots * slotsize * sizeof(T)) || continue
                idx = (ptr - block.ptr) ÷ (slotsize * sizeof(T)) + 1
                if block.freelist[idx]
                    error("free: use after free")
                end
                block.freelist[idx] = true
                block.n_avail += 1
                return
            end
        end
    end
    error("invalid pointer: not malloc'd in this arena")
end

function realloc(arena::ArenaAllocator4{T}, ptr::SizedPtr{T}, newsz::Int) where T
    # @info "realloc"
    @assert newsz >= 8
    blocksize = nextpow2(ptr.size)
    newblocksize = nextpow2(newsz)
    # @info "realloc" blocksize newblocksize
    @assert newblocksize >= blocksize
    if newsz <= blocksize
        # @info "fast path yo"
        error("need to check size")
        return SizedPtr{T}(ptr.ptr, newsz)
    end
    poolidx = poolindex_from_blocksize(blocksize)
    if !isassigned(arena.pools, poolidx)
        error("realloc: pointer not malloc'd in this heap")
    end
    pool = arena.pools[poolidx]
    # find the page XXX: Try the alignment trick from mimalloc
    # pageidx = findfirst(pool.pages) do page
    #     page.ptr <= ptr.ptr < (page.ptr + CLASS_PAGE_SIZE_BYTES)
    # end
    # Pool pointer are aligned at 64KiB
    ptr_base = UInt(ptr.ptr) & ~(64KiB - 1)
    k = findfirst(p -> UInt(p.ptr.ptr) == ptr_base, pool.pages)
    if k === nothing
        error("realloc: pointer not malloc'd in this heap")
    end
    page = pool.pages[k]
    blocksizebytes = blocksize * sizeof(T)
    idx = Int((UInt(ptr.ptr) - UInt(page.ptr.ptr)) ÷ blocksizebytes + 1)
    if page.freelist[idx]
        # @info "ERROR: Freeing block: poolidx = $(poolidx) pageidx = $(k) blockidx = $(idx)"
        error("realloc: use after free")
    elseif newsz <= blocksize
        @assert false
    end
    newptr = malloc(arena, newsz)
    Libc.memcpy(newptr.ptr, ptr.ptr, blocksizebytes) # TODO: Faster without memcpy?
    page.freelist[idx] = true
    # if poolidx == 4 && k == 1 && idx == 8
    #     @info "Freeing block: poolidx = $(poolidx) pageidx = $(k) blockidx = $(idx)"
    # end
    page.n_avail += 1
    if page.n_avail == 64KiB ÷ blocksizebytes
        let osbaseptr = Ptr{T}(UInt(page.ptr.ptr) & (~(4MiB - 1)))
            ospage = findfirst(p -> p.ptr.ptr == osbaseptr, arena.pages)
            freeidx = Int((page.ptr.ptr - osbaseptr) ÷ 64KiB) + 1
            arena.pages[ospage].freelist[freeidx] = true
            arena.pages[ospage].n_avail += 1
            deleteat!(pool.pages, k)
        end
    end
    return newptr
    # end
    # error("invalid pointer: not malloc'd in this arena")
end

# function calloc(allocator::ArenaAllocator4, count::Int, size::Int)
# end

# function realloc(allocator::ArenaAllocator4, size::Int)
# end

# const ROWS_PER_BUFFER = 1024
# const BUFFER_GROWTH_FACTOR = 1.5

# using UnsafeArrays

# struct UnsafeVector
#     x::UnsafeArray{Int, 1}
#     size::Int # the malloc'd size
# end


using ..Ferrite: AbstractSparsityPattern

struct MallocDSP4 <: AbstractSparsityPattern
    nrows::Int
    ncols::Int
    arena::ArenaAllocator4{Int}
    rows::Vector{MallocVector3{Int}}
    # rows_per_chunk::Int
    # growth_factor::Float64
    # rowptr::Vector{Vector{Int}}
    # rowlength::Vector{Vector{Int}}
    # colval::Vector{Vector{Int}}
    function MallocDSP4(
            nrows::Int, ncols::Int;
            nnz_per_row::Int = 8,
            # rows_per_chunk::Int = 1024,
            # growth_factor::Float64 = 1.5,
        )
        arena = ArenaAllocator4{Int}()
        rows = Vector{MallocVector3{Int}}(undef, nrows)
        for i in 1:nrows
            ptr = malloc(arena, nnz_per_row)
            @assert ptr.size == nnz_per_row
            if i == 2
                # @show ptr.ptr - rows[1].ptr.ptr
                # @show nnz_per_row * sizeof(Int)
                @assert ptr.ptr - rows[1].ptr.ptr == nnz_per_row * sizeof(Int)
            end
            rows[i] = MallocVector3(ptr, 0)
        end
        return new(nrows, ncols, arena, rows)
    end
end


Ferrite.n_rows(dsp::MallocDSP4) = dsp.nrows
Ferrite.n_cols(dsp::MallocDSP4) = dsp.ncols

# # hits::Int = 0
# # misses::Int = 0
# # buffer_growths::Int = 0



# # Like mod1
# # function divrem1(x::Int, y::Int)
# #     a, b = divrem(x - 1, y)
# #     return a + 1, b + 1
# # end

function check_arena(arena::ArenaAllocator4)
    bytes = UInt[]
    for poolidx = 1:length(arena.pools)
        isassigned(arena.pools, poolidx) || continue
        pool = arena.pools[poolidx]
        for (pageidx, page) in pairs(pool.pages)
            for (idx, isfree) in pairs(page.freelist)
                isfree && continue
                bytes_in_block = (page.ptr.ptr + (idx - 1) * page.blocksize * sizeof(Int)) : (page.ptr.ptr + (idx) * page.blocksize * sizeof(Int) - 1)
                if any(x -> UInt(x) in bytes, bytes_in_block)
                    error("overlapping bytes for poolidx = $poolidx pageidx = $pageidx idx = $idx")
                end
                append!(bytes, bytes_in_block)
            end
        end
    end
    @assert length(bytes) == length(unique(bytes))
    return
end


function Ferrite.add_entry!(dsp::MallocDSP4, row::Int, col::Int)
    # if row == 2 && col == 9
    #     error("breakpoitn")
    # end
    # @show row, col
    @boundscheck (1 <= row <= n_rows(dsp) && 1 <= col <= n_cols(dsp)) || throw(BoundsError())
    r = @inbounds dsp.rows[row]
    k = searchsortedfirst(r, col)
    len = length(r)
    if k > len || @inbounds(r[k]) != col
        # TODO: This assumes we only grow by single entry every time
        ptr = r.ptr
        allocated_length = r.ptr.size
        if len == allocated_length
            # c = ceil(Int, allocated_length * dsp.growth_factor)
            # o = allocated_length + 1
            # allocated_length = max(
            #     ceil(Int, allocated_length * dsp.growth_factor),
            #     allocated_length + 1,
            # )
            # allocated_length = nextpow2(allocated_length + 1)
            # println("Increasing size from $len to either $c or $o, picking $allocated_length")
            # @show row, col
            # check_arena(dsp.arena)
            ptr = realloc(dsp.arena, r.ptr, nextpow2(len+1))
            # @show ret
            # @show typeof(ret)
            # ptr = convert(Ptr{Int}, ret)
        end
        r = MallocVector3(ptr, len + 1)
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

Ferrite.eachrow(dsp::MallocDSP4) = dsp.rows
function Ferrite.eachrow(dsp::MallocDSP4, row::Int)
    return dsp.rows[row]
end

# function add_entry!(dsp::MallocDSP4, row::Int, col::Int)
#     @boundscheck (1 <= row <= n_rows(dsp) && 1 <= col <= n_cols(dsp)) || throw(BoundsError())
#     # println("adding entry: $row, $col")
#     r = dsp.rows[row]
#     rs = r.size
#     rx = r.x
#     ptr = rx.pointer
#     k = searchsortedfirst(rx, col)
#     if k == lastindex(rx) + 1 || @inbounds(rx[k]) != col
#         # TODO: This assumes we only grow by single entry every time
#         if length(rx) == rs
#             rs = Ferrite.nextpow2(rs + 1)
#             ptr = Ferrite.realloc(dsp.arena, rx.pointer, rs)
#         else
#             # @assert length(rx) < rs
#             # ptr = rx.pointer
#         end
#         rx = UnsafeArray(ptr, (length(rx) + 1, ))
#         xxx = UnsafeVector(rx, rs)
#         dsp.rows[row] = xxx
#         # Shift elements after the insertion point to the back
#         @inbounds for i in (length(rx)-1):-1:k
#             rx[i+1] = rx[i]
#         end
#         # Insert the new element
#         rx[k] = col
#     else
#         # global hits += 1
#     end
#     return
# end

# # struct RowIterator
# #     colval::Vector{Int}
# #     rowptr::Int
# #     rowlength::Int
# #     function RowIterator(dsp::MallocDSP, row::Int)
# #         @assert 1 <= row <= n_rows(dsp)
# #         chunkidx, idx = divrem(row-1, dsp.rows_per_chunk)
# #         chunkidx += 1
# #         idx      += 1
# #         rowptr    = dsp.rowptr[chunkidx]
# #         rowlength = dsp.rowlength[chunkidx]
# #         colval    = dsp.colval[chunkidx]
# #         # # Construct the colvalview for this row
# #         # colvalview = view(colval, rowptr[idx]:rowptr[idx] + rowlength[idx] - 1)
# #         return new(colval, rowptr[idx], rowlength[idx])
# #     end
# # end

# # function Base.iterate(it::RowIterator, i = 1)
# #     if i > it.rowlength || it.rowlength == 0
# #         return nothing
# #     else
# #         return it.colval[it.rowptr + i - 1], i + 1
# #     end
# # end

# # eachrow(dsp::MallocDSP) = (RowIterator(dsp, row) for row in 1:n_rows(dsp))
# # eachrow(dsp::MallocDSP, row::Int) = RowIterator(dsp, row)

# # View version
# eachrow(dsp::MallocDSP) = (eachrow(dsp, row) for row in 1:n_rows(dsp))
# function eachrow(dsp::MallocDSP, row::Int)
#     return dsp.rows[row].x
# end
# # @inline function rowview(dsp::MallocDSP, row::Int)
# #     chunkidx, idx = divrem1(row, dsp.rows_per_chunk)
# #     rowptr    = dsp.rowptr[chunkidx]
# #     rowlength = dsp.rowlength[chunkidx]
# #     colval    = dsp.colval[chunkidx]
# #     nzrange = rowptr[idx]:rowptr[idx] + rowlength[idx] - 1
# #     return view(colval, nzrange)
# # end
end # module Europe

using .Europe
