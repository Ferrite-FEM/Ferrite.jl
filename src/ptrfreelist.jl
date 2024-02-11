module Europe2

using ..Ferrite: Ferrite, n_rows, n_cols, add_entry!

const KiB =        1024 % UInt
const MiB = 1024 * 1024 % UInt

const OS_MALLOC_SIZE_BYTES  = (4 * 1024 * 1024) % UInt #  4 MiB
const CLASS_PAGE_SIZE_BYTES = (      64 * 1024) % UInt # 64 KiB

struct SizedPtr{T}
    ptr::Ptr{T}
    size::UInt
end

# Ptr{T}(ptr::SizedPtr) where {T} = Ptr{T}(ptr.ptr)

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
    free::SizedPtr{T}
    n_avail::UInt
end
function MemoryPage(ptr::SizedPtr{T}, blocksize::UInt) where T
    @assert rem(ptr.size, blocksize) == 0
    n_avail = ptr.size ÷ blocksize
    # Initialize freelist
    free = SizedPtr{T}(ptr.ptr, blocksize)
    # n_stores = 0
    for i in 0:(n_avail-2)
        # @show Int(i)
        ptr_i  = ptr.ptr + i * blocksize
        free_i = ptr_i + blocksize
        # @show free_i - ptr_i
        # @show Int(blocksize)
        # @show typeof(ptr_i)
        # @show typeof(free_i)
        unsafe_store!(Ptr{Ptr{T}}(ptr_i), free_i)
        # n_stores += 1
    end
    ptr_n = ptr.ptr + (n_avail - 1) * blocksize
    free_n = Ptr{T}()
    unsafe_store!(Ptr{Ptr{T}}(ptr_n), free_n)
    # n_stores += 1
    # @show ptr
    # @show blocksize
    # @show n_avail
    # @show n_stores
    # error()

    return MemoryPage{T}(ptr, blocksize, free, n_avail)
end

# Collection of pages for a fixed size
mutable struct MemPool{T}
    const blocksize::UInt
    const pages::Vector{MemoryPage{T}}
    has_unused::Bool
end

# All allocations up to 64KiB end up in pools
# A pool is aligned at 64KiB to find it faster
# A pool gets memory from an 4MiB (aligned) OS allocation
# Larger allocs use malloc directly.

# pools[i] contains slabs for size 2^(i - 3)
mutable struct ArenaAllocator4{T}
    const pools::Vector{MemPool{T}} # 8, 16, 32, ...
    const pages::Vector{MemoryPage{T}} # Libc.malloc'd chunks
    const available_64KiB::Vector{MemoryPage{T}}

    function ArenaAllocator4{T}() where T
        pools = MemPool{T}[]
        pages = MemoryPage{T}[]
        available_64KiB = MemoryPage{T}[]
        arena = new{T}(pools, pages, available_64KiB)
        finalizer(arena) do a
            ccall(:jl_safe_printf, Cvoid, (Cstring, ), "Finalizing arena...")
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

function aligned_alloc(alignment::UInt, size::UInt)
    return @ccall aligned_alloc(alignment::Csize_t, size::Csize_t)::Ptr{Cvoid}
end

using ..Ferrite: nextpow2

# entrypoint
function malloc(arena::ArenaAllocator4{T}, sz::Int) where T
    @assert sz >= 8
    blocksize = nextpow2(sz % UInt)
    @assert blocksize <= 64KiB
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
        pool = MemPool{T}(blocksize, MemoryPage{T}[], false)
        arena.pools[idx] = pool
    end
    # Allocate in the pool
    ptr = malloc(arena, pool, blocksize)
    return ptr
end

@inline function assert_aligned(ptr::SizedPtr, alignment::Csize_t)
    Csize_t(ptr.ptr) == Csize_t(ptr.ptr) & ~(alignment - 1) || error("bad alignment")
end

function __init__()
    @eval Base begin
        function show(io::IO, ptr::Ptr{T}) where T
            print(io, "Ptr{", T, "}(")
            show(io, UInt(ptr))
            print(io, ")")
            return
        end
    end
end

function malloc(arena::ArenaAllocator4, mempool::MemPool{T}, size::UInt) where T
    size === mempool.blocksize || error("malloc: size mismatch")
    # Find a page with a free slot
    pageidx = findlast(x -> x.n_avail > 0, mempool.pages)
    # pageidx = findfirst(x -> x.n_avail > 0, mempool.pages)
    if pageidx === nothing && length(arena.available_64KiB) > 0
        # @info "reuse'"
        oldpage = pop!(arena.available_64KiB)
        # @show length(arena.available_64KiB)
        # n_avail = 64KiB ÷ (mempool.blocksize * sizeof(T))
        # flist = resize!(oldpage.freelist, n_avail)
        # for i in 1:(length(flist) - 1)
        #     flist[i] = i + 1
        # end
        # flist[end] = 0
        # fill!(flist, true)
        page = MemoryPage(oldpage.ptr, mempool.blocksize)
        # push!(mempool.pages, page)
        kk = searchsortedfirst(mempool.pages, page; by = x -> x.ptr.ptr)
        insert!(mempool.pages, kk, page)
    elseif pageidx === nothing
        # Need to request a new page from the arena
        ospageidx = findlast(x -> x.n_avail > 0, arena.pages)
        if ospageidx === nothing
            # Need to request a new page from the OS
            osptr = SizedPtr{T}(aligned_alloc(4MiB, 4MiB), 4MiB)
            # @info "Allocating 4MiB" osptr
            assert_aligned(osptr, 4MiB)
            # os_n_avail = 4MiB ÷ 64KiB
            # os_freelist = trues(os_n_avail)
            # os_freelist = push!(collect(Int, 2:Int(os_n_avail)), 0)
            # os_free = 1
            ospage = MemoryPage(osptr, 64KiB)
            push!(arena.pages, ospage)
        else
            # Use existing page
            ospage = arena.pages[ospageidx]
        end
        @assert ospage.n_avail > 0
        # blockidx = findfirst(ospage.freelist)
        # ospage.freelist[blockidx] = false
        blockptr = ospage.free
        @assert blockptr.ptr != C_NULL
        ospage.free = SizedPtr{T}(
            unsafe_load(Ptr{Ptr{T}}(blockptr.ptr)),
            blockptr.size,
        )
        # @show blockptr
        # @show ospage.free
        # if ospage.free.ptr != C_NULL
        #     # @show ospage.free.ptr
        #     # @show blockptr.ptr
        #     # @show ospage.free.ptr - blockptr.ptr
        #     # @show 64KiB
        #     @assert ospage.free.ptr - blockptr.ptr == 64KiB
        # end
        # error()
        # ospage.freelist[blockidx] = false
        ospage.n_avail -= 1
        # blockptr = SizedPtr{T}(ospage.ptr.ptr + (blockidx - 1) * 64KiB, 64KiB ÷ sizeof(T))
        # @info "Allocating 64KiB" ospage.ptr.ptr blockidx blockptr
        assert_aligned(blockptr, 64KiB)
        # n_avail = 64KiB ÷ (mempool.blocksize * sizeof(T))
        # freelist = trues(n_avail)
        # freelist = push!(collect(Int, 2:Int(n_avail)), 0)
        # free = 1
        page = MemoryPage(blockptr, mempool.blocksize)
        # push!(mempool.pages, page)
        kk = searchsortedfirst(mempool.pages, page; by = x -> x.ptr.ptr)
        insert!(mempool.pages, kk, page)
    else
        # Use existing page
        page = mempool.pages[pageidx]
    end
    # Find available slot in the page
    # block_idx = findfirst(page.freelist)
    # page.freelist[block_idx] = false
    # block_idx = page.free
    ptr = page.free
    @assert ptr.ptr != C_NULL
    page.free = SizedPtr{T}(
        unsafe_load(Ptr{Ptr{T}}(ptr.ptr)),
        ptr.size,
    )
    # page.free = page.freelist[block_idx]
    page.n_avail -= 1
    if page.free.ptr == C_NULL
        @assert page.n_avail == 0
    #     @info "Next is null hehe"
    end
    # ptr = SizedPtr{T}(page.ptr.ptr + (block_idx - 1) * mempool.blocksize * sizeof(T), mempool.blocksize)
    return ptr
end

ptr_by(p::Ptr{T}) where {T} = p
ptr_by(p::MemoryPage) = p.ptr.ptr

function realloc(arena::ArenaAllocator4{T}, ptr::SizedPtr{T}, xxx::Int) where T
    newsz = UInt(xxx)
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
    ptr_base = Ptr{T}(UInt(ptr.ptr) & ~(64KiB - 1))
    # k = findfirst(p -> UInt(p.ptr.ptr) == ptr_base, pool.pages)
    k = searchsortedfirst(pool.pages, ptr_base; by = ptr_by)
    @assert pool.pages[k].ptr.ptr == ptr_base
    if k === nothing
        error("realloc: pointer not malloc'd in this heap")
    end
    page = pool.pages[k]
    idx = Int((UInt(ptr.ptr) - UInt(page.ptr.ptr)) ÷ blocksize + 1)
    if false # page.freelist[idx]
        # @info "ERROR: Freeing block: poolidx = $(poolidx) pageidx = $(k) blockidx = $(idx)"
        error("realloc: use after free")
    elseif newsz <= blocksize
        @assert false
    end
    newptr = malloc(arena, newsz % Int)
    Libc.memcpy(newptr.ptr, ptr.ptr, blocksize) # TODO: Faster without memcpy?
    # page.freelist[idx] = true
    # Store the current free in the ptr
    unsafe_store!(Ptr{Ptr{T}}(ptr.ptr), page.free.ptr)
    # Set the current free to the one we just freed
    page.free = ptr
    # page.
    # page.freelist[idx] = page.free
    # page.free = idx
    page.n_avail += 1
    # @show page.n_avail
    # @show 64KiB ÷ blocksize
    if page.n_avail == 64KiB ÷ blocksize
        let osbaseptr = Ptr{T}(UInt(page.ptr.ptr) & (~(4MiB - 1)))
            ospageidx = findfirst(p -> p.ptr.ptr == osbaseptr, arena.pages)
            ospageidx === nothing && error()
            ospage = arena.pages[ospageidx]
            # Store current free in ptr
            unsafe_store!(Ptr{Ptr{T}}(page.ptr.ptr), ospage.free.ptr)
            # Set the current free to the one we are freeing
            ospage.free = page.ptr
            ospage.n_avail += 1
            # @info "Freeing OS page block"
            # freeidx = Int((page.ptr.ptr - osbaseptr) ÷ 64KiB) + 1
            # arena.pages[ospage].freelist[freeidx] = true
            # arena.pages[ospage].n_avail += 1
            deleteat!(pool.pages, k)
        end
        # Cache the 64KiB block
        # @info "pushing available_64KiB"
        # push!(arena.available_64KiB, page)
        # deleteat!(pool.pages, k)
    end
    return newptr
    # end
    # error("invalid pointer: not malloc'd in this arena")
end


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
            ptr = malloc(arena, nnz_per_row * sizeof(Int))
            @assert ptr.size == nnz_per_row * sizeof(Int)
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
        allocated_length = r.ptr.size ÷ sizeof(Int)
        if len == allocated_length
            # @show row, col
            # @info "realloc: len=$len"
            # if len == 64 && row == 3 && col == 65
            #     error("bp")
            # end
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
            ptr = realloc(dsp.arena, r.ptr, Int(nextpow2(r.ptr.size + 1)))
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

end # module Europe

using .Europe2
