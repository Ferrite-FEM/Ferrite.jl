const PAGESIZE = Int(Sys.isunix() ? ccall(:jl_getpagesize, Clong, ()) : ccall(:jl_getallocationgranularity, Clong, ()))


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
function nextpow2(v::Int64)
    v -= 1
    v |= v >>> 1
    v |= v >>> 2
    v |= v >>> 4
    v |= v >>> 8
    v |= v >>> 16
    v |= v >>> 32
    v += 1
    return v
end


# struct MemorySlab
#     buf::Vector{Int}
#     function MemorySlab(n)
#         buf = zeros(Int, n)
#     end
# end

# struct SizedMemoryPool
#     slabs
# end

# using UnsafeArrays

# struct ResizeableUnsafeVector{T}
#     x::UnsafeArray{T, 1}
#     slotsize::Int
# end

# function realloc(x::ResizeableUnsafeVector{T}, newsize::Int) where T
#     if newsize <= x.slotsize
#         return ResizeableUnsafeVector{T}(UnsafeArray(x.x.pointer, (newsize, )), x.slotsize)
#     else
#         error("what now")
#     end
# end

struct MemoryBlock{T}
    nslots::Int
    slotsize::Int
    ptr::Ptr{T}
    freelist::BitVector
    function MemoryBlock{T}(nslots::Int, slotsize::Int) where T
        ptr = convert(Ptr{T}, Libc.malloc(nslots * slotsize * sizeof(T)))
        freelist = trues(nslots)
        return new{T}(nslots, slotsize, ptr, freelist)
    end
end

function malloc(memblock::MemoryBlock{T}, size::Int) where T
    nextpow2(size) === memblock.slotsize || error("!!")
    idx = findfirst(memblock.freelist)
    idx === nothing && return Ptr{T}()
    memblock.freelist[idx] = false
    ptr = memblock.ptr + (idx - 1) * memblock.slotsize * sizeof(T)
    return ptr
end

struct MemoryPool{T}
    nslots::Int
    slotsize::Int
    blocks::Vector{MemoryBlock{T}}
end

function malloc(mempool::MemoryPool{T}, size::Int) where T
    for block in mempool.blocks
        ptr = malloc(block, size)
        if ptr != C_NULL
            return ptr
        end
    end
    block = Ferrite.MemoryBlock{T}(mempool.nslots, mempool.slotsize)
    push!(mempool.blocks, block)
    ptr = Ferrite.malloc(block, size)
    ptr == C_NULL && error("malloc: could not allocate in new block")
    return ptr
end

# pools[i] contains slabs for size 2^i
mutable struct ArenaAllocator{T}
    pools::Vector{MemoryPool{T}} # 2, 4, 6, 8
    function ArenaAllocator{T}() where T
        arena = new{T}(MemoryPool{T}[])
        finalizer(arena) do a
            for i in 1:length(a.pools)
                isassigned(a.pools, i) || continue
                for block in a.pools[i].blocks
                    Libc.free(block.ptr)
                end
            end
        end

        return arena
    end
end

function malloc(arena::ArenaAllocator{T}, size::Int) where T
    poolsize = Ferrite.nextpow2(size)
    poolidx = 64 - leading_zeros(poolsize)
    length(arena.pools) < poolidx && resize!(arena.pools, poolidx)
    if !isassigned(arena.pools, poolidx)
        # TODO: Compute stuff here I guess
        nslots = 1024
        pool = Ferrite.MemoryPool{T}(nslots, poolsize, Ferrite.MemoryBlock{T}[])
        arena.pools[poolidx] = pool
    else
        pool = arena.pools[poolidx]
    end
    return malloc(pool, size)
end

function free(arena::ArenaAllocator{T}, ptr::Ptr{T}) where T
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
                return
            end
        end
    end
    error("invalid pointer: not malloc'd in this arena")
end

function realloc(arena::ArenaAllocator{T}, ptr::Ptr{T}, size::Int) where T
    for poolidx in 1:length(arena.pools)
        isassigned(arena.pools, poolidx) || continue
        pool = arena.pools[poolidx]
        nslots = pool.nslots
        slotsize = pool.slotsize
        for block in pool.blocks
            block.ptr <= ptr <= (block.ptr + nslots * slotsize * sizeof(T)) || continue
            # TODO: XXX
            idx = (ptr - block.ptr) ÷ (slotsize * sizeof(T)) + 1
            if block.freelist[idx]
                error("realloc: use after free")
            elseif size <= slotsize
                return ptr
            else
                newptr = malloc(arena, size)
                Libc.memcpy(newptr, ptr, slotsize * sizeof(T))
                block.freelist[idx] = true
                return newptr
            end
        end
    end
    error("invalid pointer: not malloc'd in this arena")
end

# function calloc(allocator::ArenaAllocator, count::Int, size::Int)
# end

# function realloc(allocator::ArenaAllocator, size::Int)
# end

# const ROWS_PER_BUFFER = 1024
# const BUFFER_GROWTH_FACTOR = 1.5

using UnsafeArrays

struct UnsafeVector
    x::UnsafeArray{Int, 1}
    size::Int # the malloc'd size
end

struct MallocDSP <: AbstractSparsityPattern
    nrows::Int
    ncols::Int
    arena::ArenaAllocator{Int}
    rows::Vector{UnsafeVector}
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
        arena = ArenaAllocator{Int}()
        rows = Vector{UnsafeVector}(undef, nrows)
        for i in 1:nrows
            ptr = malloc(arena, nnz_per_row)
            ua = UnsafeArray(ptr, (0, ))
            rows[i] = UnsafeVector(ua, nnz_per_row)
        end
        return new(nrows, ncols, arena, rows)
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

function add_entry!(dsp::MallocDSP, row::Int, col::Int)
    @boundscheck (1 <= row <= n_rows(dsp) && 1 <= col <= n_cols(dsp)) || throw(BoundsError())
    println("adding entry: $row, $col")
    r = dsp.rows[row]
    rs = r.size
    rx = r.x
    ptr = rx.pointer
    k = searchsortedfirst(rx, col)
    if k == lastindex(rx) + 1 || rx[k] != col
        # TODO: This assumes we only grow by single entry every time
        if length(rx) == rs
            rs = Ferrite.nextpow2(rs + 1)
            ptr = Ferrite.realloc(dsp.arena, rx.pointer, rs)
        else
            # @assert length(rx) < rs
            # ptr = rx.pointer
        end
        rx = UnsafeArray(ptr, (length(rx) + 1, ))
        dsp.rows[row] = UnsafeVector(rx, rs)
        # Shift elements after the insertion point to the back
        for i in (length(rx)-1):-1:k
            rx[i+1] = rx[i]
        end
        # Insert the new element
        rx[k] = col
    else
        # global hits += 1
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
    return dsp.rows[row].x
end
# @inline function rowview(dsp::MallocDSP, row::Int)
#     chunkidx, idx = divrem1(row, dsp.rows_per_chunk)
#     rowptr    = dsp.rowptr[chunkidx]
#     rowlength = dsp.rowlength[chunkidx]
#     colval    = dsp.colval[chunkidx]
#     nzrange = rowptr[idx]:rowptr[idx] + rowlength[idx] - 1
#     return view(colval, nzrange)
# end
