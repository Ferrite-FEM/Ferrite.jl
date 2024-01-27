# Heap implementation


# mutable struct Heap{T}
#     const buf::Vector{T}
#     chunksize::Int
# end

mutable struct MallocVector{T} <: AbstractVector{T}
    buf::Ptr{T}
    buflen::Int32
    len::Int32
end

function calloca(::Type{T}, n = 8)
    buflen = 8 * sizeof(Int)
    ptr = Libc.calloc(8, sizeof(Int))
    return new
end

function alloca()
end




# https://github.com/JuliaLang/julia/blob/2ab41059eff87625e7a6aaf1e08b75e700bc7b48/stdlib/Mmap/src/Mmap.jl#L12
const PAGESIZE = Int(Sys.isunix() ? ccall(:jl_getpagesize, Clong, ()) : ccall(:jl_getallocationgranularity, Clong, ()))


const POOLSIZE = 1024 * 1024

# A pool is a block of continuous memory that can hold 1024 rows
const ROWS_PER_POOL = 1024
const MIN_ROWS_PER_POOL = 1 << 7

# Fixed buffer size (1MiB)
# Rowsize  | size | # Rows
#       4  | 1MiB |  32768
#       8  | 1MiB |  16384
#      16  | 1MiB |   8192
#      32  | 1MiB |   4096
#      64  | 1MiB |   2048
#     128  | 1MiB |   1024
#     256  | 1MiB |    512

# Fixed number of rows (1024)
# Rowsize  |   size | # Rows
#       4  |  32KiB |   1024
#       8  |  64KiB |   1024
#      16  | 128KiB |   1024
#      32  | 256KiB |   1024
#      64  | 512KiB |   1024
#     128  |   1MiB |   1024
#     256  |   2MiB |   1024



# 8 pages
pool8 = 512 * 8 * sizeof(Int)
pool8 ÷ PAGESIZE

pool16 = 1024 * 16 * sizeof(Int)
pool16 / PAGESIZE



# struct DSP
#     nrows::Int
#     ncols::Int
#     rows::Vector{Row}
# end


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

for i in 1:(typemax(Int32)+100)
    @assert nextpow2(i) == nextpow(2, i)
end

struct MemoryBlock
    usage::BitVector
    buf::Vector{Int}
end

struct MemoryPool
    bytes_per_slot::Int
    blocks::Vector{MemoryBlock}
    function MemoryPool(bytes_per_slot, n_slots)
        # buf = zeros(Int, bytes_per_slot * n_slots ÷ sizeof(Int))
        # return new(bytes_per_slot, 0, Vector{Int}[buf])
    end
end

bytes_per_slot = 4 * sizeof(Int)
n_slots = 1024
_ndofs = 1025

struct PoolAllocator
    pools::Vector{MemoryPool}
end


struct Row
    len::Int
    poolidx::Int
end
struct DSP
    nrows::Int
    ncols::Int
    rows::Vector{Row}
    allocator::PoolAllocator
    function DSP(nrows::Int, ncols::Int)
        allocator = PoolAllocator(MemoryPool[])
        rows = Row[Row(0, i)]
    end
end


