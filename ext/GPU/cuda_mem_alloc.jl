struct DynamicSharedMemFunction{N, Tv <: Real, Ti <: Integer}
    mem_size::NTuple{N, Ti}
    offset::Ti
end


function (dsf::DynamicSharedMemFunction{N, Tv, Ti})() where {N, Tv, Ti}
    mem_size = dsf.mem_size
    offset = dsf.offset
    return @cuDynamicSharedMem(Tv, mem_size, offset)
end

abstract type AbstractCudaMemAlloc <: AbstractMemAlloc end

struct SharedMemAlloc{N, M, Tv <: Real, Ti <: Integer} <: AbstractCudaMemAlloc
    Ke::DynamicSharedMemFunction{N, Tv, Ti} ## block level allocation (i.e. each block will execute this function)
    fe::DynamicSharedMemFunction{M, Tv, Ti} ## block level allocation (i.e. each block will execute this function)
    tot_mem_size::Ti
end

mem_size(alloc::SharedMemAlloc) = alloc.tot_mem_size


function _can_use_dynshmem(required_shmem::Integer)
    dev = device()
    MAX_DYN_SHMEM = CUDA.attribute(dev, CUDA.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK)
    return required_shmem < MAX_DYN_SHMEM
end


function try_allocate_shared_mem(::Type{Tv}, block_dim::Ti, n_basefuncs::Ti) where {Ti <: Integer, Tv <: Real}
    shared_mem = convert(Ti, sizeof(Tv) * (block_dim) * (n_basefuncs) * n_basefuncs + sizeof(Tv) * (block_dim) * n_basefuncs)
    _can_use_dynshmem(shared_mem) || return nothing
    Ke = DynamicSharedMemFunction{3, Tv, Ti}((block_dim, n_basefuncs, n_basefuncs), convert(Ti, 0))
    fe = DynamicSharedMemFunction{2, Tv, Ti}((block_dim, n_basefuncs), convert(Ti, sizeof(Tv) * block_dim * n_basefuncs * n_basefuncs))
    return SharedMemAlloc(Ke, fe, shared_mem)
end


struct GlobalMemAlloc{LOCAL_MATRICES, LOCAL_VECTORS} <: AbstractCudaMemAlloc
    Kes::LOCAL_MATRICES ## global level allocation (i.e. memory for all blocks -> 3rd order tensor)
    fes::LOCAL_VECTORS  ## global level allocation (i.e. memory for all blocks -> 2nd order tensor)
end

cellke(alloc::GlobalMemAlloc, e::Ti) where {Ti <: Integer} = @view alloc.Kes[e, :, :]
cellfe(alloc::GlobalMemAlloc, e::Ti) where {Ti <: Integer} = @view alloc.fes[e, :]


function allocate_global_mem(::Type{Tv}, n_cells::Ti, n_basefuncs::Ti) where {Ti <: Integer, Tv <: Real}
    Kes = CUDA.zeros(Tv, n_cells, n_basefuncs, n_basefuncs)
    fes = CUDA.zeros(Tv, n_cells, n_basefuncs)
    return GlobalMemAlloc(Kes, fes)
end
