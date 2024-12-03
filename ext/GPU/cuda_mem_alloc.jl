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

struct GlobalMemAlloc{LOCAL_MATRICES, LOCAL_VECTORS} <: AbstractCudaMemAlloc
    Kes::LOCAL_MATRICES ## global level allocation (i.e. memory for all blocks -> 3rd order tensor)
    fes::LOCAL_VECTORS  ## global level allocation (i.e. memory for all blocks -> 2nd order tensor)
end

cellke(alloc::GlobalMemAlloc, e::Ti) where {Ti <: Integer} = @view alloc.Kes[e, :, :]
cellfe(alloc::GlobalMemAlloc, e::Ti) where {Ti <: Integer} = @view alloc.fes[e, :]
