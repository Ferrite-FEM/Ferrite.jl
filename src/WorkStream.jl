# module WorkStream

using StaticArrays

function mesh_loop(
    dh::DofHandler,
    colors::Vector{Vector{Int}},
    cell_worker::CW,
    copier::CC,
    sample_scratch_data,
    sample_copy_data;
    queue_length = 2 * Threads.nthreads(),
    chunk_size = 8,
   ) where {CW<:Function, CC<:Function}

    return Main.@descend mesh_loop(dh, colors, cell_worker, copier, sample_scratch_data, sample_copy_data, queue_length, Val(chunk_size))
end

function mesh_loop(
    dh::DofHandler,
    colors::Vector{Vector{Int}},
    cell_worker::CW,
    copier::CC,
    sample_scratch_data,
    sample_copy_data,
    queue_length,
    ::Val{chunk_size},
   ) where {CW<:Function, CC<:Function, chunk_size}

    ntasks = Threads.nthreads()
    scratch_datas = [copy(sample_scratch_data) for _ in 1:ntasks]
    copy_datas = [copy(sample_copy_data) for _ in 1:ntasks]
    cell_caches = [CellCache(dh) for _ in 1:ntasks]

    for color in colors

        chunks = Channel{SVector{chunk_size,Int}}(queue_length)

        Base.Experimental.@sync begin
            # Spawn the job producer
            Threads.@spawn begin
                idx = 0
                mv = zero(MVector{chunk_size,Int})
                for cid in color
                    idx += 1
                    mv[idx] = cid
                    if idx == chunk_size
                        put!(chunks, SVector(mv))
                        idx = 0
                    end
                end
                # Finalize
                if idx != 0
                    for i in (idx+1):chunk_size
                        mv[i] = 0
                    end
                    put!(chunks, SVector(mv))
                end
                close(chunks)
            end
            # Spawn the workers
            for taskid in 1:Threads.nthreads()
                scratch_data = scratch_datas[taskid]
                copy_data = copy_datas[taskid]
                cc = cell_caches[taskid]
                Threads.@spawn begin
                    for chunk in chunks
                        for cid in chunk
                            cid == 0 && break
                            reinit!(cc, cid)
                            cell_worker(cc, scratch_data, copy_data)
                            copier(copy_data)
                        end
                    end
                end
            end
        end # @sync
    end # colors
end


# end # module WorkStream
