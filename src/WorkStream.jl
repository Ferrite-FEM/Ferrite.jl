# module WorkStream

using StaticArrays
using TaskLocalValues: TaskLocalValue
using ChunkSplitters: ChunkSplitters

# With colors
"""
    mesh_loop(
        dh::DofHandler,
        colors::Vector{Vector{Int}},
        cell_worker::CW,
        copier::CC,
        sample_scratch_data,
        sample_copy_data;
        chunk_size = 8,
        ntasks = Threads.nthreads(),
        queue_length = 2 * ntasks,
    ) where {CW <: Function, CC <: Function}

Loop over the cells of the mesh in the DofHandler and execute `cell_worker` and then
`copier` on each cell.
"""
function mesh_loop(
        dh::DofHandler,
        colors::Vector{Vector{Int}},
        cell_worker::CW,
        copier::CC,
        sample_scratch_data::SSD,
        sample_copy_data::SCD;
        ntasks = Threads.nthreads(),
        queue_length = 2 * ntasks,
        chunk_size = 2^8, # TODO: What is a good default?
    ) where {CW <: Function, CC <: Function, SSD, SCD}

    # Create TakLocalValues from input
    scratch_datas = TaskLocalValue{SSD}(() -> copy(sample_scratch_data)::SSD)
    copy_datas = TaskLocalValue{SCD}(() -> copy(sample_copy_data)::SCD)
    cell_caches = TaskLocalValue{typeof(CellCache(dh))}(() -> CellCache(dh))

    for color in colors
        # Job producer
        # TODO: Re-compute chunk size here depending on length(color)?
        chunks = ChunkSplitters.chunks(color; size = chunk_size)
        chunk_channel = Channel{eltype(chunks)}(queue_length; spawn = true) do ch
            for chunk in chunks
                put!(ch, chunk)
            end
        end
        # Job consumers
        Base.Experimental.@sync begin
            for _ in 1:ntasks
                Threads.@spawn begin
                    scratch_data = scratch_datas[]
                    copy_data = copy_datas[]
                    cc = cell_caches[]
                    for chunk in chunk_channel
                        for idx in chunk
                            cid = color[idx]
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

# Without colors (this can't be used because ordering isn't stable and the copy task can't
# keep up...).
"""
    mesh_loop(
        dh::DofHandler,
        cell_worker::CW,
        copier::CC,
        sample_scratch_data,
        sample_copy_data;
        chunk_size = 8,
        ntasks = Threads.nthreads(),
        queue_length = 2 * ntasks,
    ) where {CW <: Function, CC <: Function}

Loop over the cells of the mesh in the DofHandler and execute `cell_worker` and then
`copier` on each cell.
"""
function mesh_loop(
        dh::DofHandler,
        cell_worker::CW,
        copier::CC,
        sample_scratch_data::SSD,
        sample_copy_data::SCD;
        ntasks = Threads.nthreads(),
        queue_length = 2 * ntasks,
        chunk_size = 2^8, # TODO: What is a good default?
    ) where {CW <: Function, CC <: Function, SSD, SCD}

    # Create TakLocalValues from input
    scratch_datas = TaskLocalValue{SSD}(() -> copy(sample_scratch_data)::SSD)
    cell_caches = TaskLocalValue{typeof(CellCache(dh))}(() -> CellCache(dh))
    # Create channel of copy data for all tasks to share
    available_copy_datas = Channel{SCD}(queue_length)
    computed_copy_datas = Channel{SCD}(queue_length)
    for _ in 1:queue_length
        put!(available_copy_datas, copy(sample_copy_data))
    end

    # Job producer
    # TODO: Re-compute chunk size here depending on length(color)?
    color = 1:getncells(dh.grid)
    chunks = ChunkSplitters.chunks(color; size = chunk_size)
    chunk_channel = Channel{eltype(chunks)}(queue_length; spawn = true) do ch
        for chunk in chunks
            put!(ch, chunk)
        end
    end
    # Copier
    copier_task = Threads.@spawn begin
        local copy_data
        for copy_data in computed_copy_datas
            copier(copy_data)
            put!(available_copy_datas, copy_data)
        end
    end
    # Job consumers
    Base.Experimental.@sync begin
        # Cell workers
        for _ in 1:ntasks
            Threads.@spawn begin
                local copy_data
                scratch_data = scratch_datas[]
                cc = cell_caches[]
                for chunk in chunk_channel
                    for idx in chunk
                        cid = color[idx]
                        reinit!(cc, cid)
                        copy_data = take!(available_copy_datas)
                        cell_worker(cc, scratch_data, copy_data)
                        put!(computed_copy_datas, copy_data)
                    end
                end
            end
        end
    end # @sync
    # We can now close the computed_copy_datas channel and wait for the copier to finish
    close(computed_copy_datas)
    wait(copier_task)
    return
end

# end # module WorkStream
