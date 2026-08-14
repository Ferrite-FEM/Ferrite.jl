# module WorkStream

using TaskLocalValues: TaskLocalValue
using ChunkSplitters: ChunkSplitters

# With colors
"""
    mesh_loop(
        dh::DofHandler,
        colors::Vector{Vector{Int}},
        cell_worker,
        copier,
        sample_scratch_data,
        sample_copy_data;
        ntasks = Threads.nthreads(),
        chunk_size = nothing,
    )

Loop over the cells of the mesh in the DofHandler, color by color, and execute
`cell_worker(cell_cache, scratch_data, copy_data)` followed by `copier(copy_data)` for
each cell. Since cells within a color don't share dofs it is safe to run the copier
concurrently from all worker tasks, so both worker and copier run in parallel.

`sample_scratch_data` and `sample_copy_data` are duplicated with `Base.copy` at most
`ntasks` times: the copies live in a pool from which the worker tasks check out one
entry per chunk, so the total number of copies is bounded even though new tasks are
spawned for every color.

By default each color is split statically into `ntasks` chunks (one task per chunk).
Passing `chunk_size` instead splits into chunks of that size, which gives load balancing
(one task is spawned per chunk, but the resource pool bounds the number of concurrently
allocated resources to `ntasks`).
"""
function mesh_loop(
        dh::DofHandler,
        colors::Vector{Vector{Int}},
        cell_worker::CW,
        copier::CC,
        sample_scratch_data::SSD,
        sample_copy_data::SCD;
        ntasks::Int = Threads.nthreads(),
        chunk_size::Union{Int, Nothing} = nothing,
    ) where {CW <: Function, CC <: Function, SSD, SCD}

    # Pool of task resources, bounded by ntasks. Note that task local storage would not
    # bound the number of allocated copies since fresh tasks are spawned for every color.
    cc1 = CellCache(dh)
    resources = Channel{Tuple{SSD, SCD, typeof(cc1)}}(ntasks)
    put!(resources, (copy(sample_scratch_data)::SSD, copy(sample_copy_data)::SCD, cc1))
    for _ in 2:ntasks
        put!(resources, (copy(sample_scratch_data)::SSD, copy(sample_copy_data)::SCD, CellCache(dh)))
    end

    for color in colors
        chunks = if chunk_size === nothing
            ChunkSplitters.chunks(color; n = min(ntasks, length(color)))
        else
            ChunkSplitters.chunks(color; size = chunk_size)
        end
        @sync for chunk in chunks
            Threads.@spawn begin
                resource = take!(resources)
                try
                    (scratch_data, copy_data, cc) = resource
                    # chunk is a view of color, i.e. the elements are cell ids
                    for cid in chunk
                        reinit!(cc, cid)
                        cell_worker(cc, scratch_data, copy_data)
                        copier(copy_data)
                    end
                finally
                    put!(resources, resource)
                end
            end
        end # @sync
    end # colors
    return
end

# Without colors
"""
    mesh_loop(
        dh::DofHandler,
        cell_worker,
        copier,
        sample_scratch_data,
        sample_copy_data;
        ntasks = max(1, Threads.nthreads() - 1),
        queue_length = 2 * ntasks,
        chunk_size = 2^6,
    )

Loop over all cells of the mesh in the DofHandler and execute
`cell_worker(cell_cache, scratch_data, copy_data)` in parallel on `ntasks` tasks, and
`copier(copy_data)` on a single dedicated copier task. No coloring of the mesh is
required: the copier never runs concurrently with itself so it is safe for it to mutate
shared data (e.g. assemble into the global matrix).

The copier processes the cells *in their natural order* (worker tasks hand off computed
copy data in chunks tagged with a sequence number and the copier task processes the
chunks in sequence), so the result is deterministic and identical to a serial loop.

`ntasks` defaults to `Threads.nthreads() - 1` to leave one thread for the copier task.
`sample_copy_data` is duplicated `queue_length * chunk_size` times (the batches of copy
data circulating between the workers and the copier), so `copy_data` should be kept
small; put per-task buffers in `scratch_data` instead.
"""
function mesh_loop(
        dh::DofHandler,
        cell_worker::CW,
        copier::CC,
        sample_scratch_data::SSD,
        sample_copy_data::SCD;
        ntasks::Int = max(1, Threads.nthreads() - 1),
        queue_length::Int = 2 * ntasks,
        chunk_size::Int = 2^6,
    ) where {CW <: Function, CC <: Function, SSD, SCD}

    # Task local scratch data and cell caches. Unlike the colored version the worker tasks
    # are spawned once, so task local storage allocates exactly ntasks copies.
    scratch_datas = TaskLocalValue{SSD}(() -> copy(sample_scratch_data)::SSD)
    cell_caches = TaskLocalValue{typeof(CellCache(dh))}(() -> CellCache(dh))

    # Chunks of cells, tagged with a sequence number for ordered copying
    cells = 1:getncells(dh.grid)
    chunks = ChunkSplitters.chunks(cells; size = chunk_size)
    chunk_channel = Channel{Tuple{Int, eltype(chunks)}}(queue_length; spawn = true) do ch
        for seq_chunk in enumerate(chunks)
            put!(ch, seq_chunk)
        end
    end

    # Pool of copy data batches. Workers check out one batch per chunk and hand it to the
    # copier task, which returns it to the pool. Batching amortizes the channel operations
    # over chunk_size cells (per-cell handoff makes the channels the bottleneck).
    batch_pool = Channel{Vector{SCD}}(queue_length)
    for _ in 1:queue_length
        put!(batch_pool, SCD[copy(sample_copy_data)::SCD for _ in 1:chunk_size])
    end

    # Computed batches on their way to the copier: (sequence number, #cells, batch)
    computed = Channel{Tuple{Int, Int, Vector{SCD}}}(queue_length)

    # Copier task, processing batches in sequence order (results are thus identical to a
    # serial loop). Out-of-order batches are parked in `pending` until it is their turn.
    copier_task = Threads.@spawn begin
        next = 1
        pending = Dict{Int, Tuple{Int, Vector{SCD}}}()
        for (seq, ncells, batch) in computed
            pending[seq] = (ncells, batch)
            while (item = get(pending, next, nothing); item !== nothing)
                delete!(pending, next)
                (m, b) = item
                for i in 1:m
                    copier(b[i])
                end
                put!(batch_pool, b)
                next += 1
            end
        end
    end
    # If the copier task dies, close the channels it feeds so that blocked workers error
    # out instead of hanging
    bind(computed, copier_task)
    bind(batch_pool, copier_task)

    try
        @sync for _ in 1:ntasks
            Threads.@spawn begin
                scratch_data = scratch_datas[]
                cc = cell_caches[]
                while true
                    # Note: check out the batch *before* the chunk. Otherwise the worker
                    # holding the next chunk in sequence may find the pool empty (all
                    # batches parked in the copier's reorder buffer waiting for exactly
                    # this chunk) and the pipeline deadlocks.
                    batch = take!(batch_pool)
                    local seq, chunk
                    try
                        (seq, chunk) = take!(chunk_channel)
                    catch err
                        put!(batch_pool, batch)
                        err isa InvalidStateException && break # channel closed: no more work
                        rethrow()
                    end
                    # chunk is a view of cells, i.e. the elements are cell ids
                    for (i, cid) in enumerate(chunk)
                        reinit!(cc, cid)
                        cell_worker(cc, scratch_data, batch[i])
                    end
                    put!(computed, (seq, length(chunk), batch))
                end
            end
        end # @sync
    finally
        # Closing chunk_channel kills the producer task if workers errored out early, and
        # closing computed lets the copier task finish draining and terminate.
        close(chunk_channel)
        close(computed)
    end
    wait(copier_task)
    return
end

# end # module WorkStream
