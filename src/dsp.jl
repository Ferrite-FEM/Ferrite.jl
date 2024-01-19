
# const ROWS_PER_BUFFER = 1024
# const BUFFER_GROWTH_FACTOR = 1.5

struct DSP <: AbstractSparsityPattern
    nrows::Int
    ncols::Int
    rows_per_chunk::Int
    growth_factor::Float64
    rowptr::Vector{Vector{Int}}
    rowlength::Vector{Vector{Int}}
    colval::Vector{Vector{Int}}
    function DSP(
            nrows::Int, ncols::Int;
            nnz_per_row::Int = 8,
            rows_per_chunk::Int = 1024,
            growth_factor::Float64 = 1.5,
        )
        n_full_buffers, n_remaining_rows = divrem(nrows, rows_per_chunk)
        # chunked rowptr
        rowptr = [collect(Int, 1:nnz_per_row:(rows_per_chunk+1)*nnz_per_row) for _ in 1:n_full_buffers]
        push!(rowptr, collect(Int, 1:nnz_per_row:(n_remaining_rows+1)*nnz_per_row))
        # chunked rowlength
        rowlength = [zeros(Int, rows_per_chunk) for _ in 1:n_full_buffers]
        push!(rowlength, zeros(Int, n_remaining_rows))
        # chunked colval
        colval = Vector{Int}[zeros(Int, nnz_per_row * rows_per_chunk) for _ in 1:n_full_buffers]
        push!(colval, zeros(Int, nnz_per_row * n_remaining_rows))
        return new(nrows, ncols, rows_per_chunk, growth_factor, rowptr, rowlength, colval)
    end
end


n_rows(dsp::DSP) = dsp.nrows
n_cols(dsp::DSP) = dsp.ncols

# hits::Int = 0
# misses::Int = 0
# buffer_growths::Int = 0



# Like mod1
function divrem1(x::Int, y::Int)
    a, b = divrem(x - 1, y)
    return a + 1, b + 1
end

function add_entry!(dsp::DSP, row::Int, col::Int)
    @boundscheck (1 <= row <= n_rows(dsp) && 1 <= col <= n_cols(dsp)) || throw(BoundsError())
    # chunkidx, idx = divrem1(row, dsp.rows_per_chunk)
    # rowptr    = dsp.rowptr[chunkidx]
    # rowlength = dsp.rowlength[chunkidx]
    # colval    = dsp.colval[chunkidx]
    # Construct the view for this row
    # rview = view(colval, rowptr[idx]:rowptr[idx] + rowlength[idx] - 1)
    rview = rowview(dsp, row)
    # Look and see if col already exist in this row
    k = searchsortedfirst(rview, col)
    if k == lastindex(rview) + 1 || col != rview[k]
        # global misses += 1
        chunkidx, idx = divrem1(row, dsp.rows_per_chunk)
        rowptr    = dsp.rowptr[chunkidx]
        rowlength = dsp.rowlength[chunkidx]
        colval    = dsp.colval[chunkidx]
        # col does not exist, need to insert it
        buflength = rowptr[idx + 1] - rowptr[idx]
        if rowlength[idx] == buflength
            # global buffer_growths += 1
            # Need to expand the storage
            n_new = max(ceil(Int, buflength * dsp.growth_factor), buflength + 1) - buflength
            Base._growat!(colval, rowptr[idx + 1], n_new)
            for i in idx+1:length(rowptr)
                rowptr[i] += n_new
            end
        end
        # Storage exist: insert and update the length
        rowlength[idx] += 1
        # rview = view(colval, rowptr[idx]:rowptr[idx] + rowlength[idx] - 1)
        rview = rowview(dsp, row)
        # Shift elements after the insertion point to the back
        for i in (length(rview)-1):-1:k
            rview[i+1] = rview[i]
        end
        # Insert the new element
        rview[k] = col
    else
        # global hits += 1
    end
    return
end

# struct RowIterator
#     colval::Vector{Int}
#     rowptr::Int
#     rowlength::Int
#     function RowIterator(dsp::DSP, row::Int)
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

# eachrow(dsp::DSP) = (RowIterator(dsp, row) for row in 1:n_rows(dsp))
# eachrow(dsp::DSP, row::Int) = RowIterator(dsp, row)

# View version
eachrow(dsp::DSP) = (eachrow(dsp, row) for row in 1:n_rows(dsp))
function eachrow(dsp::DSP, row::Int)
    return rowview(dsp, row)
end
@inline function rowview(dsp::DSP, row::Int)
    chunkidx, idx = divrem1(row, dsp.rows_per_chunk)
    rowptr    = dsp.rowptr[chunkidx]
    rowlength = dsp.rowlength[chunkidx]
    colval    = dsp.colval[chunkidx]
    nzrange = rowptr[idx]:rowptr[idx] + rowlength[idx] - 1
    return view(colval, nzrange)
end
