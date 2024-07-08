########################
# BlockSparsityPattern #
########################

"""
    struct BlockSparsityPattern <: AbstractSparsityPattern

Data structure representing non-zero entries for an eventual *blocked* sparse matrix.

See the constructor [`BlockSparsityPattern(::Vector{Int})`](@ref
BlockSparsityPattern(::Vector{Int})) for the user-facing documentation.

# Struct fields
 - `nrows::Int`: number of rows
 - `ncols::Int`: number of column
 - `block_sizes::Vector{Int}`: row and column block sizes
 - `blocks::Matrix{SparsityPattern}`: matrix of size `length(block_sizes) ×
   length(block_sizes)` where `blocks[i, j]` is a [`SparsityPattern`](@ref) corresponding to
   block `(i, j)`.

!!! warning "Internal struct"
    The specific implementation of this struct, such as struct fields, type layout and type
    parameters, are internal and should not be relied upon.
"""
struct BlockSparsityPattern <: AbstractSparsityPattern
    nrows::Int
    ncols::Int
    block_sizes::Vector{Int}
    blocks::Matrix{SparsityPattern}
end

"""
    BlockSparsityPattern(block_sizes::Vector{Int})

Create an empty `BlockSparsityPattern` with row and column block sizes given by
`block_sizes`.

# Examples
```julia
# Create a block sparsity pattern with block size 10 x 5
sparsity_pattern = BlockSparsityPattern([10, 5])
```

# Methods
The following methods apply to `BlockSparsityPattern` (see their respective documentation
for more details):

 - [`add_sparsity_entries!`](@ref): convenience method for calling
   [`add_cell_entries!`](@ref), [`add_interface_entries!`](@ref), and
   [`add_constraint_entries!`](@ref).
 - [`add_cell_entries!`](@ref): add entries corresponding to DoF couplings within the cells.
 - [`add_interface_entries!`](@ref): add entries corresponding to DoF couplings on the
   interface between cells.
 - [`add_constraint_entries!`](@ref): add entries resulting from constraints.
 - [`allocate_matrix`](@ref allocate_matrix(::SparsityPattern)): instantiate a (block)
   matrix from the pattern. The default matrix type is `BlockMatrix{Float64,
   Matrix{SparseMatrixCSC{Float64, Int}}}`, i.e. a `BlockMatrix`, where the individual
   blocks are of type `SparseMatrixCSC{Float64, Int}`.

!!! note "Package extension"
    This functionality is only enabled when the package
    [BlockArrays.jl](https://github.com/JuliaArrays/BlockArrays.jl) is installed (`pkg> add
    BlockArrays`) and loaded (`using BlockArrays`) in the session.
"""
function BlockSparsityPattern(blk_sizes::AbstractVector{<:Integer})
    block_sizes = collect(Int, blk_sizes)
    nrows = ncols = sum(block_sizes)
    nblocks = length(block_sizes)
    # TODO: Maybe all of these could/should share the same PoolAllocator?
    blocks = [SparsityPattern(block_sizes[i], block_sizes[j]) for i in 1:nblocks, j in 1:nblocks]
    return BlockSparsityPattern(nrows, ncols, block_sizes, blocks)
end

getnrows(bsp::BlockSparsityPattern) = bsp.nrows
getncols(bsp::BlockSparsityPattern) = bsp.ncols

# Compute block index and local index into that block
@inline function _find_block(block_sizes::Vector{Int}, i::Int)
    accumulated_block_size = 0
    block_index = 1
    while !(accumulated_block_size < i <= accumulated_block_size + block_sizes[block_index])
        accumulated_block_size += block_sizes[block_index]
        block_index += 1
    end
    local_index = i - accumulated_block_size
    return block_index, local_index
end

@inline function add_entry!(bsp::BlockSparsityPattern, row::Int, col::Int)
    @boundscheck 1 <= row <= getnrows(bsp) && 1 <= col <= getncols(bsp)
    row_block, row_local = _find_block(bsp.block_sizes, row)
    col_block, col_local = _find_block(bsp.block_sizes, col)
    add_entry!(bsp.blocks[row_block, col_block], row_local, col_local)
    return
end

# Helper struct to iterate over the rows. Behaves similar to
# Iterators.flatten([eachrow(bsp.blocks[row_block, col_block) for col_block in 1:nblocks])
# but we need to add the offset to the iterated values.
struct BSPRowIterator
    bsp::BlockSparsityPattern
    row::Int
    row_block::Int
    row_local::Int
    function BSPRowIterator(bsp::BlockSparsityPattern, row::Int)
        @assert 1 <= row <= getnrows(bsp)
        row_block, row_local = _find_block(bsp.block_sizes, row)
        return new(bsp, row, row_block, row_local)
    end
end

function Base.iterate(it::BSPRowIterator, state = (1, 1))
    col_block, idx = state
    bsp = it.bsp
    col_block > length(bsp.block_sizes) && return nothing
    block = bsp.blocks[it.row_block, col_block]
    colidxs = eachrow(block, it.row_local)
    if idx > length(colidxs)
        # Advance the col_block and reset idx to 1
        return iterate(it, (col_block + 1, 1))
    else
        # Compute global col idx and advance idx
        col_local = colidxs[idx]
        offset = sum((bsp.block_sizes[i] for i in 1:col_block-1); init = 0)
        return offset + col_local, (col_block, idx + 1)
    end
end

# TODO: eltype of the generator do not infer; might need another auxiliary struct.
eachrow(bsp::BlockSparsityPattern) = (BSPRowIterator(bsp, row) for row in 1:getnrows(bsp))
eachrow(bsp::BlockSparsityPattern, row::Int) = BSPRowIterator(bsp, row)
