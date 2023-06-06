module FerriteBlockArrays

using BlockArrays: BlockArray, BlockIndex, BlockMatrix, BlockVector, block, blockaxes,
    blockindex, blocks, findblockindex, undef_blocks, Block
using Ferrite
using SparseArrays: SparseMatrixCSC
using Ferrite: addindex!, fillzero!, n_rows, n_cols, add_entry!, eachrow

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

function BlockSparsityPattern(block_sizes::Vector{Int})
    nrows = ncols = sum(block_sizes)
    nblocks = length(block_sizes)
    blocks = [SparsityPattern(block_sizes[i], block_sizes[j]) for i in 1:nblocks, j in 1:nblocks]
    return BlockSparsityPattern(nrows, ncols, block_sizes, blocks)
end

# This is the exported hook from Ferrite. Keep this signature in sync with the one in
# Ferrite (just slightly more specific).
Ferrite.BlockSparsityPattern(block_sizes::Vector{Int}) = BlockSparsityPattern(block_sizes)

let doc = """
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
 - [`create_sparsity_pattern!`](@ref): add entries corresponding to DoF couplings.
 - [`condense_sparsity_pattern!`](@ref): add entries resulting from constraints.
 - [`create_matrix`](@ref create_matrix(::Main.FerriteBlockArrays.BlockSparsityPattern)): instantiate a (block) matrix from the pattern. The default
   matrix type is `BlockMatrix{Float64, Matrix{SparseMatrixCSC{Float64, Int}}}`, i.e. a
   `BlockMatrix`, where the individual blocks are of type `SparseMatrixCSC{Float64, Int}`.

!!! note "Package extension"
    This functionality is only enabled when the package
    [BlockArrays.jl](https://github.com/JuliaArrays/BlockArrays.jl) is installed (`pkg> add
    BlockArrays`) and loaded (`using BlockArrays`) in the session.
"""
    @doc doc Ferrite.BlockSparsityPattern(::Vector{Int})
    @doc doc BlockSparsityPattern(::Vector{Int})
end

Ferrite.n_rows(bsp::BlockSparsityPattern) = bsp.nrows
Ferrite.n_cols(bsp::BlockSparsityPattern) = bsp.ncols

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

@inline function Ferrite.add_entry!(bsp::BlockSparsityPattern, row::Int, col::Int)
    @boundscheck 1 <= row <= n_rows(bsp) && 1 <= col <= n_cols(bsp)
    row_block, row_local = _find_block(bsp.block_sizes, row)
    col_block, col_local = _find_block(bsp.block_sizes, col)
    add_entry!(bsp.blocks[row_block, col_block], row_local, col_local)
    return
end

# Helper struct to iterate over the rows. Behaves similar to
# Iterators.flatten([eachrow(bsp.blocks[row_block, col_block) for col_block in 1:nblocks])
# but we need to add the offset to the iterated values.
struct RowIterator
    bsp::BlockSparsityPattern
    row::Int
    row_block::Int
    row_local::Int
    function RowIterator(bsp::BlockSparsityPattern, row::Int)
        # @info " Making for row $row"
        @assert 1 <= row <= n_rows(bsp)
        row_block, row_local = _find_block(bsp.block_sizes, row)
        return new(bsp, row, row_block, row_local)
    end
end

function Base.iterate(it::RowIterator, state = (1, 1))
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

Ferrite.eachrow(bsp::BlockSparsityPattern) = (RowIterator(bsp, row) for row in 1:n_rows(bsp))
Ferrite.eachrow(bsp::BlockSparsityPattern, row::Int) = RowIterator(bsp, row)

##############################
## Instantiating the matrix ##
##############################

"""
    create_matrix(sp::BlockSparsityPattern)

Instantiate a blocked sparse matrix from the blocked sparsity pattern `sp`.

The type of the returned matrix is `BlockMatrix{Float64, Matrix{SparseMatrixCSC{Float64,
Int}}}`, i.e. a `BlockMatrix` where the individual blocks are of type
`SparseMatrixCSC{Float64, Int}`.

This method is a shorthand for the equivalent
[`create_matrix(BlockMatrix{Float64, Matrix{SparseMatrixCSC{Float64, Int}}}, sp)`]
(@ref create_matrix(::Type{<:BlockMatrix{T, Matrix{S}}}, ::BlockSparsityPattern) where {T, S <: AbstractMatrix{T}}).

"""
function Ferrite.create_matrix(sp::BlockSparsityPattern)
    create_matrix(BlockMatrix, sp)
end

# Fill in missing matrix type, this allows create_matrix(BlockMatrix, sp)
function Ferrite.create_matrix(::Type{<:BlockMatrix}, sp::BlockSparsityPattern)
    return create_matrix(BlockMatrix{Float64, Matrix{SparseMatrixCSC{Float64, Int}}}, sp)
end

"""
    create_matrix(::Type{BlockMatrix}, sp::BlockSparsityPattern)
    create_matrix(::Type{BlockMatrix{T, Matrix{S}}}, sp::BlockSparsityPattern)

Instantiate a blocked sparse matrix from the blocked sparsity pattern `sp`.

The type of the returned matrix is a `BlockMatrix` with blocks of type `S` (defaults to
`SparseMatrixCSC{T, Int}`).

# Examples
```
# Create a sparse matrix with default block type
create_matrix(BlockMatrix, sparsity_pattern)

# Create a sparse matrix with blocks of type SparseMatrixCSC{Float32, Int}
create_matrix(BlockMatrix{Float32, Matrix{SparseMatrixCSC{Float32, Int}}}, sparsity_pattern)
```

!!! note "Package extension"
    This functionality is only enabled when the package
    [BlockArrays.jl](https://github.com/JuliaArrays/BlockArrays.jl) is installed (`pkg> add
    BlockArrays`) and loaded (`using BlockArrays`) in the session.
"""
function Ferrite.create_matrix(::Type{<:BlockMatrix{T, Matrix{S}}}, sp::BlockSparsityPattern) where {T, S <: AbstractMatrix{T}}
    @assert isconcretetype(S)
    block_sizes = sp.block_sizes
    K = BlockArray(undef_blocks, S, block_sizes, block_sizes)
    for j in 1:length(block_sizes), i in 1:length(block_sizes)
        K[Block(i), Block(j)] = create_matrix(S, sp.blocks[i, j])
    end
    return K
end


###########################################
## BlockAssembler and associated methods ##
###########################################

struct BlockAssembler{BM, Bv} <: Ferrite.AbstractSparseAssembler
    K::BM
    f::Bv
    blockindices::Vector{BlockIndex{1}}
end

Ferrite.matrix_handle(ba::BlockAssembler) = ba.K
Ferrite.vector_handle(ba::BlockAssembler) = ba.f

function Ferrite.start_assemble(K::BlockMatrix, f; fillzero::Bool=true)
    fillzero && (fillzero!(K); fillzero!(f))
    return BlockAssembler(K, f, BlockIndex{1}[])
end

# Split into the block and the local index
splindex(idx::BlockIndex{1}) = (block(idx), blockindex(idx))

function Ferrite.assemble!(assembler::BlockAssembler, dofs::AbstractVector{<:Integer}, ke::AbstractMatrix, fe::AbstractVector)
    K = assembler.K
    f = assembler.f
    blockindices = assembler.blockindices

    @assert blockaxes(K, 1) == blockaxes(K, 2)
    @assert axes(K, 1) == axes(K, 2) == axes(f, 1)
    @boundscheck checkbounds(K, dofs, dofs)
    @boundscheck checkbounds(f, dofs)

    # Update the cached the block indices
    resize!(blockindices, length(dofs))
    @inbounds for (i, I) in pairs(dofs)
        blockindices[i] = findblockindex(axes(K, 1), I)
    end

    # Assemble matrix entries
    @inbounds for (j, blockindex_j) in pairs(blockindices)
        Bj, lj = splindex(blockindex_j)
        for (i, blockindex_i) in pairs(blockindices)
            Bi, li = splindex(blockindex_i)
            KB = @view K[Bi, Bj]
            addindex!(KB, ke[i, j], li, lj)
        end
    end

    # Assemble vector entries
    if blockaxes(f, 1) == blockaxes(K, 1)
        # If f::BlockVector with the same axes the same blockindex cache can be used...
        @inbounds for (i, blockindex_i) in pairs(blockindices)
            Bi, li = splindex(blockindex_i)
            fB = @view f[Bi]
            addindex!(fB, fe[i], li)
        end
    else
        # ... otherwise, use regular indexing in fallback assemble!
        @inbounds assemble!(f, dofs, fe)
    end
    return
end

function Ferrite.apply!(::BlockMatrix, ::AbstractVector, ::ConstraintHandler)
    error(
        "Condensation of constraints with `apply!` after assembling not supported yet " *
        "for BlockMatrix, use local condensation with `apply_assemble!` instead."
    )
end


#######################################################
## Overloaded assembly pieces from src/arrayutils.jl ##
#######################################################

function Ferrite.addindex!(B::BlockMatrix{Tv}, v::Tv, i::Int, j::Int) where Tv
    @boundscheck checkbounds(B, i, j)
    Bi, li = splindex(findblockindex(axes(B, 1), i))
    Bj, lj = splindex(findblockindex(axes(B, 2), j))
    BB = @view B[Bi, Bj]
    @inbounds addindex!(BB, v, li, lj)
    return B
end

function Ferrite.fillzero!(B::Union{BlockVector,BlockMatrix})
    for blk in blocks(B)
        fillzero!(blk)
    end
    return B
end

end # module FerriteBlockArrays
