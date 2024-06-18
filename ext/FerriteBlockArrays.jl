module FerriteBlockArrays

using BlockArrays: Block, BlockArray, BlockIndex, BlockMatrix, BlockVector, block,
    blockaxes, blockindex, blocks, findblockindex, undef_blocks
using Ferrite
using Ferrite: addindex!, fillzero!
using SparseArrays: SparseMatrixCSC


##############################
## Instantiating the matrix ##
##############################

# function Ferrite.allocate_matrix(::Type{B}, dh, ch, ...) where B <: BlockMatrix
#     # TODO: Create BSP from the induced field blocks in dh
# end

# Fill in missing matrix type, this allows allocate_matrix(BlockMatrix, sp)
function Ferrite.allocate_matrix(::Type{<:BlockMatrix}, sp::BlockSparsityPattern)
    return allocate_matrix(BlockMatrix{Float64, Matrix{SparseMatrixCSC{Float64, Int}}}, sp)
end

"""
    allocate_matrix(::Type{BlockMatrix}, sp::BlockSparsityPattern)
    allocate_matrix(::Type{BlockMatrix{T, Matrix{S}}}, sp::BlockSparsityPattern)

Instantiate a blocked sparse matrix from the blocked sparsity pattern `sp`.

The type of the returned matrix is a `BlockMatrix` with blocks of type `S` (defaults to
`SparseMatrixCSC{T, Int}`).

# Examples
```
# Create a sparse matrix with default block type
allocate_matrix(BlockMatrix, sparsity_pattern)

# Create a sparse matrix with blocks of type SparseMatrixCSC{Float32, Int}
allocate_matrix(BlockMatrix{Float32, Matrix{SparseMatrixCSC{Float32, Int}}}, sparsity_pattern)
```

!!! note "Package extension"
    This functionality is only enabled when the package
    [BlockArrays.jl](https://github.com/JuliaArrays/BlockArrays.jl) is installed (`pkg> add
    BlockArrays`) and loaded (`using BlockArrays`) in the session.
"""
function Ferrite.allocate_matrix(::Type{<:BlockMatrix{T, Matrix{S}}}, sp::BlockSparsityPattern) where {T, S <: AbstractMatrix{T}}
    @assert isconcretetype(S)
    block_sizes = sp.block_sizes
    K = BlockArray(undef_blocks, S, block_sizes, block_sizes)
    for j in 1:length(block_sizes), i in 1:length(block_sizes)
        K[Block(i), Block(j)] = allocate_matrix(S, sp.blocks[i, j])
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
