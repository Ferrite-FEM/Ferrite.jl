module FerriteBlockArrays

using BlockArrays: BlockArray, BlockIndex, BlockMatrix, BlockVector, block, blockaxes,
    blockindex, blocks, findblockindex
using Ferrite
using Ferrite: addindex!, fillzero!

# TODO: Move into Ferrite and enable for MixedDofHandler
function global_dof_range(dh::DofHandler, f::Symbol)
    set = Set{Int}()
    frange = dof_range(dh, f)
    for cc in CellIterator(dh)
        union!(set, @view celldofs(cc)[frange])
    end
    dofmin, dofmax = extrema(set)
    r = dofmin:dofmax
    if length(set) != length(r)
        error("renumber by blocks you donkey")
    end
    return r
end

###################################
## Creating the sparsity pattern ##
###################################

# Note:
# Creating the full unblocked matrix and then splitting into blocks inside the BlockArray
# constructor (i.e. by `getindex(::SparseMatrixCSC, ::UnitRange, ::UnitRange)`) is
# consistently faster than creating individual blocks directly. However, the latter approach
# uses less than half of the memory (measured for a 2x2 block system and various problem
# sizes), so might be useful in the future to provide an option on what algorithm to use.

# TODO: Could potentially extract the element type and matrix type for the individual blocks
#       by allowing e.g. create_sparsity_pattern(BlockMatrix{Float32}, ...) but that is not
#       even supported by regular pattern right now.
function Ferrite.create_sparsity_pattern(::Type{<:BlockMatrix}, dh, ch; kwargs...)
    K = create_sparsity_pattern(dh, ch; kwargs...)
    # Infer block sizes from the fields in the DofHandler
    block_sizes = [length(global_dof_range(dh, f)) for f in dh.field_names]
    return BlockArray(K, block_sizes, block_sizes)
end

function Ferrite.create_sparsity_pattern(B::BlockMatrix, dh, ch; kwargs...)
    if !(size(B, 1) == size(B, 2) == ndofs(dh))
        error("size of input matrix ($(size(B))) does not match number of dofs ($(ndofs(dh)))")
    end
    K = create_sparsity_pattern(dh, ch; kwargs...)
    ax = axes(B)
    for block_j in blockaxes(B, 2), block_i in blockaxes(B, 1)
        range_j = ax[2][block_j]
        range_i = ax[1][block_i]
        B[block_i, block_j] = K[range_i, range_j]
    end
    return B
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
