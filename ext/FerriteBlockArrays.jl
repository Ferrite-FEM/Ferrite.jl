module FerriteBlockArrays

using BlockArrays: BlockArray, BlockIndex, BlockMatrix, BlockVector, block, blockaxes,
    blockindex, blocks, findblockindex
using BlockArrays: Block, BlockArray, BlockIndex, BlockMatrix, BlockVector, block,
    blockaxes, blockindex, blocks, findblockindex, undef_blocks
using Ferrite:
    Ferrite, BlockSparsityPattern, ConstraintHandler, addindex!, allocate_matrix, assemble!,
    fillzero!, DofCoefficients, _add_inhomogeneities_cols!, _condense_column!,
    _condense_rhs_column!, _is_atomic, _zero_out_columns!, _zero_out_rows!,
    coefficients_for_dof
using SparseArrays: AbstractSparseMatrixCSC, SparseMatrixCSC


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

struct BlockAssembler{Tv, BM <: BlockMatrix{Tv}, Bv <: AbstractVector{Tv}, atomic} <: Ferrite.AbstractAssembler{Tv}
    K::BM
    f::Bv
    blockindices::Vector{BlockIndex{1}}
end

Ferrite.matrix_handle(ba::BlockAssembler) = ba.K
Ferrite.vector_handle(ba::BlockAssembler) = ba.f

# See src/assembler.jl for the `@constprop` annotation (concrete return type for literal `atomic`).
Base.@constprop :aggressive function Ferrite.start_assemble(K::BlockMatrix, f; fillzero::Bool = true, atomic::Bool = false)
    Ferrite._check_atomic_eltype(atomic, eltype(K))
    fillzero && (fillzero!(K); fillzero!(f))
    return BlockAssembler{eltype(K), typeof(K), typeof(f), atomic}(K, f, BlockIndex{1}[])
end

# Split into the block and the local index
splindex(idx::BlockIndex{1}) = (block(idx), blockindex(idx))

function Ferrite.assemble!(assembler::BlockAssembler, dofs::AbstractVector{<:Integer}, ke::AbstractMatrix, fe::AbstractVector)
    atomic = Val(_is_atomic(assembler))

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
            addindex!(KB, ke[i, j], li, lj, atomic)
        end
    end

    # Assemble vector entries
    if blockaxes(f, 1) == blockaxes(K, 1)
        # If f::BlockVector with the same axes the same blockindex cache can be used...
        @inbounds for (i, blockindex_i) in pairs(blockindices)
            Bi, li = splindex(blockindex_i)
            fB = @view f[Bi]
            addindex!(fB, fe[i], li, atomic)
        end
    else
        # ... otherwise, use regular indexing in fallback assemble!
        @inbounds assemble!(f, dofs, fe, atomic)
    end
    return
end

Ferrite._is_atomic(::BlockAssembler{<:Any, <:Any, <:Any, atomic}) where {atomic} = atomic::Bool


########################################################
## Application of constraints (see `Ferrite.apply!`)  ##
########################################################

# `Ferrite.apply!` itself is generic over the matrix type; what a blocked matrix has to
# provide are the hooks below. Every block is a sparse matrix in CSC storage (that is what
# `allocate_matrix` produces above), so each hook is a loop over the blocks that delegates to
# the corresponding Ferrite kernel with block local indices. Blocks in another storage format
# fall back to Ferrite's generic methods, which error rather than silently running a dense
# algorithm.
const SparseBlockMatrix = BlockMatrix{<:Any, <:AbstractMatrix{<:AbstractSparseMatrixCSC}}

# The global index range of block `B` along axis `d`.
blockrange(K::BlockMatrix, d::Int, B::Block{1}) = axes(K, d)[B]

# `prescribed_dofs` is sorted, so the dofs inside the (contiguous) index range `rng` form a
# contiguous slice of it. Returns the indices into `prescribed_dofs`, i.e. also the indices
# into `ch.inhomogeneities`.
function prescribed_range(prescribed_dofs::AbstractVector{<:Integer}, rng::AbstractUnitRange)
    lo = searchsortedfirst(prescribed_dofs, first(rng))
    hi = searchsortedlast(prescribed_dofs, last(rng))
    return lo:hi
end

function Ferrite.zero_out_columns!(K::SparseBlockMatrix, ch::ConstraintHandler)
    for Bj in blockaxes(K, 2)
        cols = blockrange(K, 2, Bj)
        idxs = prescribed_range(ch.prescribed_dofs, cols)
        isempty(idxs) && continue
        local_dofs = ch.prescribed_dofs[idxs] .- (first(cols) - 1)
        for Bi in blockaxes(K, 1)
            _zero_out_columns!(@view(K[Bi, Bj]), local_dofs)
        end
    end
    return
end

function Ferrite.zero_out_rows!(K::SparseBlockMatrix, ch::ConstraintHandler)
    for Bi in blockaxes(K, 1)
        isconstrained = view(ch.isconstrained, blockrange(K, 1, Bi))
        for Bj in blockaxes(K, 2)
            _zero_out_rows!(@view(K[Bi, Bj]), isconstrained)
        end
    end
    return
end

# Compute "f -= K*inhomogeneities" block by block. The generic fallback would go through
# LinearAlgebra's scalar indexing matvec for a BlockMatrix, which is O(ndofs^2).
function Ferrite.add_inhomogeneities!(f::AbstractVector, K::SparseBlockMatrix, ch::ConstraintHandler)
    for Bj in blockaxes(K, 2)
        cols = blockrange(K, 2, Bj)
        idxs = prescribed_range(ch.prescribed_dofs, cols)
        isempty(idxs) && continue
        local_dofs = ch.prescribed_dofs[idxs] .- (first(cols) - 1)
        inhomogeneities = view(ch.inhomogeneities, idxs)
        for Bi in blockaxes(K, 1)
            fi = view(f, blockrange(K, 1, Bi))
            _add_inhomogeneities_cols!(fi, @view(K[Bi, Bj]), local_dofs, inhomogeneities, false)
        end
    end
    return f
end

# Condense K := (C' * K * C) and f := (C' * f). The blocks are visited in the order of the
# global dofs, so the entries are traversed exactly like in the `SparseMatrixCSC` version.
function Ferrite._condense!(K::SparseBlockMatrix, f::AbstractVector, dofcoefficients::Vector{Union{Nothing, DofCoefficients{T, Ti}}}, dofmapping::Dict{<:Integer, <:Integer}, sym::Bool = false) where {T, Ti}
    condense_f = !(length(f) == 0)
    condense_f && @assert(length(f) == size(K, 1))

    # Return early if there are no non-trivial affine constraints
    any(i -> !(i === nothing || isempty(i)), dofcoefficients) || return

    if sym
        error("condensation of ::Symmetric matrix not supported")
    end

    for Bj in blockaxes(K, 2)
        cols = blockrange(K, 2, Bj)
        coloffset = first(cols) - 1
        for localcol in 1:length(cols)
            col_coeffs = coefficients_for_dof(dofmapping, dofcoefficients, localcol + coloffset)
            for Bi in blockaxes(K, 1)
                rowoffset = first(blockrange(K, 1, Bi)) - 1
                _condense_column!(
                    K, @view(K[Bi, Bj]), localcol, rowoffset, coloffset,
                    col_coeffs, dofcoefficients, dofmapping,
                )
            end
            if col_coeffs !== nothing && condense_f
                _condense_rhs_column!(f, localcol + coloffset, col_coeffs)
            end
        end
    end
    return
end

#######################################################
## Overloaded assembly pieces from src/arrayutils.jl ##
#######################################################

function Ferrite.addindex!(B::BlockMatrix{Tv}, v::Tv, i::Int, j::Int, ::Val{atomic} = Val(false)) where {Tv, atomic}
    @boundscheck checkbounds(B, i, j)
    Bi, li = splindex(findblockindex(axes(B, 1), i))
    Bj, lj = splindex(findblockindex(axes(B, 2), j))
    BB = @view B[Bi, Bj]
    @inbounds addindex!(BB, v, li, lj, Val{atomic}())
    return B
end

# Vector analogue of the method above, needed so that the global vector writes in
# `Ferrite._condense_local!` reach the atomic accumulation in the underlying block.
function Ferrite.addindex!(B::BlockVector{Tv}, v::Tv, i::Int, ::Val{atomic} = Val(false)) where {Tv, atomic}
    @boundscheck checkbounds(B, i)
    Bi, li = splindex(findblockindex(axes(B, 1), i))
    BB = @view B[Bi]
    @inbounds addindex!(BB, v, li, Val{atomic}())
    return B
end

function Ferrite.fillzero!(B::Union{BlockVector, BlockMatrix})
    for blk in blocks(B)
        fillzero!(blk)
    end
    return B
end

end # module FerriteBlockArrays
