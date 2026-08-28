module FerriteBlockArrays

using BlockArrays: Block, BlockArray, BlockIndex, BlockMatrix, BlockVector, block,
    blockaxes, blockindex, blocks, blocksize, findblockindex, undef_blocks
using Ferrite:
    Ferrite, BlockSparsityPattern, ConstraintHandler, addindex!, allocate_matrix, assemble!,
    fillzero!, DofCoefficients, _is_atomic
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

struct BlockAssembler{Tv, BM <: BlockMatrix{Tv}, Bv <: AbstractVector{Tv}, atomic} <: Ferrite.AbstractAssembler{Tv}
    K::BM
    f::Bv
    sorteddofs::Vector{Int}  # cell dofs, sorted and made local to their block
    permutation::Vector{Int} # sorted position -> index into the element matrix
    blockstops::Vector{Int}  # block -> last position in `sorteddofs` belonging to it
end

Ferrite.matrix_handle(ba::BlockAssembler) = ba.K
Ferrite.vector_handle(ba::BlockAssembler) = ba.f

# See src/assembler.jl for the `@constprop` annotation (concrete return type for literal `atomic`).
# As for the other formats, `f` may be left out to assemble the matrix only.
Base.@constprop :aggressive function Ferrite.start_assemble(K::BlockMatrix{Tv}, f::AbstractVector = Tv[]; fillzero::Bool = true, atomic::Bool = false) where {Tv}
    Ferrite._check_atomic_eltype(atomic, eltype(K))
    fillzero && (fillzero!(K); fillzero!(f))
    return BlockAssembler{eltype(K), typeof(K), typeof(f), atomic}(K, f, Int[], Int[], Int[])
end

# The global index range of block `B` along axis `d`.
blockrange(K::BlockMatrix, d::Int, B::Block{1}) = axes(K, d)[B]

# Split the (sorted) cell dofs into the per-block, block local dof lists that each block's
# scatter needs. Since the blocks partition the axis into consecutive index ranges, dofs
# sorted ascending are also grouped by block, so each block's dofs are a consecutive slice
# `blockstops[b - 1] + 1 : blockstops[b]` of `sorteddofs`. Returns the number of blocks.
function split_into_blocks!(sorteddofs::Vector{Int}, blockstops::Vector{Int}, K::BlockMatrix)
    nblocks = blocksize(K, 1)
    resize!(blockstops, nblocks)
    stop = 0
    @inbounds for b in 1:nblocks
        rng = blockrange(K, 1, Block(b))
        start = stop + 1
        stop = searchsortedlast(sorteddofs, last(rng))
        offset = first(rng) - 1
        for k in start:stop
            sorteddofs[k] -= offset
        end
        blockstops[b] = stop
    end
    return nblocks
end

# The slice of `sorteddofs` belonging to block `b`.
blockslice(blockstops::Vector{Int}, b::Int) = @inbounds (b == 1 ? 1 : blockstops[b - 1] + 1):blockstops[b]

function Ferrite.assemble!(assembler::BlockAssembler, dofs::AbstractVector{<:Integer}, ke::AbstractMatrix, fe::Union{AbstractVector, Nothing} = nothing)
    atomic = Val(_is_atomic(assembler))

    K = assembler.K
    f = assembler.f
    blockstops = assembler.blockstops

    @assert blockaxes(K, 1) == blockaxes(K, 2)
    @assert axes(K, 1) == axes(K, 2)
    @assert fe === nothing || axes(f, 1) == axes(K, 1)
    @boundscheck checkbounds(K, dofs, dofs)
    @boundscheck fe === nothing || checkbounds(f, dofs)

    sorteddofs, permutation = Ferrite._sortdofs_for_assembly!(assembler.permutation, assembler.sorteddofs, dofs)
    nblocks = split_into_blocks!(sorteddofs, blockstops, K)

    # Assemble matrix entries. Each block scatters the part of `ke` that belongs to it
    # through the same `_assemble_inner!` the unblocked assemblers use, so a block format
    # needs nothing beyond what it already implements to be assembled into as a block.
    Kblocks = blocks(K)
    @inbounds for bj in 1:nblocks
        cols = blockslice(blockstops, bj)
        isempty(cols) && continue
        coloffset = first(blockrange(K, 2, Block(bj))) - 1
        sortedcoldofs = view(sorteddofs, cols)
        colpermutation = view(permutation, cols)
        for bi in 1:nblocks
            rows = blockslice(blockstops, bi)
            isempty(rows) && continue
            rowoffset = first(blockrange(K, 1, Block(bi))) - 1
            sortedrowdofs = view(sorteddofs, rows)
            rowpermutation = view(permutation, rows)
            Ferrite._assemble_inner!(
                Kblocks[bi, bj], ke,
                sortedrowdofs, sortedrowdofs, rowpermutation,
                sortedcoldofs, sortedcoldofs, colpermutation,
                false, atomic, rowoffset, coloffset,
            )
        end
    end

    # Assemble vector entries
    if fe !== nothing
        if blockaxes(f, 1) == blockaxes(K, 1)
            # If f::BlockVector with the same axes the block splitting above can be reused...
            @inbounds for bi in 1:nblocks
                fb = view(f, Block(bi))
                for k in blockslice(blockstops, bi)
                    addindex!(fb, fe[permutation[k]], sorteddofs[k], atomic)
                end
            end
        else
            # ... otherwise, use regular indexing in fallback assemble!
            @inbounds assemble!(f, dofs, fe, atomic)
        end
    end
    return
end

Ferrite._is_atomic(::BlockAssembler{<:Any, <:Any, <:Any, atomic}) where {atomic} = atomic::Bool


########################################################
## Application of constraints (see `Ferrite.apply!`)  ##
########################################################

# `Ferrite.apply!` is generic over the matrix type; what a blocked matrix has to provide are the
# hooks below. Each is a loop over the blocks that slices the constraint data down to the block
# and forwards to the *same* interface function on the block, so this file knows nothing about
# how a block stores its entries -- supporting a new format is a matter of implementing that
# interface for it, with no change here. See the devdocs on assembly for the interface.

# `prescribed_dofs` is sorted, so the dofs inside the (contiguous) index range `rng` form a
# contiguous slice of it. Returns the indices into `prescribed_dofs`, i.e. also the indices
# into `ch.inhomogeneities`.
function prescribed_range(prescribed_dofs::AbstractVector{<:Integer}, rng::AbstractUnitRange)
    lo = searchsortedfirst(prescribed_dofs, first(rng))
    hi = searchsortedlast(prescribed_dofs, last(rng))
    return lo:hi
end

# The prescribed dofs inside the index range `rng`, renumbered to be local to it.
function local_prescribed_dofs(prescribed_dofs::AbstractVector{<:Integer}, rng::AbstractUnitRange)
    return prescribed_dofs[prescribed_range(prescribed_dofs, rng)] .- (first(rng) - 1)
end

function Ferrite.zero_out_columns!(K::BlockMatrix, ch::ConstraintHandler)
    for Bj in blockaxes(K, 2)
        cols = blockrange(K, 2, Bj)
        local_dofs = local_prescribed_dofs(ch.prescribed_dofs, cols)
        mask = view(ch.isconstrained, cols)
        isempty(local_dofs) && continue
        for Bi in blockaxes(K, 1)
            Ferrite.zero_out_columns!(@view(K[Bi, Bj]), local_dofs, mask)
        end
    end
    return
end

function Ferrite.zero_out_rows!(K::BlockMatrix, ch::ConstraintHandler)
    for Bi in blockaxes(K, 1)
        rows = blockrange(K, 1, Bi)
        local_dofs = local_prescribed_dofs(ch.prescribed_dofs, rows)
        mask = view(ch.isconstrained, rows)
        isempty(local_dofs) && continue
        for Bj in blockaxes(K, 2)
            Ferrite.zero_out_rows!(@view(K[Bi, Bj]), local_dofs, mask)
        end
    end
    return
end

# Compute "f -= K*g" block by block. The generic fallback would go through LinearAlgebra's
# scalar indexing matvec for a BlockMatrix, which is O(ndofs^2).
function Ferrite.add_inhomogeneities!(f::AbstractVector, K::BlockMatrix, columns::AbstractVector{<:Integer}, inhomogeneities::AbstractVector)
    for Bj in blockaxes(K, 2)
        cols = blockrange(K, 2, Bj)
        idxs = prescribed_range(columns, cols)
        isempty(idxs) && continue
        local_dofs = columns[idxs] .- (first(cols) - 1)
        local_inhomogeneities = view(inhomogeneities, idxs)
        for Bi in blockaxes(K, 1)
            fi = view(f, blockrange(K, 1, Bi))
            Ferrite.add_inhomogeneities!(fi, @view(K[Bi, Bj]), local_dofs, local_inhomogeneities)
        end
    end
    return f
end

# Condense K := (C' * K * C) and f := (C' * f). Each block condenses its own stored entries
# into `K`, in whichever order suits its storage; the rhs is condensed once at the end, which
# is what makes the block (and within-block) traversal order irrelevant.
function Ferrite._condense!(K::BlockMatrix, f::AbstractVector, dofcoefficients::Vector{Union{Nothing, DofCoefficients{T, Ti}}}, dofmapping::Dict{<:Integer, <:Integer}, sym::Bool = false) where {T, Ti}
    condense_f = !(length(f) == 0)
    condense_f && @assert(length(f) == size(K, 1))

    # Return early if there are no non-trivial affine constraints
    any(i -> !(i === nothing || isempty(i)), dofcoefficients) || return

    if sym
        error("condensation of ::Symmetric matrix not supported")
    end

    for Bj in blockaxes(K, 2)
        coloffset = first(blockrange(K, 2, Bj)) - 1
        for Bi in blockaxes(K, 1)
            rowoffset = first(blockrange(K, 1, Bi)) - 1
            Ferrite.condense_into!(K, @view(K[Bi, Bj]), rowoffset, coloffset, dofcoefficients, dofmapping)
        end
    end
    condense_f && Ferrite._condense_rhs_columns!(f, dofcoefficients, dofmapping)
    return
end

#######################################################
## Overloaded assembly pieces from src/arrayutils.jl ##
#######################################################

# Split a global index into the number of the block holding it and its index within that block
splindex(idx::BlockIndex{1}) = (Int(block(idx).n[1]), Int(blockindex(idx)))

function Ferrite.addindex!(B::BlockMatrix{Tv}, v::Tv, i::Int, j::Int, ::Val{atomic} = Val(false)) where {Tv, atomic}
    @boundscheck checkbounds(B, i, j)
    bi, li = splindex(findblockindex(axes(B, 1), i))
    bj, lj = splindex(findblockindex(axes(B, 2), j))
    @inbounds addindex!(blocks(B)[bi, bj], v, li, lj, Val{atomic}())
    return B
end

# Vector analogue of the method above, needed so that the global vector writes in
# `Ferrite._condense_local!` reach the atomic accumulation in the underlying block.
function Ferrite.addindex!(B::BlockVector{Tv}, v::Tv, i::Int, ::Val{atomic} = Val(false)) where {Tv, atomic}
    @boundscheck checkbounds(B, i)
    bi, li = splindex(findblockindex(axes(B, 1), i))
    @inbounds addindex!(blocks(B)[bi], v, li, Val{atomic}())
    return B
end

function Ferrite.fillzero!(B::Union{BlockVector, BlockMatrix})
    for blk in blocks(B)
        fillzero!(blk)
    end
    return B
end

end # module FerriteBlockArrays
