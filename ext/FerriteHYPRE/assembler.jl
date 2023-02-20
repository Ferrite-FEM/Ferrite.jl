function Ferrite.create_sparsity_pattern(::Type{<:HYPREMatrix}, dh::Ferrite.AbstractDofHandler, ch::Union{ConstraintHandler,Nothing}=nothing; kwargs...)
    K = create_sparsity_pattern(dh, ch; kwargs...)
    fill!(K.nzval, 1)
    return HYPREMatrix(K)
end

###########################################
## HYPREAssembler and associated methods ##
###########################################

struct HYPREAssembler <: Ferrite.AbstractSparseAssembler
    A::HYPRE.HYPREAssembler
end

Ferrite.matrix_handle(a::HYPREAssembler) = a.A.A.A # :)
Ferrite.vector_handle(a::HYPREAssembler) = a.A.b.b # :)

function Ferrite.start_assemble(K::HYPREMatrix, f::HYPREVector)
    return HYPREAssembler(HYPRE.start_assemble!(K, f))
end

function Ferrite.assemble!(a::HYPREAssembler, dofs::AbstractVector{<:Integer}, ke::AbstractMatrix, fe::AbstractVector)
    HYPRE.assemble!(a.A, dofs, ke, fe)
end

function Ferrite.end_assemble(a::HYPREAssembler)
    HYPRE.finish_assemble!(a.A)
end

## Methods for arrayutils.jl ##

function Ferrite.addindex!(A::HYPREMatrix, v, i::Int, j::Int)
    nrows = HYPRE_Int(1)
    ncols = Ref{HYPRE_Int}(1)
    rows = Ref{HYPRE_BigInt}(i)
    cols = Ref{HYPRE_BigInt}(j)
    values = Ref{HYPRE_Complex}(v)
    HYPRE.@check HYPRE_IJMatrixAddToValues(A.ijmatrix, nrows, ncols, rows, cols, values)
    return A
end

function Ferrite.addindex!(b::HYPREVector, v, i::Int)
    nvalues = HYPRE_Int(1)
    indices = Ref{HYPRE_BigInt}(i)
    values = Ref{HYPRE_Complex}(v)
    HYPRE.@check HYPRE_IJVectorAddToValues(b.ijvector, nvalues, indices, values)
    return b
end