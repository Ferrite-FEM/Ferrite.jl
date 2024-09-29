module FerriteHYPRE

using Ferrite: Ferrite, ConstraintHandler, SparsityPattern
using HYPRE.LibHYPRE: @check, HYPRE_BigInt, HYPRE_Complex, HYPRE_IJMatrixAddToValues,
    HYPRE_IJMatrixSetValues, HYPRE_IJVectorAddToValues, HYPRE_Int
using HYPRE: HYPRE, HYPREMatrix, HYPREVector
using MPI: MPI

###################################
## Creating the sparsity pattern ##
###################################

function Ferrite.allocate_matrix(::Type{<:HYPREMatrix}, sp::SparsityPattern)
    # Create a new matrix
    ilower = HYPRE_BigInt(1)
    iupper = HYPRE_BigInt(Ferrite.getnrows(sp))
    @assert Ferrite.getnrows(sp) == Ferrite.getncols(sp)
    A = HYPREMatrix(MPI.COMM_SELF, ilower, iupper)
    # Add the rows, one at a time
    nrows = HYPRE_Int(1)
    ncols = HYPRE_Int[0]
    rows = HYPRE_BigInt[0]
    cols = HYPRE_BigInt[]
    values = HYPRE_Complex[]
    for (rowidx, colidxs) in zip(1:Ferrite.getnrows(sp), Ferrite.eachrow(sp))
        rows[1] = rowidx
        n = length(colidxs)
        ncols[1] = n
        resize!(cols, n)
        copyto!(cols, colidxs)
        resize!(values, n)
        fill!(values, 0)
        @check HYPRE_IJMatrixSetValues(A, nrows, ncols, rows, cols, values)
    end
    HYPRE.Internals.assemble_matrix(A)
    return A
end

############################################
### HYPREAssembler and associated methods ##
############################################

struct HYPREAssembler <: Ferrite.AbstractAssembler
    A::HYPRE.HYPREAssembler
end

Ferrite.matrix_handle(a::HYPREAssembler) = a.A.A.A # :)
Ferrite.vector_handle(a::HYPREAssembler) = a.A.b.b # :)

function Ferrite.start_assemble(K::HYPREMatrix, f::HYPREVector)
    return HYPREAssembler(HYPRE.start_assemble!(K, f))
end

function Ferrite.assemble!(a::HYPREAssembler, dofs::AbstractVector{<:Integer}, ke::AbstractMatrix, fe::AbstractVector)
    HYPRE.assemble!(a.A, dofs, ke, fe)
    return
end

function Ferrite.finish_assemble(assembler::HYPREAssembler)
    HYPRE.finish_assemble!(assembler.A)
    return
end

function Ferrite.apply!(
        ::HYPREMatrix, ::Union{HYPREVector, AbstractVector}, ::ConstraintHandler
    )
    msg = "Condensation of constraints with `apply!` after assembling not supported " *
        "for HYPREMatrix, use local condensation with `apply_assemble!` instead."
    error(msg)
end


### Methods for arrayutils.jl ##

function Ferrite.addindex!(A::HYPREMatrix, v, i::Int, j::Int)
    nrows = HYPRE_Int(1)
    ncols = Ref{HYPRE_Int}(1)
    rows = Ref{HYPRE_BigInt}(i)
    cols = Ref{HYPRE_BigInt}(j)
    values = Ref{HYPRE_Complex}(v)
    @check HYPRE_IJMatrixAddToValues(A.ijmatrix, nrows, ncols, rows, cols, values)
    return A
end

function Ferrite.addindex!(b::HYPREVector, v, i::Int)
    nvalues = HYPRE_Int(1)
    indices = Ref{HYPRE_BigInt}(i)
    values = Ref{HYPRE_Complex}(v)
    @check HYPRE_IJVectorAddToValues(b.ijvector, nvalues, indices, values)
    return b
end

end # module FerriteHYPRE
