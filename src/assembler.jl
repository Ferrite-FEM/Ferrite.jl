abstract type AbstractSparseAssembler end

struct IJVAssembler{T} <: AbstractSparseAssembler
    I::Vector{Int}
    J::Vector{Int}
    V::Vector{T}
    f::Vector{T}
end

# Matrix-view of the assembler for matrix_handle
struct IJVMatrix{T} <: AbstractMatrix{T}
    I::Vector{Int}
    J::Vector{Int}
    V::Vector{T}
end
function addindex!(A::IJVMatrix{T}, v::T, i::Int, j::Int) where T
    iszero(v) && return A
    push!(A.I, i)
    push!(A.J, j)
    push!(A.V, v)
    return A
end
matrix_handle(a::IJVAssembler) = IJVMatrix(a.I, a.J, a.V)
vector_handle(a::IJVAssembler) = a.f

"""
    start_assemble([N=0]) -> IJVAssembler
    start_assemble(f::AbstractVector, [N=0]) -> IJVAssembler

Create an `IJVAssembler` object which can be used to assemble element contributions to the
global sparse matrix and optionally the global vector `f`. Use [`assemble!`](@ref) for each
element, and [`finish_assemble`](@ref), to finalize the assembly and return the sparse
matrix.

!!! note
    When the same matrix pattern is used multiple times (for e.g. multiple time steps or
    Newton iterations) it is more efficient to create the sparse matrix **once** and reuse
    the same pattern. See the [manual section](@ref man-assembly) on assembly.
"""
function start_assemble(f::Vector{T}, N::Int=0) where T
    I = Int[]; sizehint!(I, N)
    J = Int[]; sizehint!(J, N)
    V = T[];   sizehint!(V, N)
    return IJVAssembler(I, J, V, f)
end
start_assemble(N::Int=0) = start_assemble(Float64[], N)

"""
    assemble!(a::IJVAssembler, dofs, Ke)
    assemble!(a::IJVAssembler, dofs, Ke, fe)

Assembles the element matrix `Ke` and element vector `fe` into the assembler `a`.
"""
function assemble!(a::IJVAssembler{T}, dofs::AbstractVector{Int}, Ke::AbstractMatrix{T}, fe::Union{AbstractVector{T},Nothing}=nothing) where {T}
    if fe !== nothing
        assemble!(vector_handle(a), dofs, fe)
    end
    assemble!(a, dofs, dofs, Ke)
end

"""
    assemble!(a::IJVAssembler, rowdofs, coldofs, Ke)

Assembles the matrix `Ke` into `a` according to the dofs specified by `rowdofs` and `coldofs`.
"""
function assemble!(a::IJVAssembler{T}, rowdofs::AbstractVector{Int}, coldofs::AbstractVector{Int}, Ke::AbstractMatrix{T}) where {T}
    nrows = length(rowdofs)
    ncols = length(coldofs)

    @assert(size(Ke,1) == nrows)
    @assert(size(Ke,2) == ncols)

    append!(a.V, Ke)
    @inbounds for i in 1:ncols
        append!(a.I, rowdofs)
        for _ in 1:nrows
            push!(a.J, coldofs[i])
        end
    end
end

"""
    finish_assemble(a::IJVAssembler) -> K, f

Finalizes a `IJVAssembler` assembly. Return a sparse matrix `K :: SparseMatrixCSC` with the
assembled matrix values, and the dense vector `f :: Vector` with the assembled vector
values.

Note that the vector `f` is empty (but still returned) if the assembler was used for matrix
entries only.
"""
function finish_assemble(a::IJVAssembler)
    A = sparse(a.I, a.J, a.V)
    f = a.f
    if !(length(f) == 0 || length(f) == size(A, 1) == size(A, 2))
        error("size mismatch between matrix (size $(size(A))) and vector (length $(length(f)))")
    end
    return A, f
end

"""
    assemble!(g, dofs, ge)

Assembles the element residual `ge` into the global residual vector `g`.
"""
@propagate_inbounds function assemble!(g::AbstractVector{T}, dofs::AbstractVector{Int}, ge::AbstractVector{T}) where {T}
    @boundscheck checkbounds(g, dofs)
    @inbounds for i in 1:length(dofs)
        val = ge[i]
        if !iszero(val)
            g[dofs[i]] += val
        end
    end
end

"""
    matrix_handle(a::AbstractSparseAssembler)
    vector_handle(a::AbstractSparseAssembler)

Return a reference to the underlying matrix/vector of the assembler.
"""
matrix_handle, vector_handle

struct AssemblerSparsityPattern{Tv,Ti} <: AbstractSparseAssembler
    K::SparseMatrixCSC{Tv,Ti}
    f::Vector{Tv}
    permutation::Vector{Int}
    sorteddofs::Vector{Int}
end
struct AssemblerSymmetricSparsityPattern{Tv,Ti} <: AbstractSparseAssembler
    K::Symmetric{Tv,SparseMatrixCSC{Tv,Ti}}
    f::Vector{Tv}
    permutation::Vector{Int}
    sorteddofs::Vector{Int}
end

matrix_handle(a::AssemblerSparsityPattern) = a.K
matrix_handle(a::AssemblerSymmetricSparsityPattern) = a.K.data
vector_handle(a::Union{AssemblerSparsityPattern, AssemblerSymmetricSparsityPattern}) = a.f

"""
    start_assemble(K::SparseMatrixCSC;            fillzero::Bool=true) -> AssemblerSparsityPattern
    start_assemble(K::SparseMatrixCSC, f::Vector; fillzero::Bool=true) -> AssemblerSparsityPattern

Create a `AssemblerSparsityPattern` from the matrix `K` and optional vector `f`.

    start_assemble(K::Symmetric{SparseMatrixCSC};                 fillzero::Bool=true) -> AssemblerSymmetricSparsityPattern
    start_assemble(K::Symmetric{SparseMatrixCSC}, f::Vector=Td[]; fillzero::Bool=true) -> AssemblerSymmetricSparsityPattern

Create a `AssemblerSymmetricSparsityPattern` from the matrix `K` and optional vector `f`.

`AssemblerSparsityPattern` and `AssemblerSymmetricSparsityPattern` allocate workspace
necessary for efficient matrix assembly. To assemble the contribution from an element, use
[`assemble!`](@ref).

The keyword argument `fillzero` can be set to `false` if `K` and `f` should not be zeroed
out, but instead keep their current values.
"""
start_assemble(K::Union{SparseMatrixCSC, Symmetric{<:Any,SparseMatrixCSC}}, f::Vector; fillzero::Bool)

start_assemble(f::Vector, K::Union{SparseMatrixCSC, Symmetric}; fillzero::Bool=true) = start_assemble(K, f; fillzero=fillzero)
function start_assemble(K::SparseMatrixCSC{T}, f::Vector=T[]; fillzero::Bool=true) where {T}
    fillzero && (fillzero!(K); fillzero!(f))
    return AssemblerSparsityPattern(K, f, Int[], Int[])
end
function start_assemble(K::Symmetric{T,<:SparseMatrixCSC}, f::Vector=T[]; fillzero::Bool=true) where T
    fillzero && (fillzero!(K); fillzero!(f))
    return AssemblerSymmetricSparsityPattern(K, f, Int[], Int[])
end

"""
    assemble!(A::AbstractSparseAssembler, dofs::AbstractVector{Int}, Ke::AbstractMatrix)
    assemble!(A::AbstractSparseAssembler, dofs::AbstractVector{Int}, Ke::AbstractMatrix, fe::AbstractVector)

Assemble the element stiffness matrix `Ke` (and optional force vector `fe`) into the global
stiffness (and force) in `A`, given the element degrees of freedom `dofs`.

This is equivalent to `K[dofs, dofs] += Ke` and `f[dofs] += fe`, where `K` is the global
stiffness matrix and `f` the global force/residual vector, but more efficient.
"""
assemble!(::AbstractSparseAssembler, ::AbstractVector{Int}, ::AbstractMatrix, ::AbstractVector)

@propagate_inbounds function assemble!(A::AbstractSparseAssembler, dofs::AbstractVector{Int}, Ke::AbstractMatrix)
    assemble!(A, dofs, Ke, eltype(Ke)[])
end
@propagate_inbounds function assemble!(A::AbstractSparseAssembler, dofs::AbstractVector{Int}, fe::AbstractVector, Ke::AbstractMatrix)
    assemble!(A, dofs, Ke, fe)
end
@propagate_inbounds function assemble!(A::AssemblerSparsityPattern, dofs::AbstractVector{Int}, Ke::AbstractMatrix, fe::AbstractVector)
    _assemble!(A, dofs, Ke, fe, false)
end
@propagate_inbounds function assemble!(A::AssemblerSymmetricSparsityPattern, dofs::AbstractVector{Int}, Ke::AbstractMatrix, fe::AbstractVector)
    _assemble!(A, dofs, Ke, fe, true)
end

@propagate_inbounds function _assemble!(A::AbstractSparseAssembler, dofs::AbstractVector{Int}, Ke::AbstractMatrix, fe::AbstractVector, sym::Bool)
    ld = length(dofs)
    @assert size(Ke, 1) == ld
    @assert size(Ke, 2) == ld
    if length(fe) != 0
        @assert length(fe) == ld
        @boundscheck checkbounds(A.f, dofs)
        @inbounds assemble!(A.f, dofs, fe)
    end

    K = matrix_handle(A)
    permutation = A.permutation
    sorteddofs = A.sorteddofs
    @boundscheck checkbounds(K, dofs, dofs)
    resize!(permutation, ld)
    resize!(sorteddofs, ld)
    copyto!(sorteddofs, dofs)
    sortperm2!(sorteddofs, permutation)

    current_col = 1
    @inbounds for Kcol in sorteddofs
        maxlookups = sym ? current_col : ld
        Kecol = permutation[current_col]
        ri = 1 # row index pointer for the local matrix
        Ri = 1 # row index pointer for the global matrix
        nzr = nzrange(K, Kcol)
        while Ri <= length(nzr) && ri <= maxlookups
            R = nzr[Ri]
            Krow = K.rowval[R]
            Kerow = permutation[ri]
            val = Ke[Kerow, Kecol]
            if Krow == dofs[Kerow]
                # Match: add the value (if non-zero) and advance the pointers
                if !iszero(val)
                    K.nzval[R] += val
                end
                ri += 1
                Ri += 1
            elseif Krow < dofs[Kerow]
                # No match yet: advance the global matrix row pointer
                Ri += 1
            else # Krow > dofs[Kerow]
                # No match: no entry exist in the global matrix for this row. This is
                # allowed as long as the value which would have been inserted is zero.
                iszero(val) || error("some row indices were not found")
                # Advance the local matrix row pointer
                ri += 1
            end
        end
        # Make sure that remaining entries in this column of the local matrix are all zero
        for i in ri:maxlookups
            if !iszero(Ke[permutation[i], Kecol])
                error("some row indices were not found")
            end
        end
        current_col += 1
    end
end


## assemble! with local condensation ##

"""
    apply_assemble!(
        assembler::AbstractSparseAssembler, ch::ConstraintHandler,
        global_dofs::AbstractVector{Int},
        local_matrix::AbstractMatrix, local_vector::AbstractVector;
        apply_zero::Bool = false
    )

Assemble `local_matrix` and `local_vector` into the global system in `assembler` by first
doing constraint condensation using [`apply_local!`](@ref).

This is similar to using [`apply_local!`](@ref) followed by [`assemble!`](@ref) with the
advantage that non-local constraints can be handled, since this method can write to entries
of the global matrix and vector outside of the indices in `global_dofs`.

When the keyword argument `apply_zero` is `true` all inhomogeneities are set to `0` (cf.
[`apply!`](@ref) vs [`apply_zero!`](@ref)).

Note that this method is destructive since it modifies `local_matrix` and `local_vector`.
"""
function apply_assemble!(
        assembler::AbstractSparseAssembler, ch::ConstraintHandler,
        global_dofs::AbstractVector{Int},
        local_matrix::AbstractMatrix, local_vector::AbstractVector;
        apply_zero::Bool = false
    )
    _apply_local!(
        local_matrix, local_vector, global_dofs, ch, apply_zero,
        matrix_handle(assembler), vector_handle(assembler),
    )
    assemble!(assembler, global_dofs, local_matrix, local_vector)
    return
end


# Sort utilities

function sortperm2!(B, ii)
   @inbounds for i = 1:length(B)
      ii[i] = i
   end
   quicksort!(B, ii)
   return
end

function quicksort!(A, order, i=1,j=length(A))
    @inbounds if j > i
        if  j - i <= 12
           # Insertion sort for small groups is faster than Quicksort
           InsertionSort!(A, order, i, j)
           return A
        end

        pivot = A[div(i+j,2)]
        left, right = i, j
        while left <= right
            while A[left] < pivot
                left += 1
            end
            while A[right] > pivot
                right -= 1
            end
            if left <= right
                A[left], A[right] = A[right], A[left]
                order[left], order[right] = order[right], order[left]

                left += 1
                right -= 1
            end
        end  # left <= right

        quicksort!(A,order, i,   right)
        quicksort!(A,order, left,j)
    end  # j > i

    return A
end

function InsertionSort!(A, order, ii=1, jj=length(A))
    @inbounds for i = ii+1 : jj
        j = i - 1
        temp  = A[i]
        itemp = order[i]

        while true
            if j == ii-1
                break
            end
            if A[j] <= temp
                break
            end
            A[j+1] = A[j]
            order[j+1] = order[j]
            j -= 1
        end

        A[j+1] = temp
        order[j+1] = itemp
    end  # i
    return
end
