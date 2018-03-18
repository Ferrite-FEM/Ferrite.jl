abstract type Assembler end

struct AssemblerSparsityPattern{Tv,Ti} <: Assembler
    K::SparseMatrixCSC{Tv,Ti}
    f::Vector{Tv}
    permutation::Vector{Int}
    sorteddofs::Vector{Int}
end
struct AssemblerSymmetricSparsityPattern{Tv,Ti} <: Assembler
    K::Symmetric{Tv,SparseMatrixCSC{Tv,Ti}}
    f::Vector{Tv}
    permutation::Vector{Int}
    sorteddofs::Vector{Int}
end

@inline getsparsemat(a::AssemblerSparsityPattern) = a.K
@inline getsparsemat(a::AssemblerSymmetricSparsityPattern) = a.K.data

"""
    Assembler(K::SparseMatrixCSC, f=[]) -> AssemblerSparsityPattern
    Assembler(K::::Symmetric{SparseMatrixCSC}, f=[]) -> AssemblerSymmetricSparsityPattern

Create an appropriate `Assembler`. Return an `AssemblerSparsityPattern`
object that caches some values needed when assembling into the global
stiffness matrix and residual vector.
"""
Assembler

Assembler(f::Vector, K::Union{SparseMatrixCSC, Symmetric}) = Assembler(K, f)
function Assembler(K::SparseMatrixCSC, f::Vector=Float64[])
    fill!(K.nzval, 0.0)
    fill!(f, 0.0)
    AssemblerSparsityPattern(K, f, Int[], Int[])
end
function Assembler(K::Symmetric, f::Vector=Float64[])
    fill!(K.data.nzval, 0.0)
    fill!(f, 0.0)
    AssemblerSymmetricSparsityPattern(K, f, Int[], Int[])
end

@propagate_inbounds function assemble!(A::Assembler, dofs::AbstractVector{Int}, Ke::AbstractMatrix)
    assemble!(A, dofs, Ke, eltype(Ke)[])
end
@propagate_inbounds function assemble!(A::Assembler, dofs::AbstractVector{Int}, fe::AbstractVector, Ke::AbstractMatrix)
    assemble!(A, dofs, Ke, fe)
end
@propagate_inbounds function assemble!(A::AssemblerSparsityPattern, dofs::AbstractVector{Int}, Ke::AbstractMatrix, fe::AbstractVector)
    _assemble!(A, dofs, Ke, fe, false)
end
@propagate_inbounds function assemble!(A::AssemblerSymmetricSparsityPattern, dofs::AbstractVector{Int}, Ke::AbstractMatrix, fe::AbstractVector)
    _assemble!(A, dofs, Ke, fe, true)
end

@propagate_inbounds function _assemble!(A::Assembler, dofs::AbstractVector{Int}, Ke::AbstractMatrix, fe::AbstractVector, sym::Bool)
    if length(fe) != 0
        assemble!(A.f, dofs, fe)
    end

    K = getsparsemat(A)
    permutation = A.permutation
    sorteddofs = A.sorteddofs
    @boundscheck checkbounds(K, dofs, dofs)
    resize!(permutation, length(dofs))
    resize!(sorteddofs, length(dofs))
    copy!(sorteddofs, dofs)
    sortperm2!(sorteddofs, permutation)

    current_col = 1
    @inbounds for Kcol in sorteddofs
        maxlookups = sym ? current_col : length(dofs)
        current_idx = 1
        for r in nzrange(K, Kcol)
            Kerow = permutation[current_idx]
            if K.rowval[r] == dofs[Kerow]
                Kecol = permutation[current_col]
                K.nzval[r] += Ke[Kerow, Kecol]
                current_idx += 1
            end
            current_idx > maxlookups && break
        end
        if current_idx <= maxlookups
            error("some row indices were not found")
        end
        current_col += 1
    end
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
