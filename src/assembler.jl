struct Assembler{T}
    I::Vector{Int}
    J::Vector{Int}
    V::Vector{T}
end

function Assembler(N)
    I = Int[]
    J = Int[]
    V = Float64[]
    sizehint!(I, N)
    sizehint!(J, N)
    sizehint!(V, N)

    Assembler(I, J, V)
end

"""
    start_assemble([N=0]) -> Assembler

Call before starting an assembly.

Returns an `Assembler` type that is used to hold the intermediate
data before an assembly is finished.
"""
function start_assemble(N::Int=0)
    return Assembler(N)
end

"""
    assemble!(a, Ke, edof)

Assembles the element matrix `Ke` into `a`.
"""
function assemble!(a::Assembler{T}, edof::AbstractVector{Int}, Ke::AbstractMatrix{T}) where {T}
    n_dofs = length(edof)
    append!(a.V, Ke)
    @inbounds for j in 1:n_dofs
        append!(a.I, edof)
        for i in 1:n_dofs
            push!(a.J, edof[j])
        end
    end
end

"""
    end_assemble(a::Assembler) -> K

Finalizes an assembly. Returns a sparse matrix with the
assembled values.
"""
function end_assemble(a::Assembler)
    return sparse(a.I, a.J, a.V)
end

"""
    assemble!(g, ge, edof)

Assembles the element residual `ge` into the global residual vector `g`.
"""
@propagate_inbounds function assemble!(g::AbstractVector{T}, edof::AbstractVector{Int}, ge::AbstractVector{T}) where {T}
    @boundscheck checkbounds(g, edof)
    @inbounds for i in 1:length(edof)
        g[edof[i]] += ge[i]
    end
end

abstract type AbstractSparseAssembler end

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

@inline getsparsemat(a::AssemblerSparsityPattern) = a.K
@inline getsparsemat(a::AssemblerSymmetricSparsityPattern) = a.K.data

start_assemble(f::Vector, K::Union{SparseMatrixCSC, Symmetric}) = start_assemble(K, f)
function start_assemble(K::SparseMatrixCSC, f::Vector=Float64[])
    fill!(K.nzval, 0.0)
    fill!(f, 0.0)
    AssemblerSparsityPattern(K, f, Int[], Int[])
end
function start_assemble(K::Symmetric, f::Vector=Float64[])
    fill!(K.data.nzval, 0.0)
    fill!(f, 0.0)
    AssemblerSymmetricSparsityPattern(K, f, Int[], Int[])
end

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


struct LinearAssembler{Tv,Ti}
    K::SparseMatrixCSC{Tv,Ti}
    dict::Dict{Tuple{Ti,Ti},Ti}
end

function LinearAssembler(dh::DofHandler)
    K, d = _create_linear_sparsity_pattern(dh,false)
    return LinearAssembler(K,d)
end

function _create_linear_sparsity_pattern(dh::DofHandler, sym::Bool)
    ncells = getncells(dh.grid)
    n = ndofs_per_cell(dh)
    N = sym ? div(n*(n+1), 2) * ncells : n^2 * ncells
    N += ndofs(dh) # always add the diagonal elements
    I = Int[]; sizehint!(I, N)
    J = Int[]; sizehint!(J, N)
    V = UInt64[]; sizehint!(V, N)
    global_dofs = zeros(Int, n)
    for element_id in 1:ncells
        celldofs!(global_dofs, dh, element_id)
        @inbounds for j in 1:n, i in 1:n
            dofi = global_dofs[i]
            dofj = global_dofs[j]
            sym && (dofi > dofj && continue)
            push!(I, dofi)
            push!(J, dofj)
            push!(V, hash((dofi, dofj)))
        end
    end
    for d in 1:ndofs(dh)
        push!(I, d)
        push!(J, d)
        push!(V, hash((d,d)))
    end
    K = sparse(I, J, V, maximum(I), maximum(J), (x,y) -> x)

    # loop again, and construct the map from (dofi, dofj)
    # to a linear index into K.nzval
    dict = Dict{Tuple{Int,Int},Int}()
    for element_id in 1:ncells
        celldofs!(global_dofs, dh, element_id)
        @inbounds for j in 1:n, i in 1:n
            dofi = global_dofs[i]
            dofj = global_dofs[j]
            sym && (dofi > dofj && continue)
            dict[hash((dofi, dofj))] = findfirst(K.nzval, hash((dofi, dofj)))
        end
    end
    for d in 1:ndofs(dh)
        dict[hash((d, d))] = findfirst(K.nzval, hash((d, d)))
    end
    W = zeros(length(I))
    return sparse(I, J, W, maximum(I), maximum(J)), dict
end

function assemble!(a::LinearAssembler, ke::AbstractMatrix, dofs::AbstractVector)
    _ndofs = length(dofs)
    @inbounds for dofj in 1:_ndofs, dofi in 1:_ndofs
        h = hash((dofs[dofi], dofs[dofj]))
        a.K.nzval[a.d[h]] += ke[dofi, dofj]
    end
end

@propagate_inbounds function getnzvalindex(A::SparseMatrixCSC{T}, i0::Integer, i1::Integer) where T
    @boundscheck !(1 <= i0 <= A.m && 1 <= i1 <= A.n) && throw(BoundsError())
    r1 = Int(A.colptr[i1])
    r2 = Int(A.colptr[i1+1]-1)
    @boundscheck (r1 > r2) && throw(ArgumentError("invalid index"))
    r1 = searchsortedfirst(A.rowval, i0, r1, r2, Forward)
    @boundscheck ((r1 > r2) || (A.rowval[r1] != i0)) && throw(ArgumentError("invalid index"))
    return r1
end
