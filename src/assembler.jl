immutable Assembler{T}
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
function assemble!{T}(a::Assembler{T}, Ke::AbstractMatrix{T}, edof::AbstractVector{Int})
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
function assemble!{T}(g::AbstractVector{T}, ge::AbstractVector{T}, edof::AbstractVector{Int})
    @boundscheck checkbounds(g, edof)
    @inbounds for i in 1:length(edof)
        g[edof[i]] += ge[i]
    end
end

immutable AssemblerSparsityPattern{Tv, Ti}
    K::SparseMatrixCSC{Tv, Ti}
    f::Vector{Tv}
    tmpi::Vector{Int}
    tmpf::Vector{Tv}
end


function start_assemble(K::SparseMatrixCSC, f::Vector=Float64[])
    AssemblerSparsityPattern(K, f, Int[], eltype(K)[])
end

assemble!(A::AssemblerSparsityPattern, Ke::AbstractMatrix, dofs::AbstractVector{Int}) = assemble!(A, eltype(Ke)[], Ke, dofs)
function assemble!(A::AssemblerSparsityPattern, fe::AbstractVector, Ke::AbstractMatrix, dofs::AbstractVector{Int})
    if length(fe) != 0
        assemble!(A.f, fe, dofs)
    end

    K = A.K
    tmpi = A.tmpi
    tmpf = A.tmpf
    @boundscheck checkbounds(K, dofs, dofs)
    resize!(A.tmpi, length(dofs))
    resize!(A.tmpf, length(dofs))
    copy!(A.tmpf, dofs)
    sortperm2!(tmpf, tmpi)

    current_col = 1
    @inbounds for col in dofs
        current_idx = 1
        l = length(dofs)
        for r in nzrange(K, col)
            row = tmpi[current_idx]
            if K.rowval[r] == dofs[row]
                K.nzval[r] += Ke[row, current_col]
                current_idx += 1
            end
            current_idx > l && break
        end
        if current_idx <= l
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
end # function mysortperm

#----------------------------------------------------

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
end # function quicksort!


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
end # function InsertionSort!