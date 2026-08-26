abstract type AbstractAssembler{Tv} end
abstract type AbstractCSCAssembler{Tv} <: AbstractAssembler{Tv} end
abstract type AbstractCSRAssembler{Tv} <: AbstractAssembler{Tv} end

"""
    struct COOAssembler{Tv, Ti}

This assembler creates a COO (**coo**rdinate format) representation of a sparse matrix
during assembly and converts it into a `SparseMatrixCSC{Tv, Ti}` on finalization.
"""
struct COOAssembler{Tv, Ti} # <: AbstractAssembler{Tv}
    nrows::Int
    ncols::Int
    f::Vector{Tv}
    I::Vector{Ti}
    J::Vector{Ti}
    V::Vector{Tv}
end

function COOAssembler{Tv, Ti}(nrows::Int, ncols::Int; sizehint::Int = 0) where {Tv, Ti}
    I = Int[]
    J = Int[]
    V = Tv[]
    sizehint!(I, sizehint)
    sizehint!(J, sizehint)
    sizehint!(V, sizehint)
    f = Tv[]
    return COOAssembler{Tv, Ti}(nrows, ncols, f, I, J, V)
end

"""
    COOAssembler(nrows::Int, ncols::Int; sizehint::Int = 0)

Create a new assembler.
"""
function COOAssembler(nrows::Int, ncols::Int; sizehint::Int = 0)
    return COOAssembler{Float64, Int}(nrows, ncols; sizehint = sizehint)
end
COOAssembler(; sizehint::Int = 0) = COOAssembler(-1, -1; sizehint = sizehint)

# """
#     start_assemble(; sizehint::Int = 0) -> COOAssembler
#     start_assemble(nrows::Int, ncols::Int; sizehint::Int = 0) -> COOAssembler

# Create an `COOAssembler` which can be used to assemble element contributions to a global
# sparse matrix and vector. `nrows` is the number of rows in the final matrix and `ncols` the
# number of columns. If `nrows` and `ncols` are not passed they are inferred from the
# maximum indices added to the assembler during assembly. `sizehint` is a hint for how many
# entries in total will be added to the assembler and can be passed to optimize for
# allocations.

# Use [`assemble!`](@ref) to insert element contributions, and [`finish_assemble`](@ref), to
# finalize the assembly and return the sparse matrix (and optionally vector).

# Note that allocating a sparse matrix and assemble into it is generally preferred. See below
# and the [manual section on assembly](@ref man-assembly).

# !!! note
#     When the same matrix pattern is used multiple times (for e.g. multiple time steps or
#     Newton iterations) it is more efficient to create the sparse matrix **once** and reuse
#     the same pattern. See the [manual section](@ref man-assembly) on assembly.
# """
# function start_assemble(nrows::Int, ncols::Int; sizehint::Int = 0)
#     return COOAssembler{Float64, Int}(nrows, ncols; sizehint = sizehint)
# end

# function start_assemble(; sizehint::Int = 0)
#     return COOAssembler{Float64, Int}(-1, -1; sizehint = sizehint)
# end

"""
    assemble!(a::COOAssembler, dofs, Ke)
    assemble!(a::COOAssembler, dofs, Ke, fe)

Assembles the element matrix `Ke` and element vector `fe` into `a`.
"""
function assemble!(a::COOAssembler{T}, dofs::AbstractVector{Int}, Ke::AbstractMatrix{T}, fe::Union{AbstractVector{T}, Nothing} = nothing) where {T}
    assemble!(a, dofs, dofs, Ke)
    if fe !== nothing
        # If the final number of rows is unknown we grow the vector lazily,
        # otherwise we resize it directly to nrows.
        if a.nrows == -1
            m = maximum(dofs; init = 0)
            lf = length(a.f)
            if lf < m
                resize!(a.f, m)
                for i in (lf + 1):m
                    a.f[i] = 0
                end
            end
        elseif isempty(a.f)
            resize!(a.f, a.nrows)
            fill!(a.f, 0)
        end
        assemble!(a.f, dofs, fe)
    end
    return
end

"""
    assemble!(a::COOAssembler, rowdofs, coldofs, Ke)

Assembles the matrix `Ke` into `a` according to the dofs specified by `rowdofs` and `coldofs`.
"""
function assemble!(a::COOAssembler{T}, rowdofs::AbstractVector{Int}, coldofs::AbstractVector{Int}, Ke::AbstractMatrix{T}) where {T}
    nrows = length(rowdofs)
    ncols = length(coldofs)

    @assert(size(Ke, 1) == nrows)
    @assert(size(Ke, 2) == ncols)

    append!(a.V, Ke)
    @inbounds for i in 1:ncols
        append!(a.I, rowdofs)
        for _ in 1:nrows
            push!(a.J, coldofs[i])
        end
    end
    return
end

"""
    finish_assemble(a::COOAssembler) -> K, f

Finalize the assembly and return the sparse matrix `K::SparseMatrixCSC` and vector
`f::Vector`. If the assembler has not been used for vector assembly, `f` is an empty
vector.
"""
function finish_assemble(a::COOAssembler)
    # Create the matrix
    nrows = a.nrows == -1 ? maximum(a.I) : a.nrows
    ncols = a.ncols == -1 ? maximum(a.J) : a.ncols
    K = SparseArrays.sparse!(a.I, a.J, a.V, nrows, ncols)
    # Finalize the vector
    f = a.f
    if !isempty(f)
        # There have been things assembled, make sure it is resized correctly
        lf = length(f)
        @assert lf <= nrows
        if lf < nrows
            resize!(f, nrows)
            for i in (lf + 1):nrows
                f[i] = 0
            end
        end
    end
    return K, f
end

"""
    assemble!(g, dofs, ge, atomic = Val(false))

Assembles the element residual `ge` into the global residual vector `g`.
"""
@propagate_inbounds function assemble!(g::AbstractVector{T}, dofs::AbstractVector{Int}, ge::AbstractVector{T}, ::Val{atomic} = Val(false)) where {T, atomic}
    @boundscheck checkbounds(g, dofs)
    @boundscheck checkbounds(ge, keys(dofs))
    @inbounds for (i, dof) in pairs(dofs)
        addindex!(g, ge[i], dof, Val(atomic))
    end
    return
end

"""
    matrix_handle(a::AbstractAssembler)
    vector_handle(a::AbstractAssembler)

Return a reference to the underlying matrix/vector of the assembler used during
assembly operations.
"""
matrix_handle, vector_handle

# The `atomic` type parameter of the assemblers below is a `Bool` deciding whether the
# accumulation into the global matrix and vector uses atomic additions (see
# `start_assemble` and `addindex!`). It is a type parameter, and not a field, so that
# the atomic and non-atomic assembly paths compile to separate specializations: with a
# runtime flag the never-executed atomic code slows down the non-atomic path by 5-10%
# (LLVM neither unswitches a branch at the accumulation site out of the assembly loops,
# nor generates as good code when both paths are inlined next to each other).

# The scratch buffers used to sort the local dofs are typed by `ST`, which is `Vector{Int}`
# for the regular assemblers and `Nothing` for assemblers that cannot allocate per-instance
# buffers (GPU kernels). In the latter case `assemble!` looks up the local dofs with a
# binary search instead of sorting them first, see `_assemble_compressed_unsorted!`.

"""
Assembler for sparse matrix with CSC storage type.
"""
struct CSCAssembler{Tv, Ti, MT <: AbstractSparseMatrix{Tv, Ti}, atomic, FT <: AbstractVector{Tv}, ST <: Union{Nothing, Vector{Int}}} <: AbstractCSCAssembler{Tv}
    K::MT
    f::FT
    rowpermutation::ST
    colpermutation::ST
    sortedrowdofs::ST
    sortedcoldofs::ST
end

"""
Assembler for sparse matrix with CSR storage type.
"""
struct CSRAssembler{Tv, Ti, MT <: AbstractSparseMatrix{Tv, Ti}, atomic, FT <: AbstractVector{Tv}, ST <: Union{Nothing, Vector{Int}}} <: AbstractCSRAssembler{Tv} #AbstractSparseMatrixCSR does not exist
    K::MT
    f::FT
    rowpermutation::ST
    colpermutation::ST
    sortedrowdofs::ST
    sortedcoldofs::ST
end

"""
Assembler for symmetric sparse matrix with CSC storage type.
"""
struct SymmetricCSCAssembler{Tv, Ti, MT <: Symmetric{Tv, <:AbstractSparseMatrixCSC{Tv, Ti}}, atomic} <: AbstractCSCAssembler{Tv}
    K::MT
    f::Vector{Tv}
    rowpermutation::Vector{Int} # Symmetric assembly doesn't need separate row and
    colpermutation::Vector{Int} # col permutation and dofs, but simplifies code reuse
    sortedrowdofs::Vector{Int}  # reuse with non-symmetric cases. sortedrowdofs and
    sortedcoldofs::Vector{Int}  # rowpermutation always aliased to sortedcoldofs and colpermutation.
end

# Whether accumulation into the global matrix and vector uses atomic additions. This is a
# compile time constant (see comment above) and the fallback covers assemblers that do
# not support atomic assembly.
_is_atomic(::CSCAssembler{<:Any, <:Any, <:Any, atomic}) where {atomic} = atomic::Bool
_is_atomic(::CSRAssembler{<:Any, <:Any, <:Any, atomic}) where {atomic} = atomic::Bool
_is_atomic(::SymmetricCSCAssembler{<:Any, <:Any, <:Any, atomic}) where {atomic} = atomic::Bool
_is_atomic(::AbstractAssembler) = false

function _check_atomic_eltype(atomic::Bool, ::Type{T}) where {T}
    if atomic && !(T <: AtomicEltypes)
        throw(ArgumentError("atomic assembly is only supported for eltypes Float16, Float32, Float64, and Complex of these, got $T"))
    end
    return
end

function Base.show(io::IO, ::MIME"text/plain", a::Union{CSCAssembler, CSRAssembler, SymmetricCSCAssembler})
    print(io, typeof(a), " for assembling into:\n - ")
    summary(io, a.K)
    f = a.f
    if !isempty(f)
        print(io, "\n - ")
        summary(io, f)
    end
    return
end

matrix_handle(a::Union{AbstractCSCAssembler, AbstractCSRAssembler}) = a.K
matrix_handle(a::SymmetricCSCAssembler) = a.K.data
vector_handle(a::Union{AbstractCSCAssembler, AbstractCSRAssembler}) = a.f

"""
    start_assemble(K::AbstractSparseMatrixCSC{Tv}; fillzero = true, atomic = false) -> CSCAssembler{Tv}
    start_assemble(K::AbstractSparseMatrixCSC{Tv}, f::Vector{Tv}; fillzero = true, atomic = false) -> CSCAssembler{Tv}

Create a `CSCAssembler{Tv}` from the matrix `K` and optional vector `f` with value type `Tv`.

    start_assemble(K::Symmetric{AbstractSparseMatrixCSC{Tv}}; fillzero = true, atomic = false) -> SymmetricCSCAssembler{Tv}
    start_assemble(K::Symmetric{AbstractSparseMatrixCSC{Tv}}, f::Vector = Tv[]; fillzero = true, atomic = false) -> SymmetricCSCAssembler{Tv}

Create a `SymmetricCSCAssembler{Tv}` from the matrix `K` and optional vector `f` with value type `Tv`.

`CSCAssembler` and `SymmetricCSCAssembler` allocate workspace
necessary for efficient matrix assembly. To assemble the contribution from an element, use
[`assemble!`](@ref).

The keyword argument `fillzero` can be set to `false` if `K` and `f` should not be zeroed
out, but instead keep their current values.

The keyword argument `atomic` can be set to `true` to make the accumulation into `K` and
`f` use atomic additions. This makes it safe to assemble from multiple concurrent tasks
*without* partitioning the cells into independent sets ("grid coloring"), at the cost of
some overhead and a non-deterministic result: the order in which contributions are added
to a given entry depends on the task scheduling, and floating point addition is not
associative. Atomic accumulation is only supported for the value types `Float16`,
`Float32`, and `Float64`, and `Complex` of these (other value types throw an
`ArgumentError`). Note that each task still needs
its own assembler since the assembler contains buffers that are modified during
`assemble!`. Note also that the value of `atomic` determines a type parameter of the
returned assembler, so for a type stable setup the value should be a literal (or
otherwise a compile time constant). See the [howto on multithreaded assembly](@ref
howto-threaded-assembly) for more details.

Depending on the loaded extensions more assembly formats become available through this interface.
"""
start_assemble(K::Union{AbstractSparseMatrixCSC, Symmetric{<:Any, <:AbstractSparseMatrixCSC}}, f::Vector; fillzero::Bool)

# The `@constprop :aggressive` makes sure that a literal `atomic = true/false` keyword
# argument propagates into the `atomic` type parameter of the returned assembler, i.e.
# that the return type is concrete (the default constprop heuristics give up here).
Base.@constprop :aggressive function start_assemble(K::AbstractSparseMatrixCSC{T, Ti}, f::Vector = T[]; fillzero::Bool = true, maxcelldofs_hint::Int = 0, atomic::Bool = false) where {T, Ti}
    _check_atomic_eltype(atomic, T)
    fillzero && (fillzero!(K); fillzero!(f))
    return CSCAssembler{T, Ti, typeof(K), atomic, typeof(f), Vector{Int}}(K, f, zeros(Int, maxcelldofs_hint), zeros(Int, maxcelldofs_hint), zeros(Int, maxcelldofs_hint), zeros(Int, maxcelldofs_hint))
end
Base.@constprop :aggressive function start_assemble(K::Symmetric{T, <:SparseMatrixCSC{T, Ti}}, f::Vector = T[]; fillzero::Bool = true, maxcelldofs_hint::Int = 0, atomic::Bool = false) where {T, Ti}
    _check_atomic_eltype(atomic, T)
    fillzero && (fillzero!(K); fillzero!(f))
    permutation = zeros(Int, maxcelldofs_hint)
    sorteddofs = zeros(Int, maxcelldofs_hint)
    return SymmetricCSCAssembler{T, Ti, typeof(K), atomic}(K, f, permutation, permutation, sorteddofs, sorteddofs)
end

# Atomic accumulation on a device uses `KernelAbstractions.@atomic`, which updates a single
# array element, so there is no way to update the real and the imaginary part of a complex
# number in one atomic operation.
function _check_device_atomic_eltype(atomic::Bool, ::Type{Tv}) where {Tv}
    _check_atomic_eltype(atomic, Tv)
    if atomic && Tv <: Complex
        throw(ArgumentError("atomic assembly on device is not supported for complex value types, got $Tv"))
    end
    return
end

# Assemblers for sparse matrices living on a device, shared by the `start_assemble` methods
# of the GPU extensions. The assemblers carry no sorting scratch (`ST === Nothing`), so a
# single instance can be shared by all workers and the kernels look up every entry with a
# binary search, see `_assemble_compressed_unsorted!`. The `atomic` flag is passed as a
# `Val` so that the returned type does not depend on constant propagation into this
# function.
for AT in (:CSCAssembler, :CSRAssembler)
    @eval function _device_assembler(::Type{$AT}, K::AbstractSparseMatrix{Tv, Ti}, f::AbstractVector{Tv}, fillzero::Bool, ::Val{atomic}) where {Tv, Ti, atomic}
        _check_device_atomic_eltype(atomic, Tv)
        fillzero && (fillzero!(K); fillzero!(f))
        return $AT{Tv, Ti, typeof(K), atomic, typeof(f), Nothing}(K, f, nothing, nothing, nothing, nothing)
    end
end

function finish_assemble(a::Union{CSCAssembler, CSRAssembler, SymmetricCSCAssembler})
    return a.K, a.f
end

"""
    assemble!(A::Ferrite.AbstractAssembler, dofs::AbstractVector{Int}, Ke::AbstractMatrix)
    assemble!(A::Ferrite.AbstractAssembler, dofs::AbstractVector{Int}, Ke::AbstractMatrix, fe::AbstractVector)

Assemble the square element stiffness matrix `Ke` (and optional force vector `fe`) into the global
stiffness (and force) in `A`, given the element degrees of freedom `dofs`.

This is equivalent to `K[dofs, dofs] += Ke` and `f[dofs] += fe`, where `K` is the global stiffness matrix and `f` the global force/residual vector, but more efficient.

    assemble!(A::Ferrite.AbstractAssembler, rowdofs::AbstractVector{Int}, coldofs::AbstractVector{Int}, Ke::AbstractMatrix)
    assemble!(A::Ferrite.AbstractAssembler, rowdofs::AbstractVector{Int}, coldofs::AbstractVector{Int}, Ke::AbstractMatrix, fe::AbstractVector)

Assemble the element stiffness matrix `Ke` (and optional force vector `fe`) into the global
stiffness (and force) in `A`, given the element row degrees of freedom, `rowdofs`, and element column degrees of freedom, `coldofs`.
This is equivalent to `K[rowdofs, coldofs] += Ke` and `f[rowdofs] += fe`, but more efficient.
"""
assemble!(::AbstractAssembler, ::AbstractVector{<:Integer}, ::AbstractMatrix, ::AbstractVector)

@propagate_inbounds function assemble!(A::AbstractAssembler, dofs::AbstractVector{<:Integer}, Ke::AbstractMatrix, fe::Union{AbstractVector, Nothing} = nothing)
    size(Ke, 1) == size(Ke, 2) || throw(ArgumentError("Ke is rectangular, but only a single `dofs` vector is provided. Please call assemble!(A, rowdofs, coldofs, Ke, fe) instead."))
    return _assemble!(A, dofs, dofs, Ke, fe, false)
end
@propagate_inbounds function assemble!(A::AbstractAssembler, rowdofs::AbstractVector{<:Integer}, coldofs::AbstractVector{<:Integer}, Ke::AbstractMatrix, fe::Union{AbstractVector, Nothing} = nothing)
    return _assemble!(A, rowdofs, coldofs, Ke, fe, false)
end
@propagate_inbounds function assemble!(A::SymmetricCSCAssembler, dofs::AbstractVector{<:Integer}, Ke::AbstractMatrix, fe::Union{AbstractVector, Nothing} = nothing)
    return _assemble!(A, dofs, dofs, Ke, fe, true)
end

"""
    _sortdofs_for_assembly!(permutation::Vector{Int}, sorteddofs::Vector{Int}, dofs::AbstractVector)

Sorts the dofs into a separate buffer and returns it together with a permutation vector.
"""
@propagate_inbounds function _sortdofs_for_assembly!(permutation::Vector{Int}, sorteddofs::Vector{Int}, dofs::AbstractVector)
    ld = length(dofs)
    resize!(permutation, ld)
    resize!(sorteddofs, ld)
    copyto!(sorteddofs, dofs)
    sortperm2!(sorteddofs, permutation)
    return sorteddofs, permutation
end

@propagate_inbounds function _assemble!(A::Union{AbstractCSCAssembler, AbstractCSRAssembler}, rowdofs::AbstractVector{<:Integer}, coldofs::AbstractVector{<:Integer}, Ke::AbstractMatrix, fe::Union{AbstractVector, Nothing}, sym::Bool)
    atomic = Val(_is_atomic(A))
    @boundscheck checkbounds(Ke, keys(rowdofs), keys(coldofs))
    if fe !== nothing
        @boundscheck checkbounds(fe, keys(rowdofs))
        @boundscheck checkbounds(A.f, rowdofs)
        @inbounds assemble!(A.f, rowdofs, fe, atomic)
    end

    K = matrix_handle(A)
    @boundscheck checkbounds(K, rowdofs, coldofs)

    # Assemblers without sorting scratch (`ST === Nothing`) look up the dofs one by one
    if A.sortedcoldofs === nothing
        return _assemble_inner_unsorted!(K, Ke, rowdofs, coldofs, sym, atomic)
    end

    # We assume that the input dofs are not sorted, because the cells need the dofs in
    # a specific order, which might not be the sorted order. Hence we sort them.
    # Note that we are not allowed to mutate `dofs` in the process.
    sortedcoldofs, colpermutation = _sortdofs_for_assembly!(A.colpermutation, A.sortedcoldofs, coldofs)
    sortedrowdofs, rowpermutation = if rowdofs !== coldofs
        _sortdofs_for_assembly!(A.rowpermutation, A.sortedrowdofs, rowdofs)
    else
        sortedcoldofs, colpermutation
    end

    return _assemble_inner!(K, Ke, rowdofs, sortedrowdofs, rowpermutation, coldofs, sortedcoldofs, colpermutation, sym, atomic)
end

# Number of stored entries per local minor dof above which a major slice is searched with
# binary search instead of the linear merge walk in `_assemble_compressed!`.
const SPARSE_COLUMN_SEARCH_RATIO = 8

# The assembly kernels below operate on the raw arrays of a compressed sparse matrix and are
# phrased in terms of the *major* index -- the one whose entries are stored contiguously,
# i.e. the column for CSC and the row for CSR -- and the *minor* index -- the one stored in
# the index array, i.e. the row for CSC and the column for CSR. These singletons name that
# orientation; they are shared with the constraint application kernels.
struct MajorIsColumn end
struct MajorIsRow end

_missing_sparsity_pattern_error(::MajorIsColumn, major::Integer, minor::Integer) = _missing_sparsity_pattern_error(minor, major)
_missing_sparsity_pattern_error(::MajorIsRow, major::Integer, minor::Integer) = _missing_sparsity_pattern_error(major, minor)

"""
    _assemble_compressed!(orient, majorptr, minoridx, nzval, nminor, Ke, sortedmajordofs, majorperm, sortedminordofs, minorperm, sym, atomic)

Assemble the local matrix `Ke` into the compressed sparse matrix given by the pointer array
`majorptr` (`colptr` for CSC, `rowptr` for CSR), the index array `minoridx` (`rowvals` for
CSC, `colvals` for CSR) and the stored values `nzval`. `nminor` is the number of possible
minor indices, i.e. the number of rows for CSC and the number of columns for CSR, and
`Ke[i, j]` is the local value for the `i`th minor and the `j`th major dof.

The local dofs are expected to be sorted (`sortedmajordofs`/`sortedminordofs`) together with
the permutations mapping them back to the local indices of `Ke`. If `sym` is `true` only the
`minor <= major` triangle is assembled (the upper triangle for CSC storage).
"""
@propagate_inbounds function _assemble_compressed!(
        orient, majorptr::AbstractVector, minoridx::AbstractVector, nzval::AbstractVector, nminor::Int,
        Ke::AbstractMatrix,
        sortedmajordofs::AbstractVector, majorperm::AbstractVector,
        sortedminordofs::AbstractVector, minorperm::AbstractVector,
        sym::Bool, atomic::Val = Val(false)
    )
    current_major = 1
    ld = length(sortedminordofs)
    @inbounds for Kmajor in sortedmajordofs
        maxlookups = sym ? current_major : ld
        Kemajor = majorperm[current_major]
        nzr = majorptr[Kmajor]:(majorptr[Kmajor + 1] - 1)
        # Fast path for a fully dense major slice
        if length(nzr) == nminor
            offset = first(nzr) - 1
            for mi in 1:maxlookups
                val = Ke[minorperm[mi], Kemajor]
                iszero(val) || addindex!(nzval, val, offset + sortedminordofs[mi], atomic)
            end
            current_major += 1
            continue
        end
        # Fast path for a major slice with many entries per local minor dof, but not dense
        # enough for the branch above
        if length(nzr) > SPARSE_COLUMN_SEARCH_RATIO * maxlookups
            lo = first(nzr)
            hi = last(nzr)
            for mi in 1:maxlookups
                Keminor_dof = sortedminordofs[mi]
                R = searchsortedfirst(minoridx, Keminor_dof, lo, hi, Base.Order.Forward)
                if R <= hi && minoridx[R] == Keminor_dof
                    val = Ke[minorperm[mi], Kemajor]
                    iszero(val) || addindex!(nzval, val, R, atomic)
                    lo = R + 1
                else
                    # No entry exists in the global matrix for this minor index, which is
                    # allowed as long as the value which would have been inserted is zero.
                    iszero(Ke[minorperm[mi], Kemajor]) || _missing_sparsity_pattern_error(orient, Kmajor, Keminor_dof)
                    lo = R
                end
            end
            current_major += 1
            continue
        end
        mi = 1 # minor index pointer for the local matrix
        Mi = 1 # minor index pointer for the global matrix
        while Mi <= length(nzr) && mi <= maxlookups
            R = nzr[Mi]
            Kminor = minoridx[R]
            Keminor_dof = sortedminordofs[mi]
            if Kminor == Keminor_dof
                # Match: add the value (if non-zero) and advance the pointers
                val = Ke[minorperm[mi], Kemajor]
                if !iszero(val)
                    addindex!(nzval, val, R, atomic)
                end
                mi += 1
                Mi += 1
            elseif Kminor < Keminor_dof
                # No match yet: advance the global matrix pointer
                Mi += 1
            else # Kminor > Keminor_dof
                # No match: no entry exist in the global matrix for this minor index. This
                # is allowed as long as the value which would have been inserted is zero.
                iszero(Ke[minorperm[mi], Kemajor]) || _missing_sparsity_pattern_error(orient, Kmajor, Keminor_dof)
                # Advance the local matrix pointer
                mi += 1
            end
        end
        # Make sure that remaining entries in this slice of the local matrix are all zero
        for i in mi:maxlookups
            if !iszero(Ke[minorperm[i], Kemajor])
                _missing_sparsity_pattern_error(orient, Kmajor, sortedminordofs[i])
            end
        end
        current_major += 1
    end
    return
end

"""
    _assemble_compressed_unsorted!(orient, majorptr, minoridx, nzval, Ke, majordofs, minordofs, sym, atomic)

Scratch-free variant of [`_assemble_compressed!`](@ref) taking the local dofs in the order
given by the cell: every stored local value is looked up with a binary search in the minor
index array. Allocation- and recursion-free so that it can be called from a GPU kernel.
"""
@propagate_inbounds function _assemble_compressed_unsorted!(
        orient, majorptr::AbstractVector, minoridx::AbstractVector, nzval::AbstractVector,
        Ke::AbstractMatrix, majordofs::AbstractVector, minordofs::AbstractVector,
        sym::Bool, atomic::Val = Val(false)
    )
    @inbounds for j in eachindex(majordofs)
        majordof = majordofs[j]
        # `Int` conversion so that the two bounds have the same type also for matrices with
        # a narrow index type (`searchsortedfirst` requires that).
        lo = Int(majorptr[majordof])
        hi = Int(majorptr[majordof + 1]) - 1
        for i in eachindex(minordofs)
            minordof = minordofs[i]
            # Symmetric assembly only stores the `minor <= major` triangle
            sym && minordof > majordof && continue
            val = Ke[i, j]
            iszero(val) && continue
            k = searchsortedfirst(minoridx, minordof, lo, hi, Base.Order.Forward)
            if k <= hi && minoridx[k] == minordof
                addindex!(nzval, val, k, atomic)
            else
                _missing_sparsity_pattern_error(orient, majordof, minordof)
            end
        end
    end
    return
end

@propagate_inbounds function _assemble_inner!(
        K::SparseMatrixCSC, Ke::AbstractMatrix,
        rowdofs::AbstractVector, sortedrowdofs::AbstractVector, rowpermutation::AbstractVector,
        coldofs::AbstractVector, sortedcoldofs::AbstractVector, colpermutation::AbstractVector,
        sym::Bool, atomic::Val = Val(false)
    )
    return _assemble_compressed!(
        MajorIsColumn(), SparseArrays.getcolptr(K), rowvals(K), nonzeros(K), size(K, 1), Ke,
        sortedcoldofs, colpermutation, sortedrowdofs, rowpermutation, sym, atomic
    )
end

@propagate_inbounds function _assemble_inner_unsorted!(
        K::SparseMatrixCSC, Ke::AbstractMatrix,
        rowdofs::AbstractVector, coldofs::AbstractVector,
        sym::Bool, atomic::Val = Val(false)
    )
    return _assemble_compressed_unsorted!(
        MajorIsColumn(), SparseArrays.getcolptr(K), rowvals(K), nonzeros(K), Ke,
        coldofs, rowdofs, sym, atomic
    )
end

function _missing_sparsity_pattern_error(Krow::Integer, Kcol::Integer)
    msg = "You are trying to assemble values in to K[$(Krow), $(Kcol)], but K[$(Krow), " *
        "$(Kcol)] is missing in the sparsity pattern. Make sure you have called `K = " *
        "allocate_matrix(dh)` or `K = allocate_matrix(dh, ch)` if you " *
        "have affine constraints. This error might also happen if you are using " *
        "the assembler in a threaded assembly loop (you need to create one " *
        "`assembler` for each task)."
    throw(ErrorException(msg))
end

## assemble! with local condensation ##

"""
    apply_assemble!(
        assembler::AbstractAssembler, ch::ConstraintHandler,
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
        assembler::AbstractAssembler, ch::ConstraintHandler,
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

"""
    sortperm2!(data::AbstractVector, permutation::AbstractVector)

Sort the input vector inplace and compute the corresponding permutation.
"""
function sortperm2!(B, ii)
    @inbounds for i in 1:length(B)
        ii[i] = i
    end
    quicksort!(B, ii)
    return
end

function quicksort!(A, order, i = 1, j = length(A))
    @inbounds if j > i
        if j - i <= 12
            # Insertion sort for small groups is faster than Quicksort
            InsertionSort!(A, order, i, j)
            return A
        end

        pivot = A[div(i + j, 2)]
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

        quicksort!(A, order, i, right)
        quicksort!(A, order, left, j)
    end  # j > i

    return A
end

function InsertionSort!(A, order, ii = 1, jj = length(A))
    @inbounds for i in (ii + 1):jj
        j = i - 1
        temp = A[i]
        itemp = order[i]

        while true
            if j == ii - 1
                break
            end
            if A[j] <= temp
                break
            end
            A[j + 1] = A[j]
            order[j + 1] = order[j]
            j -= 1
        end

        A[j + 1] = temp
        order[j + 1] = itemp
    end  # i
    return
end
