abstract type AbstractAssembler{Tv} end
abstract type AbstractCSCAssembler{Tv} <: AbstractAssembler{Tv} end
abstract type AbstractCSRAssembler{Tv} <: AbstractAssembler{Tv} end
abstract type AbstractThreadSafeAssembler{Tv} <: AbstractAssembler{Tv} end

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

"""
Assembler for sparse matrix with CSC storage type.
"""
struct CSCAssembler{Tv, Ti, MT <: AbstractSparseMatrixCSC{Tv, Ti}, atomic} <: AbstractCSCAssembler{Tv}
    K::MT
    f::Vector{Tv}
    rowpermutation::Vector{Int}
    colpermutation::Vector{Int}
    sortedrowdofs::Vector{Int}
    sortedcoldofs::Vector{Int}
end

"""
Assembler for sparse matrix with CSR storage type.
"""
struct CSRAssembler{Tv, Ti, MT <: AbstractSparseMatrix{Tv, Ti}, atomic} <: AbstractCSRAssembler{Tv} #AbstractSparseMatrixCSR does not exist
    K::MT
    f::Vector{Tv}
    rowpermutation::Vector{Int}
    colpermutation::Vector{Int}
    sortedrowdofs::Vector{Int}
    sortedcoldofs::Vector{Int}
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
    return CSCAssembler{T, Ti, typeof(K), atomic}(K, f, zeros(Int, maxcelldofs_hint), zeros(Int, maxcelldofs_hint), zeros(Int, maxcelldofs_hint), zeros(Int, maxcelldofs_hint))
end
Base.@constprop :aggressive function start_assemble(K::Symmetric{T, <:SparseMatrixCSC{T, Ti}}, f::Vector = T[]; fillzero::Bool = true, maxcelldofs_hint::Int = 0, atomic::Bool = false) where {T, Ti}
    _check_atomic_eltype(atomic, T)
    fillzero && (fillzero!(K); fillzero!(f))
    permutation = zeros(Int, maxcelldofs_hint)
    sorteddofs = zeros(Int, maxcelldofs_hint)
    return SymmetricCSCAssembler{T, Ti, typeof(K), atomic}(K, f, permutation, permutation, sorteddofs, sorteddofs)
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

# Number of stored entries per local row above which a column is searched with binary
# search instead of the linear merge walk in `_assemble_inner!`.
const SPARSE_COLUMN_SEARCH_RATIO = 8

@propagate_inbounds function _assemble_inner!(
        K::SparseMatrixCSC, Ke::AbstractMatrix,
        rowdofs::AbstractVector, sortedrowdofs::AbstractVector, rowpermutation::AbstractVector,
        coldofs::AbstractVector, sortedcoldofs::AbstractVector, colpermutation::AbstractVector,
        sym::Bool, atomic::Val = Val(false)
    )
    current_col = 1
    Krows = rowvals(K)
    Kvals = nonzeros(K)
    ld = length(rowdofs)
    nrows = size(K, 1)
    @inbounds for Kcol in sortedcoldofs
        maxlookups = sym ? current_col : ld
        Kecol = colpermutation[current_col]
        nzr = nzrange(K, Kcol)
        # Fast path for a fully dense column
        if length(nzr) == nrows
            offset = first(nzr) - 1
            for ri in 1:maxlookups
                val = Ke[rowpermutation[ri], Kecol]
                iszero(val) || addindex!(Kvals, val, offset + sortedrowdofs[ri], atomic)
            end
            current_col += 1
            continue
        end
        # Fast path for a column with many entries per local row, but not dense enough for
        # the branch above
        if length(nzr) > SPARSE_COLUMN_SEARCH_RATIO * maxlookups
            lo = first(nzr)
            hi = last(nzr)
            for ri in 1:maxlookups
                Kerow_dof = sortedrowdofs[ri]
                R = searchsortedfirst(Krows, Kerow_dof, lo, hi, Base.Order.Forward)
                if R <= hi && Krows[R] == Kerow_dof
                    val = Ke[rowpermutation[ri], Kecol]
                    iszero(val) || addindex!(Kvals, val, R, atomic)
                    lo = R + 1
                else
                    # No entry exists in the global matrix for this row, which is allowed
                    # as long as the value which would have been inserted is zero.
                    iszero(Ke[rowpermutation[ri], Kecol]) || _missing_sparsity_pattern_error(Kerow_dof, Kcol, sortedrowdofs)
                    lo = R
                end
            end
            current_col += 1
            continue
        end
        ri = 1 # row index pointer for the local matrix
        Ri = 1 # row index pointer for the global matrix
        while Ri <= length(nzr) && ri <= maxlookups
            R = nzr[Ri]
            Krow = Krows[R]
            Kerow_dof = sortedrowdofs[ri]
            if Krow == Kerow_dof
                # Match: add the value (if non-zero) and advance the pointers
                val = Ke[rowpermutation[ri], Kecol]
                if !iszero(val)
                    addindex!(Kvals, val, R, atomic)
                end
                ri += 1
                Ri += 1
            elseif Krow < Kerow_dof
                # No match yet: advance the global matrix row pointer
                Ri += 1
            else # Krow > Kerow_dof
                # No match: no entry exist in the global matrix for this row. This is
                # allowed as long as the value which would have been inserted is zero.
                iszero(Ke[rowpermutation[ri], Kecol]) || _missing_sparsity_pattern_error(Kerow_dof, Kcol, sortedrowdofs)
                # Advance the local matrix row pointer
                ri += 1
            end
        end
        # Make sure that remaining entries in this column of the local matrix are all zero
        for i in ri:maxlookups
            if !iszero(Ke[rowpermutation[i], Kecol])
                _missing_sparsity_pattern_error(sortedrowdofs[i], Kcol, sortedrowdofs)
            end
        end
        current_col += 1
    end
    return
end

function _missing_sparsity_pattern_error(Krow::Integer, Kcol::Integer, sorteddofs::Union{AbstractVector{<:Integer}, Nothing} = nothing)
    msg = "You are trying to assemble values in to K[$(Krow), $(Kcol)], but K[$(Krow), " *
        "$(Kcol)] is missing in the sparsity pattern. Make sure you have called `K = " *
        "allocate_matrix(dh)` or `K = allocate_matrix(dh, ch)` if you " *
        "have affine constraints. This error might also happen if you are using " *
        "the assembler in a threaded assembly loop (you need to create one " *
        "`assembler` for each task)."
    if sorteddofs !== nothing && _has_duplicates_sorted(sorteddofs)
        msg *= " Note: the dof index vector passed to `assemble!` contains duplicated " *
            "entries. This happens in interface assembly when a field has dofs that are " *
            "shared between the two cells (e.g. a continuous interpolation): the local " *
            "matrix must then be condensed onto the unique dofs before assembly, see " *
            "`condense_interface!`. (The reported entry may nevertheless also be " *
            "genuinely missing from the sparsity pattern.)"
    end
    throw(ErrorException(msg))
end

function _has_duplicates_sorted(sorteddofs::AbstractVector{<:Integer})
    for i in 2:length(sorteddofs)
        sorteddofs[i] == sorteddofs[i - 1] && return true
    end
    return false
end

## Interface assembly: condensation of stacked local matrices onto unique dofs ##

"""
    InterfaceAssemblyBuffer{T}(max_ndofs::Int = 0)

Scratch storage for [`condense_interface!`](@ref). The element type `T` should match the
element type of the local matrix/vector to be condensed (e.g. a dual number type when the
local matrix is produced by automatic differentiation); it is deliberately independent of
both the `InterfaceCache` and the assembler. `max_ndofs` (e.g.
[`max_nstacked_interface_dofs`](@ref)) presizes the storage; the buffer grows as needed.
"""
struct InterfaceAssemblyBuffer{T}
    Kc::Vector{T}
    fc::Vector{T}
end
function InterfaceAssemblyBuffer{T}(max_ndofs::Int = 0) where {T}
    return InterfaceAssemblyBuffer{T}(Vector{T}(undef, max_ndofs * max_ndofs), Vector{T}(undef, max_ndofs))
end

"""
    condense_interface!(buf::InterfaceAssemblyBuffer, ic::InterfaceCache, Ke::AbstractMatrix) -> (udofs, Kc)
    condense_interface!(buf::InterfaceAssemblyBuffer, ic::InterfaceCache, Ke::AbstractMatrix, fe::AbstractVector) -> (udofs, Kc, fc)

Condense a local interface matrix `Ke` (and optionally vector `fe`), computed in the
*stacked* layout of [`interfacedofs`](@ref), onto the unique interface dofs, such that the
result can be assembled with the ordinary `assemble!(assembler, udofs, Kc, fc)` (or
`apply_assemble!(assembler, ch, udofs, Kc, fc)` for constrained problems).

With `T` the map from unique to stacked dofs, this computes `Kc = Tᵀ Ke T` and
`fc = Tᵀ fe`: the two stacked copies of a dof shared between the cells are summed onto its
single unique position. Note that condensation *preserves* the local bilinear/linear form
supplied by the kernel — a kernel that weights the two copies of a shared dof incorrectly
(e.g. summing raw side values of a continuous field) is not repaired by it. See the
documentation on interface assembly for the weighting rules.

When the interface has no shared dofs (e.g. pure discontinuous interpolations) the inputs
are returned unchanged, without copying.

The returned dof vector, matrix, and vector are *borrowed* storage (from the cache and the
buffer): valid until the next `reinit!` of the cache or the next `condense_interface!`
call with the same buffer, must not be mutated, and must be copied before storing.
"""
function condense_interface!(
        buf::InterfaceAssemblyBuffer, ic::InterfaceCache,
        Ke::AbstractMatrix, fe::Union{AbstractVector, Nothing} = nothing
    )
    ns = nstacked_interface_dofs(ic)
    if size(Ke) != (ns, ns)
        throw(DimensionMismatch("size(Ke) = $(size(Ke)) does not match the stacked interface size ($ns, $ns)"))
    end
    if fe !== nothing && length(fe) != ns
        throw(DimensionMismatch("length(fe) = $(length(fe)) does not match the stacked interface size $ns"))
    end
    if !ic.any_shared
        # No shared dofs: stacked == unique, pass the inputs through without copying
        return fe === nothing ? (interfacedofs(ic), Ke) : (interfacedofs(ic), Ke, fe)
    end
    m = ic.stacked_to_unique
    nu = nunique_interface_dofs(ic)
    length(buf.Kc) < nu * nu && resize!(buf.Kc, nu * nu)
    Kc = reshape(view(buf.Kc, 1:(nu * nu)), nu, nu)
    fill!(Kc, zero(eltype(Kc)))
    @inbounds for j in 1:ns
        mj = m[j]
        for i in 1:ns
            Kc[m[i], mj] += Ke[i, j]
        end
    end
    fe === nothing && return unique_interfacedofs(ic), Kc
    length(buf.fc) < nu && resize!(buf.fc, nu)
    fc = view(buf.fc, 1:nu)
    fill!(fc, zero(eltype(fc)))
    @inbounds for i in 1:ns
        fc[m[i]] += fe[i]
    end
    return unique_interfacedofs(ic), Kc, fc
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

!!! note
    `global_dofs` must not contain duplicated entries: the constraint condensation assumes
    a one-to-one map between local indices and global dofs. For interface assembly with
    fields that share dofs between the two cells, condense the stacked local system with
    [`condense_interface!`](@ref) first and pass the result to this function.
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
