import Ferrite: ConstraintHandler

# -------------------- adapt ----------------------------------------

"""
    DeviceConstraintHandler{Tv, Ti, PD, IH, IP}

A flattened, device-friendly representation of a [`ConstraintHandler`](@ref),
suitable for use in GPU kernels. Create it by `adapt`-ing a closed
`ConstraintHandler` to a KernelAbstractions backend.

# Fields
- `prescribed_dofs`: indices of the prescribed (constrained) DOFs
- `inhomogeneities`: the prescribed values at those DOFs
- `is_prescribed`: a Boolean mask of length `ndofs`, with `true` at each prescribed DOF
"""
struct DeviceConstraintHandler{Tv, Ti, PD <: AbstractVector{Ti}, IH <: AbstractVector{Tv}, IP <: AbstractVector{Bool}}
    prescribed_dofs::PD
    inhomogeneities::IH
    is_prescribed::IP
end

function adapt_structure(backend::KA.Backend, ch::ConstraintHandler)
    @assert Ferrite.isclosed(ch)
    n = Ferrite.ndofs(ch.dh)
    is_prescribed = zeros(Bool, n)
    for d in ch.prescribed_dofs
        is_prescribed[d] = true
    end
    return DeviceConstraintHandler(
        adapt(backend, ch.prescribed_dofs),
        adapt(backend, ch.inhomogeneities),
        adapt(backend, is_prescribed),
    )
end

# -------------------- KA kernels -----------------------------------

@kernel function _meandiag_kernel!(diag, colptr, rowval, nzval)
    j = @index(Global, Linear)
    for k in colptr[j]:(colptr[j + 1] - 1)
        if rowval[k] == j
            diag[j] = abs(nzval[k])
            break
        end
    end
end

# Combined: f -= K[:, d] * inhom[d], then zero column d
@kernel function _apply_inhom_zero_cols_kernel!(f, colptr, rowval, nzval, prescribed_dofs, inhomogeneities, applyzero)
    idx = @index(Global, Linear)
    d = prescribed_dofs[idx]
    v = inhomogeneities[idx]
    for k in colptr[d]:(colptr[d + 1] - 1)
        if !applyzero && !iszero(v)
            KA.@atomic f[rowval[k]] -= v * nzval[k]
        end
        nzval[k] = zero(eltype(nzval))
    end
end

# Zero out all nonzeros whose row index is a prescribed DOF
@kernel function _apply_zero_rows_kernel!(rowval, nzval, is_prescribed)
    idx = @index(Global, Linear)
    if is_prescribed[rowval[idx]]
        nzval[idx] = zero(eltype(nzval))
    end
end

# Set K[d,d] = m and f[d] = inhom[d] * m for each prescribed DOF d
@kernel function _apply_set_diag_kernel!(colptr, rowval, nzval, f, prescribed_dofs, inhomogeneities, m, applyzero)
    idx = @index(Global, Linear)
    d = prescribed_dofs[idx]
    for k in colptr[d]:(colptr[d + 1] - 1)
        if rowval[k] == d
            nzval[k] = m
            break
        end
    end
    f[d] = (applyzero ? zero(eltype(f)) : inhomogeneities[idx]) * m
end


function meandiag(K::GPUArrays.AbstractGPUSparseMatrixCSC{T}) where {T}
    n = size(K, 1)
    backend = get_backend(nonzeros(K))
    diag = KA.zeros(backend, T, n)
    _meandiag_kernel!(backend)(
        diag, SparseArrays.getcolptr(K), rowvals(K), nonzeros(K);
        ndrange = n
    )
    KA.synchronize(backend)
    return sum(diag) / n
end

function Ferrite.apply!(K::GPUArrays.AbstractGPUSparseMatrixCSC{T}, f::GPUArrays.AbstractGPUVector{T}, ch::DeviceConstraintHandler{T}, applyzero::Bool = false) where {T}
    n_prescribed = length(ch.prescribed_dofs)
    n_prescribed == 0 && return

    colptr = SparseArrays.getcolptr(K)
    rv = rowvals(K)
    nz = nonzeros(K)
    nnz_K = length(nz)
    m = meandiag(K)
    backend = get_backend(nz)

    # Step 1+2: f -= K[:,d]*v, then zero prescribed columns
    _apply_inhom_zero_cols_kernel!(backend)(
        f, colptr, rv, nz, ch.prescribed_dofs, ch.inhomogeneities, applyzero;
        ndrange = n_prescribed
    )
    KA.synchronize(backend)

    # Step 3: zero prescribed rows
    _apply_zero_rows_kernel!(backend)(
        rv, nz, ch.is_prescribed;
        ndrange = nnz_K
    )
    KA.synchronize(backend)

    # Step 4: set diagonal and rhs
    _apply_set_diag_kernel!(backend)(
        colptr, rv, nz, f, ch.prescribed_dofs, ch.inhomogeneities, T(m), applyzero;
        ndrange = n_prescribed
    )
    KA.synchronize(backend)

    return
end

function Ferrite.apply_zero!(K::GPUArrays.AbstractGPUSparseMatrixCSC{T}, f::GPUArrays.AbstractGPUVector{T}, ch::DeviceConstraintHandler{T}) where {T}
    return Ferrite.apply!(K, f, ch, true)
end
