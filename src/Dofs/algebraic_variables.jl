# Algebraic (mesh-free) variables.

# Marker value-shape parameter for scalar algebraic variables, see `AlgebraicVariable()`.
struct ScalarValueShape end

"""
    AlgebraicVariable()
    AlgebraicVariable{V}()

Declaration of a typed algebraic (mesh-free) unknown that can be added to a
[`DofHandler`](@ref) with [`add!`](@ref add!(::DofHandler, ::Symbol, ::AlgebraicVariable)).
Its dofs are not attached to the mesh and do not appear in `celldofs`.

`AlgebraicVariable()` declares a scalar. Supported value shapes are `Vec{dim}`,
`Tensor{2, dim}`, `SymmetricTensor{2, dim}`, `Tensor{4, dim}`, and
`SymmetricTensor{4, dim}`, where `dim` is in `1:3`. Every independent component of the
value shape receives one dof; to prescribe individual components, use an
[`AffineConstraint`](@ref) on the corresponding dof.

# Examples
```julia
add!(dh, :p0, AlgebraicVariable())                            # one scalar unknown
add!(dh, :z, AlgebraicVariable{Vec{3}}())                     # three unknowns
add!(dh, :σ̄, AlgebraicVariable{SymmetricTensor{2, 2}}())      # three unknowns
```

See also [`AlgebraicValues`](@ref) and [`algebraic_value`](@ref).
"""
struct AlgebraicVariable{V, N, CI}
    components::NTuple{N, CI}
end

function AlgebraicVariable{V}() where {V}
    components = _canonical_components(V)
    return AlgebraicVariable{V, length(components), eltype(typeof(components))}(components)
end

AlgebraicVariable() = AlgebraicVariable{ScalarValueShape}()

# Number of dofs owned by the variable (one per independent component).
n_algebraic_dofs(v::AlgebraicVariable) = length(v.components)

function Base.show(io::IO, ::AlgebraicVariable{V}) where {V}
    if V === ScalarValueShape
        print(io, "AlgebraicVariable()")
    else
        print(io, "AlgebraicVariable{", V, "}()")
    end
    return
end

##############################
# Canonical component orders #
##############################

# The Cartesian index space of a value shape, e.g. `(dim, dim)` for a second order
# tensor. This is the only place where the supported shapes are enumerated; everything
# below derives from `_index_space` and `_canonical_index`.
_index_space(::Type{ScalarValueShape}) = (1,)
_index_space(::Type{Vec{dim}}) where {dim} = _checked_dims(Vec{dim}, dim, 1)
_index_space(::Type{Tensor{2, dim}}) where {dim} = _checked_dims(Tensor{2, dim}, dim, 2)
_index_space(::Type{SymmetricTensor{2, dim}}) where {dim} = _checked_dims(SymmetricTensor{2, dim}, dim, 2)
_index_space(::Type{Tensor{4, dim}}) where {dim} = _checked_dims(Tensor{4, dim}, dim, 4)
_index_space(::Type{SymmetricTensor{4, dim}}) where {dim} = _checked_dims(SymmetricTensor{4, dim}, dim, 4)
function _index_space(::Type{V}) where {V}
    if V isa Type && V <: AbstractTensor
        params = Base.unwrap_unionall(V).parameters
        if length(params) >= 3 && params[1] isa Int && params[2] isa Int && !(params[3] isa TypeVar)
            order, dim = params[1], params[2]
            # Suggest the corresponding free-scalar spelling, but only if that spelling is
            # itself a supported shape (order-1 tensors are only supported as `Vec`)
            shape = if order == 1 && !(V <: SymmetricTensor)
                "Vec{$dim}"
            elseif order == 2 || order == 4
                "$(nameof(V)){$order, $dim}"
            else
                nothing
            end
            if shape !== nothing
                error(
                    "AlgebraicVariable value shapes must not fix the coefficient scalar type: " *
                        "use `AlgebraicVariable{$(shape)}()` instead of `AlgebraicVariable{$(V)}()`."
                )
            end
        end
    end
    error(
        "unsupported value shape `$(V)` for AlgebraicVariable. Supported shapes are " *
            "scalar (`AlgebraicVariable()`), `Vec{dim}`, `Tensor{2, dim}`, `SymmetricTensor{2, dim}`, " *
            "`Tensor{4, dim}`, and `SymmetricTensor{4, dim}`."
    )
end

function _checked_dims(::Type{V}, dim::Int, order::Int) where {V}
    # Tensors.jl only supports dimensions 1, 2, and 3
    if !(1 <= dim <= 3)
        error("unsupported dimension $dim in value shape `$(V)` for AlgebraicVariable: supported dimensions are 1, 2, and 3")
    end
    return ntuple(_ -> dim, order)
end

# Fold an index tuple to its canonical (symmetry-representative) component: a plain index
# for scalars and `Vec`s, and the lower triangle entries for the symmetric tensors.
_canonical_index(::Type{ScalarValueShape}, idx::Tuple) = only(idx)
_canonical_index(::Type{<:Vec}, idx::Tuple) = only(idx)
_canonical_index(::Type{<:Tensor}, idx::Tuple) = idx
_canonical_index(::Type{<:SymmetricTensor{2}}, (i, j)::Tuple) = (max(i, j), min(i, j))
_canonical_index(::Type{<:SymmetricTensor{4}}, (i, j, k, l)::Tuple) = (max(i, j), min(i, j), max(k, l), min(k, l))

# Canonical ordering of the components of a value shape: first occurrence of each
# canonical component in column major iteration of the index space. This matches the data
# storage order of the corresponding Tensors.jl type (e.g. lower triangle column major
# for symmetric tensors). The same order is shared by dof numbering, value
# reconstruction, basis directions, and component-wise renumbering.
function _canonical_components(::Type{V}) where {V}
    dims = _index_space(V)
    return Tuple(unique(_canonical_index(V, Tuple(I)) for I in CartesianIndices(dims)))
end

#####################################
# Value reconstruction and basis    #
#####################################

# One-hot basis direction for the canonical component `c`. `_reconstruct_value`
# canonicalizes the indices it queries, so comparing against `c` is exact.
function _algebraic_basis_value(::Type{V}, c, ::Type{T}) where {V, T}
    return _reconstruct_value(V, comp -> comp == c ? one(T) : zero(T))
end

# Reconstruct the typed value of `v`: component `k` reads `a[dofs[k]]`.
function _reconstruct_algebraic_value(v::AlgebraicVariable{V}, a::AbstractVector{T}, dofs::AbstractVector{Int}) where {V, T}
    comps = v.components
    coefficient = function (c)
        k = findfirst(==(c), comps)::Int # canonical indices are always found
        return a[dofs[k]]
    end
    return _reconstruct_value(V, coefficient)
end

_reconstruct_value(::Type{ScalarValueShape}, coefficient::F) where {F} = coefficient(1)
function _reconstruct_value(::Type{V}, coefficient::F) where {V, F}
    return V((idx...) -> coefficient(_canonical_index(V, idx)))
end

###################
# AlgebraicValues #
###################

"""
    AlgebraicValues([T::Type = Float64], variable::AlgebraicVariable)

Evaluation data for an [`AlgebraicVariable`](@ref), analogous to [`CellValues`](@ref) and
[`FacetValues`](@ref). It caches the constant basis directions and is intended to be
constructed once and passed to element routines.
"""
struct AlgebraicValues{AV <: AlgebraicVariable, N, VT}
    variable::AV
    basis::NTuple{N, VT}
end

function AlgebraicValues(::Type{T}, v::AlgebraicVariable{V, N}) where {T <: Number, V, N}
    basis = ntuple(i -> _algebraic_basis_value(V, v.components[i], T), Val(N))
    return AlgebraicValues(v, basis)
end
AlgebraicValues(v::AlgebraicVariable) = AlgebraicValues(Float64, v)

"""
    getnbasefunctions(av::AlgebraicValues)

Return the number of algebraic basis directions.
"""
getnbasefunctions(av::AlgebraicValues) = length(av.basis)

"""
    algebraic_basis_value(av::AlgebraicValues, i::Int)

Return the cached basis direction for the `i`th algebraic dof.
"""
function algebraic_basis_value(av::AlgebraicValues, i::Int)
    if !(1 <= i <= getnbasefunctions(av))
        error("dof index $i out of bounds for AlgebraicValues with $(getnbasefunctions(av)) dof(s)")
    end
    return av.basis[i]
end

"""
    algebraic_value(av::AlgebraicValues, u::AbstractVector, dof_range = eachindex(u))

Reconstruct the algebraic value from `u[dof_range]`. The scalar type follows `eltype(u)`.
"""
function algebraic_value(av::AlgebraicValues, u::AbstractVector, dof_range::AbstractVector{Int} = eachindex(u))
    if length(dof_range) != getnbasefunctions(av)
        error("the length of `dof_range` ($(length(dof_range))) does not match the number of algebraic dofs ($(getnbasefunctions(av)))")
    end
    return _reconstruct_algebraic_value(av.variable, u, dof_range)
end

function Base.show(io::IO, ::MIME"text/plain", av::AlgebraicValues{<:Any, N, VT}) where {N, VT}
    print(io, "AlgebraicValues(", av.variable, "): ", N, " basis value(s) of type ", VT)
    return
end

#########################
# Coupling descriptors  #
#########################

# Supertype for the coupling descriptors in algebraic_coupling.jl, defined here so that
# earlier included files can reference the type.
abstract type AbstractCoupling end
