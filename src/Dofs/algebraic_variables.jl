# Algebraic (mesh-free) variables.
#
# An `AlgebraicVariable` represents a typed unknown that is not attached to the mesh: it
# has no domain, interpolation, or spatial support. Its dofs are numbered globally by the
# `DofHandler` (after all spatial dofs) but never appear in `celldofs`. Spatial coupling is
# introduced explicitly through coupling descriptors (see `CellCoupling`, `FacetCoupling`,
# and `AlgebraicCoupling` in algebraic_coupling.jl).

# Private marker type used as the value-shape parameter for scalar algebraic variables,
# see `AlgebraicVariable()`.
struct ScalarValueShape end

"""
    AlgebraicVariable()
    AlgebraicVariable{V}(; active_components = nothing)

Declaration of a typed algebraic (mesh-free) unknown that can be added to a
[`DofHandler`](@ref) with [`add!`](@ref add!(::DofHandler, ::Symbol, ::AlgebraicVariable)).
An algebraic variable has no mesh domain and no interpolation; its degrees of freedom are
numbered globally (after all spatial dofs) but do not appear in `celldofs`. Where the
variable couples to spatial fields is declared explicitly with coupling descriptors, see
[`CellCoupling`](@ref), [`FacetCoupling`](@ref), and [`AlgebraicCoupling`](@ref).

`AlgebraicVariable()` declares a single scalar unknown. The value shape `V` declares a
structured value, without fixing the coefficient scalar type:

 - `AlgebraicVariable{Vec{dim}}()`: a vector with `dim` components,
 - `AlgebraicVariable{Tensor{2, dim}}()`: a second order tensor,
 - `AlgebraicVariable{SymmetricTensor{2, dim}}()`: a symmetric second order tensor,
 - `AlgebraicVariable{Tensor{4, dim}}()`: a fourth order tensor,
 - `AlgebraicVariable{SymmetricTensor{4, dim}}()`: a minor symmetric fourth order tensor,

with `dim` in `1:3` (the dimensions supported by Tensors.jl).

By default every component of the declared value is an unknown and receives a dof. The
keyword `active_components` selects a nonempty subset of the components as unknowns using
natural indices: plain indices such as `(1, 3)` for a vector, and one Cartesian index
tuple per component, such as `((1, 1), (2, 2))` for a second order tensor or
`((1, 1, 2, 2),)` for a fourth order tensor. Symmetry-equivalent indices (e.g. `(1, 2)`
and `(2, 1)` for a `SymmetricTensor`) identify the same component. The selection is stored
canonicalized in the order of the value type's components, independent of the order in
which it was supplied; inactive components own no dofs.

# Examples
```julia
add!(dh, :p0, AlgebraicVariable())                            # one scalar unknown
add!(dh, :z, AlgebraicVariable{Vec{3}}())                     # three unknowns
add!(dh, :σ̄, AlgebraicVariable{SymmetricTensor{2, 2}}())      # three unknowns
add!(dh, :σ̄₁, AlgebraicVariable{SymmetricTensor{2, 2}}(
    active_components = ((1, 1), (2, 2)),                     # two unknowns
))
```

See also [`AlgebraicValues`](@ref), [`active_components`](@ref), and
[`algebraic_value`](@ref).
"""
struct AlgebraicVariable{V, N, CI}
    active_components::NTuple{N, CI}
    function AlgebraicVariable{V}(; active_components::Union{Nothing, Tuple} = nothing) where {V}
        components = _canonicalize_active_components(V, active_components)
        return new{V, length(components), eltype(typeof(components))}(components)
    end
end

AlgebraicVariable(; kwargs...) = AlgebraicVariable{ScalarValueShape}(; kwargs...)

"""
    active_components(variable::AlgebraicVariable)
    active_components(av::AlgebraicValues)

Return the active components of the algebraic variable as a tuple of natural component
indices, canonicalized to the component order of the declared value type. Each active
component owns exactly one global dof. For a scalar variable this returns `(1,)`, for a
fully active `Vec{3}` variable `(1, 2, 3)`, and for e.g.
`AlgebraicVariable{SymmetricTensor{2, 2}}(active_components = ((1, 1), (2, 2)))` it returns
`((1, 1), (2, 2))`.
"""
active_components(v::AlgebraicVariable) = v.active_components

# Number of dofs owned by the variable (one per active component).
n_algebraic_dofs(v::AlgebraicVariable) = length(active_components(v))

_value_shape_string(::Type{ScalarValueShape}) = "scalar"
_value_shape_string(::Type{V}) where {V} = string(V)

function Base.show(io::IO, v::AlgebraicVariable{V}) where {V}
    if V === ScalarValueShape
        print(io, "AlgebraicVariable()")
    else
        print(io, "AlgebraicVariable{", V, "}(")
        if active_components(v) != _canonical_components(V)
            print(io, "active_components = ", active_components(v))
        end
        print(io, ")")
    end
    return
end

##############################
# Canonical component orders #
##############################

# Tensors.jl only supports dimensions 1, 2, and 3 (a 0-dimensional shape would also own
# no dofs, violating the nonempty active-component requirement).
function _check_supported_dim(dim::Int, ::Type{V}) where {V}
    if !(1 <= dim <= 3)
        error("unsupported dimension $dim in value shape `$(V)` for AlgebraicVariable: supported dimensions are 1, 2, and 3")
    end
    return
end

# Canonical natural-index ordering of the components of a declared value shape. This order
# matches the data storage order of the corresponding `Tensors.jl` type:
#  - `Vec{dim}`: 1, ..., dim
#  - `Tensor{2, dim}`: column major, i.e. (1,1), (2,1), ..., (dim,dim)
#  - `SymmetricTensor{2, dim}`: lower triangle column major, i.e. for `dim = 3`:
#    (1,1), (2,1), (3,1), (2,2), (3,2), (3,3)
#  - `Tensor{4, dim}`: column major with the first index fastest
#  - `SymmetricTensor{4, dim}`: one entry per pair of symmetric second order components,
#    with the (i,j) pair fastest, i.e. the storage order of the n×n matrix representation
# The same order is shared by dof numbering, value reconstruction, basis directions,
# component-level coupling expansion, and component-wise renumbering.
_canonical_components(::Type{ScalarValueShape}) = (1,)
function _canonical_components(::Type{Vec{dim}}) where {dim}
    _check_supported_dim(dim, Vec{dim})
    return ntuple(identity, dim)
end
function _canonical_components(::Type{Tensor{2, dim}}) where {dim}
    _check_supported_dim(dim, Tensor{2, dim})
    return Tuple([(i, j) for j in 1:dim for i in 1:dim])
end
function _canonical_components(::Type{SymmetricTensor{2, dim}}) where {dim}
    _check_supported_dim(dim, SymmetricTensor{2, dim})
    return Tuple([(i, j) for j in 1:dim for i in j:dim])
end
function _canonical_components(::Type{Tensor{4, dim}}) where {dim}
    _check_supported_dim(dim, Tensor{4, dim})
    return Tuple([(i, j, k, l) for l in 1:dim for k in 1:dim for j in 1:dim for i in 1:dim])
end
function _canonical_components(::Type{SymmetricTensor{4, dim}}) where {dim}
    _check_supported_dim(dim, SymmetricTensor{4, dim})
    pairs2 = [(i, j) for j in 1:dim for i in j:dim]
    return Tuple([(ij[1], ij[2], kl[1], kl[2]) for kl in pairs2 for ij in pairs2])
end
function _canonical_components(::Type{V}) where {V}
    if V isa Type && V <: AbstractTensor && isconcretetype(V)
        error(
            "AlgebraicVariable value shapes must not fix the coefficient scalar type: " *
                "use e.g. `AlgebraicVariable{$(V.name.wrapper){$(join(V.parameters[1:2], ", "))}}()` instead of `AlgebraicVariable{$(V)}()`."
        )
    end
    error(
        "unsupported value shape `$(V)` for AlgebraicVariable. Supported shapes are " *
            "scalar (`AlgebraicVariable()`), `Vec{dim}`, `Tensor{2, dim}`, `SymmetricTensor{2, dim}`, " *
            "`Tensor{4, dim}`, and `SymmetricTensor{4, dim}`."
    )
end

# Normalize a user supplied natural component index to its canonical representative and
# check rank and bounds.
function _normalize_component(::Type{ScalarValueShape}, c)
    c isa Integer || error("a scalar AlgebraicVariable has a single component selected by the index 1, got $(repr(c))")
    c == 1 || error("a scalar AlgebraicVariable has a single component selected by the index 1, got $c")
    return 1
end
function _normalize_component(::Type{Vec{dim}}, c) where {dim}
    c isa Integer || error("components of a `Vec{$dim}` valued AlgebraicVariable are selected with plain indices (e.g. `active_components = (1, 3)`), got $(repr(c))")
    1 <= c <= dim || error("component index $c out of bounds for value shape `Vec{$dim}`")
    return Int(c)
end
function _normalize_component(::Type{Tensor{2, dim}}, c) where {dim}
    c isa Tuple{Integer, Integer} || error("components of a `Tensor{2, $dim}` valued AlgebraicVariable are selected with index 2-tuples (e.g. `active_components = ((1, 1), (2, 2))`), got $(repr(c))")
    i, j = Int(c[1]), Int(c[2])
    (1 <= i <= dim && 1 <= j <= dim) || error("component index ($i, $j) out of bounds for value shape `Tensor{2, $dim}`")
    return (i, j)
end
function _normalize_component(::Type{SymmetricTensor{2, dim}}, c) where {dim}
    c isa Tuple{Integer, Integer} || error("components of a `SymmetricTensor{2, $dim}` valued AlgebraicVariable are selected with index 2-tuples (e.g. `active_components = ((1, 1), (2, 2))`), got $(repr(c))")
    i, j = Int(c[1]), Int(c[2])
    (1 <= i <= dim && 1 <= j <= dim) || error("component index ($i, $j) out of bounds for value shape `SymmetricTensor{2, $dim}`")
    # (i, j) and (j, i) identify the same component; the canonical representative is the
    # lower triangle entry (i >= j), matching the Tensors.jl storage.
    return (max(i, j), min(i, j))
end
function _normalize_component(::Type{Tensor{4, dim}}, c) where {dim}
    c isa NTuple{4, Integer} || error("components of a `Tensor{4, $dim}` valued AlgebraicVariable are selected with index 4-tuples (e.g. `active_components = ((1, 1, 2, 2),)`), got $(repr(c))")
    idx = map(Int, c)
    all(x -> 1 <= x <= dim, idx) || error("component index $idx out of bounds for value shape `Tensor{4, $dim}`")
    return idx
end
function _normalize_component(::Type{SymmetricTensor{4, dim}}, c) where {dim}
    c isa NTuple{4, Integer} || error("components of a `SymmetricTensor{4, $dim}` valued AlgebraicVariable are selected with index 4-tuples (e.g. `active_components = ((1, 1, 2, 2),)`), got $(repr(c))")
    i, j, k, l = map(Int, c)
    all(x -> 1 <= x <= dim, (i, j, k, l)) || error("component index ($i, $j, $k, $l) out of bounds for value shape `SymmetricTensor{4, $dim}`")
    # Minor symmetries: (i, j) and (j, i), as well as (k, l) and (l, k), identify the same
    # component; the canonical representatives are the lower triangle entries.
    return (max(i, j), min(i, j), max(k, l), min(k, l))
end

function _canonicalize_active_components(::Type{V}, selection::Union{Nothing, Tuple}) where {V}
    canonical = _canonical_components(V)
    selection === nothing && return canonical
    isempty(selection) && error("active_components must select at least one component")
    normalized = map(c -> _normalize_component(V, c), selection)
    if !allunique(normalized)
        error("duplicate (or symmetry-equivalent duplicate) entries in active_components = $(selection)")
    end
    # Store in canonical order, independent of the supplied order
    return Tuple(filter(in(normalized), collect(canonical)))
end

#####################################
# Value reconstruction and basis    #
#####################################

# Constant basis direction with scalar type `T` associated with the declared component `c`
# of value shape `V`: the full typed value with a one in component `c` and zeros elsewhere.
_algebraic_basis_value(::Type{ScalarValueShape}, ::Int, ::Type{T}) where {T} = one(T)
function _algebraic_basis_value(::Type{Vec{dim}}, c::Int, ::Type{T}) where {dim, T}
    return Vec{dim}(i -> i == c ? one(T) : zero(T))
end
function _algebraic_basis_value(::Type{Tensor{2, dim}}, c::Tuple{Int, Int}, ::Type{T}) where {dim, T}
    return Tensor{2, dim}((i, j) -> (i, j) == c ? one(T) : zero(T))
end
function _algebraic_basis_value(::Type{SymmetricTensor{2, dim}}, c::Tuple{Int, Int}, ::Type{T}) where {dim, T}
    return SymmetricTensor{2, dim}((i, j) -> (max(i, j), min(i, j)) == c ? one(T) : zero(T))
end
function _algebraic_basis_value(::Type{Tensor{4, dim}}, c::NTuple{4, Int}, ::Type{T}) where {dim, T}
    return Tensor{4, dim}((i, j, k, l) -> (i, j, k, l) == c ? one(T) : zero(T))
end
function _algebraic_basis_value(::Type{SymmetricTensor{4, dim}}, c::NTuple{4, Int}, ::Type{T}) where {dim, T}
    return SymmetricTensor{4, dim}((i, j, k, l) -> (max(i, j), min(i, j), max(k, l), min(k, l)) == c ? one(T) : zero(T))
end

# Reconstruct the typed value of `v` from the coefficient vector `a`, where the `k`th
# active component reads `a[dofs[k]]` and inactive components are `zero(T)`.
function _reconstruct_algebraic_value(v::AlgebraicVariable{V}, a::AbstractVector{T}, dofs::AbstractVector{Int}) where {V, T}
    comps = active_components(v)
    coefficient = function (c)
        k = findfirst(==(c), comps)
        return k === nothing ? zero(T) : a[dofs[k]]
    end
    return _reconstruct_value(V, coefficient)
end

_reconstruct_value(::Type{ScalarValueShape}, coefficient::F) where {F} = coefficient(1)
function _reconstruct_value(::Type{Vec{dim}}, coefficient::F) where {dim, F}
    return Vec{dim}(i -> coefficient(i))
end
function _reconstruct_value(::Type{Tensor{2, dim}}, coefficient::F) where {dim, F}
    return Tensor{2, dim}((i, j) -> coefficient((i, j)))
end
function _reconstruct_value(::Type{SymmetricTensor{2, dim}}, coefficient::F) where {dim, F}
    return SymmetricTensor{2, dim}((i, j) -> coefficient((max(i, j), min(i, j))))
end
function _reconstruct_value(::Type{Tensor{4, dim}}, coefficient::F) where {dim, F}
    return Tensor{4, dim}((i, j, k, l) -> coefficient((i, j, k, l)))
end
function _reconstruct_value(::Type{SymmetricTensor{4, dim}}, coefficient::F) where {dim, F}
    return SymmetricTensor{4, dim}((i, j, k, l) -> coefficient((max(i, j), min(i, j), max(k, l), min(k, l))))
end

####################
# AlgebraicValues  #
####################

"""
    AlgebraicValues([T::Type = Float64], variable::AlgebraicVariable)

Evaluation object for an [`AlgebraicVariable`](@ref): the algebraic counterpart of e.g.
[`CellValues`](@ref) for spatial fields. Like `CellValues`, which is constructed from
the interpolation, it is constructed from the variable once during setup and passed into
the assembly kernel, where the variable is queried on it:

 - [`algebraic_value(av, u, [dof_range])`](@ref algebraic_value): the typed value
   reconstructed from coefficients,
 - [`algebraic_basis_value(av, i)`](@ref algebraic_basis_value): the constant basis
   direction associated with active dof `i`,
 - [`ndofs(av)`](@ref ndofs): the number of active dofs,
 - [`active_components(av)`](@ref active_components): the active components.

Since an algebraic variable has no mesh domain there is no quadrature, geometry, or
`reinit!`; the basis directions are constants of scalar type `T`, fixed at construction.

The queries are type stable, in contrast to looking up the variable inside a kernel with
[`algebraic_variable`](@ref), which goes through an untyped registry in the `DofHandler`.

# Examples
```julia
σ̄var = AlgebraicVariable{SymmetricTensor{2, 2}}()
dh = DofHandler(grid)
add!(dh, :u, ip_u)
add!(dh, :σ̄, σ̄var)
close!(dh)

cv = CellValues(qr, ip_u)
av = AlgebraicValues(σ̄var) # or AlgebraicValues(algebraic_variable(dh, :σ̄))
# ... pass both into the assembly kernel ...
```
"""
struct AlgebraicValues{AV <: AlgebraicVariable, N, VT}
    variable::AV
    basis::NTuple{N, VT} # constant basis directions, in active-component order
end

function AlgebraicValues(::Type{T}, v::AlgebraicVariable{V, N}) where {T <: Number, V, N}
    comps = active_components(v)
    basis = ntuple(i -> _algebraic_basis_value(V, comps[i], T), Val(N))
    return AlgebraicValues(v, basis)
end
AlgebraicValues(v::AlgebraicVariable) = AlgebraicValues(Float64, v)

active_components(av::AlgebraicValues) = active_components(av.variable)

"""
    ndofs(av::AlgebraicValues)

Return the number of active dofs of the algebraic variable in `av`, i.e. the number of
coefficients that its value is reconstructed from.
"""
ndofs(av::AlgebraicValues) = length(av.basis)

"""
    algebraic_basis_value(av::AlgebraicValues, i::Int)

Return the constant direction associated with the `i`th active dof of the algebraic
variable: the full typed value with a one in the corresponding declared component and
zeros elsewhere. This is the derivative of the value reconstructed by
[`algebraic_value`](@ref) with respect to coefficient `i`, and takes the place of the
test/trial functions of the variable in linearizations. It is not a shape function,
however: there is no spatial argument, gradient, or quadrature point. For a scalar
variable it is `one(T)`, and for e.g. a fully active `SymmetricTensor{2, 2}` variable,
`i = 2` gives the symmetric tensor with ones in the `(2, 1)` and `(1, 2)` components.

Here `i` indexes the active dofs, i.e. `1 <= i <= ndofs(av)`; inactive components have no
basis direction since they own no dof.
"""
function algebraic_basis_value(av::AlgebraicValues, i::Int)
    if !(1 <= i <= ndofs(av))
        error("active dof index $i out of bounds for AlgebraicValues with $(ndofs(av)) active dof(s)")
    end
    return av.basis[i]
end

"""
    algebraic_value(av::AlgebraicValues, u::AbstractVector, dof_range = eachindex(u))

Reconstruct the typed value of the algebraic variable from the coefficients
`u[dof_range]`, in active-component order (see [`active_components`](@ref)). For a scalar
variable this returns the single coefficient; for vector and tensor valued variables the
full typed value is reconstructed with `zero(eltype(u))` inserted in inactive components.

This is the natural form inside an assembly kernel, where `u` holds the local unknowns of
the augmented local system and `dof_range` is the local range of the variable:

```julia
layout = local_dofs(cell, descriptor)
ae = a[layout] # gather the augmented local unknowns
σ̄ = algebraic_value(av, ae, dof_range(layout, :σ̄))
```

The scalar type of the result follows `eltype(u)`, so dual numbers pass through when the
kernel is differentiated with automatic differentiation. Passing the global solution
vector with the global dof numbers, `algebraic_value(av, a, algebraic_dofs(dh, :σ̄))`, is
equivalent to `algebraic_value(dh, a, :σ̄)` but type stable.
"""
function algebraic_value(av::AlgebraicValues, u::AbstractVector, dof_range::AbstractVector{Int} = eachindex(u))
    if length(dof_range) != ndofs(av)
        error("the length of `dof_range` ($(length(dof_range))) does not match the number of active dofs of the algebraic variable ($(ndofs(av)))")
    end
    return _reconstruct_algebraic_value(av.variable, u, dof_range)
end

function Base.show(io::IO, ::MIME"text/plain", av::AlgebraicValues{<:Any, N, VT}) where {N, VT}
    print(io, "AlgebraicValues(", av.variable, "): ", N, " active dof(s), basis type ", VT)
    return
end

#########################
# Coupling descriptors  #
#########################

# Supertype for the coupling descriptors `CellCoupling`, `FacetCoupling`, and
# `AlgebraicCoupling` (implemented in algebraic_coupling.jl). Defined here so that earlier
# included files can reference the type.
abstract type AbstractCoupling end
