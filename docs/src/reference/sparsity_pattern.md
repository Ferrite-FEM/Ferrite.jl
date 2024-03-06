# Sparsity pattern and sparse matrices

This is the reference documentation for sparsity patterns and sparse matrix instantiation.

A general overview of the usage is as follows:
 1. Create an empty pattern using a constructor, for example [`SparsityPattern`](@ref
    SparsityPattern(::Int, ::Int))
 2. Add entries to the pattern from the DofHandler using [`create_sparsity_pattern!`](@ref)
 3. Condense the pattern using the ConstraintHandler using
    [`condense_sparsity_pattern!`](@ref)
 4. Instantiate sparse matrices from the sparsity pattern using [`create_matrix`](@ref)

For example, the steps above could look as follows when creating a `SparseMatrixCSC` matrix:
```julia
dh = DofHandler(...)
ch = ConstraintHandler(...)
# 1. Create empty pattern
sparsity_pattern = SparsityPattern(ndofs(dh), ndofs(dh))
# 2. Add entries
create_sparsity_pattern!(sparsity_pattern, dh)
# 3. Condense the pattern
condense_sparsity_pattern!(sparsity_pattern, ch)
# 4. Instantiate the matrix
K = create_matrix(SparseMatrixCSC{Float64, Int}, sparsity_pattern)
```

For the standard cases there exist various convenience shortcuts, as documented below, see
e.g. the method [`create_matrix(dh, ch)`](@ref create_matrix(::DofHandler)).


## Sparsity patterns

### `AbstractSparsityPattern`

The following applies to all subtypes of `AbstractSparsityPattern`:

```@docs
AbstractSparsityPattern
create_sparsity_pattern!
condense_sparsity_pattern!
```

### `SparsityPattern`

```@docs
SparsityPattern(::Int, ::Int)
create_sparsity_pattern
create_matrix(::SparsityPattern)
SparsityPattern
```

### `BlockSparsityPattern`

!!! note "Package extension"
    This functionality is only enabled when the package
    [BlockArrays.jl](https://github.com/JuliaArrays/BlockArrays.jl) is installed (`pkg> add
    BlockArrays`) and loaded (`using BlockArrays`) in the session.

```@docs
BlockSparsityPattern(::Vector{Int})
Main.FerriteBlockArrays.BlockSparsityPattern
create_matrix(::Main.FerriteBlockArrays.BlockSparsityPattern)
create_matrix(::Type{<:BlockMatrix{T, Matrix{S}}}, sp::Main.FerriteBlockArrays.BlockSparsityPattern) where {T, S <: AbstractMatrix{T}}
```

## Sparse matrices

### Creating matrix from `SparsityPattern`

```@docs
create_matrix(::Type{S}, ::AbstractSparsityPattern) where {Tv, Ti, S <: SparseMatrixCSC{Tv, Ti}}
create_matrix(::Type{Symmetric{Tv, S}}, ::AbstractSparsityPattern) where {Tv, Ti, S <: SparseMatrixCSC{Tv, Ti}}
```

### Creating matrix from `DofHandler`

```@docs
create_matrix(::Type{MatrixType}, ::DofHandler, args...; kwargs...) where {MatrixType}
create_matrix(::DofHandler, args...; kwargs...)
```
