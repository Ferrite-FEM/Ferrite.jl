# Sparsity pattern and sparse matrices

This is the reference documentation for sparsity patterns and sparse matrix instantiation.
See the topic section on [Sparsity pattern and sparse matrices](@ref topic-sparse-matrix).

## Sparsity patterns

### `AbstractSparsityPattern`

The following applies to all subtypes of `AbstractSparsityPattern`:

```@docs
AbstractSparsityPattern
create_sparsity_pattern!
add_cell_entries!
add_interface_entries!
add_constraint_entries!
Ferrite.add_entry!
```

### `SparsityPattern`

```@docs
SparsityPattern(::Int, ::Int)
allocate_matrix(::SparsityPattern)
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
allocate_matrix(::Main.FerriteBlockArrays.BlockSparsityPattern)
allocate_matrix(::Type{<:BlockMatrix{T, Matrix{S}}}, sp::Main.FerriteBlockArrays.BlockSparsityPattern) where {T, S <: AbstractMatrix{T}}
```

## Sparse matrices

### Creating matrix from `SparsityPattern`

```@docs
allocate_matrix(::Type{S}, ::AbstractSparsityPattern) where {Tv, Ti, S <: SparseMatrixCSC{Tv, Ti}}
allocate_matrix(::Type{Symmetric{Tv, S}}, ::AbstractSparsityPattern) where {Tv, Ti, S <: SparseMatrixCSC{Tv, Ti}}
```

### Creating matrix from `DofHandler`

```@docs
allocate_matrix(::Type{MatrixType}, ::DofHandler, args...; kwargs...) where {MatrixType}
allocate_matrix(::DofHandler, args...; kwargs...)
```
