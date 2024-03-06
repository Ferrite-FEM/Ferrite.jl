# Sparsity pattern and sparse matrix

## Sparsity Pattern

Given a `DofHandler` we can obtain the corresponding sparse matrix by using the
[`create_sparsity_pattern`](@ref) function. This will setup a `SparseMatrixCSC`
with stored values on all the places corresponding to the degree of freedom numbering
in the `DofHandler`. This means that when we assemble into the global stiffness matrix
there is no need to change the internal representation of the sparse matrix since the
sparse structure will not change.

Often the finite element problem is symmetric and will result in a symmetric sparse
matrix. This information is often something that the sparse solver can take advantage of.
If the solver only needs half the matrix there is no need to assemble both halves.
For this purpose there is a [`create_symmetric_sparsity_pattern`](@ref) function that
will only create the upper half of the matrix, and return a `Symmetric` wrapped
`SparseMatrixCSC`.

Given a `DofHandler` `dh` we can obtain the (symmetric) sparsity pattern as

```julia
K = create_matrix(dh)
K = create_symmetric_sparsity_pattern(dh)
```

The returned sparse matrix will be used together with an `Assembler`, which
assembles efficiently into the matrix, without modifying the internal representation.

## Degree of freedom coupling

Discuss the `coupling` keyword argument.


## Eliminating constrained entries

Discuss the `keep_constrained` keyword argument.


## Blocked sparsity pattern

Discuss `BlockSparsityPattern` and `BlockArrays` extension.
