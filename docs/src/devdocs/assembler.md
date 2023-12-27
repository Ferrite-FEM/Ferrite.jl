# [Assembler](@id devdocs-assembler)

The assembler handles the insertion of the element matrices and element vectors into the system matrix and right hand side. While the CSC and CSR formats are the most common sparse matrix formats in practice, users might want to have optimized custom matrix formats for their specific use-case. The default assembler [`AssemblerSparsityPattern`](@ref) should be able to handle most cases in practice. To support a custom format users have to dispatch the following functions:

```@docs
Ferrite._assemble_inner!
Ferrite.zero_out_rows!
Ferrite.zero_out_columns!
```

and the `AbstractSparseMatrix` interface for their custom matrix type.

## Custom Assembler

In case the default assembler is insufficient, users can implement a custom assemblers. For this, they can create a custom type and dispatch the following functions.

```@docs
Ferrite.matrix_handle
Ferrite.vector_handle
start_assemble!
finish_assemble!
assemble!
```

For local elimination support the following functions might also need custom dispatches

```@docs
Ferrite._condense_local!
```

## Custom Matrix Type Sparsity Pattern

TODO `create_sparsity_pattern`
