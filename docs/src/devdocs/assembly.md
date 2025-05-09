# [Assembly](@id devdocs-assembly)

The assembler handles the insertion of the element matrices and element vectors into the system matrix and right hand side. While the CSC and CSR formats are the most common sparse matrix formats in practice, users might want to have optimized custom matrix formats for their specific use-case. The default assemblers [`Ferrite.CSCAssembler`](@ref) and [`Ferrite.CSRAssembler`](@ref) should be able to handle most cases in practice. To support a custom format users have to dispatch the following functions on their new assembler and matrix type. There is the public interface

```@docs; canonical=false
Ferrite.allocate_matrix
```

the internal interface
```@docs
Ferrite.zero_out_rows!
Ferrite.zero_out_columns!
Ferrite._condense!
```

and the `AbstractSparseMatrix` interface for their custom matrix type. Optional dispatches to speed up operations might be

```@docs
Ferrite.add_inhomogeneities!
```

## Custom Assembler

In case the default assembler is insufficient, users can implement a custom assemblers. For this, they can create a custom type and dispatch the following functions.

```@docs; canonical=false
start_assemble
assemble!
```

For local elimination support the following functions might also need custom dispatches

```@docs
Ferrite._condense_local!
```

## Type definitions

```@docs
Ferrite.COOAssembler
Ferrite.CSCAssembler
Ferrite.CSRAssembler
Ferrite.SymmetricCSCAssembler
```

## Utility functions

```@docs
Ferrite.matrix_handle
Ferrite.vector_handle
Ferrite._sortdofs_for_assembly!
Ferrite.sortperm2!
```
