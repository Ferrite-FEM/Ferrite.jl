# [Assembly](@id devdocs-assembly)

An assembler handles the insertion of the element matrices and element vectors into the system matrix and vector,
and should *normally* (the exact interface is yet to be fully established) subtype `AbstractAssembler{T}`. Here `T` is the
`eltype` of the contained system matrix and vector. This allows the user to infer the eltype when preallocating the element
matrix and vector, e.g.
```julia
function doassemble!(assembler::Ferrite.AbstractAssembler{T}, ...) where {T}
    Ke = zeros(T, n, n) # n = dofs per cell
    fe = zeros(T, n)
    for cell in CellIterator(...)
        element_routine!(Ke, fe, cell, ...)
        assemble!(assembler, celldofs(cell), Ke, fe)
    end
end
```

## Custom matrix formats
While the CSC and CSR formats are the most common sparse matrix formats in practice, users might want to have optimized custom matrix formats for their specific use-case. The default assemblers [`Ferrite.CSCAssembler`](@ref) and [`Ferrite.CSRAssembler`](@ref) should be able to handle most cases in practice. To support a custom format users have to dispatch the following functions on their matrix type. There is the public interface

```@docs; canonical=false
Ferrite.allocate_matrix
```

the internal interface
```@docs
Ferrite.zero_out_rows!
Ferrite.zero_out_columns!
Ferrite._condense!
```

and the `AbstractMatrix` interface for their custom matrix type. [`apply!`](@ref) itself is
generic and dispatches on `AbstractMatrix`, so a custom format is supported as soon as the
functions above are dispatched -- there is no need to add an `apply!` method. Optional
dispatches to speed up operations might be

```@docs
Ferrite.add_inhomogeneities!
```

The kernels backing the CSC implementations of the above take the prescribed dofs and the
constrained mask directly instead of the [`ConstraintHandler`](@ref), so that a matrix built
from CSC blocks can reuse them per block by passing block local indices. See
`Ferrite._zero_out_columns!`, `Ferrite._zero_out_rows!`,
`Ferrite._add_inhomogeneities_cols!` and `Ferrite._condense_column!`, and the BlockArrays
extension for an example of how they are composed.

## Custom assembler

In case the default assembler is insufficient, users can implement a custom assembler. For this, they can create a custom type and dispatch the following functions.

```@docs; canonical=false
start_assemble
assemble!
```

For local elimination support the following functions might also need custom dispatches

```@docs
Ferrite._condense_local!
```

Note that [`apply_assemble!`](@ref) passes the assembler's `atomic` flag on to
`Ferrite._condense_local!`, so a custom assembler supporting atomic accumulation should
report it through `Ferrite._is_atomic` in order to make the global writes of non-local
constraints concurrency-safe as well.

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
