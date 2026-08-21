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
Ferrite.add_inhomogeneities!
Ferrite.condense_into!
Ferrite.addindex!
```

and the `AbstractMatrix` interface for their custom matrix type. [`apply!`](@ref) itself is
generic and dispatches on `AbstractMatrix`, so a custom format is supported as soon as the
functions above are dispatched -- there is no need to add an `apply!` method.

Each of these takes the constraint data explicitly and, where relevant, index offsets, rather
than a [`ConstraintHandler`](@ref). That is deliberate: the very same methods are then used
both for a matrix on its own and for a matrix used as a *block* of a blocked matrix, where the
indices are block local and the offsets place the block in the global system. A format that
implements them therefore works with the BlockArrays extension without any further work, and
without that extension having to know anything about it.

Two conventions are worth calling out:

- `zero_out_rows!` and `zero_out_columns!` receive the set of indices to zero **twice**, once
  as a sorted list and once as a boolean mask. Which one can be used efficiently depends on the
  storage: a column-compressed format walks the listed columns directly, while a row-compressed
  format has to scan its stored column indices against the mask. Passing both avoids forcing
  every format to build the representation it does not have.
- `condense_into!` writes into a *destination* matrix that is not necessarily the matrix it
  reads, which is what lets a block condense into the blocked matrix it belongs to. It only has
  to handle the matrix; the right-hand side is condensed separately by
  [`Ferrite._condense!`](@ref), so the order in which stored entries are visited does not
  matter.

Finally, [`Ferrite._condense!`](@ref) itself is dispatched per format, but is a one-liner over
`Ferrite.condense_into!` for anything that implements it:

```@docs
Ferrite._condense!
```

CSC and CSR are mirror images of one another -- one stores columns contiguously, the other
rows -- so their implementations of the above are shared, parameterised by which index is the
contiguous one. `Ferrite.minor_indices` is the accessor that abstracts the difference:

```@docs
Ferrite.minor_indices
```

This is an implementation detail of those two formats, not part of the interface. A format that
does not store scalar entries in flat arrays parallel to `nonzeros` -- a blocked format such as
BSR, say -- simply implements the interface functions directly and never defines it.

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
