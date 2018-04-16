```@meta
DocTestSetup = :(using JuAFEM)
```

# Assembly

When the local stiffness matrix and force vector have been calculated
they should be assembled into the global stiffness matrix and the
global force vector. This is just a matter of adding the local
matrix and vector to the global one, at the correct place. Consider e.g.
assembling the local stiffness matrix `ke` and the local force vector `fe`
into the global `K` and `f` respectively. These should be assembled into
the row/column which corresponds to the degrees of freedom for the cell:

```julia
K[celldofs, celldofs] += ke
f[celldofs]           += fe
```

where `celldofs` is the vector containing the degrees of freedom for the cell.
The method above is very inefficient -- it is especially costly to index
into the sparse matrix `K` directly. Therefore we will instead use an
`Assembler` that will help with the assembling of both the global stiffness
matrix and the global force vector. It is also often convenient to create the
sparse matrix just once, and reuse the allocated matrix. This is useful for
e.g. iterative solvers or time dependent problems where the sparse matrix
structure, or [Sparsity Pattern](@ref) will stay the same in every iteration/
time step.

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
K = create_sparsity_pattern(dh)
K = create_symmetric_sparsity_pattern(dh)
```

The returned sparse matrix will be used together with an `Assembler`, which
assembles efficiently into the matrix, without modifying the internal representation.

## `Assembler`

Assembling efficiently into the sparse matrix requires some extra workspace.
This workspace is allocated in an `Assembler`. [`start_assemble`](@ref) is
used to create an `Assembler`:

```julia
A = start_assemble(K)
A = start_assemble(K, f)
```

where `K` is the global stiffness matrix, and `f` the global force vector.
It is optional to give the force vector to the assembler -- sometimes
there is no need to assemble a global force vector.

fds

```julia
assemble!(A, celldofs, ke)
assemble!(A, celldofs, ke, fe)
```


To give a more

```julia
K = create_sparsity_pattern(dh)
f = zeros(ndofs(dh))
A = start_assemble(K, f)

for cell in CellIterator(dh)
    ke, fe = ...
    assemble!(A, celldofs(cell), ke, fe)
end
```
