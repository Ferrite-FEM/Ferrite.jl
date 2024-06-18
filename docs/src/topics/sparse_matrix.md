# [Sparsity pattern and sparse matrices](@id topic-sparse-matrix)

An important property of the finite element method is that it results in *sparse matrices*
for the linear systems to be solved. On this page the topic of sparsity and sparse matrices
are discussed.

```@contents
Pages = ["sparse_matrix.md"]
Depth = 2:2
```

## Sparsity pattern

The sparse structure of the linear system depends on many factors such as e.g. the weak
form, the discretization, and the choice of interpolation(s). In the end it boils down to
how the degrees of freedom (DoFs) *couple* with each other. The most common reason that two
DoFs couple is because they belong to the same element. Note, however, that this is not
guaranteed to result in a coupling since it depends on the specific weak form that is being
discretized, see e.g. [Increasing the sparsity](@ref). Boundary conditions and constraints
can also result in additional DoF couplings.

If DoFs `i` and `j` couple, then the computed value in the eventual matrix will be
*structurally nonzero*[^1]. In this case the entry `(i, j)` should be included in the
sparsity pattern. Conversely, if DoFs `i` and `j` *don't* couple, then the computed value
will be *zero*. In this case the entry `(i, j)` should *not* be included in the sparsity
pattern since there is no need to allocate memory for entries that will be zero.

The sparsity, i.e. the ratio of zero-entries to the total number of entries, is often *very*
high and taking advantage of this results in huge savings in terms of memory. For example,
in a problem with ``10^6`` DoFs there will be a matrix of size ``10^6 \times 10^6``. If all
``10^{12}`` entries of this matrix had to be stored (0% sparsity) as double precision
(`Float64`, 8 bytes) it would require 8 TB of memory. If instead the sparsity is 99.9973%
(which is the case when solving the heat equation on a three dimensional hypercube with
linear Lagrange interpolation) this would be reduced to 216 MB.

[1]: Structurally nonzero means that there is a possibility of a nonzero value even though
     the computed value might become zero in the end for various reasons.


!!! details "Sparsity pattern example"

    To give an example, in this one-dimensional heat problem (see the [Heat
    equation](../tutorials/heat_equation.md) tutorial for the weak form) we have 4 nodes
    with 3 elements in between. For simplicitly DoF numbers and node numbers are the same
    but this is not true in general since nodes and DoFs can be numbered independently (and
    in fact are numbered independently in Ferrite).

    ```
    1 ----- 2 ----- 3 ----- 4
    ```

    Assuming we use linear Lagrange interpolation (the "hat functions") this will give the
    following connections according to the weak form:
     - Trial function 1 couples with test functions 1 and 2 (entries `(1, 1)` and `(1, 2)`
       included in the sparsity pattern)
     - Trial function 2 couples with test functions 1, 2, and 3 (entries `(2, 1)`, `(2, 2)`,
       and `(2, 3)` included in the sparsity pattern)
     - Trial function 3 couples with test functions 2, 3, and 4 (entries `(3, 2)`, `(3, 3)`,
       and `(3, 4)` included in the sparsity pattern)
     - Trial function 4 couples with test functions 3 and 4 (entries `(4, 3)` and `(4, 4)`
       included in the sparsity pattern)

    The resulting sparsity pattern would look like this:

    ```
    4×4 SparseArrays.SparseMatrixCSC{Float64, Int64} with 10 stored entries:
     0.0  0.0   ⋅    ⋅
     0.0  0.0  0.0   ⋅
      ⋅   0.0  0.0  0.0
      ⋅    ⋅   0.0  0.0
    ```

    Moreover, if the problem is solved with periodic boundary conditions, for example by
    constraining the value on the right side to the value on the left side, there will be
    additional couplings. In the example above, this means that DoF 4 should be equal to DoF
    1. Since DoF 4 is constrained it has to be eliminated from the system. Existing entries
    that include DoF 4 are `(3, 4)`, `(4, 3)`, and `(4, 4)`. Given the simple constraint in
    this case we can simply replace DoF 4 with DoF 1 in these entries and we end up with
    entries `(3, 1)`, `(1, 3)`, and `(1, 1)`. This results in two new entries: `(3, 1)` and
    `(1, 3)` (entry `(1, 1)` is already included).

## Creating sparsity patterns

Creating a sparsity pattern can be quite expensive if not done properly and therefore
Ferrite provides efficient methods and datastructures for this. In general the sparsity
pattern is not known in advance and has to be created incrementally. To make this
incremental construction efficient it is necessary to use a dynamic data structure which
allow for fast insertions.

The sparsity pattern also serves as a "matrix builder". When all entries are inserted into
the sparsity pattern the dynamic data structure is typically converted, or "compressed",
into a sparse matrix format such as e.g. the [*compressed sparse row
(CSR)*](https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_(CSR,_CRS_or_Yale_format))
format or the [*compressed sparse column
(CSC)*](https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_column_(CSC_or_CCS))
format, where the latter is the default sparse matrix type implemented in the SparseArrays
standard library. These matrix formats allow for fast linear algebra operations, such as
factorizations and matrix-vector multiplications, that are needed when the linear system is
solved. See [Instantiating the sparse matrix](@ref) for more details.

In summary, a dynamic structure is more efficient when incrementally building the pattern by
inserting new entries, and a static or compressed structure is more efficient for linear
algebra operations.

### Basic sparsity patterns construction

Working with the sparsity pattern explicitly is in many cases not necessary. For basic
usage (e.g. when only one matrix needed, when no customization of the pattern is
required, etc) there exist convenience methods of [`allocate_matrix`](@ref) that return
the matrix directly. Most examples in this documentation don't deal with the sparsity
pattern explicitly because the basic method suffice.
See also [Instantiating the sparse matrix](@ref) for more details.

### Custom sparsity pattern construction

In more advanced cases there might be a need for more fine grained control of the sparsity
pattern. The following steps are typically taken when constructing a sparsity pattern in
Ferrite:

 1. **Initialize an empty pattern:** This can be done by either using the
    [`init_sparsity_pattern(dh)`](@ref) function or by using a constructor directly.
    `init_sparsity_pattern` will return a default pattern type that is compatible with the
    DofHandler. In some cases you might require another type of pattern (for example a
    blocked pattern, see [Blocked sparsity pattern](@ref)) and in that case you can use the
    constructor directly.

 2. **Add entries to the pattern:** There are a number of functions that add entries to the
    pattern:
     - [`create_sparsity_pattern!`](@ref) is a convenience method for performing the common
       task of calling `add_cell_entries!`, `add_interface_entries!`, and
       `add_constraint_entries!` after each other (see below).
     - [`add_cell_entries!`](@ref) adds entries for all couplings between the DoFs within
       each element. These entries correspond to assembling the standard element matrix and
       is thus almost always required.
     - [`add_interface_entries!`](@ref) adds entries for couplings between the DoFs in
       neighboring elements. These entries are required when integrating along internal
       interfaces between elements (e.g. for discontinuous Galerkin methods).
     - [`add_constraint_entries!`](@ref) adds entries required from constraints and boundary
       conditions in the ConstraintHandler. Note that this operation depends on existing
       entries in the pattern and *must* be called as the last operation on the pattern.
     - [`Ferrite.add_entry!`](@ref) adds a single entry to the pattern. This can be used if
       you need to add custom entries that are not covered by the other functions.

 3. **Instantiate the matrix:** A sparse matrix can be created from the sparsity pattern
    using [`allocate_matrix`](@ref), see [Instantiating the sparse matrix](@ref) below for
    more details.

### Increasing the sparsity

By default, when creating a sparsity pattern, it is assumed that each DoF within an element
couple with with *all* other DoFs in the element.

!!! todo
     - Discuss the `coupling` keyword argument.
     - Discuss the `keep_constrained` keyword argument.

### Blocked sparsity pattern

!!! todo
    Discuss `BlockSparsityPattern` and `BlockArrays` extension.

## Instantiating the sparse matrix

!!! todo
    Write text.
