```@setup dofs
using Ferrite
```

# Degrees of freedom

The distribution and numbering of degrees of freedom (dofs) are handled by the `DofHandler`.
The DofHandler will be used to query information about the dofs. For example we can obtain
the dofs for a particular cell, which we need when assembling the system.

The `DofHandler` is based on the grid. Here we create a simple grid
with Triangle cells, and then create a `DofHandler` based on the grid

```@example dofs
grid = generate_grid(Triangle, (20, 20))
dh = DofHandler(grid)
# hide
```

## Fields

Before we can distribute the dofs we need to specify fields. A field is simply the unknown
function(s) we are solving for. To add a field we need a name (a `Symbol`) and the the
interpolation describing the shape functions for the field. Here we add a scalar field `:u`,
interpolated using linear (degree 1) shape functions on a triangle, and a vector field `:v`,
also interpolated with linear shape functions on a triangle, but raised to the power 2 to
indicate that it is a vector field with 2 components (for a 2D problem).

```@example dofs
interpolation_u = Lagrange{RefTriangle, 1}()
interpolation_v = Lagrange{RefTriangle, 1}() ^ 2

add!(dh, :u, interpolation_u)
add!(dh, :v, interpolation_v)
# hide
```

Finally, when we have added all the fields, we have to `close!` the `DofHandler`.
When the `DofHandler` is closed it will traverse the grid and distribute all the
dofs for the fields we added.

```@example dofs
close!(dh)
```

## Local DoF indices

Locally on each element the DoFs are ordered by field, in the same order they were added
to the DofHandler. Within each field the DoFs follow the order of the interpolation.
Concretely this means that the local matrix is a block matrix (and the local vector a block
vector).

For the example DofHandler defined above, with the two fields `:u` and `:v`, the local
system would look something like:

```math
\begin{bmatrix}
    K^e_{uu} & K^e_{uv} \\
    K^e_{vu} & K^e_{vv}
\end{bmatrix}
\begin{bmatrix}
    u^e \\
    v^e
\end{bmatrix} =
\begin{bmatrix}
    f^e_u \\
    f^e_v
\end{bmatrix}
```

where

 - ``K^e_{uu}`` couples test and trial functions for the `:u` field
 - ``K^e_{uv}`` couples test functions for the `:u` field with trial functions for the `:v` field
 - ``K^e_{vu}`` couples test functions for the `:v` field with trial functions for the `:u` field
 - ``K^e_{vv}`` couples test and trial functions for the `:v` field

We can query the local index ranges for the dofs of the two fields with the
[`dof_range`](@ref) function:

```@example dofs
u_range = dof_range(dh, :u)
v_range = dof_range(dh, :v)

u_range, v_range
```

i.e. the local indices for the `:u` field are `1:3` and for the `:v` field `4:9`. This
matches directly with the number of dofs for each field: 3 for `:u` and 6 for `:v`. The
ranges are used when assembling the blocks of the matrix, i.e. if `Ke` is the local matrix,
then `Ke[u_range, u_range]` corresponds to the ``K_{uu}``, `Ke[u_range, v_range]` to
``K_{uv}``, etc. See for example [Tutorial 8: Stokes flow](../tutorials/stokes-flow.md) for
how the ranges are used.


## Global DoF indices

The global ordering of dofs, however, does *not* follow any specific order by default. In
particular, they are *not* ordered by field, so while the local system is a block system,
the global system is not. For all intents and purposes, the default global dof ordering
should be considered an implementation detail.

!!! warning "DoF numbering is decoupled from the node numbering"
    A common pitfall for new Ferrite users is to assume that the numbering of DoFs follows the
    numbering of the nodes in the grid. This is *not* the case. While it would be possible
    to align these two numberings in some special cases (single isoparametric scalar field)
    it is not possible in general.

We can query the global dofs for a cell with the `celldofs` function. For example, for cell
#45 the global dofs are:

```@example dofs
global_dofs = celldofs(dh, 45)
```

which looks more or less random. We can also look at the mapping between local and global
dofs for field `:u`:

```@example dofs
u_range .=> global_dofs[u_range]
```

and for field `:v`:

```@example dofs
v_range .=> global_dofs[v_range]
```

which makes it clear that `:u`-dofs and `:v`-dofs are interleaved in the global numbering.


## Renumbering global DoF indices

The global DoF order determines the sparsity pattern of the global matrix. In some cases,
mostly depending on the linear solve strategy, it can be beneficial to reorder the global
DoFs. For example:

 - For a direct solver the sparsity pattern determines the
   [fill-in](https://en.wikipedia.org/wiki/Sparse_matrix#Reducing_fill-in). An optimized
   order can reduce the fill-in and thus the memory consumption and the computational cost.
   (For a iterative solver the order doesn't matter as much since the order doesn't affect
   number of operations in a matrix-vector product.)
 - Using block-based solvers is possible for multi-field problems. In this case it is
   simpler if the global system is also ordered by field and/or components similarly to the
   local system.

It is important to note that renumbering the global dofs does *not* affect the local order.
The local system is *always* blocked by fields as described above.

The default ordering (which, again, should be considered a black box) results in the
following sparsity pattern:

```@example dofs
allocate_matrix(dh)
nothing # hide
```

!!! details "Default sparsity pattern"
    ```@example dofs
    allocate_matrix(dh) # hide
    ```

This looks pretty good (all entries are concentrated around the diagonal) but it is
important to note that this is just a happy coincidence because i) we used the built-in grid
generator (which numbers neighboring cells consecutively) and because ii) Ferrite by default
distributes global dofs cell by cell.

### Renumbering for a global block system

In order to obtain a global block system it is possible to renumber by fields, or even by
component, using the [`renumber!`](@ref) function with the [`DofOrder.FieldWise`](@ref) and
[`DofOrder.ComponentWise`](@ref) orders, respectively.

For example, renumbering by fields gives the global block system
```math
\begin{bmatrix}
    K_{uu} & K_{uv} \\
    K_{vu} & K_{vv}
\end{bmatrix}
```

```@example dofs
renumber!(dh, DofOrder.FieldWise())
allocate_matrix(dh)
nothing # hide
```

!!! details "Sparsity pattern after field-wise renumbering"
    ```@example dofs
    allocate_matrix(dh) # hide
    ```

Similarly, global blocking can be done by components, and fields and/or component can be
permuted in the global system. See [`DofOrder.FieldWise`](@ref) and
[`DofOrder.ComponentWise`](@ref) for more details.

### Renumbering to reduce fill-in

Ferrite can be used together with the [Metis.jl](https://github.com/JuliaSparse/Metis.jl)
package to optimize the global DoF ordering w.r.t. reduced fill-in:

```@example dofs
using Metis
renumber!(dh, DofOrder.Ext{Metis}())
allocate_matrix(dh)
nothing # hide
```

!!! details "Sparsity pattern after Metis renumbering"
    ```@example dofs
    allocate_matrix(dh) # hide
    ```

The [`DofOrder.FieldWise`](@ref) and [`DofOrder.ComponentWise`](@ref) orders preserve
internal ordering within the fields/components so they can be combined with Metis to reduce
fill-in for the individual blocks, for example:

```@example dofs
renumber!(dh, DofOrder.Ext{Metis}())
renumber!(dh, DofOrder.FieldWise())
allocate_matrix(dh)
nothing # hide
```

!!! details "Sparsity pattern after Metis and field-wise renumbering"
    ```@example dofs
    allocate_matrix(dh) # hide
    ```
