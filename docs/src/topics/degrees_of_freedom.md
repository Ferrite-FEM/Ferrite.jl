```@setup dofs
using Ferrite
```

# Degrees of Freedom

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
interpolation describing the shape functions for the field. Here we add a scalar field `:p`,
interpolated using linear (degree 1) shape functions on a triangle, and a vector field `:u`,
also interpolated with linear shape functions on a triangle, but raised to the power 2 to
indicate that it is a vector field with 2 components (for a 2D problem).

```@example dofs
add!(dh, :p, Lagrange{RefTriangle, 1}())
add!(dh, :u, Lagrange{RefTriangle, 1}()^2)
# hide
```

Finally, when we have added all the fields, we have to `close!` the `DofHandler`.
When the `DofHandler` is closed it will traverse the grid and distribute all the
dofs for the fields we added.

```@example dofs
close!(dh)
```

## Ordering of Dofs

ordered in the same order as we add to dofhandler
vertices -> edges -> faces -> volumes
