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
function(s) we are solving for. To add a field we need a name (a `Symbol`) and we also
need to specify number of components for the field. Here we add a vector field `:u`
(2 components for a 2D problem) and a scalar field `:p`.

```@example dofs
add!(dh, :u, Lagrange{RefTriangle, 1}()^2)
add!(dh, :p, Lagrange{RefTriangle, 1}())
# hide
```

Finally, when we have added all the fields, we have to `close!` the `DofHandler`.
When the `DofHandler` is closed it will traverse the grid and distribute all the
dofs for the fields we added.

```@example dofs
close!(dh)
```

### Specifying interpolation for a field

In the example above we did not specify which interpolation should be used for our fields
`:u` and `:p`. By default iso-parametric elements will be used meaning that the
interpolation that matches the grid will be used -- for a linear grid a linear
interpolation will be used etc. It is sometimes useful to separate the grid interpolation
from the interpolation that is used to approximate our fields
(e.g. sub- and super-parametric elements).

We can specify which interpolation that should be used for the approximation when we add
the fields to the dofhandler. For example, here we add our vector field `:u` with a
quadratic interpolation, and our `:p` field with a linear approximation.

```@example dofs
dh = DofHandler(grid) # hide
add!(dh, :u, Lagrange{RefTriangle, 2}()^2)
add!(dh, :p, Lagrange{RefTriangle, 1}())
# hide
```

## Ordering of Dofs

ordered in the same order as we add to dofhandler
nodes -> (edges ->) faces -> cells
