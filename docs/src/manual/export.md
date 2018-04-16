```@setup export
using JuAFEM
grid = generate_grid(Triangle, (2, 2))
dh = DofHandler(grid); push!(dh, :u, 1); close!(dh)
u = rand(ndofs(dh)); σ = rand(getncells(grid))
```

# Export

When the problem is solved, and the solution vector `u` is known we typically
want to visualize it. The simplest way to do this is to write the solution to a
VTK-file, which can be viewed in e.g. [`Paraview`](https://www.paraview.org/).
To write VTK-files, JuAFEM wraps, functions from the
[`WriteVTK.jl`](https://github.com/jipolanco/WriteVTK.jl) in a submodule `VTK`.

First we need to create a file, based on the grid. This is done with the
[`VTK.grid`](@ref) function:

```@example export
vtk = VTK.grid("my-solution", grid)
# hide
```

Next we have to add data to the file. We may add different kinds of data;
point data using [`VTK.point_data`](@ref) or cell data using
[`VTK.cell_data`](@ref). Point data is data for each nodal coordinate in the
grid, for example our solution vector. Point data can be either scalars
or vectors. Cell data is -- as the name suggests -- data for each cell. This
can be for example the stress. As an example, lets add a solution vector `u`
as point data, and a vector with stress for each cell, `σ`, as cell data:

```@example export
VTK.point_data(vtk, u, "my-point-data")
VTK.cell_data(vtk,  σ, "my-cell-data")
# hide
```

Finally, we need to save the file to disk, using [`VTK.save`](@ref)

```@example export
VTK.save(vtk)
rm("my-solution.vtu") # hide
```

Alternatively, all of the above can be done using a `do` block:

```@example export
VTK.grid("my-solution", grid) do vtk
    VTK.point_data(vtk, u, "my-point-data")
    VTK.cell_data(vtk, σ, "my-cell-data")
end
rm("my-solution.vtu") # hide
```

For other functionality, and more information refer to the
[`WriteVTK.jl` README](https://github.com/jipolanco/WriteVTK.jl/blob/master/README.md).
In particular, for exporting the solution at multiple time steps, the
[section on PVD files](https://github.com/jipolanco/WriteVTK.jl#paraview-data-pvd-file-format)
is useful.

## Exporting with `DofHandler`

There is an even more convenient way to export a solution vector `u` -- using the
`DofHandler`. The `DofHandler` already contains all of the information needed,
such as the names of our fields and if they are scalar or vector fields. But most
importantly the `DofHandler` knows about the numbering and distribution of
degrees of freedom, and thus knows how to "distribute" the solution vector on
the grid. For example, lets say we have a `DofHandler` `dh` and a solution
vector `u`:

```@example export
vtk = VTK.grid("my-solution", dh)
VTK.point_data(vtk, dh, u)
VTK.save(vtk)
rm("my-solution.vtu") # hide
```

or with a `do`-block:

```@example export
VTK.grid("my-solution", dh) do vtk
    VTK.point_data(vtk, dh, u)
    VTK.cell_data(vtk, σ, "my-cell-data")
end
rm("my-solution.vtu") # hide
```

When `VTK.point_data` is used with a `DofHandler` all of the fields will be
written to the VTK file, and the names will be determined by the fieldname
symbol that was used when the field was added to the `DofHandler`.

## Exporting Boundary Conditions

There is also a `VTK.point_data` which accepts a `ConstraintHandler`.
This method is useful to verify that the boundary conditions are
applied where they are supposed to. For a `ConstraintHandler` `ch`
we can export the boundary conditions as

```julia
VTK.grid("boundary-conditions", grid) do vtk
    VTK.point_data(vtk, ch)
end
```

This will export zero-valued fields with ones on the parts where the
boundary conditions are active.
