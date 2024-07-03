```@setup export
using Ferrite
grid = generate_grid(Triangle, (2, 2))
dh = DofHandler(grid); add!(dh, :u, Lagrange{RefTriangle,1}()); close!(dh)
u = rand(ndofs(dh)); σ = rand(getncells(grid))
```

# Export

When the problem is solved, and the solution vector `u` is known we typically
want to visualize it. The simplest way to do this is to write the solution to a
VTK-file, which can be viewed in e.g. [`Paraview`](https://www.paraview.org/).
To write VTK-files, Ferrite comes with an export interface with a
[`WriteVTK.jl`](https://github.com/jipolanco/WriteVTK.jl) backend to simplify
the exporting.

The following structure can be used to write various output to a vtk-file:
```@example export
VTKGridFile("my_solution", grid) do vtk
    write_solution(vtk, dh, u)
end;
```
where `write_solution` is just one example of the following functions that can be used

* [`write_solution`](@ref)
* [`write_cell_data`](@ref)
* [`write_node_data`](@ref)
* [`write_projection`](@ref)
* [`Ferrite.write_cellset`](@ref)
* [`Ferrite.write_nodeset`](@ref)
* [`Ferrite.write_constraints`](@ref)
* [`Ferrite.write_cell_colors`](@ref)

Instead of using the `do`-block, it is also possible to do
```@example export
vtk = VTKGridFile("my_solution", grid)
write_solution(vtk, dh, u)
# etc.
close(vtk);
```

The data written by `write_solution`, `write_cell_data`, `write_node_data`, and `write_projection` may be either scalar (`Vector{<:Number}`) or tensor (`Vector{<:AbstractTensor}`) data.

For simulations with multiple time steps, typically one `VTK` (`.vtu`) file is written
for each time step. In order to connect the actual time with each of these files,
a `VTKFileCollection` can be used, which will write one paraview datafile (`.pvd`)
file and one `VTKGridFile` (`.vtu`) for each time step.

```@example export
pvd = VTKFileCollection("my_results", grid)
for t in range(0, 1, 5)
    # Do calculations to update u
    addstep!(pvd, t) do vtk
        write_solution(vtk, dh, u)
    end
end
close(pvd);
```
See [Transient heat equation](@ref tutorial-transient-heat-equation) for an example
