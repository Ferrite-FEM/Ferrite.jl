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
* [`Ferrite.write_facetset`](@ref)
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
the `paraview_collection` can function from `WriteVTK.jl` can be used. This will create
one paraview datafile (`.pvd`) file and one `VTKGridFile` (`.vtu`) for each time step.

```@example export
using WriteVTK
pvd = paraview_collection("my_results")
for (step, t) in enumerate(range(0, 1, 5))
    # Do calculations to update u
    VTKGridFile("my_results_$step", dh) do vtk
        write_solution(vtk, dh, u)
        pvd[t] = vtk
    end
end
vtk_save(pvd);
```
See [Transient heat equation](@ref tutorial-transient-heat-equation) for an example

# Evaluating the Solution at a list of Points for Plotting using Plots.jl
The solution can also be evaluated at a list of points and plotted using `Plots.jl`. For the [heat equation](@ref tutorial-heat-equation) example it can be done like this:
```
using Plots
# The domain extends from -1 to 1 by default
xrange = yrange = range(-1.0, 1.0, length=100)
# evaluating outside of the domain returns NaN
points = [Vec((x, y)) for x in xrange for y in yrange];
ph = PointEvalHandler(grid, points)
u_points = evaluate_at_points(ph, dh, u, :u)
# reorganize the data for plotting
u_points = reshape(u_points, length(xrange), length(yrange))
heatmap(xrange, yrange, u_points, xlabel="x", ylabel="y", title="u(x, y)", aspect_ratio=:equal)
```
