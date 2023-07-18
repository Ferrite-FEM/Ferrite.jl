```@setup export
using Ferrite
grid = generate_grid(Triangle, (2, 2))
dh = DofHandler(grid); add!(dh, :u, Lagrange{2,RefTetrahedron,1}()); close!(dh)
u = rand(ndofs(dh)); σ = rand(getncells(grid))
```

# Export

When the problem is solved, and the solution vector `u` is known we typically
want to visualize it. The simplest way to do this is to write the solution to a
VTK-file, which can be viewed in e.g. [`Paraview`](https://www.paraview.org/).
To write VTK-files, Ferrite comes with a filestream with a 
[`WriteVTK.jl`](https://github.com/jipolanco/WriteVTK.jl) backend to simplify
the exporting.

The following structure can be used to write various output to a vtk-file:

```@example export
VTKStream("my_solution", grid) do vtks
    write_solution(vtks, dh, u)
end
```
where `write_solution` is just one example of the following functions that can be used 

* [`write_solution`](@ref)
* [`write_celldata`](@ref)
* [`write_nodedata`](@ref)
* [`write_projected`](@ref)
* [`write_cellset`](@ref)
* [`write_nodeset`](@ref)
* [`write_dirichlet`](@ref)
* [`write_cell_colors`](@ref)

Instead of using the `do`-block, it is also possible to do
```@example export
vtks = VTKStream("my_solution", grid)
write_solution(vtks, dh, u)
# etc.
close(vtks)
```

The data written by `write_solution`, `write_celldata`, `write_nodedata`, and `write_projected` may be either scalar (`Vector{<:Number}`) or tensor (`Vector{<:AbstractTensor}`) data. 