# VTKHDF export: the struct lives here so `Ferrite.VTKHDFGridFile` always
# exists; all functionality is implemented in the FerriteVTKHDF package
# extension, which loads together with VTKHDF.jl.

"""
    VTKHDFGridFile(filename::AbstractString, grid::AbstractGrid; kwargs...)
    VTKHDFGridFile(filename::AbstractString, dh::DofHandler; kwargs...)

Create a `VTKHDFGridFile`, the [VTKHDF file format](https://docs.vtk.org/en/latest/vtk_file_formats/vtkhdf_file_format/index.html)
counterpart of [`VTKGridFile`](@ref). Requires
[VTKHDF.jl](https://github.com/Ferrite-FEM/VTKHDF.jl) to be loaded.
Keyword arguments are forwarded to `VTKHDF.vtkhdf_grid`.

The same data functions as for [`VTKGridFile`](@ref) are supported
([`write_solution`](@ref), [`write_cell_data`](@ref),
[`write_projection`](@ref), [`write_node_data`](@ref),
[`Ferrite.write_cellset`](@ref), ...), as well as the `do`-block form.

Unlike the XML-based formats, a VTKHDF file can hold a whole time series in
one file, with the grid stored only once. Open the file with `temporal = true`
and write each step with `VTKHDF.write_timestep`:

```julia
vtkhdf = VTKHDFGridFile("simulation", dh; temporal = true)
for (t, u) in timesteps
    write_timestep(vtkhdf, t) do vtk
        write_solution(vtk, dh, u)
    end
end
close(vtkhdf)
```
"""
struct VTKHDFGridFile{FT}
    vtk::FT  # ::FT <: VTKHDF.VTKHDFFile (FT unrestricted since VTKHDF is a weak dependency)
    cellnodes::Union{Vector{UnitRange{Int}}, Nothing}
    node_mapping::Union{Vector{Int}, Nothing}
end

write_discontinuous(vtk::VTKHDFGridFile) = vtk.cellnodes !== nothing
