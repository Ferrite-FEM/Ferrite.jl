```@meta
DocTestSetup = :(using Ferrite)
```
# Postprocessing

## Project to nodes
```@docs
L2Projector
project
```

```@docs
reshape_to_nodes
```

## VTK Export

```@docs
vtk_grid(filename::AbstractString, grid::Grid{dim,C,T}; compress::Bool) where {dim,C,T} 
vtk_point_data(vtk::WriteVTK.DatasetFile, data::Union{Vector{SymmetricTensor{2,dim,T,M}}},name::AbstractString) where {dim,T,M}
vtk_point_data(vtk::WriteVTK.DatasetFile, data::Union{ Vector{Tensor{order,dim,T,M}}, Vector{SymmetricTensor{order,dim,T,M}}}, name::AbstractString) where {order,dim,T,M}
vtk_cellset
```