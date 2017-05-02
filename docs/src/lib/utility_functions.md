```@meta
CurrentModule = JuAFEM
DocTestSetup = quote
    using JuAFEM
end
```

# Utilities

```@index
Pages = ["utility_functions.md"]
```

## QuadratureRule

```@docs
getpoints
getweights
```

## Interpolation

```@docs
getnbasefunctions
getdim
getrefshape
getorder
```

## CellValues

```@docs
reinit!
getnquadpoints
getdetJdV

shape_value
shape_gradient
shape_symmetric_gradient
shape_divergence

function_value
function_gradient
function_symmetric_gradient
function_divergence
spatial_coordinate
```

## FaceValues

All of the methods for [`CellValues`](@ref) apply for `FaceValues` as well.
In addition, there are some methods that are unique for `FaecValues`:

```@docs
getcurrentface
```

## Assembling

```@docs
start_assemble
assemble!
end_assemble
```


## VTK

```@docs
vtk_grid
getVTKtype
```
