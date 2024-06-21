# Special data structures

## `ArrayOfVectorViews`
`ArrayOfVectorViews` is a data structure representing an `Array` of
vector views (specifically `SubArray{T, 1} where T`). By arranging all
data (of type `T`) continuously in memory, this will significantly reduce
the garbage collection time compared to using an `Array{AbstractVector{T}}`. While the data in each view can be mutated, the length of each view is
fixed after construction.
This data structure provides two features not provided by `ArraysOfArrays.jl`: Support of matrices and higher order arrays for storing vectors
of different dimensions and efficient construction when the number of elements in each view is not known in advance.

```@docs
Ferrite.ArrayOfVectorViews
Ferrite.ConstructionBuffer
Ferrite.push_at_index!
```
