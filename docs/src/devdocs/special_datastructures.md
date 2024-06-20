# Special data structures

## `ArrayOfVectorViews`
`ArrayOfVectorViews` is a data structure representing an `Array` of
vector views (specifically `SubArray{T, 1} where T`). By arranging all
data (of type `T`) continuously in memory, this will significantly reduce
the garbage collection time compared to using an `Array{AbstractVector{T}}`. While the data in each view can be mutated, the length of each view is
fixed after construction.

```@docs
Ferrite.ArrayOfVectorViews
Ferrite.ConstructionBuffer
Ferrite.push_at_index!
```
