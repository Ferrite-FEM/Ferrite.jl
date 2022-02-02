```@meta
DocTestSetup = :(using Ferrite)
```

# Grid & AbstractGrid

## Grid

```@docs
Node
Cell
CellIndex
VertexIndex
EdgeIndex
FaceIndex
Grid
```

### Utility Functions

```@docs
getcells
getncells
getnodes
getnnodes
Ferrite.nnodes_per_cell
getcellset
getcellsets
getnodeset
getnodesets
getfaceset
getfacesets
getedgeset
getedgesets
getvertexset
getvertexsets
compute_vertex_values
transform!
getcoordinates
getcoordinates!
Ferrite.getneighbors
Ferrite.faceskeleton
```

### Grid Sets Utility

```@docs
addcellset!
addfaceset!
addnodeset!
```

### Multithreaded Assembly
```@docs
create_coloring
```


## AbstractGrid

It can be very useful to use a grid type for a certain special case, e.g. mixed cell types, adaptivity, IGA, etc.
In order to define your own `<: AbstractGrid` you need to fulfill the `AbstractGrid` interface.
In case that certain structures are preserved from the `Ferrite.Grid` type, you don't need to dispatch on your own type, but rather rely on the fallback `AbstractGrid` dispatch.

### Example

As a starting point, we choose a minimal working example from the test suite:

```julia
struct SmallGrid{dim,N,C<:Ferrite.AbstractCell} <: Ferrite.AbstractGrid{dim}
    nodes_test::Vector{NTuple{dim,Float64}}
    cells_test::NTuple{N,C}
end
```

Here, the names of the fields as well as their underlying datastructure changed compared to the `Grid` type. This would lead to the fact, that any usage
with the utility functions and DoF management will not work. So, we need to feed into the interface how to handle this subtyped datastructure.
We start with the utility functions that are associated with the cells of the grid:

```julia
Ferrite.getcells(grid::SmallGrid) = grid.cells_test
Ferrite.getcells(grid::SmallGrid, v::Union{Int, Vector{Int}}) = grid.cells_test[v]
Ferrite.getncells(grid::SmallGrid{dim,N}) where {dim,N} = N
Ferrite.getcelltype(grid::SmallGrid) = eltype(grid.cells_test)
Ferrite.getcelltype(grid::SmallGrid, i::Int) = typeof(grid.cells_test[i])
```

Next, we define some helper functions that take care of the node handling.

```julia
Ferrite.getnodes(grid::SmallGrid) = grid.nodes_test
Ferrite.getnodes(grid::SmallGrid, v::Union{Int, Vector{Int}}) = grid.nodes_test[v]
Ferrite.getnnodes(grid::SmallGrid) = length(grid.nodes_test)
Ferrite.nnodes_per_cell(grid::SmallGrid, i::Int=1) = Ferrite.nnodes(grid.cells_test[i])
Ferrite.n_faces_per_cell(grid::SmallGrid) = nfaces(eltype(grid.cells_test))
```

Finally, we define `getcoordinates`, which is an important function, if we want to assemble a problem.
The transformation from the reference space to the physical one requires information about the coordinates in order to construct the
Jacobian. The return of this part is later handled over to `reinit!`.

```julia
function Ferrite.getcoordinates!(x::Vector{Vec{dim,T}}, grid::SmallGrid, cell::Int) where {dim,T}
    for i in 1:length(x)
        x[i] = Vec{dim,T}(grid.nodes_test[grid.cells_test[cell].nodes[i]])
    end
end

function Ferrite.getcoordinates(grid::SmallGrid{dim}, cell::Int) where dim
    nodeidx = grid.cells_test[cell].nodes
    return [Vec{dim,Float64}(grid.nodes_test[i]) for i in nodeidx]::Vector{Vec{dim,Float64}}
end
```

Now, you would be able to assemble the heat equation example over the new custom `SmallGrid` type.
Note that this particular subtype isn't able to handle boundary entity sets and so, you can't describe boundaries with it.
In order to use boundaries, e.g. for Dirichlet constraints in the ConstraintHandler, you would need to dispatch the `AbstractGrid` sets utility functions on `SmallGrid`.
