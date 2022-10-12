```@meta
DocTestSetup = :(using Ferrite)
```

# Grid

## Mesh Reading

A Ferrite `Grid` can be generated with the [`generate_grid`](@ref) function. 
More advanced meshes can be imported with the 
[`FerriteMeshParser.jl`](https://github.com/Ferrite-FEM/FerriteMeshParser.jl) (from Abaqus input files),
or even created and translated with the [`Gmsh.jl`](https://github.com/JuliaFEM/Gmsh.jl) and [`FerriteGmsh.jl`](https://github.com/Ferrite-FEM/FerriteGmsh.jl) package, respectively.

### FerriteGmsh

`FerriteGmsh.jl` supports all defined cells with an alias in [`Ferrite.jl`](https://github.com/Ferrite-FEM/Ferrite.jl/blob/master/src/Grid/grid.jl#L39-L54) as well as the 3D Serendipity `Cell{3,20,6}`.
Either, a mesh is created on the fly with the gmsh API or a mesh in `.msh` or `.geo` format can be read and translated with the `FerriteGmsh.togrid` function.
```@docs
FerriteGmsh.togrid
```
`FerriteGmsh.jl` supports currently the translation of `cellsets` and `facesets`.
Such sets are defined in Gmsh as `PhysicalGroups` of dimension `dim` and `dim-1`, respectively.
In case only a part of the mesh is the domain, the domain can be specified by providing the keyword argument `domain` the name of the `PhysicalGroups` in the [`FerriteGmsh.togrid`](@ref) function.

!!! note "Why you should read a .msh file"
    Reading a `.msh` file is the advertised way, since otherwise you remesh whenver you run the code.
    Further, if you choose to read the grid directly from the current model of the gmsh API you get artificial nodes,
    which doesn't harm the FE computation, but maybe distort your sophisticated grid operations (if present).
    For more information, see [this issue](https://github.com/Ferrite-FEM/FerriteGmsh.jl/issues/20).

If you want to read another, not yet supported cell from gmsh, consider to open a PR at `FerriteGmsh` that extends the [`gmshtoferritecell` dict](https://github.com/Ferrite-FEM/FerriteGmsh.jl/blob/c9de4f64b3ad3c73fcb36758855a6e517c6d0d95/src/FerriteGmsh.jl#L6-L15)
and if needed, reorder the element nodes by dispatching [`FerriteGmsh.translate_elements`](https://github.com/Ferrite-FEM/FerriteGmsh.jl/blob/c9de4f64b3ad3c73fcb36758855a6e517c6d0d95/src/FerriteGmsh.jl#L17-L63).
The reordering of nodes is necessary if the Gmsh ordering doesn't match the one from Ferrite. Gmsh ordering is documented [here](https://gmsh.info/doc/texinfo/gmsh.html#Node-ordering).
For an exemplary usage of `Gmsh.jl` and `FerriteGmsh.jl`, consider the [Stokes Flow](@ref) and [Incompressible Navier-Stokes Equations via DifferentialEquations.jl](@ref) example.

### FerriteMeshParser

`FerriteMeshParser.jl` converts the mesh in an Abaqus input file (`.inp`) to a `Ferrite.Grid` with its function `get_ferrite_grid`.
The translations for most of Abaqus' standard 2d and 3d continuum elements to a `Ferrite.Cell` are defined.
Custom translations can be given as input, which can be used to import other (custom) elements or to override the default translation.
```@docs
FerriteMeshParser.get_ferrite_grid
```

If you are missing the translation of an Abaqus element that is equivalent to a `Ferrite.Cell`,
consider to open an [issue](https://github.com/Ferrite-FEM/FerriteMeshParser.jl/issues/new) or a pull request. 

## `Grid` Datastructure

In Ferrite a Grid is a collection of `Node`s and `Cell`s and is parameterized in its physical dimensionality and cell type.
`Node`s are points in the physical space and can be initialized by a N-Tuple, where N corresponds to the dimensions.

```julia
n1 = Node((0.0, 0.0))
```

`Cell`s are defined based on the `Node` IDs. Hence, they collect IDs in a N-Tuple.
Consider the following 2D mesh:

![global mesh](./assets/global_mesh.svg)

The cells of the grid can be described in the following way

```julia
julia> cells = [
              Quadrilateral((1,2,5,4)),
              Quadrilateral((2,3,6,5)),
              Quadrilateral((4,5,8,7)),
              Quadrilateral((5,6,9,8))
       ]
```

where each Quadrilateral, which is a subtype of `AbstractCell` saves in the field `nodes` the tuple of node IDs.
Additionally, the data structure `Grid` can hold node-, face- and cellsets. 
All of these three sets are defined by a dictionary that maps a string key to a `Set`. 
For the special case of node- and cellsets the dictionary's value is of type `Set{Int}`, i.e. a keyword is mapped to a node or cell ID, respectively. 

Facesets are a more elaborate construction. They map a `String` key to a `Set{FaceIndex}`, where each `FaceIndex` consists of `(global_cell_id, local_face_id)`.
In order to understand the `local_face_id` properly, one has to consider the reference space of the element, which typically is spanned by a product of the interval ``[-1, 1]`` and in this particular example ``[-1, 1] \times [-1, 1]``. 
In this space a local numbering of nodes and faces exists, i.e.


![local element](./assets/local_element.svg)


The example shows a local face ID ordering, defined as:

```julia
faces(::Lagrange{2,RefCube,1}) = ((1,2), (2,3), (3,4), (4,1))
```

Other face ID definitions [can be found in the src files](https://github.com/Ferrite-FEM/Ferrite.jl/blob/8224282ab4d67cb523ef342e4a6ceb1716764ada/src/interpolations.jl#L154) in the corresponding `faces` dispatch.


The highlighted face, i.e. the two lines from node ID 3 to 6 and from 6 to 9, on the right hand side of our test mesh can now be described as

```julia
julia> faces = [
           (3,6),
           (6,9)
       ]
```

The local ID can be constructed based on elements, corresponding faces and chosen interpolation, since the face ordering is interpolation dependent.
```julia
julia> function compute_faceset(cells, global_faces, ip::Interpolation{dim}) where {dim}
           local_faces = Ferrite.faces(ip)
           nodes_per_face = length(local_faces[1])
           d = Dict{NTuple{nodes_per_face, Int}, FaceIndex}()
           for (c, cell) in enumerate(cells) # c is global cell number
               for (f, face) in enumerate(local_faces) # f is local face number
                   # store the global nodes for the particular element, local face combination
                   d[ntuple(i-> cell.nodes[face[i]], nodes_per_face)] = FaceIndex(c, f)
               end
           end
       
           faces = Vector{FaceIndex}()
           for face in global_faces
               # lookup the element, local face combination for this face
               push!(faces, d[face])
           end
       
           return faces
       end

julia> interpolation = Lagrange{2, RefCube, 1}()

julia> compute_faceset(cells, faces, interpolation)
Vector{FaceIndex} with 2 elements:
  FaceIndex((2, 2))
  FaceIndex((4, 2))
```

Ferrite considers edges only in the three dimensional space. However, they share the concepts of faces in terms of `(global_cell_id,local_edge_id)` identifier.

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

## Topology

Ferrite.jl's `Grid` type offers experimental features w.r.t. topology information. The functions [`Ferrite.getneighborhood`](@ref) and [`Ferrite.faceskeleton`](@ref)
are the interface to obtain topological information. The [`Ferrite.getneighborhood`](@ref) can construct lists of directly connected entities based on a given entity (`CellIndex,FaceIndex,EdgeIndex,VertexIndex`).
The [`Ferrite.faceskeleton`](@ref) function can be used to evaluate integrals over material interfaces or computing element interface values such as jumps.
