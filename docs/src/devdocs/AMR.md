# AMR

## P4est

All of it is based on these papers:

- [BWG2011](@citet)
- [IBWG2015](@citet)

where almost everything is implemented in a serial way from the first paper.
Only certain specific algorithms of the second paper are implemented and there is a lot of open work to include the iterators of the second paper.
Look into the issues of Ferrite.jl and search for the AMR tag.

### Important Concepts

One of the most important concepts, where everything is based on, are space filling curves (SFC).
In particular, [Z-order (also named Morton order, Morton space-filling curves)](https://en.wikipedia.org/wiki/Z-order_curve) are used in p4est.
The basic idea is that each Octant (in 3D) or quadrant (in 2D) can be encoded by 2 quantities

- the level `l`
- the lower left (front) coordinates `xyz`

Based on them a unique identifier, the morton index, can be computed.
The mapping from (`l`, `xyz`) -> `mortonidx(l,xyz)` is bijective, meaning we can flip the approach
and can construct each octant/quadrant solely by the `mortonidx` and a given level `l`.

The current implementation of an octant looks currently like this:
```julia
struct OctantBWG{dim, N, T} <: AbstractCell{RefHypercube{dim}}
    #Refinement level
    l::T
    #x,y,z \in {0,...,2^b} where (0 ≤ l ≤ b)}
    xyz::NTuple{dim,T}
end
```
whenever coordinates are considered we follow the z order logic, meaning x before y before z.

The octree is implemented as:
```julia
struct OctreeBWG{dim,N,T} <: AbstractAdaptiveCell{RefHypercube{dim}}
    leaves::Vector{OctantBWG{dim,N,T}}
    #maximum refinement level
    b::T
    nodes::NTuple{N,Int}
end
```

So, only the leaves of the tree are stored and not any intermediate refinement level.
The field `b` is the maximum refinement level and is crucial. This parameter determines the size of the octree coordinate system.
The octree coordinate system is the coordinate system in which the coordinates `xyz` of any `octant::OctantBWG` are described.
This coordinate system goes from [0,2^b]^{dim}. The size of an octant is always 1 at the lowest possible level `b`.

### Examples

Let's say the maximum octree level is $b=3$, then the coordinate system is in 2D $[0,2^3]^2 = [0, 8]^2$.
So, our root is on level 0 of size 8 and has the lower left coordinates `(0,0)`

```julia
# different constructors available, first one OctantBWG(dim,level,mortonid,maximumlevel)
# other possibility by giving directly level and a tuple of coordinates OctantBWG(level,(x,y))
julia> oct = OctantBWG(2,0,1,3)
OctantBWG{2,4,4}
   l = 0
   xy = 0,0
```
The size of octants at a specific level can be computed by a simple operation
```julia
julia> Ferrite._compute_size(3,0)
8
```
Now, to fully understand the octree coordinate system we go a level down, i.e. we cut the space in $x$ and $y$ in half.
This means, that the octants are now of size 4.
```julia
julia> Ferrite._compute_size(3,1)
4
```
Construct all level 1 octants based on mortonid:
```julia
# note the arguments are dim,level,mortonid,maximumlevel
julia> oct = OctantBWG(2,1,1,3)
OctantBWG{2,4,4}
   l = 1
   xy = 0,0

julia> o = OctantBWG(2,1,2,3)
OctantBWG{2,4,4}
   l = 1
   xy = 4,0

julia> o = OctantBWG(2,1,3,3)
OctantBWG{2,4,4}
   l = 1
   xy = 0,4

julia> o = OctantBWG(2,1,4,3)
OctantBWG{2,4,4}
   l = 1
   xy = 4,4
```

So, the morton index is on **one** specific level just a x before y before z "cell" or "element" identifier
```
x-----------x-----------x
|           |           |
|           |           |
|     3     |     4     |
|           |           |
|           |           |
x-----------x-----------x
|           |           |
|           |           |
|     1     |     2     |
|           |           |
|           |           |
x-----------x-----------x
```

The operation to compute octants/quadrants is cheap, since it is just bitshifting.
An important aspect of the morton index is that it's only consecutive on **one** level in this specific implementation.
Note that other implementation exists that incorporate the level integer within the morton identifier and by that have a unique identifier across levels.
If you have a tree like this below:

```
x-----------x-----------x
|           |           |
|           |           |
|     9     |    10     |
|           |           |
|           |           |
x-----x--x--x-----------x
|     |6 |7 |           |
|  3  x--x--x           |
|     |4 |5 |           |
x-----x--x--x     8     |
|     |     |           |
|  1  |  2  |           |
x-----x-----x-----------x
```

you would maybe think this is the morton index, but strictly speaking it is not.
What we see above is just the `leafindex`, i.e. the index where you find this leaf in the `leaves` array of `OctreeBWG`.
Let's try to construct the lower right based on the morton index on level 1

```julia
julia> o = OctantBWG(2,1,8,3)
ERROR: AssertionError: m ≤ (one(T) + one(T)) ^ (dim * l)
Stacktrace:
 [1] OctantBWG(dim::Int64, l::Int32, m::Int32, b::Int32)
   @ Ferrite ~/repos/Ferrite.jl/src/Adaptivity/AdaptiveCells.jl:23
 [2] OctantBWG(dim::Int64, l::Int64, m::Int64, b::Int64)
   @ Ferrite ~/repos/Ferrite.jl/src/Adaptivity/AdaptiveCells.jl:43
 [3] top-level scope
   @ REPL[93]:1
```

The assertion expresses that it is not possible to construct a morton index 8 octant, since the upper bound of the morton index is 4 on level 1.
The morton index of the lower right cell is 2 on level 1.

```julia
julia> o = OctantBWG(2,1,2,3)
OctantBWG{2,4,4}
   l = 1
   xy = 4,0
```

### Octant operation

There are multiple useful functions to compute information about an octant e.g. parent, childs, etc.

```@docs
Ferrite.AMR.isancestor
Ferrite.AMR.morton
Ferrite.AMR.children
Ferrite.AMR.vertices
Ferrite.AMR.edges
Ferrite.AMR.faces
Ferrite.AMR.transform_pointBWG
```

### Intraoctree operation

Intraoctree operation stay within one octree and compute octants that are attached in some way to a pivot octant `o`.
These operations are useful to collect unique entities within a single octree or to compute possible neighbors of `o`.
[BWG2011](@citet) Algorithm 5, 6, and 7 describe the following intraoctree operations:

```@docs
Ferrite.AMR.corner_neighbor
Ferrite.AMR.edge_neighbor
Ferrite.AMR.face_neighbor
Ferrite.AMR.possibleneighbors
```

### Interoctree operation

Interoctree operation are in contrast to intraoctree operation by computing octant transformations across different octrees.
Thereby, one needs to account for topological connections between the octrees as well as possible rotations of the octrees.
[BWG2011](@citet) Algorithm 8, 10, and 12 explain the algorithms that are implemented in the following functions:

```@docs
Ferrite.AMR.transform_corner
Ferrite.AMR.transform_edge
Ferrite.AMR.transform_face
```

Note that we flipped the input and to expected output logic a bit to the proposed algorithms of the paper.
However, the original proposed versions are implemented as well in:

```@docs
Ferrite.AMR.transform_corner_remote
Ferrite.AMR.transform_edge_remote
Ferrite.AMR.transform_face_remote
```

despite being never used in the code base so far.
