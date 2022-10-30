# P4est in Julia with Ferrite

All of it is based on these papers:

- [Original p4est paper](https://p4est.github.io/papers/BursteddeWilcoxGhattas11.pdf)
- [Extension to anisotropic refinement, aka p6est](https://epubs.siam.org/doi/10.1137/140974407)
- [Extension to RefTet elements and in depth explanations](https://bonndoc.ulb.uni-bonn.de/xmlui/handle/20.500.11811/7661); basically monography about t8code

## Important Concepts

One of the most important concepts, where everything is based on, are space filling curves (SFC).
In particular, [Z-order (also named Morton order, Morton space-filling curves)](https://en.wikipedia.org/wiki/Z-order_curve) are used in p4est.
The basic idea is that each Octant (in 3D) or quadrant (in 2D) can be encoded by 2 quantities

- the level `l`
- the lower left (front) coordinates `xyz`

Based on them a unique identifier, the morton index, can be computed.
The good part is, that the mapping from (`l`, `xyz`) -> `mortonidx(l,xyz)` is bijective, meaning we can flip the approach
and can construct each octant/quadrant solely by the `mortonidx`.

The current implementation of an octant looks currently like this:
```julia
struct OctantBWG{dim, N, M, T} <: AbstractCell{dim,N,M}
    #Refinement level
    l::T
    #x,y,z \in {0,...,2^b} where (0 ≤ l ≤ b)}
    xyz::NTuple{dim,T}
end
```
whenever coordinates are considered we follow the z order logic, meaning x before y before z.

The octree is implemented as:
```julia
struct OctreeBWG{dim,N,M,T} <: AbstractAdaptiveCell{dim,N,M}
    leaves::Vector{OctantBWG{dim,N,M,T}}
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
#_compute_size(b::Integer, l::Integer) in Ferrite at /home/mkoehler/repos/Ferrite.jl/src/Adaptivity/AdaptiveCells.jl:375
julia> Ferrite._compute_size(3,0)
8
```
Now, to fully understand the octree coordinate system we go a level down, i.e. we cut the space in x and y in half.
This means, that the octrees now of size 4.
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

The good news: it super cheap to compute octants/quadrants.
An important aspect of the morton index is that it's only consecutive on **one** level.
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
The morton index of the lower right cell is of course 2 on level 1.


## Current state and open questions

I implemented basic functionality to constructs and operate on octants/octrees.
In particular, I implemented from the p4est paper the Algorithms 1-7,14,15 (and refinement equivalent).

Currently, I'm at fulfilling the `AbstractGrid` interface which has some tricky parts.
All of the functionality is serial, nothing in terms of distributed is implemented.

### What we already can do
- refine octants
- coarsen octants
- compute neighbors of octants
- take a `Grid` and make a `ForestBWG` out of it

### Open questions
- How much do we cache
    - We only save very few things and so, sometimes we need to ad hoc compute things.
      **Example**: when determining the cell with cellid `i`. We need to go through all trees, determine the length of `leaves`
                   and take the sum in a cumulative way in order to detect to which tree `i` belongs to.
                   We could of course cache the number of leaves and get rid of `length(leaves)` calls for each tree
- How to count the nodes? (Actually most important one) In Algorithm 20 of the p4est paper is, an algorithm to count the unique nodes.
  However, as far as I understand the algorithm covers only important aspects of hanging nodes and distributed stuff.
  So, how to count the unique nodes? We need this for `getcoordinates` and the like.
  Maybe by computing face_neighbors and take their coordinate (and of course transform it later)
- What datastructure for the `leaves` ? Currently it's a vector but is there something which exploits that only contiguous pieces are added/removed?

### Open TODOs
- Coordinate transformation from octree coordinate system to physical coordinate system
- Octant dispatches that are required to fulfill `<:AbstractCell`
- Morton index can be computed much faster by methods I commented above the function
- more efficient `getcells(forest,i)`
