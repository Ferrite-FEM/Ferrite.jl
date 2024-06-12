"""
    AbstractRefShape{refdim}

Supertype for all reference shapes with reference dimension `refdim`. Reference shapes are
used to define grid cells, interpolations, and quadrature rules.

Currently implemented reference shapes are: [`RefLine`](@ref), [`RefTriangle`](@ref),
[`RefQuadrilateral`](@ref), [`RefTetrahedron`](@ref), [`RefHexahedron`](@ref),
[`RefPrism`](@ref), and [`RefPyramid`](@ref).

# Examples
```julia
# Create a 1st order Lagrange interpolation on the reference triangle
interpolation = Lagrange{2, RefTriangle, 1}()

# Create a 2nd order quadrature rule for the reference quadrilateral
quad_rule = Quadrature{2, RefQuadrilateral}(2)
```

Implementation details can be found in the devdocs section on [Reference cells](@ref).
"""
abstract type AbstractRefShape{refdim} end

"""
    RefHypercube{dim} <: AbstractRefShape{dim}

Reference shape for a `dim`-dimensional hypercube. See [`AbstractRefShape`](@ref)
documentation for details.
"""
struct RefHypercube{refdim} <: AbstractRefShape{refdim} end

"""
    RefLine <: AbstractRefShape{1}

Reference line/interval, alias for [`RefHypercube{1}`](@ref). See [`AbstractRefShape`](@ref)
documentation for details.

# Extended help

```
----------------+--------------------
Vertex numbers: | Vertex coordinates:
  1-------2     | v1: 𝛏 = (-1.0,)
    --> ξ₁      | v2: 𝛏 = ( 1.0,)
----------------+--------------------
```
"""
const RefLine = RefHypercube{1}

"""
    RefQuadrilateral <: AbstractRefShape{2}

Reference quadrilateral, alias for [`RefHypercube{2}`](@ref). See [`AbstractRefShape`](@ref)
documentation for details.

# Extended help

```
----------------+---------------------
Vertex numbers: | Vertex coordinates:
    4-------3   |
    |       |   | v1: 𝛏 = (-1.0, -1.0)
    |       |   | v2: 𝛏 = ( 1.0, -1.0)
ξ₂^ |       |   | v3: 𝛏 = ( 1.0,  1.0)
  | 1-------2   | v4: 𝛏 = (-1.0,  1.0)
  +--> ξ₁       |
----------------+---------------------
Edge numbers:   | Edge identifiers:
    +---3---+   | e1: (v1, v2)
    |       |   | e2: (v2, v3)
    4       2   | e3: (v3, v4)
    |       |   | e4: (v4, v1)
    +---1---+   |
----------------+---------------------
```
"""
const RefQuadrilateral = RefHypercube{2}

"""
    RefHexahedron <: AbstractRefShape{3}

Reference hexahedron, alias for [`RefHypercube{3}`](@ref). See [`AbstractRefShape`](@ref)
documentation for details.

# Extended help

```
-----------------------------------------+-----------------------------
Vertex numbers:                          | Vertex coordinates:
            5--------8        5--------8 | v1: 𝛏 = (-1.0, -1.0, -1.0)
           /        /|       /|        | | v2: 𝛏 = ( 1.0, -1.0, -1.0)
          /        / |      / |        | | v3: 𝛏 = ( 1.0,  1.0, -1.0)
  ^ ξ₃   6--------7  |     6  |        | | v4: 𝛏 = (-1.0,  1.0, -1.0)
  |      |        |  4     |  1--------4 | v5: 𝛏 = (-1.0, -1.0,  1.0)
  +-> ξ₂ |        | /      | /        /  | v6: 𝛏 = ( 1.0, -1.0,  1.0)
 /       |        |/       |/        /   | v7: 𝛏 = ( 1.0,  1.0,  1.0)
ξ₁       2--------3        2--------3    | v8: 𝛏 = (-1.0,  1.0,  1.0)
-----------------------------------------+-----------------------------
Edge numbers:                            | Edge identifiers:
            +----8---+        +----8---+ |
          5/        /|      5/|        | |  e1: (v1, v2),  e2: (v2, v3)
          /       7/ |12    / |9     12| |  e3: (v3, v4),  e4: (v4, v1)
         +----6---+  |     +  |        | |  e5: (v5, v6),  e6: (v6, v7)
         |        |  +     |  +---4----+ |  e7: (v7, v8),  e8: (v8, v5)
       10|      11| /    10| /1       /  |  e9: (v1, v5), e10: (v2, v6)
         |        |/3      |/        /3  | e11: (v3, v7), e12: (v4, v8)
         +---2----+        +---2----+    |
-----------------------------------------+-----------------------------
Face numbers:                            | Face identifiers:
            +--------+        +--------+ |
           /   6    /|       /|        | |  f1: (v1, v4, v3, v2)
          /        / |      / |   5    | |  f2: (v1, v2, v6, v5)
         +--------+ 4|     +  |        | |  f3: (v2, v3, v7, v6)
         |        |  +     |2 +--------+ |  f4: (v3, v4, v8, v7)
         |    3   | /      | /        /  |  f5: (v1, v5, v8, v4)
         |        |/       |/    1   /   |  f6: (v5, v6, v7, v8)
         +--------+        +--------+    |
-----------------------------------------+-----------------------------
```
"""
const RefHexahedron = RefHypercube{3}

"""
    RefSimplex{dim} <: AbstractRefShape{dim}

Reference shape for a `dim`-dimensional simplex. See [`AbstractRefShape`](@ref)
documentation for details.
"""
struct RefSimplex{refdim} <: AbstractRefShape{refdim} end

@doc raw"""
    RefTriangle <: AbstractRefShape{2}

Reference triangle, alias for [`RefSimplex{2}`](@ref). See [`AbstractRefShape`](@ref)
documentation for details.

# Extended help

```
----------------+--------------------
Vertex numbers: | Vertex coordinates:
    2           |
    | \         | v1: 𝛏 = (1.0, 0.0)
    |   \       | v2: 𝛏 = (0.0, 1.0)
ξ₂^ |     \     | v3: 𝛏 = (0.0, 0.0)
  | 3-------1   |
  +--> ξ₁       |
----------------+--------------------
Edge numbers:   | Edge identifiers:
    +           |
    | \         | e1: (v1, v2)
    2   1       | e2: (v2, v3)
    |     \     | e3: (v3, v1)
    +---3---+   |
----------------+--------------------
```
"""
const RefTriangle = RefSimplex{2}

@doc raw"""
    RefTetrahedron <: AbstractRefShape{3}

Reference tetrahedron, alias for [`RefSimplex{3}`](@ref). See [`AbstractRefShape`](@ref)
documentation for details.

# Extended help

```
---------------------------------------+-------------------------
Vertex numbers:                        | Vertex coordinates:
             4                4        |
  ^ ξ₃      /  \             /| \      |  v1: 𝛏 = (0.0, 0.0, 0.0)
  |        /     \          / |   \    |  v2: 𝛏 = (1.0, 0.0, 0.0)
  +-> ξ₂  /        \       /  1___  \  |  v3: 𝛏 = (0.0, 1.0, 0.0)
 /       /      __--3     / /    __‾-3 |  v4: 𝛏 = (0.0, 0.0, 1.0)
ξ₁      2 __--‾‾         2/__--‾‾      |
---------------------------------------+-------------------------
Edge numbers:                          | Edge identifiers:
             +                +        | e1: (v1, v2)
            /  \             /| \      | e2: (v2, v3)
         5 /     \ 6      5 / |4  \ 6  | e3: (v3, v1)
          /        \       /  +__3  \  | e4: (v1, v4)
         /      __--+     / /1   __‾-+ | e5: (v2, v4)
        + __--‾‾2        +/__--‾‾2     | e6: (v3, v4)
---------------------------------------+-------------------------
Face numbers:                          | Face identifiers:
             +                +        |
            /  \             /| \      | f1: (v1, v3, v2)
           /     \          / | 4 \    | f2: (v1, v2, v4)
          /   3    \       /2 +___  \  | f3: (v2, v3, v4)
         /      __--+     / /  1 __‾-+ | f4: (v1, v4, v3)
        + __--‾‾         +/__--‾‾      |
---------------------------------------+-------------------------
```
"""
const RefTetrahedron = RefSimplex{3}

"""
    RefPrism <: AbstractRefShape{3}

Reference prism. See [`AbstractRefShape`](@ref) documentation for details.

# Extended help

```
-----------------------------------------+----------------------------
Vertex numbers:                          | Vertex coordinates:
            4-------/6       4--------6  |
           /     /   |      /|        |  |  v1: 𝛏 = (0.0, 0.0, 0.0)
          /   /      |     / |        |  |  v2: 𝛏 = (1.0, 0.0, 0.0)
  ^ ξ₃   5 /         |    5  |        |  |  v3: 𝛏 = (0.0, 1.0, 0.0)
  |      |          /3    |  1-------/3  |  v4: 𝛏 = (0.0, 0.0, 1.0)
  +-> ξ₂ |       /        | /     /      |  v5: 𝛏 = (1.0, 0.0, 1.0)
 /       |    /           |/   /         |  v6: 𝛏 = (0.0, 1.0, 1.0)
ξ₁       2 /              2 /            |
-----------------------------------------+----------------------------
Edge numbers:                            | Edge identifiers:
            +---8---/+       +---8----+  |
          7/     /   |     7/|        |  | e1: (v2, v1),  e2: (v1, v3)
          /   / 9    |6    / |3       |6 | e3: (v1, v4),  e4: (v3, v2)
         + /         |    +  |        |  | e5: (v2, v5),  e6: (v3, v6)
         |          /+    |  +--2----/+  | e7: (v4, v5),  e8: (v4, v6)
        5|       /       5| /1    /      | e9: (v6, v5)
         |    / 4         |/   / 4       |
         + /              + /            |
-----------------------------------------+----------------------------
Face numbers:                            | Face identifiers:
            +-------/+       +--------+  |
           /  5  /   |      /|        |  | f1: (v1, v3, v2)
          /   /      |     / |    3   |  | f2: (v1, v2, v5, v4)
         + /         |    +  |        |  | f3: (v3, v1, v4, v6)
         |     4    /+    |2 +-------/+  | f4: (v2, v3, v6, v5)
         |       /        | /  1  /      | f5: (v4, v5, v6)
         |    /           |/   /         |
         + /              + /            |
-----------------------------------------+----------------------------
```
"""
struct RefPrism <: AbstractRefShape{3} end

"""
    RefPyramid <: AbstractRefShape{3}

Reference pyramid. See [`AbstractRefShape`](@ref) documentation for details.

# Extended help

```
TODO: Add ascii art for the pyramid
```
"""
struct RefPyramid <: AbstractRefShape{3} end
