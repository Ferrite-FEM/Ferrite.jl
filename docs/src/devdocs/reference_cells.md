# Reference cells

The reference cells are used to i) define grid cells, ii) define shape functions, and iii)
define quadrature rules. The numbering of vertices, edges, faces are visualized below. See
also
[`FerriteViz.elementinfo`](https://ferrite-fem.github.io/FerriteViz.jl/dev/api/#FerriteViz.elementinfo).

## Numbering and identification of entities

The local numbering of vertices, edges, and faces, on the reference cells, specifies, for
example, the cell node order, and the order in which interpolations distribute their DoFs.
It is important for internal consistency and correctness that the the same convention is
used everywhere. The convention adopted by Ferrite.jl is documented below for each reference
cell.

**Edges** are identified by the 2-tuple of vertices it connects, starting with the smallest
vertex number, i.e. ``(v_i, v_j)`` where ``i < j``.

**Faces** are identified by the n-tuple of vertices constructing the face in anti-clockwise
order (viewing the face from the outside of the reference cell), starting with the smallest
vertex number, i.e. ``(v_i, v_j, v_k, ...)`` where ``i < j, i < k, i < ...``.

## Reference cell implementations

### Reference line

```@raw html
<figure>
<img src="../../assets/ref-line.svg" />
<figcaption><em>
Numbering of the vertices and the edge for the reference line. Source: <a
href="https://defelement.com/img/ref-interval.html">DefElement</a> (<a
href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a>) <sup
class="footnote-reference"><a id="citeref-1" href="#footnote-1">[1]</a></sup>.
</em></figcaption>
</figure>
```

**Vertex coordinates**
```math
\boldsymbol{\xi}_1 = (-1, ), \quad
\boldsymbol{\xi}_2 = ( 1, )
```

**Edge identifiers**
```math
e_1 = (v_1, v_2)
```

### Reference triangle

```@raw html
<figure>
<img src="../../assets/ref-triangle.svg" />
<figcaption><em>
Numbering of the vertices, edges, and the face for the reference triangle. Source: <a
href="https://defelement.com/img/ref-triangle.html">DefElement</a> (<a
href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a>) <sup
class="footnote-reference"><a id="citeref-1" href="#footnote-1">[1]</a></sup>.
</em></figcaption>
</figure>
```

**Vertex coordinates**
```math
\boldsymbol{\xi}_1 = (1, 0), \quad
\boldsymbol{\xi}_2 = (0, 1), \quad
\boldsymbol{\xi}_3 = (0, 0), \quad
```

**Edge identifiers**
```math
e_1 = (v_1, v_2), \quad
e_2 = (v_2, v_3), \quad
e_3 = (v_3, v_1)
```

### Reference quadrilateral

```@raw html
<figure>
<img src="../../assets/ref-quadrilateral.svg" />
<figcaption><em>
Numbering of the vertices, edges, and the face for the reference quadrilateral. Source: <a
href="https://defelement.com/img/ref-quadrilateral.html">DefElement</a> (<a
href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a>) <sup
class="footnote-reference"><a id="citeref-1" href="#footnote-1">[1]</a></sup>.
</em></figcaption>
</figure>
```

**Vertex coordinates**
```math
\boldsymbol{\xi}_1 = (-1, -1), \quad
\boldsymbol{\xi}_2 = ( 1, -1), \quad
\boldsymbol{\xi}_3 = ( 1,  1), \quad
\boldsymbol{\xi}_4 = (-1,  1)
```

**Edge identifiers**
```math
e_1 = (v_1, v_2), \quad
e_2 = (v_2, v_3), \quad
e_3 = (v_3, v_4), \quad
e_4 = (v_4, v_1)
```

### Reference tetrahedron

```@raw html
<figure>
<img src="../../assets/ref-tetrahedron.svg" />
<figcaption><em>
Numbering of the vertices, edges, faces, and the cell for the reference tetrahedron. Source:
<a href="https://defelement.com/img/ref-tetrahedron.html">DefElement</a> (<a
href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a>) <sup
class="footnote-reference"><a id="citeref-1" href="#footnote-1">[1]</a></sup>.
</em></figcaption>
</figure>
```

**Vertex coordinates**
```math
\boldsymbol{\xi}_1 = (0, 0, 0), \quad
\boldsymbol{\xi}_2 = (1, 0, 0), \quad
\boldsymbol{\xi}_3 = (0, 1, 0), \quad
\boldsymbol{\xi}_4 = (0, 0, 1)
```

**Edge identifiers**
```math
e_1 = (v_1, v_2), \quad
e_2 = (v_2, v_3), \quad
e_3 = (v_3, v_1), \quad
e_4 = (v_1, v_4), \quad
e_5 = (v_2, v_4), \quad
e_6 = (v_3, v_4)
```

**Face identifiers**
```math
f_1 = (v_1, v_3, v_2), \quad
f_2 = (v_1, v_2, v_4), \quad
f_3 = (v_2, v_3, v_4), \quad
f_4 = (v_1, v_4, v_3)
```

### Reference hexahedron

```@raw html
<figure>
<img src="../../assets/ref-hexahedron.svg" />
<figcaption><em>
Numbering of the vertices, edges, faces, and the cell for the reference hexahedron. Source:
<a href="https://defelement.com/img/ref-hexahedron.html">DefElement</a> (<a
href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a>) <sup
class="footnote-reference"><a id="citeref-1" href="#footnote-1">[1]</a></sup>.
</em></figcaption>
</figure>
```

**Vertex coordinates**
```math
\boldsymbol{\xi}_1 = (-1, -1, -1), \quad
\boldsymbol{\xi}_2 = ( 1, -1, -1), \quad
\boldsymbol{\xi}_3 = ( 1,  1, -1), \quad
\boldsymbol{\xi}_4 = (-1,  1, -1), \\
\boldsymbol{\xi}_5 = (-1, -1,  1), \quad
\boldsymbol{\xi}_6 = ( 1, -1,  1), \quad
\boldsymbol{\xi}_7 = ( 1,  1,  1), \quad
\boldsymbol{\xi}_8 = (-1,  1,  1)
```

**Edge identifiers**
```math
e_1 = (v_1, v_2), \quad
e_2 = (v_2, v_3), \quad
e_3 = (v_3, v_4), \quad
e_4 = (v_4, v_1), \quad
e_5 = (v_5, v_6), \quad
e_6 = (v_6, v_7), \\
e_7 = (v_7, v_8), \quad
e_8 = (v_8, v_5), \quad
e_9 = (v_1, v_5), \quad
e_{10} = (v_2, v_6), \quad
e_{11} = (v_3, v_7), \quad
e_{12} = (v_4, v_8)
```

**Face identifiers**
```math
f_1 = (v_1, v_4, v_3, v_2), \quad
f_2 = (v_1, v_2, v_6, v_5), \quad
f_3 = (v_2, v_3, v_7, v_6), \\
f_4 = (v_3, v_4, v_8, v_7), \quad
f_5 = (v_1, v_5, v_8, v_4), \quad
f_6 = (v_5, v_6, v_7, v_8)
```

### Reference prism

```@raw html
<figure>
<img src="../../assets/ref-prism.svg" />
<figcaption><em>
Numbering of the vertices, edges, faces, and the cell for the reference prism. Source:
<a href="https://defelement.com/img/ref-prism.html">DefElement</a> (<a
href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a>) <sup
class="footnote-reference"><a id="citeref-1" href="#footnote-1">[1]</a></sup>.
</em></figcaption>
</figure>
```

**Vertex coordinates**
```math
\boldsymbol{\xi}_1 = (0, 0, 0), \quad
\boldsymbol{\xi}_2 = (1, 0, 0), \quad
\boldsymbol{\xi}_3 = (0, 1, 0), \\
\boldsymbol{\xi}_4 = (0, 0, 1), \quad
\boldsymbol{\xi}_5 = (1, 0, 1), \quad
\boldsymbol{\xi}_6 = (0, 1, 1)
```

**Edge identifiers**
```math
e_1 = (v_1, v_2), \quad
e_2 = (v_1, v_3), \quad
e_3 = (v_1, v_4), \quad
e_4 = (v_2, v_3), \quad
e_5 = (v_2, v_5), \\
e_6 = (v_3, v_6), \quad
e_7 = (v_4, v_5), \quad
e_8 = (v_4, v_6), \quad
e_9 = (v_5, v_6)
```

**Face identifiers**
```math
f_1 = (v_1, v_3, v_2), \quad
f_2 = (v_1, v_2, v_5, v_4), \quad
f_3 = (v_1, v_4, v_6, v_3), \\
f_4 = (v_2, v_3, v_6, v_5), \quad
f_5 = (v_4, v_5, v_6), \quad
```

### `AbstractRefShape` subtypes

```@docs
Ferrite.AbstractRefShape
Ferrite.RefSimplex
Ferrite.RefHypercube
Ferrite.RefLine
Ferrite.RefTriangle
Ferrite.RefQuadrilateral
Ferrite.RefTetrahedron
Ferrite.RefHexahedron
Ferrite.RefPrism
```

[^1]: All figures from [DefElement](https://defelement.com/) are used under
      [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The figures are modified to
      follow Ferrite.jl numbering, to use (``\xi_1``, ``\xi_2``, ``\xi_3``) for the
      coordinate axes, and to use julia logo colors.
