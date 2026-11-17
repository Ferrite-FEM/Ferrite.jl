# Geometric quantities (volumes, centroids and area vectors) of cells and facets, computed
# by quadrature with values objects.

"""
    spatial_volume(cv::CellValues)
    spatial_volume(fv::FacetValues)

Return the volume of the cell (or facet) that the values were last reinitialized for, i.e.
``\\int \\mathrm{d}\\Omega``, computed as the sum of [`getdetJdV`](@ref) over the
quadrature points. Here, "volume" refers to the measure in the dimension of the entity, i.e.
the volume of a facet is its area (in 3D) or length (in 2D), consistent with `getdetJdV`
for `FacetValues`.

The result is exact if the quadrature rule integrates ``\\det \\boldsymbol{J}`` exactly.
For affine cells (lines, triangles, tetrahedra, parallelograms, etc.) ``\\det \\boldsymbol{J}``
is constant and any quadrature rule gives the exact volume. For other geometries the
required order depends on the geometric interpolation, see [`spatial_centroid`](@ref).

Since only the geometric mapping is used, the values do not need any function
interpolation. They can be constructed with an empty `NamedTuple`, `(;)`, and the geometric
interpolation of the cells:
```julia
ip_geo = geometric_interpolation(Quadrilateral)
cv = CellValues(QuadratureRule{RefQuadrilateral}(2), (;), ip_geo)
for cell in CellIterator(grid)
    reinit!(cv, cell)
    V = spatial_volume(cv)
    xc = spatial_centroid(cv, getcoordinates(cell))
end
```

# Examples
```jldoctest
julia> grid = generate_grid(Quadrilateral, (2, 2));

julia> cv = CellValues(QuadratureRule{RefQuadrilateral}(2), (;), Lagrange{RefQuadrilateral, 1}());

julia> reinit!(cv, getcoordinates(grid, 1));

julia> spatial_volume(cv) ≈ 1
true
```
"""
function spatial_volume(v::Union{AbstractCellValues, FacetValues})
    V = getdetJdV(v, 1)
    for q_point in 2:getnquadpoints(v)
        V += getdetJdV(v, q_point)
    end
    return V
end

"""
    spatial_centroid(cv::CellValues, x::AbstractVector{<:Vec})
    spatial_centroid(fv::FacetValues, x::AbstractVector{<:Vec})

Return the centroid,
``\\boldsymbol{x}_c = \\int \\boldsymbol{x} \\,\\mathrm{d}\\Omega / \\int \\mathrm{d}\\Omega``,
of the cell (or facet) that the values were last reinitialized for. `x` are the coordinates
of the cell.

The result is exact if the quadrature rule integrates both ``\\det \\boldsymbol{J}`` and
``\\boldsymbol{x} \\det \\boldsymbol{J}`` exactly. Note that this is a requirement on the
polynomial degree that the rule integrates exactly, and not on the location of the points.
For example, a single point rule returns the mapped quadrature point as the centroid. For
affine cells, this is the exact centroid only if the point is located at the centroid of
the reference shape (as for the default single point rules). For non-affine cells, a single
point rule is never sufficient in general. Sufficient rules for the cells in Ferrite are:
 - Affine cells (`Line`, `Triangle`, `Tetrahedron`, and parallelogram `Quadrilateral`s and
   parallelepiped `Hexahedron`s): `QuadratureRule{RefShape}(1)`, or any other rule that is
   exact for linear polynomials.
 - General (bilinear) `Quadrilateral`s and (trilinear) `Hexahedron`s:
   `QuadratureRule{RefShape}(2)`.
 - `QuadraticTriangle` and `QuadraticTetrahedron`: `QuadratureRule{RefShape}(4)` and
   `QuadratureRule{RefShape}(5)`, respectively.
 - `QuadraticQuadrilateral` and `QuadraticHexahedron`: `QuadratureRule{RefShape}(3)` and
   `QuadratureRule{RefShape}(4)`, respectively.
The same orders are sufficient for the facets, with `FacetQuadratureRule`. See
[`spatial_volume`](@ref) for how to construct values without any function interpolation.

# Examples
```jldoctest
julia> grid = generate_grid(Quadrilateral, (2, 2));

julia> cv = CellValues(QuadratureRule{RefQuadrilateral}(2), (;), Lagrange{RefQuadrilateral, 1}());

julia> x = getcoordinates(grid, 1);

julia> reinit!(cv, x);

julia> spatial_centroid(cv, x) ≈ Vec(-0.5, -0.5)
true
```
"""
function spatial_centroid(v::Union{AbstractCellValues, FacetValues}, x::AbstractVector{<:Vec})
    V = getdetJdV(v, 1)
    xV = spatial_coordinate(v, 1, x) * V
    for q_point in 2:getnquadpoints(v)
        dV = getdetJdV(v, q_point)
        V += dV
        xV += spatial_coordinate(v, q_point, x) * dV
    end
    return xV / V
end

"""
    spatial_area_vector(fv::FacetValues)

Return the area vector,
``\\boldsymbol{S} = \\int \\boldsymbol{n} \\,\\mathrm{d}\\Gamma``,
of the facet that the values were last reinitialized for, where ``\\boldsymbol{n}`` is the
outward unit normal (see [`getnormal`](@ref)).

For a flat facet ``\\boldsymbol{S}`` is the area (length in 2D) times the normal, i.e.
`spatial_volume(fv) * getnormal(fv, q_point)` for any quadrature point. For non-flat facets,
e.g. warped quadrilateral facets of a `Hexahedron` or the curved facets of quadratic cells,
the normal varies over the facet and ``\\boldsymbol{S}`` is the integral of the normal. The
area vectors of the facets of a cell always sum to zero (since the surface of the cell is
closed), which e.g. makes them suitable for flux computations in finite volume methods.

The result is exact if the quadrature rule integrates
``\\boldsymbol{n} \\,\\mathrm{d}\\Gamma`` exactly. This integrand is a polynomial for the
cells in Ferrite, and the rules listed in [`spatial_centroid`](@ref) are sufficient.

# Examples
```jldoctest
julia> grid = generate_grid(Quadrilateral, (2, 2));

julia> fv = FacetValues(FacetQuadratureRule{RefQuadrilateral}(2), (;), Lagrange{RefQuadrilateral, 1}());

julia> reinit!(fv, getcoordinates(grid, 1), 1); # The bottom facet of the bottom left cell

julia> spatial_area_vector(fv) ≈ Vec(0.0, -1.0)
true
```
"""
function spatial_area_vector(fv::FacetValues)
    S = getnormal(fv, 1) * getdetJdV(fv, 1)
    for q_point in 2:getnquadpoints(fv)
        S += getnormal(fv, q_point) * getdetJdV(fv, q_point)
    end
    return S
end
