# Imports for parallel (isolated) test execution:
using LinearAlgebra
using Random

# Reference cell volumes and centroids
const REFERENCE_GEOMETRY = Dict(
    RefLine => (2.0, Vec((0.0,))),
    RefQuadrilateral => (4.0, Vec((0.0, 0.0))),
    RefHexahedron => (8.0, Vec((0.0, 0.0, 0.0))),
    RefTriangle => (1 / 2, Vec((1 / 3, 1 / 3))),
    RefTetrahedron => (1 / 6, Vec((1 / 4, 1 / 4, 1 / 4))),
    RefPrism => (1 / 2, Vec((1 / 3, 1 / 3, 1 / 2))),
    RefPyramid => (1 / 3, Vec((3 / 8, 3 / 8, 1 / 4))),
)

# Geometry-only cell and facet values
function geometry_values(ip_geo, order)
    RefShape = getrefshape(ip_geo)
    cv = CellValues(QuadratureRule{RefShape}(order), (;), ip_geo)
    fv = FacetValues(FacetQuadratureRule{RefShape}(order), (;), ip_geo)
    return cv, fv
end

random_vec(rng, dim) = Vec{dim}(Tuple(rand(rng, dim)))

@testset "geometry-only values" begin
    # Values objects without function interpolations (empty NamedTuple, `(;)`) work with the
    # iterators, and agree with values with a function interpolation
    grid = generate_grid(Quadrilateral, (2, 3))
    topology = ExclusiveTopology(grid)
    ip_geo = Lagrange{RefQuadrilateral, 1}()
    ip = DiscontinuousLagrange{RefQuadrilateral, 0}()
    qr = QuadratureRule{RefQuadrilateral}(2)
    fqr = FacetQuadratureRule{RefQuadrilateral}(2)
    cv, cv_ip = CellValues(qr, (;), ip_geo), CellValues(qr, ip)
    fv, fv_ip = FacetValues(fqr, (;), ip_geo), FacetValues(fqr, ip)
    iv = InterfaceValues(fqr, (;), ip_geo)
    for cell in CellIterator(grid)
        reinit!(cv, cell)
        reinit!(cv_ip, cell)
        x = getcoordinates(cell)
        @test spatial_volume(cv) ≈ spatial_volume(cv_ip) ≈ 2 / 3
        @test spatial_centroid(cv, x) ≈ spatial_centroid(cv_ip, x) ≈ sum(x) / 4
    end
    for fc in FacetIterator(grid, getfacetset(grid, "left"))
        reinit!(fv, fc)
        reinit!(fv_ip, fc)
        x = getcoordinates(fc)
        @test spatial_volume(fv) ≈ spatial_volume(fv_ip) ≈ 2 / 3
        @test spatial_centroid(fv, x) ≈ spatial_centroid(fv_ip, x)
        @test spatial_centroid(fv, x)[1] ≈ -1
        @test all(q -> getnormal(fv, q) ≈ Vec(-1.0, 0.0), 1:getnquadpoints(fv))
    end
    for ic in InterfaceIterator(grid, topology)
        reinit!(iv, ic)
        @test spatial_volume(iv.here) ≈ spatial_volume(iv.there)
        @test spatial_centroid(iv.here, getcoordinates(ic.a)) ≈ spatial_centroid(iv.there, getcoordinates(ic.b))
    end
end

@testset "affine cells ($CT)" for CT in (
        Line, QuadraticLine, Triangle, QuadraticTriangle, Quadrilateral, QuadraticQuadrilateral,
        Tetrahedron, QuadraticTetrahedron, Hexahedron, QuadraticHexahedron,
        Wedge, QuadraticWedge, Pyramid,
    )
    rng = Xoshiro(1)
    ip_geo = geometric_interpolation(CT)
    RefShape = getrefshape(ip_geo)
    rdim = Ferrite.getrefdim(ip_geo)
    # Random affine map x = A ξ + b (with positive determinant)
    A = one(Tensor{2, rdim}) + Tensor{2, rdim}(Tuple(rand(rng, rdim * rdim))) / 3
    b = random_vec(rng, rdim)
    @test det(A) > 0
    x = [A ⋅ ξ + b for ξ in Ferrite.reference_coordinates(ip_geo)]
    V_ref, xc_ref = REFERENCE_GEOMETRY[RefShape]
    # Any rule that is exact for linear polynomials is sufficient for affine cells
    cv, fv = geometry_values(ip_geo, 2)
    reinit!(cv, x)
    V = spatial_volume(cv)
    xc = spatial_centroid(cv, x)
    @test V ≈ det(A) * V_ref
    @test xc ≈ A ⋅ xc_ref + b
    # Facets: closed surface, ∑ |Γ| n = 0, and the divergence theorem ∑ |Γ| xc ⋅ n = rdim V
    S = zero(Vec{rdim})
    divsum = 0.0
    for f in 1:Ferrite.nfacets(RefShape)
        reinit!(fv, x, f)
        n = getnormal(fv, 1) # planar facets
        @test (spatial_centroid(fv, x) - xc) ⋅ n > 0 # pointing out of the cell
        @test spatial_area_vector(fv) ≈ spatial_volume(fv) * n
        S += spatial_area_vector(fv)
        divsum += spatial_volume(fv) * (spatial_centroid(fv, x) ⋅ n)
    end
    @test norm(S) < 1.0e-12 * V
    @test divsum ≈ rdim * V
end

@testset "quadrature dependence" begin
    # The volume of an affine cell is exact with any rule, but the centroid requires a rule
    # that is exact for linear polynomials (here: a single point in the reference centroid)
    ip_geo = Lagrange{RefQuadrilateral, 1}()
    x = [Vec(0.0, 0.0), Vec(2.0, 0.0), Vec(3.0, 1.0), Vec(1.0, 1.0)] # parallelogram
    cv_center = CellValues(QuadratureRule{RefQuadrilateral}(1), (;), ip_geo)
    cv_corner = CellValues(QuadratureRule{RefQuadrilateral}([4.0], [Vec(-1.0, -1.0)]), (;), ip_geo)
    reinit!(cv_center, x)
    reinit!(cv_corner, x)
    @test spatial_volume(cv_center) ≈ spatial_volume(cv_corner) ≈ 2
    @test spatial_centroid(cv_center, x) ≈ Vec(1.5, 0.5)
    @test spatial_centroid(cv_corner, x) ≈ x[1]
    # For a (non-affine) trapezoid the single point rule gives the exact volume, but not the
    # exact centroid, which requires a 2x2 rule
    x = [Vec(0.0, 0.0), Vec(2.0, 0.0), Vec(1.5, 1.0), Vec(0.5, 1.0)]
    cv2 = CellValues(QuadratureRule{RefQuadrilateral}(2), (;), ip_geo)
    reinit!(cv_center, x)
    reinit!(cv2, x)
    @test spatial_volume(cv_center) ≈ spatial_volume(cv2) ≈ 3 / 2
    @test spatial_centroid(cv2, x) ≈ Vec(1.0, 4 / 9)
    @test !(spatial_centroid(cv_center, x) ≈ spatial_centroid(cv2, x))
end

@testset "curved cells ($CT)" for (CT, order, qr_ref) in (
        (Quadrilateral, 2, QuadratureRule{RefQuadrilateral}(8)),
        (Hexahedron, 2, QuadratureRule{RefHexahedron}(8)),
        (QuadraticLine, 2, QuadratureRule{RefLine}(8)),
        (QuadraticTriangle, 4, QuadratureRule{RefTriangle}(:gaussjacobi, 15)),
        (QuadraticQuadrilateral, 3, QuadratureRule{RefQuadrilateral}(8)),
        (QuadraticTetrahedron, 5, QuadratureRule{RefTetrahedron}(:polyquad, 10)),
        (QuadraticHexahedron, 4, QuadratureRule{RefHexahedron}(8)),
    )
    # Randomly perturbed nodes give non-affine (bilinear, trilinear) and curved (quadratic)
    # cells. The orders listed in the docstring of `spatial_centroid` should give the same result
    # as a high order quadrature rule.
    rng = Xoshiro(2)
    ip_geo = geometric_interpolation(CT)
    rdim = Ferrite.getrefdim(ip_geo)
    x = [ξ + (random_vec(rng, rdim) - Vec{rdim}(_ -> 0.5)) / 5 for ξ in Ferrite.reference_coordinates(ip_geo)]
    cv, _ = geometry_values(ip_geo, order)
    cv_ref = CellValues(qr_ref, (;), ip_geo)
    reinit!(cv, x)
    reinit!(cv_ref, x)
    @test spatial_volume(cv) ≈ spatial_volume(cv_ref) rtol = 1.0e-13
    @test spatial_centroid(cv, x) ≈ spatial_centroid(cv_ref, x) rtol = 1.0e-13
    # The area vectors of the (non-flat) facets sum to zero over the closed surface
    _, fv = geometry_values(ip_geo, order)
    S = sum(1:Ferrite.nfacets(getrefshape(ip_geo))) do f
        reinit!(fv, x, f)
        spatial_area_vector(fv)
    end
    @test norm(S) < 1.0e-13 * spatial_volume(cv)
end

@testset "warped facet" begin
    # Lifting one corner of a hexahedron makes the top facet non-planar: the normal varies
    # over the facet, and only the integrated area vectors sum to zero
    ip_geo = Lagrange{RefHexahedron, 1}()
    x = Ferrite.reference_coordinates(ip_geo)
    x[7] += Vec(0.0, 0.0, 0.5)
    _, fv = geometry_values(ip_geo, 2)
    reinit!(fv, x, 6)
    @test !(getnormal(fv, 1) ≈ getnormal(fv, getnquadpoints(fv)))
    S = zero(Vec{3})
    S_first = zero(Vec{3})
    for f in 1:6
        reinit!(fv, x, f)
        S += spatial_area_vector(fv)
        S_first += spatial_volume(fv) * getnormal(fv, 1)
    end
    @test norm(S) < 1.0e-14
    @test norm(S_first) > 0.1
end

@testset "embedded cells" begin
    # Quadrilateral in 3D, in the plane z = 1
    x = [Vec(0.0, 0.0, 1.0), Vec(2.0, 0.0, 1.0), Vec(2.0, 1.0, 1.0), Vec(0.0, 1.0, 1.0)]
    cv = CellValues(QuadratureRule{RefQuadrilateral}(2), (;), Lagrange{RefQuadrilateral, 1}()^3)
    reinit!(cv, x)
    @test spatial_volume(cv) ≈ 2
    @test spatial_centroid(cv, x) ≈ Vec(1.0, 0.5, 1.0)
end
