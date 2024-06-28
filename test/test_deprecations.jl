using Ferrite, Test

@testset "Deprecations" begin

@testset "Deprecation of auto-vectorized methods" begin
    # Deprecation of auto-selecting the interpolation
    grid = generate_grid(Quadrilateral, (1, 1))
    dh = DofHandler(grid)
    @test_deprecated r"interpolation explicitly, and vectorize it" add!(dh, :u, 2)
    @test_deprecated r"interpolation explicitly, and vectorize it" add!(dh, :p, 1)
    close!(dh)
    @test ndofs(dh) == 12
    # Deprecation of auto-vectorizing
    dh = DofHandler(grid)
    ip = Lagrange{RefQuadrilateral,1}()
    @test_deprecated r"vectorize the interpolation" add!(dh, :u, 2, ip)
    @test_deprecated r"vectorize the interpolation" add!(dh, :p, 1, ip)
    close!(dh)
    @test ndofs(dh) == 12
end

@testset "Deprecation of (Cell|Face)(Scalar|Vector)Values" begin
    ip = Lagrange{RefQuadrilateral, 1}()
    qr = QuadratureRule{RefQuadrilateral}(2)
    for CVType in (
            CellScalarValues, CellVectorValues,
            FaceScalarValues, FaceVectorValues,
            PointScalarValues, PointVectorValues,
        )
        err = try CVType(qr, ip) catch e sprint(showerror, e) end
        @test occursin("merged into a single type", err)
        @test occursin("CHANGELOG", err)
    end
    # Smoke tests to see that old code still loads
    function foo(cv::CellScalarValues{D,T,R}) where {D, T, R} end
    function bar(cv::CellVectorValues{D,T}) where {D, T} end
    function baz(cv::PointVectorValues{D}) where {D} end
end

@testset "Deprecation of old RefShapes" begin
    # Interpolations
    for order in 1:2
        @test (@test_deprecated r"RefLine" Lagrange{1, RefCube, order}()) === Lagrange{RefLine, order}()
    end
    for order in 1:5
        @test (@test_deprecated r"RefTriangle" Lagrange{2, RefTetrahedron, order}()) === Lagrange{RefTriangle, order}()
    end
    for order in 1:2
        @test (@test_deprecated r"RefQuadrilateral" Lagrange{2, RefCube, order}()) === Lagrange{RefQuadrilateral, order}()
    end
    for order in 1:2
        @test (@test_deprecated r"RefHexahedron" Lagrange{3, RefCube, order}()) === Lagrange{RefHexahedron, order}()
    end
    @test (@test_deprecated r"RefQuadrilateral" Serendipity{2, RefCube, 2}()) === Serendipity{RefQuadrilateral, 2}()
    @test (@test_deprecated r"RefHexahedron" Serendipity{3, RefCube, 2}()) === Serendipity{RefHexahedron, 2}()
    @test (@test_deprecated r"RefTriangle" CrouzeixRaviart{2, 1}()) === CrouzeixRaviart{RefTriangle, 1}()
    @test (@test_deprecated r"RefTriangle" BubbleEnrichedLagrange{2, RefTetrahedron, 1}()) === BubbleEnrichedLagrange{RefTriangle, 1}()
    # Quadrature/(Cell|Face)Value combinations (sometimes warns in the QR constructor, sometimes it the FEValues constructor)
    function test_combo(constructor, qdim, qshape, qargs, ip)
        qr = QuadratureRule{qdim, qshape}(qargs...)
        constructor(qr, ip)
    end
    @test (@test_deprecated r"RefLine.*RefQuadrilateral" test_combo(CellValues, 1, RefCube, (1,), Lagrange{RefLine, 1}())) isa CellValues
    @test (@test_deprecated r"RefLine.*RefQuadrilateral" test_combo(CellValues, 1, RefCube, (:legendre, 1), Lagrange{RefLine, 1}())) isa CellValues
    @test (@test_deprecated r"RefQuadrilateral.*RefHexahedron" test_combo(CellValues, 2, RefCube, (1,), Lagrange{RefQuadrilateral, 1}())) isa CellValues
    @test (@test_deprecated r"RefQuadrilateral.*RefHexahedron" test_combo(CellValues, 2, RefCube, (:legendre, 1), Lagrange{RefQuadrilateral, 1}())) isa CellValues
    @test (@test_deprecated r"RefHexahedron" test_combo(CellValues, 3, RefCube, (1,), Lagrange{RefHexahedron, 1}())) isa CellValues
    @test (@test_deprecated r"RefHexahedron" test_combo(CellValues, 3, RefCube, (:legendre, 1), Lagrange{RefHexahedron, 1}())) isa CellValues
    @test (@test_deprecated r"RefLine" test_combo(FacetValues, 0, RefCube, (1,), Lagrange{RefLine, 1}())) isa FacetValues
    @test (@test_deprecated r"RefLine" test_combo(FacetValues, 0, RefCube, (:legendre, 1), Lagrange{RefLine, 1}())) isa FacetValues
    @test (@test_deprecated r"(RefLine.*RefQuadrilateral)" test_combo(FacetValues, 1, RefCube, (1,), Lagrange{RefQuadrilateral, 1}())) isa FacetValues
    @test (@test_deprecated r"likely this comes" test_combo(FacetValues, 1, RefCube, (1,), Lagrange{RefQuadrilateral, 1}())) isa FacetValues
    @test (@test_deprecated r"(RefLine.*RefQuadrilateral)" test_combo(FacetValues, 1, RefCube, (:legendre, 1), Lagrange{RefQuadrilateral, 1}())) isa FacetValues
    @test (@test_deprecated r"likely this comes" test_combo(FacetValues, 1, RefCube, (:legendre, 1), Lagrange{RefQuadrilateral, 1}())) isa FacetValues
    @test (@test_deprecated r"RefQuadrilateral.*RefHexahedron" test_combo(FacetValues, 2, RefCube, (1,), Lagrange{RefHexahedron, 1}())) isa FacetValues
    @test (@test_deprecated r"likely this comes" test_combo(FacetValues, 2, RefCube, (1,), Lagrange{RefHexahedron, 1}())) isa FacetValues
    @test (@test_deprecated r"RefQuadrilateral.*RefHexahedron" test_combo(FacetValues, 2, RefCube, (:legendre, 1), Lagrange{RefHexahedron, 1}())) isa FacetValues
    @test (@test_deprecated r"likely this comes" test_combo(FacetValues, 2, RefCube, (:legendre, 1), Lagrange{RefHexahedron, 1}())) isa FacetValues
    @test (@test_deprecated r"RefTriangle" test_combo(FacetValues, 1, RefTetrahedron, (1,), Lagrange{RefTriangle, 1}())) isa FacetValues
    @test (@test_deprecated r"RefTriangle" test_combo(FacetValues, 1, RefTetrahedron, (:legendre, 1), Lagrange{RefTriangle, 1}())) isa FacetValues
end

@testset "Ferrite.value and Ferrite.derivative" begin
    ip = Lagrange{RefQuadrilateral, 1}()
    ξ = zero(Vec{2})
    @test (@test_deprecated Ferrite.value(ip, ξ)) == [Ferrite.reference_shape_value(ip, ξ, i) for i in 1:getnbasefunctions(ip)]
    @test (@test_deprecated Ferrite.derivative(ip, ξ)) == [Ferrite.reference_shape_gradient(ip, ξ, i) for i in 1:getnbasefunctions(ip)]
    @test (@test_deprecated Ferrite.value(ip, 1, ξ)) == Ferrite.reference_shape_value(ip, ξ, 1)
end

@testset "facesets" begin
grid = generate_grid(Quadrilateral, (2,2))
addfaceset!(grid, "right_face", x -> x[1] ≈ 1)
@test getfacetset(grid, "right_face") == getfacetset(grid, "right")
addfaceset!(grid, "right_face_explicit", Set(Ferrite.FaceIndex(fi[1], fi[2]) for fi in getfacetset(grid, "right")))
@test getfacetset(grid, "right_face_explicit") == getfacetset(grid, "right")
end

@testset "vtk_grid" begin
    # Ensure no MethodError on pre v1.
    @test_throws ErrorException vtk_grid("old", generate_grid(Line, (1,)))
end

@testset "onboundary" begin
    msg = "`onboundary` is deprecated, check just the facetset instead of first checking `onboundary`."
    @test_throws ErrorException(msg) onboundary(first(CellIterator(generate_grid(Line, (2,)))), 1)
    msg = "`boundary_matrix` is not part of the Grid anymore and thus not a supported keyword argument."
    @test_throws ErrorException(msg) Grid(Triangle[], Node{2,Float64}[]; boundary_matrix = something)
end

@testset "getdim" begin
    msg = "`Ferrite.getdim` is deprecated, use `getrefdim` or `getspatialdim` instead"
    @test_throws ErrorException(msg) Ferrite.getdim(generate_grid(Line, (1,)))
    @test_throws ErrorException(msg) Ferrite.getdim(Lagrange{RefTriangle,1}())
    @test_throws ErrorException(msg) Ferrite.getdim(Line((1,2)))
end

@testset "getfielddim" begin
    msg = "`Ferrite.getfielddim(::AbstractDofHandler, args...) is deprecated, use `n_components` instead"
    dh = close!(add!(DofHandler(generate_grid(Triangle, (1,1))), :u, Lagrange{RefTriangle,1}()))
    @test_throws ErrorException(msg) Ferrite.getfielddim(dh, Ferrite.find_field(dh, :u))
    @test_throws ErrorException(msg) Ferrite.getfielddim(dh.subdofhandlers[1], :u)
end

@testset "default_interpolation" begin
    @test Ferrite.default_interpolation(Triangle) == geometric_interpolation(Triangle)
end

end # testset deprecations
