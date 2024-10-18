using Ferrite, Test

@testset "Deprecations" begin

@testset "Deprecation of auto-vectorized methods" begin
    # Deprecation of auto-selecting the interpolation
    grid = generate_grid(Quadrilateral, (1, 1))
    dh = DofHandler(grid)
    @test_throws Ferrite.DeprecationError add!(dh, :u, 2)
    @test_throws Ferrite.DeprecationError add!(dh, :p, 1)
    close!(dh)
    @test ndofs(dh) == 0
    # Deprecation of auto-vectorizing
    dh = DofHandler(grid)
    ip = Lagrange{RefQuadrilateral,1}()
    @test_throws Ferrite.DeprecationError add!(dh, :u, 2, ip)
    @test_throws Ferrite.DeprecationError add!(dh, :p, 1, ip)
    close!(dh)
    @test ndofs(dh) == 0
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
    # Quadrature/(Cell|Face)Value combinations (sometimes warns in the QR constructor, sometimes it the FEValues constructor)
    function test_combo(constructor, qdim, qshape, qargs, ip)
        qr = QuadratureRule{qdim, qshape}(qargs...)
        constructor(qr, ip)
    end
    @test_throws Ferrite.DeprecationError test_combo(CellValues, 1, RefCube, (1,), Lagrange{RefLine, 1}())
    @test_throws Ferrite.DeprecationError test_combo(CellValues, 1, RefCube, (:legendre, 1), Lagrange{RefLine, 1}())
    @test_throws Ferrite.DeprecationError test_combo(CellValues, 2, RefCube, (1,), Lagrange{RefQuadrilateral, 1}())
    @test_throws Ferrite.DeprecationError test_combo(CellValues, 2, RefCube, (:legendre, 1), Lagrange{RefQuadrilateral, 1}())
    @test_throws Ferrite.DeprecationError test_combo(CellValues, 3, RefCube, (1,), Lagrange{RefHexahedron, 1}())
    @test_throws Ferrite.DeprecationError test_combo(CellValues, 3, RefCube, (:legendre, 1), Lagrange{RefHexahedron, 1}())
    @test_throws Ferrite.DeprecationError test_combo(FacetValues, 0, RefCube, (1,), Lagrange{RefLine, 1}())
    @test_throws Ferrite.DeprecationError test_combo(FacetValues, 0, RefCube, (:legendre, 1), Lagrange{RefLine, 1}())
    @test_throws Ferrite.DeprecationError test_combo(FacetValues, 1, RefCube, (1,), Lagrange{RefQuadrilateral, 1}())
    @test_throws Ferrite.DeprecationError test_combo(FacetValues, 1, RefCube, (1,), Lagrange{RefQuadrilateral, 1}())
    @test_throws Ferrite.DeprecationError test_combo(FacetValues, 1, RefCube, (:legendre, 1), Lagrange{RefQuadrilateral, 1}())
    @test_throws Ferrite.DeprecationError test_combo(FacetValues, 1, RefCube, (:legendre, 1), Lagrange{RefQuadrilateral, 1}())
    @test_throws Ferrite.DeprecationError test_combo(FacetValues, 2, RefCube, (1,), Lagrange{RefHexahedron, 1}())
    @test_throws Ferrite.DeprecationError test_combo(FacetValues, 2, RefCube, (1,), Lagrange{RefHexahedron, 1}())
    @test_throws Ferrite.DeprecationError test_combo(FacetValues, 2, RefCube, (:legendre, 1), Lagrange{RefHexahedron, 1}())
    @test_throws Ferrite.DeprecationError test_combo(FacetValues, 2, RefCube, (:legendre, 1), Lagrange{RefHexahedron, 1}())
    @test_throws Ferrite.DeprecationError test_combo(FacetValues, 1, RefTetrahedron, (1,), Lagrange{RefTriangle, 1}())
    @test_throws Ferrite.DeprecationError test_combo(FacetValues, 1, RefTetrahedron, (:legendre, 1), Lagrange{RefTriangle, 1}())
end

@testset "Ferrite.value and Ferrite.derivative" begin
    ip = Lagrange{RefQuadrilateral, 1}()
    ξ = zero(Vec{2})
    @test_throws Ferrite.DeprecationError Ferrite.value(ip, ξ)
    @test_throws Ferrite.DeprecationError Ferrite.derivative(ip, ξ)
    @test_throws Ferrite.DeprecationError Ferrite.value(ip, 1, ξ)
end

@testset "facesets" begin
    grid = generate_grid(Quadrilateral, (2,2))
    @test_throws Ferrite.DeprecationError addfaceset!(grid, "right_face", x -> x[1] ≈ 1)
    @test_throws Ferrite.DeprecationError addfaceset!(grid, "right_face_explicit", Set(Ferrite.FaceIndex(fi[1], fi[2]) for fi in getfacetset(grid, "right")))
end

@testset "vtk_grid" begin
    # Ensure no MethodError on pre v1.
    @test_throws Ferrite.DeprecationError vtk_grid("old", generate_grid(Line, (1,)))
end

@testset "onboundary" begin
    msg = "`onboundary` is deprecated, check just the facetset instead of first checking `onboundary`."
    @test_throws Ferrite.DeprecationError(msg) onboundary(first(CellIterator(generate_grid(Line, (2,)))), 1)
    msg = "`boundary_matrix` is not part of the Grid anymore and thus not a supported keyword argument."
    @test_throws Ferrite.DeprecationError(msg) Grid(Triangle[], Node{2,Float64}[]; boundary_matrix = something)
end

@testset "getdim" begin
    msg = "`Ferrite.getdim` is deprecated, use `getrefdim` or `getspatialdim` instead"
    @test_throws Ferrite.DeprecationError(msg) Ferrite.getdim(generate_grid(Line, (1,)))
    @test_throws Ferrite.DeprecationError(msg) Ferrite.getdim(Lagrange{RefTriangle,1}())
    @test_throws Ferrite.DeprecationError(msg) Ferrite.getdim(Line((1,2)))
end

@testset "getfielddim" begin
    msg = "`Ferrite.getfielddim(::AbstractDofHandler, args...) is deprecated, use `n_components` instead"
    dh = close!(add!(DofHandler(generate_grid(Triangle, (1,1))), :u, Lagrange{RefTriangle,1}()))
    @test_throws Ferrite.DeprecationError(msg) Ferrite.getfielddim(dh, Ferrite.find_field(dh, :u))
    @test_throws Ferrite.DeprecationError(msg) Ferrite.getfielddim(dh.subdofhandlers[1], :u)
end

@testset "default_interpolation" begin
    @test_throws Ferrite.DeprecationError Ferrite.default_interpolation(Triangle)
end

@testset "start_assemble" begin
    @test_throws Ferrite.DeprecationError start_assemble()
    @test_throws Ferrite.DeprecationError start_assemble(10)
end

@testset "celldofs!(::Vector, ::Cache)" begin
    grid = generate_grid(Quadrilateral, (1, 1))
    dh = DofHandler(grid)
    ip = Lagrange{RefQuadrilateral, 1}()
    add!(dh, :u, ip)
    close!(dh)
    cc = CellCache(dh)
    reinit!(cc, 1)
    v = Int[]
    @test_throws Ferrite.DeprecationError celldofs!(v, cc)
    fc = FacetCache(dh)
    reinit!(fc, FacetIndex(1, 1))
    @test_throws Ferrite.DeprecationError celldofs!(v, fc)
end

end # testset deprecations
