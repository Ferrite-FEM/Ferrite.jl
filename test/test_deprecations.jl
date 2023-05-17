using Ferrite, Test

@testset "Deprecations" begin

@testset "Deprecation of auto-vectorized methods" begin
    # Deprecation of auto-selecing the interpolation
    grid = generate_grid(Quadrilateral, (1, 1))
    dh = DofHandler(grid)
    @test_deprecated r"interpolation explicitly, and vectorize it" add!(dh, :u, 2)
    @test_deprecated r"interpolation explicitly, and vectorize it" add!(dh, :p, 1)
    close!(dh)
    @test ndofs(dh) == 12
    # Deprecation of auto-vectorizing
    dh = DofHandler(grid)
    ip = Lagrange{2,RefCube,1}()
    @test_deprecated r"vectorize the interpolation" add!(dh, :u, 2, ip)
    @test_deprecated r"vectorize the interpolation" add!(dh, :p, 1, ip)
    close!(dh)
    @test ndofs(dh) == 12
end

@testset "Deprecation of (Cell|Face)(Scalar|Vector)Values" begin
    ip = Lagrange{2, RefCube, 1}()
    qr = QuadratureRule{2, RefCube}(2)
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
        @test (@test_deprecated r"RefLine" Lagrange{1, RefCube, order}()) === Lagrange{1, RefLine, order}()
    end
    for order in 1:5
        @test (@test_deprecated r"RefTriangle" Lagrange{2, RefTetrahedron, order}()) === Lagrange{2, RefTriangle, order}()
    end
    for order in 1:2
        @test (@test_deprecated r"RefQuadrilateral" Lagrange{2, RefCube, order}()) === Lagrange{2, RefQuadrilateral, order}()
    end
    for order in 1:2
        @test (@test_deprecated r"RefHexahedron" Lagrange{3, RefCube, order}()) === Lagrange{3, RefHexahedron, order}()
    end
    @test (@test_deprecated r"RefQuadrilateral" Serendipity{2, RefCube, 2}()) === Serendipity{2, RefQuadrilateral, 2}()
    @test (@test_deprecated r"RefHexahedron" Serendipity{3, RefCube, 2}()) === Serendipity{3, RefHexahedron, 2}()
    @test (@test_deprecated r"RefTriangle" CrouzeixRaviart{2, 1}()) === CrouzeixRaviart{2, RefTriangle, 1}()
    @test (@test_deprecated r"RefTriangle" BubbleEnrichedLagrange{2, RefTetrahedron, 1}()) === BubbleEnrichedLagrange{2, RefTriangle, 1}()
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
    @test (@test_deprecated r"RefLine" test_combo(FaceValues, 0, RefCube, (1,), Lagrange{RefLine, 1}())) isa FaceValues
    @test (@test_deprecated r"RefLine" test_combo(FaceValues, 0, RefCube, (:legendre, 1), Lagrange{RefLine, 1}())) isa FaceValues
    @test (@test_deprecated r"(RefLine.*RefQuadrilateral)" test_combo(FaceValues, 1, RefCube, (1,), Lagrange{RefQuadrilateral, 1}())) isa FaceValues
    @test (@test_deprecated r"likely this comes" test_combo(FaceValues, 1, RefCube, (1,), Lagrange{RefQuadrilateral, 1}())) isa FaceValues
    @test (@test_deprecated r"(RefLine.*RefQuadrilateral)" test_combo(FaceValues, 1, RefCube, (:legendre, 1), Lagrange{RefQuadrilateral, 1}())) isa FaceValues
    @test (@test_deprecated r"likely this comes" test_combo(FaceValues, 1, RefCube, (:legendre, 1), Lagrange{RefQuadrilateral, 1}())) isa FaceValues
    @test (@test_deprecated r"RefQuadrilateral.*RefHexahedron" test_combo(FaceValues, 2, RefCube, (1,), Lagrange{RefHexahedron, 1}())) isa FaceValues
    @test (@test_deprecated r"likely this comes" test_combo(FaceValues, 2, RefCube, (1,), Lagrange{RefHexahedron, 1}())) isa FaceValues
    @test (@test_deprecated r"RefQuadrilateral.*RefHexahedron" test_combo(FaceValues, 2, RefCube, (:legendre, 1), Lagrange{RefHexahedron, 1}())) isa FaceValues
    @test (@test_deprecated r"likely this comes" test_combo(FaceValues, 2, RefCube, (:legendre, 1), Lagrange{RefHexahedron, 1}())) isa FaceValues
    @test (@test_deprecated r"likely this comes" test_combo(CellValues, 2, RefTetrahedron, (1,), Lagrange{RefTriangle, 1}())) isa CellValues
    @test (@test_deprecated r"likely this comes" test_combo(CellValues, 2, RefTetrahedron, (:legendre, 1), Lagrange{RefTriangle, 1}())) isa CellValues
    @test (@test_deprecated r"RefTriangle" test_combo(FaceValues, 1, RefTetrahedron, (1,), Lagrange{RefTriangle, 1}())) isa FaceValues
    @test (@test_deprecated r"RefTriangle" test_combo(FaceValues, 1, RefTetrahedron, (:legendre, 1), Lagrange{RefTriangle, 1}())) isa FaceValues
end

end # testset deprecations
