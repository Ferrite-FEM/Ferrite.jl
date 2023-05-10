using Ferrite, Test

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
    # Deprecation of (Cell|Face)VectorValues(::ScalarInterpolation)
    ip = Lagrange{2,RefCube,1}()
    qr = QuadratureRule{2,RefCube}(1)
    @test_deprecated r"scalar interpolations to CellVectorValues" CellVectorValues(qr, ip)
    @test_deprecated r"scalar interpolations to CellVectorValues" CellVectorValues(Float64, qr, ip)
    qr_f = QuadratureRule{1,RefCube}(1)
    @test_deprecated r"scalar interpolations to FaceVectorValues" FaceVectorValues(qr_f, ip)
    @test_deprecated r"scalar interpolations to FaceVectorValues" FaceVectorValues(Float64, qr_f, ip)
end
