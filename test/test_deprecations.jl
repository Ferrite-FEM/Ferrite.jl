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

end # testset deprecations
