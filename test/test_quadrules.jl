@testset "Quadrature testing" begin
    ref_tet_vol(dim) = 1 / factorial(dim)
    ref_square_vol(dim) = 2^dim


    function integrate(qr::QuadratureRule, f::Function)
        I = 0.0
        for (w, x) in zip(weights(qr), points(qr))
            I += w * f(x)
        end
        return I
    end

    # Cube
    for dim = (1,2,3)
        for order in (1,2,3,4)
            f = (x, p) -> sum([x[i]^p for i in 1:length(x)])
            qr = QuadratureRule(:legendre, Dim{dim}, RefCube(), order)
            @test integrate(qr, (x) -> f(x, 2*order-1)) < 1e-14
            @test sum(qr.weights) ≈ ref_square_vol(dim)
        end
    end

    # Tetrahedron
    g = (x) -> sqrt(sum(x))
    dim = 2
    for order in (2, 3)
        qr = QuadratureRule(:legendre, Dim{dim}, RefTetrahedron(), order)
        # http://www.wolframalpha.com/input/?i=integrate+sqrt(x%2By)+from+x+%3D+0+to+1,+y+%3D+0+to+1-x
        @test integrate(qr, g) - 0.4 < 0.01
        @test sum(qr.weights) ≈ ref_tet_vol(dim)
    end

    dim = 3
    for order in (2, 3)
        qr = QuadratureRule(:legendre, Dim{dim}, RefTetrahedron(), order)
        # Table 1:
        # http://www.m-hikari.com/ijma/ijma-2011/ijma-1-4-2011/venkateshIJMA1-4-2011.pdf
        @test integrate(qr, g) - 0.14 < 0.01
        @test sum(qr.weights) ≈ ref_tet_vol(dim)
    end

end
