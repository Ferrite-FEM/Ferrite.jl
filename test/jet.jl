# JET.jl tests that run on CI only for released julia versions
using Ferrite
using JET: @test_call
using Test

include("test_utils.jl")

@testset "CellValues" begin
    @testset "ip=$scalar_interpol" for (scalar_interpol, quad_rule) in (
            (Lagrange{RefLine, 1}(), QuadratureRule{RefLine}(2)),
            (Lagrange{RefLine, 2}(), QuadratureRule{RefLine}(2)),
            (Lagrange{RefQuadrilateral, 1}(), QuadratureRule{RefQuadrilateral}(2)),
            (Lagrange{RefQuadrilateral, 2}(), QuadratureRule{RefQuadrilateral}(2)),
            (Lagrange{RefTriangle, 1}(), QuadratureRule{RefTriangle}(2)),
            (Lagrange{RefTriangle, 2}(), QuadratureRule{RefTriangle}(2)),
            (Lagrange{RefTriangle, 3}(), QuadratureRule{RefTriangle}(2)),
            (Lagrange{RefTriangle, 4}(), QuadratureRule{RefTriangle}(2)),
            (Lagrange{RefTriangle, 5}(), QuadratureRule{RefTriangle}(2)),
            (Lagrange{RefHexahedron, 1}(), QuadratureRule{RefHexahedron}(2)),
            (Serendipity{RefQuadrilateral, 2}(), QuadratureRule{RefQuadrilateral}(2)),
            (Lagrange{RefTriangle, 1}(), QuadratureRule{RefTriangle}(2)),
            (Lagrange{RefTetrahedron, 2}(), QuadratureRule{RefTetrahedron}(2)),
            (Lagrange{RefPrism, 2}(), QuadratureRule{RefPrism}(2)),
            (Lagrange{RefPyramid, 2}(), QuadratureRule{RefPyramid}(2)),
        )
        for func_interpol in (scalar_interpol, VectorizedInterpolation(scalar_interpol)), DiffOrder in 1:2
            (DiffOrder == 2 && Ferrite.getorder(func_interpol) == 1) && continue # No need to test linear interpolations again
            geom_interpol = scalar_interpol # Tests below assume this
            update_gradients = true
            update_hessians = (DiffOrder == 2 && Ferrite.getorder(func_interpol) > 1)
            cv = CellValues(quad_rule, func_interpol, geom_interpol; update_gradients, update_hessians)
            coords, _ = valid_coordinates_and_normals(func_interpol)
            @test_call reinit!(cv, coords)
            # TODO: Also @test_call some methods that use cv?
        end
    end
    @testset "Embedded elements" begin
        @testset "Scalar/vector on curves (vdim = $vdim)" for vdim in (0, 1, 2, 3)
            ip_base = Lagrange{RefLine, 1}()
            ip = vdim > 0 ? ip_base^vdim : ip_base
            qr = QuadratureRule{RefLine}(1)
            # Reference values
            csv1 = CellValues(qr, ip)
            @test_call reinit!(csv1, [Vec((0.0,)), Vec((1.0,))])
            ## sdim = 2, Consistency with 1D
            csv2 = CellValues(qr, ip, ip_base^2)
            @test_call reinit!(csv2, [Vec((0.0, 0.0)), Vec((1.0, 0.0))])
            ## sdim = 3, Consistency with 1D
            csv3 = CellValues(qr, ip, ip_base^3)
            @test_call reinit!(csv3, [Vec((0.0, 0.0, 0.0)), Vec((1.0, 0.0, 0.0))])
        end
        @testset "Scalar/vector on surface (vdim = $vdim)" for vdim in (0, 1, 2, 3)
            ip_base = Lagrange{RefQuadrilateral, 1}()
            ip = vdim > 0 ? ip_base^vdim : ip_base
            qr = QuadratureRule{RefQuadrilateral}(1)
            csv2 = CellValues(qr, ip)
            @test_call reinit!(csv2, [Vec((-1.0, -1.0)), Vec((1.0, -1.0)), Vec((1.0, 1.0)), Vec((-1.0, 1.0))])
            csv3 = CellValues(qr, ip, ip_base^3)
            @test_call reinit!(csv3, [Vec((-1.0, -1.0, 0.0)), Vec((1.0, -1.0, 0.0)), Vec((1.0, 1.0, 0.0)), Vec((-1.0, 1.0, 0.0))])
        end
    end
end # of testset
