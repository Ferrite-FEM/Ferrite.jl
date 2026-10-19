using Ferrite, Test
import KernelAbstractions as KA

include("multifield_cellvalues_utils.jl")

test_multifield_cellvalues(KA.CPU())

@testset "CellValues worker storage" begin
    qr = QuadratureRule{RefTriangle}(2)
    ip = Lagrange{RefTriangle, 2}()
    cell = Triangle((1, 2, 3))
    x = [Vec((0.0, 0.0)), Vec((1.5, 0.0)), Vec((0.0, 2.0))]
    for ip_fun in (ip, (u = ip, T = ip), RaviartThomas{RefTriangle, 1}(), (u = ip, q = RaviartThomas{RefTriangle, 1}())),
            update_gradients in (false, true), update_detJdV in (false, true)
        cv = CellValues(qr, ip_fun; update_gradients, update_detJdV)
        cvs = Ferrite.distribute_to_workers(KA.CPU(), cv, 2)
        cv1 = @inferred cvs[1]
        cv2 = @inferred cvs[Int32(2)]
        @test typeof(cv1) === typeof(cv2) === eltype(cvs)
        reinit!(cv1, cell, x)
        reinit!(cv2, cell, 2 .* x)
        reinit!(cv, cell, x)
        # Updating the second worker must not overwrite the first worker's data.
        for (fv1, fv_ref) in zip(Ferrite.get_fun_values(cv1), Ferrite.get_fun_values(cv))
            for qp in 1:getnquadpoints(cv), i in 1:getnbasefunctions(fv_ref)
                @test shape_value(fv1, qp, i) ≈ shape_value(fv_ref, qp, i)
                if update_gradients
                    @test shape_gradient(fv1, qp, i) ≈ shape_gradient(fv_ref, qp, i)
                end
            end
        end
        if update_detJdV
            @test Ferrite.getdetJdVs(cv1) ≈ Ferrite.getdetJdVs(cv)
            @test Ferrite.getdetJdVs(cv2) ≈ 4 .* Ferrite.getdetJdVs(cv)
        else
            @test Ferrite.getdetJdVs(cv1) === nothing
        end
    end
end
