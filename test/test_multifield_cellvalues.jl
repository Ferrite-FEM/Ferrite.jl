using Ferrite, GPUArrays, Test
import Adapt: adapt
import KernelAbstractions: @kernel, @index
import KernelAbstractions as KA

# Single-cell mixed u-p element routine, usable both on CPU and device
function assemble_mixed_element!(Ke, cv)
    ndofs_u = getnbasefunctions(cv.u)
    for q_point in 1:getnquadpoints(cv)
        dΩ = getdetJdV(cv, q_point)
        for i in 1:ndofs_u
            div_Nui = shape_divergence(cv.u, q_point, i)
            for j in 1:getnbasefunctions(cv.p)
                Np = shape_value(cv.p, q_point, j)
                Ke[i, ndofs_u + j] += div_Nui * Np * dΩ
                Ke[ndofs_u + j, i] += div_Nui * Np * dΩ
            end
        end
    end
    return Ke
end

@kernel function multifield_test_kernel(Kes, cvs, coords)
    worker_index = @index(Global, Linear)
    cv = cvs[worker_index]
    reinit!(cv, view(coords, :, worker_index))
    Ke = view(Kes, worker_index, :, :)
    assemble_mixed_element!(Ke, cv)
end

function test_multifield_cellvalues(backend)
    @testset "Multi-field CellValues using $backend" begin
        ipu = Lagrange{RefQuadrilateral, 2}()^2
        ipp = Lagrange{RefQuadrilateral, 1}()
        qr = QuadratureRule{RefQuadrilateral}(Float32, 2)
        cv = CellValues(Float32, qr, (u = ipu, p = ipp, T = ipp))
        x = [Vec{2, Float32}((0.0, 0.0)), Vec{2, Float32}((1.1, 0.0)), Vec{2, Float32}((1.0, 1.2)), Vec{2, Float32}((-0.1, 1.0))]

        n_workers = 4
        cvs = Ferrite.distribute_to_workers(backend, cv, n_workers)
        @test cvs[1].p === cvs[1].T # Aliasing of equal interpolations preserved after distribution
        @test cvs[1].u !== cvs[1].p

        # Different geometries expose accidental sharing of mutable worker buffers.
        coords = hcat((Float32(w) .* x for w in 1:n_workers)...)
        n = getnbasefunctions(cv.u) + getnbasefunctions(cv.p)

        Kes = KA.zeros(backend, Float32, n_workers, n, n)
        multifield_test_kernel(backend, n_workers)(Kes, cvs, adapt(backend, coords), ndrange = n_workers)
        KA.synchronize(backend)
        Kes_h = Array(Kes)
        for w in 1:n_workers
            reinit!(cv, view(coords, :, w))
            Ke_ref = assemble_mixed_element!(zeros(Float32, n, n), cv)
            @test Kes_h[w, :, :] ≈ Ke_ref
        end
    end
    return nothing
end

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
