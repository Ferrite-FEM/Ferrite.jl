using Ferrite, CUDA, Test
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
    reinit!(cv, coords)
    Ke = view(Kes, worker_index, :, :)
    assemble_mixed_element!(Ke, cv)
end

@testset "Multi-field CellValues using $backend" for backend in [KA.CPU(), CUDABackend()]
    ipu = Lagrange{RefQuadrilateral, 2}()^2
    ipp = Lagrange{RefQuadrilateral, 1}()
    qr = QuadratureRule{RefQuadrilateral}(Float32, 2)
    cv = CellValues(Float32, qr, (u = ipu, p = ipp, T = ipp))
    x = [Vec{2, Float32}((0.0, 0.0)), Vec{2, Float32}((1.1, 0.0)), Vec{2, Float32}((1.0, 1.2)), Vec{2, Float32}((-0.1, 1.0))]

    n_workers = 4
    cvs = Ferrite.distribute_to_workers(backend, cv, n_workers)
    @test cvs[1].p === cvs[1].T # Aliasing of equal interpolations preserved after distribution
    @test cvs[1].u !== cvs[1].p

    # CPU reference
    reinit!(cv, x)
    n = getnbasefunctions(cv.u) + getnbasefunctions(cv.p)
    Ke_ref = assemble_mixed_element!(zeros(Float32, n, n), cv)

    Kes = KA.zeros(backend, Float32, n_workers, n, n)
    multifield_test_kernel(backend, n_workers)(Kes, cvs, adapt(backend, x), ndrange = n_workers)
    KA.synchronize(backend)
    Kes_h = Array(Kes)
    for w in 1:n_workers
        @test Kes_h[w, :, :] ≈ Ke_ref
    end
end
