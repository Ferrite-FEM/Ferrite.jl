# Backend-generic test of FacetValues in a KernelAbstractions kernel, shared by
# test_facevalues.jl (KA.CPU()) and GPU/runtests.jl (CUDABackend()).

using Ferrite, GPUArrays, Test
import Adapt: adapt
import KernelAbstractions: @kernel, @index
import KernelAbstractions as KA

@kernel function facetvalues_test_kernel(detJdVs, normals, shape_values, fvs, coords, facets)
    worker_index = @index(Global, Linear)
    fv = fvs[worker_index]
    facet = facets[worker_index]
    cell_coords = view(coords, :, worker_index)
    reinit!(fv, cell_coords, facet)

    nqp = getnquadpoints(fv)
    for qp in 1:nqp
        detJdVs[worker_index, qp] = getdetJdV(fv, qp)
        normals[worker_index, qp] = getnormal(fv, qp)
        for i in 1:getnbasefunctions(fv)
            shape_values[worker_index, qp, i] = shape_value(fv, qp, i)
        end
    end
end

function test_facetvalues(backend)
    @testset "FacetValues kernel test using $backend" begin
        fqr = FacetQuadratureRule{RefQuadrilateral}(Float32, 2)
        ip = Lagrange{RefQuadrilateral, 1}()
        fv = FacetValues(Float32, fqr, ip)

        x = [
            Vec{2, Float32}((0.0, 0.0)),
            Vec{2, Float32}((1.1, 0.0)),
            Vec{2, Float32}((1.0, 1.2)),
            Vec{2, Float32}((-0.1, 1.0)),
        ]

        n_workers = 4
        fvs = Ferrite.distribute_to_workers(backend, fv, n_workers)

        # Different geometries and facets expose accidental sharing of mutable worker buffers.
        coords = hcat((Float32(w) .* x for w in 1:n_workers)...)
        facets = Int[1, 2, 3, 4]

        nqp = getnquadpoints(fv)
        n_basefuncs = getnbasefunctions(fv)

        detJdVs = KA.zeros(backend, Float32, n_workers, nqp)
        normals = KA.zeros(backend, Vec{2, Float32}, n_workers, nqp)
        shape_values = KA.zeros(backend, Float32, n_workers, nqp, n_basefuncs)

        facetvalues_test_kernel(backend, n_workers)(
            detJdVs,
            normals,
            shape_values,
            fvs,
            adapt(backend, coords),
            adapt(backend, facets),
            ndrange = n_workers,
        )
        KA.synchronize(backend)

        detJdVs_h = Array(detJdVs)
        normals_h = Array(normals)
        shape_values_h = Array(shape_values)

        for w in 1:n_workers
            facet = facets[w]
            cell_coords = view(coords, :, w)
            reinit!(fv, cell_coords, facet)
            for qp in 1:nqp
                @test detJdVs_h[w, qp] ≈ getdetJdV(fv, qp)
                @test normals_h[w, qp] ≈ getnormal(fv, qp)
                for i in 1:n_basefuncs
                    @test shape_values_h[w, qp, i] ≈ shape_value(fv, qp, i)
                end
            end
        end

        # Test multi-field FacetValues distribution and aliasing
        ipu = Lagrange{RefQuadrilateral, 2}()^2
        ipp = Lagrange{RefQuadrilateral, 1}()
        fvm = FacetValues(Float32, fqr, (u = ipu, p = ipp, T = ipp))
        fvms = Ferrite.distribute_to_workers(backend, fvm, n_workers)
        GPUArrays.@allowscalar begin
            @test fvms[1].p === fvms[1].T
            @test fvms[1].u !== fvms[1].p
        end
    end
    return nothing
end
