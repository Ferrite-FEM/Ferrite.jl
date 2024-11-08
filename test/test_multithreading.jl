using Ferrite, Test
using LinearAlgebra: Symmetric
using SparseArrays: sprand

function equivalent_but_distinct(x::T, y::T) where {T}
    if isbitstype(T)
        @test x === y
    elseif T <: AbstractArray
        @test x !== y
        for i in eachindex(x, y)
            equivalent_but_distinct(x[i], y[i])
        end
    else
        # (mutable) struct, Tuple, etc.
        for s in fieldnames(T)
            equivalent_but_distinct(getfield(x, s), getfield(y, s))
        end
    end
    return
end

@testset "task_local" begin
    # General fallback for bitstypes
    for x in (1, 1.0, true, nothing)
        equivalent_but_distinct(x, task_local(x))
    end
    # task_local(::Array) behaves like copy(::Array)
    for x in (rand(1), rand(1, 1), rand(1, 1, 1))
        equivalent_but_distinct(x, task_local(x))
    end
    # task_local(::Tuple) calls task_local recursively (optionally used in FacetQuadratureRule)
    for x in ((1, 2), (rand(2), rand(2)))
        equivalent_but_distinct(x, task_local(x))
    end
    # task_local(::QuadratureRule) behaves like copy(::QuadratureRule)
    for qr in (QuadratureRule{RefTriangle}(2), FacetQuadratureRule{RefTriangle}(2))
        equivalent_but_distinct(qr, task_local(qr))
    end
    # Interpolations are are assumed to be singletons
    for ip in (Lagrange{RefTriangle, 1}(), Lagrange{RefTriangle, 2}()^2)
        equivalent_but_distinct(ip, task_local(ip))
    end
    # GeometryMapping, FunctionValues
    ip = Lagrange{RefTriangle, 2}()
    qr = QuadratureRule{RefTriangle}(2)
    for DiffOrder in (0, 1, 2)
        gm = Ferrite.GeometryMapping{DiffOrder}(Float64, ip, qr)
        equivalent_but_distinct(gm, task_local(gm))
        fv = Ferrite.FunctionValues{DiffOrder}(Float64, ip, qr, Ferrite.VectorizedInterpolation{2}(ip))
        equivalent_but_distinct(fv, task_local(fv))
    end
    # CellValues
    for cv in (CellValues(qr, ip), CellValues(qr, ip; update_hessians = true))
        equivalent_but_distinct(cv, task_local(cv))
    end
    # FacetValues
    fqr = FacetQuadratureRule{RefTriangle}(2)
    fv = FacetValues(fqr, ip)
    equivalent_but_distinct(fv, task_local(fv))
    # InterfaceValues
    iv = InterfaceValues(fqr, ip)
    equivalent_but_distinct(iv, task_local(iv))
    # CSCAssembler, SymmetricCSCAssembler
    let K = sprand(10, 10, 0.5), f = rand(10)
        for assembler in (start_assemble(K, f), start_assemble(Symmetric(K), f))
            tl = task_local(assembler)
            @test tl.K === assembler.K
            @test tl.f === assembler.f
            @test tl.permutation !== assembler.permutation
            @test tl.sorteddofs !== assembler.sorteddofs
        end
    end
    # TODO: Test CellCache
end
