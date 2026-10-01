using Ferrite, Test
using LinearAlgebra: Symmetric
using SparseArrays: sparse, sprand
import BlockArrays
import SparseMatricesCSR: SparseMatrixCSR

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

@testset "task_local_copy" begin
    # General fallback for bitstypes
    for x in (1, 1.0, true, nothing)
        equivalent_but_distinct(x, task_local_copy(x))
    end
    # Error for non-bitstypes without a specific method
    @test_throws MethodError task_local_copy(big"1.0")
    @test_throws MethodError task_local_copy(Ref(1))
    @test_throws MethodError task_local_copy(view(rand(2), :))
    # task_local_copy(::Array) behaves like copy(::Array)
    for x in (rand(1), rand(1, 1), rand(1, 1, 1))
        equivalent_but_distinct(x, task_local_copy(x))
    end
    # task_local_copy(::Tuple) calls task_local_copy recursively (optionally used in
    # FacetQuadratureRule)
    for x in ((1, 2), (rand(2), rand(2)))
        equivalent_but_distinct(x, task_local_copy(x))
    end
    # task_local_copy(::QuadratureRule) behaves like copy(::QuadratureRule)
    for qr in (QuadratureRule{RefTriangle}(2), FacetQuadratureRule{RefTriangle}(2))
        equivalent_but_distinct(qr, task_local_copy(qr))
    end
    # Interpolations are are assumed to be singletons
    for ip in (Lagrange{RefTriangle, 1}(), Lagrange{RefTriangle, 2}()^2)
        equivalent_but_distinct(ip, task_local_copy(ip))
    end
    # GeometryMapping, FunctionValues
    ip = Lagrange{RefTriangle, 2}()
    qr = QuadratureRule{RefTriangle}(2)
    for DiffOrder in (0, 1, 2)
        gm = Ferrite.GeometryMapping{DiffOrder}(Float64, ip, qr)
        equivalent_but_distinct(gm, task_local_copy(gm))
        fv = Ferrite.FunctionValues{DiffOrder}(Float64, ip, qr, Ferrite.VectorizedInterpolation{2}(ip))
        equivalent_but_distinct(fv, task_local_copy(fv))
    end
    # CellValues
    for cv in (CellValues(qr, ip), CellValues(qr, ip; update_hessians = true))
        equivalent_but_distinct(cv, task_local_copy(cv))
    end
    # MultiFieldCellValues
    let ipu = Lagrange{RefTriangle, 2}()^2, ipp = Lagrange{RefTriangle, 1}()
        cmv = MultiFieldCellValues(qr, (u = ipu, p = ipp, T = ipp))
        tl = @inferred task_local_copy(cmv)
        equivalent_but_distinct(cmv, tl)
        # Aliasing between fields with equal interpolations is preserved
        @test tl.p === tl.T
        @test tl.p !== cmv.p
    end
    # PointValues
    pv = PointValues(ip)
    equivalent_but_distinct(pv, task_local_copy(pv))
    # FacetValues
    fqr = FacetQuadratureRule{RefTriangle}(2)
    fv = FacetValues(fqr, ip)
    equivalent_but_distinct(fv, task_local_copy(fv))
    # InterfaceValues
    iv = InterfaceValues(fqr, ip)
    equivalent_but_distinct(iv, task_local_copy(iv))
    # CSCAssembler, SymmetricCSCAssembler, CSRAssembler (with and without atomics)
    let K = sprand(10, 10, 0.5), f = rand(10)
        for atomic in (false, true), assembler in (
                    start_assemble(K, f; atomic), start_assemble(Symmetric(K), f; atomic),
                    start_assemble(SparseMatrixCSR(transpose(K)), f; atomic),
                )
            tl = @inferred task_local_copy(assembler)
            @test typeof(tl) === typeof(assembler)
            @test tl.K === assembler.K
            @test tl.f === assembler.f
            for s in (:rowpermutation, :colpermutation, :sortedrowdofs, :sortedcoldofs)
                @test getfield(tl, s) !== getfield(assembler, s)
            end
            if assembler isa Ferrite.SymmetricCSCAssembler
                # Aliasing of row and col buffers is preserved
                @test tl.rowpermutation === tl.colpermutation
                @test tl.sortedrowdofs === tl.sortedcoldofs
            end
        end
    end
    # BlockAssembler
    let K = BlockArrays.mortar([sprand(5, 5, 0.5) for _ in 1:2, _ in 1:2]), f = rand(10)
        assembler = start_assemble(K, f)
        tl = @inferred task_local_copy(assembler)
        @test typeof(tl) === typeof(assembler)
        @test tl.K === assembler.K
        @test tl.f === assembler.f
        for s in (:sorteddofs, :permutation, :blockstops)
            @test getfield(tl, s) !== getfield(assembler, s)
        end
    end
    # CellCache, FacetCache, InterfaceCache
    grid = generate_grid(Triangle, (2, 2))
    dh = DofHandler(grid)
    add!(dh, :u, ip)
    close!(dh)
    for gridordh in (grid, dh)
        cc = CellCache(gridordh)
        reinit!(cc, 2)
        tl = task_local_copy(cc)
        @test tl.grid === cc.grid
        @test tl.dh === cc.dh
        for s in (:cellid, :nodes, :coords, :dofs)
            @test getfield(tl, s) !== getfield(cc, s)
            @test getfield(tl, s) == getfield(cc, s)
        end
        fc = FacetCache(gridordh)
        reinit!(fc, FacetIndex(2, 1))
        tl = task_local_copy(fc)
        @test tl.cc.cellid !== fc.cc.cellid
        @test tl.cc.dofs !== fc.cc.dofs
        @test tl.dofs === tl.cc.dofs # Aliasing is preserved
        @test tl.current_facet_id == fc.current_facet_id
        ic = InterfaceCache(gridordh)
        reinit!(ic, FacetIndex(1, 1), FacetIndex(2, 1))
        tl = task_local_copy(ic)
        @test tl.a.cc.coords !== ic.a.cc.coords
        @test tl.b.cc.coords !== ic.b.cc.coords
        @test tl.dofs !== ic.dofs
        @test tl.dofs == ic.dofs
    end
end
