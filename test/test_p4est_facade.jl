# The ForestBWG grid facade (#1413): a `ForestBWG` is used directly as the grid of a
# `DofHandler`/`ConstraintHandler`/VTK export. Grid accessors answer for the *current*
# refinement state through an epoch-guarded materialization cache; mutators bump the
# epoch, stale `DofHandler` use errors, and `reclose!` re-distributes dofs.
using Ferrite, Test
using LinearAlgebra, SparseArrays

@testset "facade accessor overrides vs creategrid reference $dim D" for (dim, CT, RS) in
    ((2, Quadrilateral, RefQuadrilateral), (3, Hexahedron, RefHexahedron))

    grid = generate_grid(CT, ntuple(_ -> 2, dim))
    addcellset!(grid, "left", x -> x[1] <= 0)
    forest = ForestBWG(grid, 3)
    Ferrite.AMR.refine!(forest, [1])
    Ferrite.AMR.balanceforest!(forest)
    ref = Ferrite.AMR.creategrid(forest)

    # cells
    @test getcells(forest) == getcells(ref)
    @test getcells(forest, 3) == getcells(ref, 3)
    @test getcells(forest, [1, 2]) == getcells(ref, [1, 2])
    @test getcells(forest, "left") == getcells(ref, "left")
    @test getncells(forest) == getncells(ref) == length(getcells(forest))
    @test getcelltype(forest) === CT === getcelltype(forest, 1)
    @test Ferrite.nnodes_per_cell(forest) == 2^dim == Ferrite.nnodes_per_cell(forest, 1)

    # nodes and coordinates
    @test getnodes(forest) == getnodes(ref)
    @test getnodes(forest, 2) == getnodes(ref, 2)
    @test getnodes(forest, [1, 2]) == getnodes(ref, [1, 2])
    @test getnnodes(forest) == getnnodes(ref)
    @test Ferrite.get_coordinate_type(forest) === Vec{dim, Float64}
    @test Ferrite.get_coordinate_eltype(forest) === Float64
    @test Ferrite.get_node_coordinate(forest, 5) == Ferrite.get_node_coordinate(ref, 5)
    for cellid in (1, getncells(ref))
        @test getcoordinates(forest, cellid) == getcoordinates(ref, cellid)
    end

    # sets: cellsets/facetsets are reconstructed, node/vertex sets are a documented gap
    @test getcellset(forest, "left") == getcellset(ref, "left")
    @test Ferrite.getcellsets(forest) == Ferrite.getcellsets(ref)
    @test getfacetset(forest, "left") == getfacetset(ref, "left")
    @test Ferrite.getfacetsets(forest) == Ferrite.getfacetsets(ref)
    @test isempty(Ferrite.getnodesets(forest))
    @test isempty(Ferrite.getvertexsets(forest))
    @test_throws KeyError getnodeset(forest, "anything")
    @test_throws KeyError getvertexset(forest, "anything")

    # conformity information through the documented accessor, never via snapshot fields
    @test Ferrite.AMR.conformity_info(forest) == ref.conformity_info
    @test Ferrite.AMR.conformity_info(ref) == ref.conformity_info
    @test Ferrite.has_hanging_nodes(forest)
    @test Ferrite.has_hanging_nodes(ref)
    @test !Ferrite.has_hanging_nodes(grid)
end

@testset "hard errors: ExclusiveTopology and transform_coordinates!" begin
    forest = ForestBWG(generate_grid(Quadrilateral, (2, 2)), 3)
    err = try
        ExclusiveTopology(forest)
        nothing
    catch e
        e
    end
    @test err isa ErrorException
    @test occursin("facetskeleton", err.msg)
    @test_throws ErrorException Ferrite.transform_coordinates!(forest, x -> 2 * x)
end

@testset "epoch semantics" begin
    # plain grids are not epoch-tracked
    @test Ferrite.grid_epoch(generate_grid(Quadrilateral, (1, 1))) == 0

    forest = ForestBWG(generate_grid(Quadrilateral, (2, 2)), 3)
    e = Ferrite.grid_epoch(forest)
    @test e >= 1

    # pure queries (including materialization) leave the epoch alone
    getcells(forest)
    getnnodes(forest)
    getcoordinates(forest, 1)
    Ferrite.AMR.facetskeleton(forest)
    @test Ferrite.grid_epoch(forest) == e

    # every mutator bumps
    Ferrite.AMR.refine!(forest, [1])
    @test Ferrite.grid_epoch(forest) > e
    e = Ferrite.grid_epoch(forest)
    Ferrite.AMR.balanceforest!(forest)
    @test Ferrite.grid_epoch(forest) > e
    e = Ferrite.grid_epoch(forest)
    Ferrite.AMR.refine!(forest, 2) # scalar refine!
    @test Ferrite.grid_epoch(forest) > e
    e = Ferrite.grid_epoch(forest)
    Ferrite.AMR.refine_all!(forest, 3)
    @test Ferrite.grid_epoch(forest) > e
    e = Ferrite.grid_epoch(forest)
    Ferrite.AMR.coarsen!(forest, collect(1:4); require_all_siblings = false)
    @test Ferrite.grid_epoch(forest) > e
    e = Ferrite.grid_epoch(forest)
    Ferrite.AMR.refine_and_coarsen!(forest, Int[], [1])
    @test Ferrite.grid_epoch(forest) > e
    # _coarsen_all! (internal) also invalidates
    f3 = ForestBWG(generate_grid(Quadrilateral, (1, 1)), 1)
    Ferrite.AMR.refine_all!(f3, 1)
    e3 = Ferrite.grid_epoch(f3)
    Ferrite.AMR._coarsen_all!(f3)
    @test Ferrite.grid_epoch(f3) > e3

    # the facade serves the refreshed state after a mutation
    f2 = ForestBWG(generate_grid(Quadrilateral, (2, 2)), 3)
    n = getncells(f2)
    @test length(getcells(f2)) == n
    Ferrite.AMR.refine!(f2, [1])
    Ferrite.AMR.balanceforest!(f2)
    @test length(getcells(f2)) == getncells(f2) > n
end

# Assemble the Laplace problem with unit load; returns the constrained solution.
function _facade_solve(dh, ch, cv)
    n = getnbasefunctions(cv)
    Ke = zeros(n, n)
    fe = zeros(n)
    K = allocate_matrix(dh, ch)
    f = zeros(ndofs(dh))
    assembler = start_assemble(K, f)
    for cell in CellIterator(dh)
        reinit!(cv, cell)
        fill!(Ke, 0)
        fill!(fe, 0)
        for qp in 1:getnquadpoints(cv)
            dΩ = getdetJdV(cv, qp)
            for i in 1:n
                fe[i] += shape_value(cv, qp, i) * dΩ
                for j in 1:n
                    Ke[i, j] += (shape_gradient(cv, qp, i) ⋅ shape_gradient(cv, qp, j)) * dΩ
                end
            end
        end
        assemble!(assembler, celldofs(cell), Ke, fe)
    end
    apply!(K, f, ch)
    u = K \ f
    apply!(u, ch)
    return u
end

@testset "tutorial-shaped adaptive loop $dim D" for (dim, CT, RS) in
    ((2, Quadrilateral, RefQuadrilateral), (3, Hexahedron, RefHexahedron))

    forest = ForestBWG(generate_grid(CT, ntuple(_ -> 2, dim)), 3)
    ip = Lagrange{RS, 1}()
    cv = CellValues(QuadratureRule{RS}(2), ip)

    dh = DofHandler(forest)
    add!(dh, :u, ip)
    close!(dh)
    @test ndofs(dh) == getnnodes(forest) == 3^dim

    for _ in 1:2
        ch = ConstraintHandler(dh)
        add!(ch, ConformityConstraint(:u))
        add!(ch, Dirichlet(:u, getfacetset(forest, "left"), x -> 0.0))
        close!(ch)

        u = _facade_solve(dh, ch, cv)
        @test length(u) == ndofs(dh)
        @test all(isfinite, u)
        @test maximum(abs, u) > 0

        # mark + refine; the DofHandler is now stale and must say so
        Ferrite.AMR.refine!(forest, [1])
        Ferrite.AMR.balanceforest!(forest)
        stale = try
            CellIterator(dh)
            nothing
        catch e
            e
        end
        @test stale isa ErrorException
        @test occursin("reclose!", stale.msg)
        @test_throws ErrorException ndofs(dh)
        @test_throws ErrorException ConstraintHandler(dh)
        @test_throws ErrorException VTKGridFile(tempname(), dh)

        # reclose! re-distributes dofs against the new epoch
        reclose!(dh)
        @test ndofs(dh) == getnnodes(forest)
        @test Ferrite.isclosed(dh)
    end

    # after the loop, a full solve on the twice-refined forest still goes through,
    # and VTK export accepts the recloseed handler
    ch = ConstraintHandler(dh)
    add!(ch, ConformityConstraint(:u))
    add!(ch, Dirichlet(:u, getfacetset(forest, "left"), x -> 0.0))
    close!(ch)
    u = _facade_solve(dh, ch, cv)
    @test all(isfinite, u)
    mktempdir() do dir
        VTKGridFile(joinpath(dir, "facade"), dh) do vtk
            write_solution(vtk, dh, u)
        end
        @test isfile(joinpath(dir, "facade.vtu"))
    end
end

@testset "reclose! guards" begin
    # not closed
    forest = ForestBWG(generate_grid(Quadrilateral, (2, 2)), 3)
    dh = DofHandler(forest)
    add!(dh, :u, Lagrange{RefQuadrilateral, 1}())
    @test_throws ErrorException reclose!(dh)

    # multiple SubDofHandlers are not supported in v1
    dh2 = DofHandler(forest)
    sdh_a = SubDofHandler(dh2, 1:2)
    add!(sdh_a, :u, Lagrange{RefQuadrilateral, 1}())
    sdh_b = SubDofHandler(dh2, 3:4)
    add!(sdh_b, :u, Lagrange{RefQuadrilateral, 1}())
    close!(dh2)
    Ferrite.AMR.refine!(forest, [1])
    Ferrite.AMR.balanceforest!(forest)
    @test_throws ErrorException reclose!(dh2)

    # a single SubDofHandler on a subdomain is not supported in v1 either
    forest3 = ForestBWG(generate_grid(Quadrilateral, (2, 2)), 3)
    dh3 = DofHandler(forest3)
    sdh_c = SubDofHandler(dh3, 1:2)
    add!(sdh_c, :u, Lagrange{RefQuadrilateral, 1}())
    close!(dh3)
    Ferrite.AMR.refine!(forest3, [1])
    Ferrite.AMR.balanceforest!(forest3)
    @test_throws ErrorException reclose!(dh3)

    # reclose! on a conforming (non-epoch-tracked) grid is a valid re-distribution no-op
    grid = generate_grid(Quadrilateral, (2, 2))
    dh4 = DofHandler(grid)
    add!(dh4, :u, Lagrange{RefQuadrilateral, 1}())
    close!(dh4)
    nd = ndofs(dh4)
    reclose!(dh4)
    @test ndofs(dh4) == nd
end

@testset "creategrid remains a thin working wrapper" begin
    forest = ForestBWG(generate_grid(Quadrilateral, (2, 2)), 3)
    Ferrite.AMR.refine!(forest, [1])
    Ferrite.AMR.balanceforest!(forest)
    g = Ferrite.AMR.creategrid(forest)
    @test g isa Ferrite.AMR.NonConformingGrid
    @test getncells(g) == getncells(forest)
    # the returned grid does not alias the facade cache
    @test getcells(g) !== getcells(forest)
    @test getnodes(g) !== getnodes(forest)
end
