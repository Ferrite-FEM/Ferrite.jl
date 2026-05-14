@testset "reorder_cells!" begin
    @testset "identity permutation leaves grid unchanged" begin
        grid = generate_grid(Quadrilateral, (3, 2))
        cells_before = deepcopy(getcells(grid))
        cellsets_before = deepcopy(Ferrite.getcellsets(grid))
        facetsets_before = deepcopy(Ferrite.getfacetsets(grid))

        reorder_cells!(grid, collect(1:getncells(grid)))

        @test getcells(grid) == cells_before
        @test Ferrite.getcellsets(grid) == cellsets_before
        @test Ferrite.getfacetsets(grid) == facetsets_before
    end

    @testset "cells are reordered according to permutation" begin
        grid = generate_grid(Line, (4,))
        # cells 1 and 2 are in the left half
        addcellset!(grid, "left", Set([1, 2]))
        original_cells = deepcopy(getcells(grid))
        perm = [4, 3, 2, 1]

        reorder_cells!(grid, perm)

        for (new_idx, old_idx) in enumerate(perm)
            @test getcells(grid, new_idx) == original_cells[old_idx]
        end

        @test getcellset(grid, "left") == OrderedSet([3, 4])
    end

    @testset "facetsets are updated by permutation" begin
        grid = generate_grid(Quadrilateral, (3, 1))
        # "left" facetset uses FacetIndex(1, 4) by default from generate_grid
        perm = [3, 2, 1]  # reverse
        # old cell 1 -> new position invperm([3,2,1])[1] = 3

        reorder_cells!(grid, perm)

        left_set = getfacetset(grid, "left")
        @test all(fi -> fi[1] == 3, left_set)

        right_set = getfacetset(grid, "right")
        @test all(fi -> fi[1] == 1, right_set)
    end

    @testset "vertexsets are updated by permutation" begin
        grid = generate_grid(Quadrilateral, (2, 2))
        addvertexset!(grid, "bottom-left-corner", x -> x[1] ≈ -1.0 && x[2] ≈ -1.0)
        # The bottom-left corner vertex belongs to cell 1, vertex 1
        @test !isempty(getvertexset(grid, "bottom-left-corner"))
        original_cell_ids = Set(vi[1] for vi in getvertexset(grid, "bottom-left-corner"))

        perm = [4, 3, 2, 1]  # reverse
        iperm = invperm(perm)

        reorder_cells!(grid, perm)

        new_cell_ids = Set(vi[1] for vi in getvertexset(grid, "bottom-left-corner"))
        @test new_cell_ids == Set(iperm[id] for id in original_cell_ids)
    end
end

@testset "compute_sfc_ordering" begin
    @testset "permutation is valid" begin
        for (CT, dim) in [(Line, 1), (Quadrilateral, 2), (Triangle, 2), (Hexahedron, 3), (Tetrahedron, 3)]
            nels = ntuple(_ -> 3, dim)
            grid = generate_grid(CT, nels)
            perm = compute_sfc_ordering(grid)
            @test isperm(perm)
            @test length(perm) == getncells(grid)
        end
    end

    @testset "2D grid: SFC ordering reduces average neighbor distance" begin
        # A shuffled 2D grid should have better locality after SFC ordering
        grid = generate_grid(Quadrilateral, (4, 4))
        Random.seed!(42)
        shuffled_perm = randperm(getncells(grid))
        reorder_cells!(grid, shuffled_perm)

        function mean_consecutive_center_dist(g)
            centers = [
                sum(get_node_coordinate(g, n) for n in Ferrite.get_node_ids(getcells(g, i))) /
                    length(Ferrite.get_node_ids(getcells(g, i))) for i in 1:getncells(g)
            ]
            dists = [norm(centers[i + 1] - centers[i]) for i in 1:(length(centers) - 1)]
            sum(dists) / length(dists)
        end

        dist_before = mean_consecutive_center_dist(grid)
        sfc_perm = compute_sfc_ordering(grid)
        reorder_cells!(grid, sfc_perm)
        dist_after = mean_consecutive_center_dist(grid)

        @test dist_after <= dist_before
    end
end
