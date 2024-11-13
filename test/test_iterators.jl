@testset "CellIterator" begin
    @testset "Single domain" begin
        grid = generate_grid(Hexahedron, (2, 2, 2))
        ip = Lagrange{RefHexahedron, 1}()
        dh = DofHandler(grid)
        add!(dh, :u, ip)
        close!(dh)

        ci_dh = CellIterator(dh)
        ci_grid = CellIterator(grid)
        @test ci_dh.set == ci_grid.set
        # Test for allocations
        # TODO: find a way to avoid resize for single sdh
        # @test (testallocs_iterate(ci_dh)) === nothing
        # Iterators over grid can allocate due to potential resize!
        @test_throws AllocCheckFailure testallocs_iterate(ci_grid)
    end

    @testset "subdomains" begin
        @testset "same cell type" begin
            grid = generate_grid(Hexahedron, (2, 2, 2))
            ip = Lagrange{RefHexahedron, 1}()
            dh = DofHandler(grid)
            sdh1 = SubDofHandler(dh, Set(collect(1:4)))
            sdh2 = SubDofHandler(dh, Set(collect(5:8)))
            add!(sdh1, :u, ip^3)
            add!(sdh2, :p, ip)
            close!(dh)

            ci_sdh1 = CellIterator(sdh1, Set(collect(1:4)))
            ci_sdh2 = CellIterator(sdh2, Set(collect(5:8)))
            ci_dh = CellIterator(dh)

            # Test for allocations
            @test (testallocs_iterate(ci_sdh1)) === nothing
            @test (testallocs_iterate(ci_sdh2)) === nothing
            @test_throws AllocCheckFailure testallocs_iterate(ci_dh)
        end

        @testset "mixed cell types" begin
            nodes = [Node((-1.0, 0.0)), Node((0.0, 0.0)), Node((1.0, 0.0)), Node((-1.0, 1.0)), Node((0.0, 1.0))]
            cells = [
                Quadrilateral((1, 2, 5, 4)),
                Triangle((3, 5, 2)),
            ]
            grid = Grid(cells, nodes)
            ip_quad = Lagrange{RefQuadrilateral, 1}()
            ip_tri = Lagrange{RefTriangle, 1}()
            dh = DofHandler(grid)
            sdh_quad = SubDofHandler(dh, Set([1]))
            sdh_tri = SubDofHandler(dh, Set([2]))
            add!(sdh_quad, :u, ip_quad)
            add!(sdh_tri, :u, ip_tri)
            close!(dh)

            ci_quad = CellIterator(sdh_quad, Set([1]))
            ci_tri = CellIterator(sdh_tri, Set([2]))
            ci_dh = CellIterator(dh, Set([1]))

            # Test for allocations
            try
                testallocs_iterate(ci_quad)

            catch err
                @info err.errors[3]
            end
            @test (testallocs_iterate(ci_quad)) === nothing
            @test (testallocs_iterate(ci_tri)) === nothing
            @test_throws AllocCheckFailure testallocs_iterate(ci_dh)
        end
    end

    @testset "error paths" begin
        nodes = [Node((-1.0, 0.0)), Node((0.0, 0.0)), Node((1.0, 0.0)), Node((-1.0, 1.0)), Node((0.0, 1.0))]
        cells = [
            Quadrilateral((1, 2, 5, 4)),
            Triangle((3, 5, 2)),
        ]
        grid = Grid(cells, nodes)
        ip_quad = Lagrange{RefQuadrilateral, 1}()
        ip_tri = Lagrange{RefTriangle, 1}()
        dh = DofHandler(grid)
        sdh_quad = SubDofHandler(dh, Set([1]))
        sdh_tri = SubDofHandler(dh, Set([2]))
        add!(sdh_quad, :u, ip_quad)
        add!(sdh_tri, :u, ip_tri)
        close!(dh)
        @test_throws ErrorException("The cells in the cellset are not all of the same celltype.") CellIterator(dh)
    end
end

# @testset "FacetIterator" begin
#     @testset "Single domain" begin
#         grid = generate_grid(Hexahedron, (1000, 2, 1))
#         ip = Lagrange{RefHexahedron, 1}()
#         dh = DofHandler(grid)
#         add!(dh, :u, ip)
#         close!(dh)
#         topology = ExclusiveTopology(grid)

#         ii_dh_top = InterfaceIterator(dh, topology)
#         ii_dh = InterfaceIterator(dh)
#         ii_grid_top = InterfaceIterator(grid, topology)
#         ii_grid = InterfaceIterator(grid)
#         @test ii_dh.set == ii_dh_top.set == ii_grid.set == ii_grid_top.set
#         # Test that topology has no effect on iterator type
#         @test typeof(ii_dh) == typeof(ii_dh_top)
#         @test typeof(ii_grid) == typeof(ii_grid_top)
#         # Test for allocations
#         # TODO: find a way to avoid resize for single sdh
#         # @test (testallocs_iterate(ii_dh_top)) === nothing
#         # @test (testallocs_iterate(ii_dh)) === nothing
#         # Iterators over grid can allocate due to potential resize!
#         @test_throws AllocCheckFailure testallocs_iterate(ii_grid)
#         @test_throws AllocCheckFailure testallocs_iterate(ii_grid_top)
#     end

#     @testset "subdomains" begin
#         @testset "same cell type" begin
#             grid = generate_grid(Hexahedron, (1000, 2, 1))
#             ip = Lagrange{RefHexahedron, 1}()
#             dh = DofHandler(grid)
#             sdh1 = SubDofHandler(dh, Set([collect(1:500)..., collect(1501:2000)...]))
#             sdh2 = SubDofHandler(dh, Set([collect(501:1500)...]))
#             add!(sdh1, :u, ip)
#             add!(sdh2, :u, ip)
#             close!(dh)
#             topology = ExclusiveTopology(grid)

#             ii_sdh_top = InterfaceIterator(sdh1, sdh2, topology)
#             ii_sdh = InterfaceIterator(sdh1, sdh2)
#             ii_same_sdh = InterfaceIterator(sdh1, sdh1)
#             @test ii_sdh.set == ii_sdh_top.set
#             @test length(ii_sdh.set) == 1002
#             @test count(interface_index -> interface_index[1] == interface_index[3] - 1000, ii_sdh.set) == 500
#             @test count(interface_index -> interface_index[1] == interface_index[3] + 1000, ii_sdh.set) == 500
#             ii_sdh_flipped = InterfaceIterator(sdh2, sdh1)
#             @test all(interface -> InterfaceIndex(interface[3], interface[4], interface[1], interface[2]) ∈ ii_sdh_flipped.set, ii_sdh.set)
#             # Test that topology has no effect on iterator type
#             @test typeof(ii_sdh) == typeof(ii_sdh_top)
#             # Test for allocations
#             @test (testallocs_iterate(ii_sdh)) === nothing
#             @test (testallocs_iterate(ii_same_sdh)) === nothing
#         end

#         @testset "mixed cell types" begin
#             nodes = [Node((-1.0, 0.0)), Node((0.0, 0.0)), Node((1.0, 0.0)), Node((-1.0, 1.0)), Node((0.0, 1.0))]
#             cells = [
#                 Quadrilateral((1, 2, 5, 4)),
#                 Triangle((3, 5, 2)),
#             ]
#             grid = Grid(cells, nodes)
#             topology = ExclusiveTopology(grid)
#             ip_quad = Lagrange{RefQuadrilateral, 1}()
#             ip_tri = Lagrange{RefTriangle, 1}()
#             dh = DofHandler(grid)
#             sdh_quad = SubDofHandler(dh, Set([1]))
#             sdh_tri = SubDofHandler(dh, Set([2]))
#             add!(sdh_quad, :u, ip_quad)
#             add!(sdh_tri, :u, ip_tri)
#             close!(dh)

#             ii_sdh_top = InterfaceIterator(sdh_quad, sdh_tri, topology)
#             ii_sdh = InterfaceIterator(sdh_quad, sdh_tri)
#             @test ii_sdh.set == ii_sdh_top.set
#             @test ii_sdh.set == Set([InterfaceIndex(1, 2, 2, 2)])
#             ii_sdh_flipped = InterfaceIterator(sdh_tri, sdh_quad)
#             @test ii_sdh_flipped.set == Set([InterfaceIndex(2, 2, 1, 2)])
#             # Test that topology has no effect on iterator type
#             @test typeof(ii_sdh) == typeof(ii_sdh_top)
#             # Test for allocations
#             @test (testallocs_iterate(ii_sdh_top)) === nothing
#             @test (testallocs_iterate(ii_sdh)) === nothing
#         end
#     end

#     @testset "InterfaceIndex" begin
#         # checkmate, codecov bot!
#         @test isequal(InterfaceIndex(1, 2, 3, 4), (1, 2, 3, 4))
#         @test isequal((1, 2, 3, 4), InterfaceIndex(1, 2, 3, 4))
#     end

#     @testset "error paths" begin
#         nodes = [Node((-1.0, 0.0)), Node((0.0, 0.0)), Node((1.0, 0.0)), Node((-1.0, 1.0)), Node((0.0, 1.0))]
#         cells = [
#             Quadrilateral((1, 2, 5, 4)),
#             Triangle((3, 5, 2)),
#         ]
#         grid = Grid(cells, nodes)
#         topology = ExclusiveTopology(grid)
#         ip_quad = Lagrange{RefQuadrilateral, 1}()
#         ip_tri = Lagrange{RefTriangle, 1}()
#         dh = DofHandler(grid)
#         sdh_quad = SubDofHandler(dh, Set([1]))
#         sdh_tri = SubDofHandler(dh, Set([2]))
#         add!(sdh_quad, :u, ip_quad)
#         add!(sdh_tri, :u, ip_tri)
#         close!(dh)
#         @test_throws ErrorException("The cells in the set (set of InterfaceIndex) are not all of the same celltype on each side.") InterfaceIterator(sdh_quad, sdh_tri, Set([InterfaceIndex(2, 2, 1, 2)]))
#     end
# end

@testset "InterfaceIterator" begin
    @testset "Single domain" begin
        grid = generate_grid(Hexahedron, (1000, 2, 1))
        ip = Lagrange{RefHexahedron, 1}()
        dh = DofHandler(grid)
        add!(dh, :u, ip)
        close!(dh)
        topology = ExclusiveTopology(grid)

        ii_dh_top = InterfaceIterator(dh, topology)
        ii_dh = InterfaceIterator(dh)
        ii_grid_top = InterfaceIterator(grid, topology)
        ii_grid = InterfaceIterator(grid)
        @test ii_dh.set == ii_dh_top.set == ii_grid.set == ii_grid_top.set
        # Test that topology has no effect on iterator type
        @test typeof(ii_dh) == typeof(ii_dh_top)
        @test typeof(ii_grid) == typeof(ii_grid_top)
        # Test for allocations
        # TODO: find a way to avoid resize for single sdh
        # @test (testallocs_iterate(ii_dh_top)) === nothing
        # @test (testallocs_iterate(ii_dh)) === nothing
        # Iterators over grid can allocate due to potential resize!
        @test_throws AllocCheckFailure testallocs_iterate(ii_grid)
        @test_throws AllocCheckFailure testallocs_iterate(ii_grid_top)
    end

    @testset "subdomains" begin
        @testset "same cell type" begin
            grid = generate_grid(Hexahedron, (1000, 2, 1))
            ip = Lagrange{RefHexahedron, 1}()
            dh = DofHandler(grid)
            sdh1 = SubDofHandler(dh, Set([collect(1:500)..., collect(1501:2000)...]))
            sdh2 = SubDofHandler(dh, Set([collect(501:1500)...]))
            add!(sdh1, :u, ip)
            add!(sdh2, :u, ip)
            close!(dh)
            topology = ExclusiveTopology(grid)

            ii_sdh_top = InterfaceIterator(sdh1, sdh2, topology)
            ii_sdh = InterfaceIterator(sdh1, sdh2)
            ii_same_sdh = InterfaceIterator(sdh1, sdh1)
            @test ii_sdh.set == ii_sdh_top.set
            @test length(ii_sdh.set) == 1002
            @test count(interface_index -> interface_index[1] == interface_index[3] - 1000, ii_sdh.set) == 500
            @test count(interface_index -> interface_index[1] == interface_index[3] + 1000, ii_sdh.set) == 500
            ii_sdh_flipped = InterfaceIterator(sdh2, sdh1)
            @test all(interface -> InterfaceIndex(interface[3], interface[4], interface[1], interface[2]) ∈ ii_sdh_flipped.set, ii_sdh.set)
            # Test that topology has no effect on iterator type
            @test typeof(ii_sdh) == typeof(ii_sdh_top)
            # Test for allocations
            @test (testallocs_iterate(ii_sdh)) === nothing
            @test (testallocs_iterate(ii_same_sdh)) === nothing
        end

        @testset "mixed cell types" begin
            nodes = [Node((-1.0, 0.0)), Node((0.0, 0.0)), Node((1.0, 0.0)), Node((-1.0, 1.0)), Node((0.0, 1.0))]
            cells = [
                Quadrilateral((1, 2, 5, 4)),
                Triangle((3, 5, 2)),
            ]
            grid = Grid(cells, nodes)
            topology = ExclusiveTopology(grid)
            ip_quad = Lagrange{RefQuadrilateral, 1}()
            ip_tri = Lagrange{RefTriangle, 1}()
            dh = DofHandler(grid)
            sdh_quad = SubDofHandler(dh, Set([1]))
            sdh_tri = SubDofHandler(dh, Set([2]))
            add!(sdh_quad, :u, ip_quad)
            add!(sdh_tri, :u, ip_tri)
            close!(dh)

            ii_sdh_top = InterfaceIterator(sdh_quad, sdh_tri, topology)
            ii_sdh = InterfaceIterator(sdh_quad, sdh_tri)
            @test ii_sdh.set == ii_sdh_top.set
            @test ii_sdh.set == Set([InterfaceIndex(1, 2, 2, 2)])
            ii_sdh_flipped = InterfaceIterator(sdh_tri, sdh_quad)
            @test ii_sdh_flipped.set == Set([InterfaceIndex(2, 2, 1, 2)])
            # Test that topology has no effect on iterator type
            @test typeof(ii_sdh) == typeof(ii_sdh_top)
            # Test for allocations
            @test (testallocs_iterate(ii_sdh_top)) === nothing
            @test (testallocs_iterate(ii_sdh)) === nothing
        end
    end

    @testset "InterfaceIndex" begin
        # checkmate, codecov bot!
        @test isequal(InterfaceIndex(1, 2, 3, 4), (1, 2, 3, 4))
        @test isequal((1, 2, 3, 4), InterfaceIndex(1, 2, 3, 4))
    end

    @testset "error paths" begin
        nodes = [Node((-1.0, 0.0)), Node((0.0, 0.0)), Node((1.0, 0.0)), Node((-1.0, 1.0)), Node((0.0, 1.0))]
        cells = [
            Quadrilateral((1, 2, 5, 4)),
            Triangle((3, 5, 2)),
        ]
        grid = Grid(cells, nodes)
        topology = ExclusiveTopology(grid)
        ip_quad = Lagrange{RefQuadrilateral, 1}()
        ip_tri = Lagrange{RefTriangle, 1}()
        dh = DofHandler(grid)
        sdh_quad = SubDofHandler(dh, Set([1]))
        sdh_tri = SubDofHandler(dh, Set([2]))
        add!(sdh_quad, :u, ip_quad)
        add!(sdh_tri, :u, ip_tri)
        close!(dh)
        @test_throws ErrorException("The cells in the set (set of InterfaceIndex) are not all of the same celltype on each side.") InterfaceIterator(sdh_quad, sdh_tri, Set([InterfaceIndex(2, 2, 1, 2)]))
    end
end
