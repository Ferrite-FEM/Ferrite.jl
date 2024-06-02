using Test

getview(vals, inds::Ferrite.AdaptiveRange) = view(vals, inds.start:(inds.start + inds.ncurrent - 1))

@testset "CompareWithExclusiveTop" begin
    for CT in (Line, Quadrilateral, Tetrahedron)
        @testset "$CT" begin
            grid = generate_grid(CT, ntuple(Returns(3), Ferrite.getrefdim(CT)))
            top = ExclusiveTopology(grid)
            top2 = Ferrite.ExclusiveTopology2(grid)
            for field in (:vertex_to_cell, :cell_neighbor, :face_face_neighbor, :edge_edge_neighbor, :vertex_vertex_neighbor)
                @testset "$field" begin
                    nbh = getproperty(top, field)
                    nbh2 = getproperty(top2, field)
                    for idx in eachindex(nbh)
                        @test Set(nbh[idx]) == Set(nbh2[idx])
                    end
                end
            end
        end
    end
end
