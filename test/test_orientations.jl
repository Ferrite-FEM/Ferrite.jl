@testset "Orientations" begin
    @testset "PathOrientationInfo" begin
        # Path [a, b], flipped
        test_paths = [
            ((1, 2), false),   # 1 -> 2, regular
            ((2, 1), true),    # 2 -> 1, inverted
            ((5, 1), true),    # 5 -> 1, inverted
            ((3, 5), false),   # 3 -> 5, regular
        ]

        for (path, flipped) in test_paths
            orientation = Ferrite.PathOrientationInfo(path)

            @test orientation.flipped == flipped
        end
    end

    @testset "SurfaceOrientationInfo" begin
        # Surface [facenodes], flipped, shift_index
        test_surfaces = [
            # 2
            # | \
            # 3 - 1, regular, rotation = 0 deg
            ((1, 2, 3), false, 0),
            # 1
            # | \
            # 2 - 3, regular, rotation = 120 deg
            ((3, 1, 2), false, 1),
            # 3
            # | \
            # 1 - 2, regular, rotation = 240 deg
            ((2, 3, 1), false, 2),
            # 3
            # | \
            # 2 - 1, inverted, rotation = 0 deg
            ((1, 3, 2), true, 0),
            # 1
            # | \
            # 3 - 2, inverted, rotation = 120 deg
            ((2, 1, 3), true, 1),
            # 2
            # | \
            # 1 - 3, inverted, rotation = 240 deg
            ((3, 2, 1), true, 2),
            # 4 - 3
            # |   |
            # 1 - 2, regular, rotation = 0 deg
            ((1, 2, 3, 4), false, 0),
            # 2 - 3
            # |   |
            # 1 - 4, inverted, rotation = 0 deg
            ((1, 4, 3, 2), true, 0),
            # 3 - 2
            # |   |
            # 4 - 1, regular, rotation = 90 deg
            ((4, 1, 2, 3), false, 1),
            # 1 - 2
            # |   |
            # 4 - 3, inverted, rotation = 270 deg
            ((4, 3, 2, 1), true, 3),
        ]


        for (surface, flipped, shift_index) in test_surfaces
            orientation = Ferrite.SurfaceOrientationInfo(surface)

            @test orientation.flipped == flipped
            @test orientation.shift_index == shift_index
        end
    end
end
