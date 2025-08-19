# Helper function to test grid generation for a given floating-point type
function test_generate_grid(T::Type)
    # Define the cell types to test
    cell_types = [
        Line, QuadraticLine,
        Quadrilateral, QuadraticQuadrilateral, Triangle, QuadraticTriangle,
        Hexahedron, Wedge, Pyramid, Tetrahedron, SerendipityQuadraticHexahedron,
    ]

    # Loop over all cell types and test grid generation
    for CT in cell_types
        rdim = Ferrite.getrefdim(CT)
        nels = ntuple(i -> 2, rdim)
        left = -ones(Vec{rdim, T})
        right = ones(Vec{rdim, T})
        grid = generate_grid(CT, nels, left, right)
        @test isa(grid, Grid{rdim, CT, T})
    end
    return
end

function test_line_grid(T::Type)
    for CT in [QuadraticLine]
        for dim in 1:3
            left = rand(Vec{dim, T})
            right = left + rand(Vec{dim, T})
            grid = generate_grid(CT, (2,), left, right)

            @test isa(grid, Grid{dim, CT, T})
            @test get_node_coordinate(grid, 1) ≈ left
            @test get_node_coordinate(grid, getnnodes(grid)) ≈ right
            @test get_node_coordinate(grid, Int(ceil(getnnodes(grid) / 2))) ≈ (left + right) / 2
            @test getncells(grid) == 2
        end
    end

    # fallback case
    @test isa(generate_grid(Line, (1,)), Grid{1, Line, Float64})
    @test isa(generate_grid(QuadraticLine, (1,)), Grid{1, QuadraticLine, Float64})

    return
end

# Run tests for different floating-point types
@testset "Generate Grid Tests" begin
    test_generate_grid(Float64)
    test_generate_grid(Float32)
    test_line_grid(Float64)
    test_line_grid(Float32)
end
