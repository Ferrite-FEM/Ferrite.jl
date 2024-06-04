using Test
using Ferrite
using Tensors

# Helper function to test grid generation for a given floating-point type
function test_generate_grid(T::Type)
    # Helper function to check the type of nodes and cells in the grid
    function check_grid_types(grid::Grid{dim, C, T}) where {dim, C <: Ferrite.AbstractCell, T}
        # Check nodes
        for node in grid.nodes
            @test typeof(get_node_coordinate(node)) == Vec{dim, T}
        end
        
        # Check cells
        @test all(typeof(cell) == C for cell in grid.cells)
    end
    
    # Testing Line grid
    nel = (10,)
    left = Vec{1,T}((-1.0,))
    right = Vec{1,T}((1.0,))
    grid = generate_grid(Line, nel, left, right)
    @test isa(grid, Grid{1, Line, T})
    check_grid_types(grid)
    
    # Testing QuadraticLine grid
    grid = generate_grid(QuadraticLine, nel, left, right)
    @test isa(grid, Grid{1, QuadraticLine, T})
    check_grid_types(grid)

    # Testing 2D grids
    nel = (10, 10)
    left_2d = Vec{2,T}((-1.0, -1.0))
    right_2d = Vec{2,T}((1.0, 1.0))
    grid = generate_grid(Quadrilateral, nel, left_2d, right_2d)
    @test isa(grid, Grid{2, Quadrilateral, T})
    check_grid_types(grid)

    grid = generate_grid(QuadraticQuadrilateral, nel, left_2d, right_2d)
    @test isa(grid, Grid{2, QuadraticQuadrilateral, T})
    check_grid_types(grid)

    grid = generate_grid(Triangle, nel, left_2d, right_2d)
    @test isa(grid, Grid{2, Triangle, T})
    check_grid_types(grid)

    grid = generate_grid(QuadraticTriangle, nel, left_2d, right_2d)
    @test isa(grid, Grid{2, QuadraticTriangle, T})
    check_grid_types(grid)

    # Testing 3D grids
    nel = (10, 10, 10)
    left_3d = Vec{3,T}((-1.0, -1.0, -1.0))
    right_3d = Vec{3,T}((1.0, 1.0, 1.0))
    grid = generate_grid(Hexahedron, nel, left_3d, right_3d)
    @test isa(grid, Grid{3, Hexahedron, T})
    check_grid_types(grid)

    grid = generate_grid(Wedge, nel, left_3d, right_3d)
    @test isa(grid, Grid{3, Wedge, T})
    check_grid_types(grid)

    grid = generate_grid(Pyramid, nel, left_3d, right_3d)
    @test isa(grid, Grid{3, Pyramid, T})
    check_grid_types(grid)

    grid = generate_grid(Tetrahedron, nel, left_3d, right_3d)
    @test isa(grid, Grid{3, Tetrahedron, T})
    check_grid_types(grid)

    grid = generate_grid(SerendipityQuadraticHexahedron, nel, left_3d, right_3d)
    @test isa(grid, Grid{3, SerendipityQuadraticHexahedron, T})
    check_grid_types(grid)
end

# Run tests for different floating-point types
@testset "Generate Grid Tests" begin
    test_generate_grid(Float64)
    test_generate_grid(Float32)
    test_generate_grid(Float16)
end
