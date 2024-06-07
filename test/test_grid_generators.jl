# Helper function to test grid generation for a given floating-point type
function test_generate_grid(T::Type)
    # Define the cell types to test
    cell_types = [
        Line, QuadraticLine,
        Quadrilateral, QuadraticQuadrilateral, Triangle, QuadraticTriangle,
        Hexahedron, Wedge, Pyramid, Tetrahedron, SerendipityQuadraticHexahedron]

    # Loop over all cell types and test grid generation
    for CT in cell_types
        rdim = Ferrite.getrefdim(CT) 
        nels = ntuple(i -> 2, rdim)  
        left = - ones(Vec{rdim,T})
        right =  ones(Vec{rdim,T})
        grid = generate_grid(CT, nels, left, right)
        @test isa(grid, Grid{rdim, CT, T})
    end
end

# Run tests for different floating-point types
@testset "Generate Grid Tests" begin
    test_generate_grid(Float64)
    test_generate_grid(Float32)
end
