using Ferrite, Test
include(joinpath(@__DIR__, "convergence_test_utils.jl"))

# These test only for convergence within margins
@testset "convergence analysis" begin
    for interpolation in (
            Lagrange{RefTriangle, 3}(),
            Lagrange{RefTriangle, 4}(),
            Lagrange{RefTriangle, 5}(),
            Lagrange{RefHexahedron, 1}(),
            Lagrange{RefHexahedron, 3}(),
            Lagrange{RefTetrahedron, 1}(),
            Lagrange{RefTetrahedron, 4}(),
            Lagrange{RefPrism, 1}(),
            Lagrange{RefPyramid, 1}(),
            #
            Serendipity{RefQuadrilateral, 2}(),
            Serendipity{RefHexahedron, 2}(),
            #
            BubbleEnrichedLagrange{RefTriangle, 1}(),
            #
            CrouzeixRaviart{RefTriangle, 1}(),
            CrouzeixRaviart{RefTetrahedron, 1}(),
            RannacherTurek{RefQuadrilateral, 1}(),
            RannacherTurek{RefHexahedron, 1}(),
        )
        ConvergenceTestHelper.run_convergence_analysis(interpolation)
    end
end
