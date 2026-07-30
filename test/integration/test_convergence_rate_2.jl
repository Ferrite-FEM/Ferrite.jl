using Ferrite, Test
include(joinpath(@__DIR__, "convergence_test_utils.jl"))

# Continuation of test_convergence_rate.jl (split to balance parallel workers).
@testset "convergence rate (part 2)" begin
    for interpolation in (
            Lagrange{RefLine, 1}(),
            Lagrange{RefLine, 2}(),
            Lagrange{RefQuadrilateral, 1}(),
            Lagrange{RefQuadrilateral, 2}(),
            Lagrange{RefQuadrilateral, 3}(),
            Lagrange{RefTriangle, 1}(),
            Lagrange{RefTriangle, 2}(),
            Lagrange{RefHexahedron, 2}(),
            Lagrange{RefHexahedron, 3}(),
            Lagrange{RefTetrahedron, 2}(),
            Lagrange{RefTetrahedron, 4}(),
            Lagrange{RefPrism, 2}(),
            CrouzeixRaviart{RefTriangle, 1}(),
            RannacherTurek{RefQuadrilateral, 1}(),
            RannacherTurek{RefHexahedron, 1}(),
        )
        ConvergenceTestHelper.run_convergence_rate(interpolation)
    end
end
