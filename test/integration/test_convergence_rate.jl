using Ferrite, Test
include(joinpath(@__DIR__, "convergence_test_utils.jl"))

# These test also for correct convergence rates.
# The interpolations are split across test_convergence_rate*.jl files to balance
# the (expensive) 3D fine-grid solves across parallel workers. CrouzeixRaviart on
# RefTetrahedron is by far the most expensive single case (fine grid = 42³ tets),
# so it gets its own file.
@testset "convergence rate" begin
    for interpolation in (
            CrouzeixRaviart{RefTetrahedron, 1}(),
        )
        ConvergenceTestHelper.run_convergence_rate(interpolation)
    end
end
