# These test only for convergence within margins
@testset "convergence analysis" begin
    @testset failfast = true "$interpolation" for interpolation in (
            Lagrange{RefTriangle, 3}(),
            Lagrange{RefTriangle, 4}(),
            Lagrange{RefTriangle, 5}(),
            Lagrange{RefHexahedron, 1}(),
            Lagrange{RefTetrahedron, 1}(),
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
        # Generate a grid ...
        geometry = ConvergenceTestHelper.get_geometry(interpolation)
        interpolation_geo = geometric_interpolation(geometry)
        N = ConvergenceTestHelper.get_num_elements(interpolation)
        grid = generate_grid(geometry, ntuple(x -> N, getrefdim(geometry)))
        # ... a suitable quadrature rule ...
        qr_order = ConvergenceTestHelper.get_quadrature_order(interpolation)
        qr = QuadratureRule{getrefshape(interpolation)}(qr_order)
        # ... and then pray to the gods of convergence.
        dh, ch, cellvalues = ConvergenceTestHelper.setup_poisson_problem(grid, interpolation, interpolation_geo, qr)
        u = ConvergenceTestHelper.solve(dh, ch, cellvalues)
        ConvergenceTestHelper.check_and_compute_convergence_norms(dh, u, cellvalues, 1.0e-2)
    end
end

# These test also for correct convergence rates
@testset "convergence rate" begin
    @testset failfast = true  "$interpolation" for interpolation in (
            Lagrange{RefLine, 1}(),
            Lagrange{RefLine, 2}(),
            Lagrange{RefQuadrilateral, 1}(),
            Lagrange{RefQuadrilateral, 2}(),
            Lagrange{RefQuadrilateral, 3}(),
            Lagrange{RefTriangle, 1}(),
            Lagrange{RefTriangle, 2}(),
            Lagrange{RefHexahedron, 2}(),
            Lagrange{RefTetrahedron, 2}(),
            Lagrange{RefPrism, 2}(),
            CrouzeixRaviart{RefTriangle, 1}(),
            CrouzeixRaviart{RefTetrahedron, 1}(),
            RannacherTurek{RefQuadrilateral, 1}(),
            RannacherTurek{RefHexahedron, 1}(),
        )
        # Generate a grid ...
        geometry = ConvergenceTestHelper.get_geometry(interpolation)
        interpolation_geo = geometric_interpolation(geometry)
        # "Coarse case"
        N₁ = ConvergenceTestHelper.get_num_elements(interpolation)
        grid = generate_grid(geometry, ntuple(x -> N₁, getrefdim(geometry)))
        # ... a suitable quadrature rule ...
        qr_order = ConvergenceTestHelper.get_quadrature_order(interpolation)
        qr = QuadratureRule{getrefshape(interpolation)}(qr_order)
        # ... and then pray to the gods of convergence.
        dh, ch, cellvalues = ConvergenceTestHelper.setup_poisson_problem(grid, interpolation, interpolation_geo, qr)
        u = ConvergenceTestHelper.solve(dh, ch, cellvalues)
        L2₁, H1₁, _ = ConvergenceTestHelper.check_and_compute_convergence_norms(dh, u, cellvalues, 1.0e-2)

        # "Fine case"
        N₂ = 2 * N₁
        grid = generate_grid(geometry, ntuple(x -> N₂, getrefdim(geometry)))
        # ... a suitable quadrature rule ...
        qr_order = ConvergenceTestHelper.get_quadrature_order(interpolation)
        qr = QuadratureRule{getrefshape(interpolation)}(qr_order)
        # ... and then pray to the gods of convergence.
        dh, ch, cellvalues = ConvergenceTestHelper.setup_poisson_problem(grid, interpolation, interpolation_geo, qr)
        u = ConvergenceTestHelper.solve(dh, ch, cellvalues)
        L2₂, H1₂, _ = ConvergenceTestHelper.check_and_compute_convergence_norms(dh, u, cellvalues, 5.0e-3)

        @test -(log(L2₂) - log(L2₁)) / (log(N₂) - log(N₁)) ≈ Ferrite.getorder(interpolation) + 1 atol = 0.1
        @test -(log(H1₂) - log(H1₁)) / (log(N₂) - log(N₁)) ≈ Ferrite.getorder(interpolation) atol = 0.1
    end
end
