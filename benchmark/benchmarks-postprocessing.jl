#----------------------------------------------------------------------#
# Postprocessing: L2 projection, point evaluation and field evaluation
#----------------------------------------------------------------------#
SUITE["postprocessing"] = BenchmarkGroup()

# L2 projection of quadrature-point data (e.g. stresses) to nodes.
SUITE["postprocessing"]["L2Projector"] = BenchmarkGroup()
let g = SUITE["postprocessing"]["L2Projector"]
    grid = generate_grid(Triangle, (30, 30))
    ip = Lagrange{RefTriangle, 1}()
    qr = QuadratureRule{RefTriangle}(2)

    build_projector = function (grid, ip, qr)
        proj = L2Projector(grid)
        add!(proj, 1:getncells(grid), ip; qr_rhs = qr)
        return close!(proj)
    end
    g["build (Triangle 30×30)"] = @benchmarkable $build_projector($grid, $ip, $qr) evals = 1

    proj = build_projector(grid, ip, qr)
    qp_data = [[rand(SymmetricTensor{2, 2}) for _ in 1:getnquadpoints(qr)] for _ in 1:getncells(grid)]
    g["project tensor data (Triangle 30×30)"] = @benchmarkable project($proj, $qp_data) evals = 1
end

# Evaluating a finite element field at arbitrary points in the domain: locating the points
# (PointEvalHandler construction) and evaluating the field at them.
SUITE["postprocessing"]["point evaluation"] = BenchmarkGroup()
let g = SUITE["postprocessing"]["point evaluation"]
    grid = generate_grid(Quadrilateral, (50, 50))
    dh = DofHandler(grid)
    add!(dh, :u, Lagrange{RefQuadrilateral, 2}())
    close!(dh)
    u = rand(ndofs(dh))
    npoints = 1000
    points = [Vec((0.99 * cos(θ), 0.99 * sin(θ))) for θ in range(0, π / 2, npoints)]
    g["PointEvalHandler ($npoints points)"] = @benchmarkable PointEvalHandler($grid, $points) evals = 1 seconds = 1.0
    ph = PointEvalHandler(grid, points)
    g["evaluate_at_points ($npoints points)"] = @benchmarkable evaluate_at_points($ph, $dh, $u, :u) evals = 1
    g["evaluate_at_grid_nodes"] = @benchmarkable evaluate_at_grid_nodes($dh, $u, :u) evals = 1
end

# Setting dof values from an analytical function (e.g. initial conditions).
let
    grid = generate_grid(Quadrilateral, (50, 50))
    dh = DofHandler(grid)
    add!(dh, :u, Lagrange{RefQuadrilateral, 2}())
    close!(dh)
    u = zeros(ndofs(dh))
    SUITE["postprocessing"]["apply_analytical!"] = @benchmarkable(
        apply_analytical!($u, $dh, :u, x -> x[1] + x[2]), evals = 1
    )
end
