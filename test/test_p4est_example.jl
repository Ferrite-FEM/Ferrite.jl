using Ferrite, Test
import Ferrite: geometric_interpolation, getrefdim

include(joinpath(@__DIR__, "integration", "convergence_test_utils.jl"))

# Squared elementwise L2 errors against the analytical solution, used as the
# refinement indicator.
function compute_cell_squared_l2_errors(dh, u, cellvalues)
    η = zeros(getncells(dh.grid))
    for (cellid, cell) in enumerate(CellIterator(dh))
        reinit!(cellvalues, cell)
        coords = getcoordinates(cell)
        uₑ = u[celldofs(cell)]
        for q_point in 1:getnquadpoints(cellvalues)
            dΩ = getdetJdV(cellvalues, q_point)
            x = spatial_coordinate(cellvalues, q_point, coords)
            uₐₙₐ = ConvergenceTestHelper.analytical_solution(x)
            uₐₚₚᵣₒₓ = function_value(cellvalues, q_point, uₑ)
            η[cellid] += (uₐₙₐ - uₐₚₚᵣₒₓ)^2 * dΩ
        end
    end
    return η
end

# Dörfler marking: mark the smallest set of cells that accounts for a fraction
# θ of the total squared error.
function dorfler_marking(η, θ)
    perm = sortperm(η; rev = true)
    threshold = θ * sum(η)
    marked = Int[]
    acc = 0.0
    for cellid in perm
        push!(marked, cellid)
        acc += η[cellid]
        acc ≥ threshold && break
    end
    return marked
end

@testset "AMR convergence analysis" begin
    @testset "$interpolation" for (interpolation, L2target) in (
            (Lagrange{RefQuadrilateral, 1}(), 1.0e-3),
            (Lagrange{RefHexahedron, 1}(), 5.0e-3),
        )
        # Start from a deliberately coarse macro grid so that the adaptive
        # loop below has to do the resolving.
        geometry = ConvergenceTestHelper.get_geometry(interpolation)
        interpolation_geo = geometric_interpolation(geometry)
        dim = getrefdim(geometry)
        grid = generate_grid(geometry, ntuple(_ -> 4, dim))
        maxiter = 15
        forest = ForestBWG(grid, maxiter + 2)
        qr_order = ConvergenceTestHelper.get_quadrature_order(interpolation)
        qr = QuadratureRule{getrefshape(interpolation)}(qr_order)

        L2errors = Float64[]
        dofcounts = Int[]
        solved_nonconforming_grid = false
        local dh, u, cellvalues
        for _ in 1:maxiter
            grid_transferred = Ferrite.creategrid(forest)
            dh, ch, cellvalues = ConvergenceTestHelper.setup_poisson_problem(grid_transferred, interpolation, interpolation_geo, qr)
            u = ConvergenceTestHelper.solve(dh, ch, cellvalues)
            η = compute_cell_squared_l2_errors(dh, u, cellvalues)
            push!(L2errors, √(sum(η)))
            push!(dofcounts, ndofs(dh))
            solved_nonconforming_grid |= !isempty(grid_transferred.conformity_info)
            last(L2errors) ≤ L2target && break
            Ferrite.refine!(forest, dorfler_marking(η, 0.5))
            Ferrite.balanceforest!(forest)
        end

        # Each refinement step must reduce the discretization error ...
        @test all(diff(L2errors) .< 0)
        # ... down to the target tolerance within the iteration budget.
        @test last(L2errors) ≤ L2target
        # The optimal convergence rate for smooth solutions is
        # ‖u - uₕ‖_L² ~ ndofs^(-(p+1)/dim).
        rate = -(log(last(L2errors)) - log(first(L2errors))) / (log(last(dofcounts)) - log(first(dofcounts)))
        @test rate ≈ (Ferrite.getorder(interpolation) + 1) / dim atol = 0.2
        # Marking only a subset of the cells must have produced hanging nodes
        # at some point, otherwise the conformity machinery was not exercised.
        @test solved_nonconforming_grid
        # The final adaptive solution has to pass the pointwise and norm
        # checks, which in particular validates the hanging node constraints.
        testatol = ConvergenceTestHelper.get_test_tolerance(interpolation)
        ConvergenceTestHelper.check_and_compute_convergence_norms(dh, u, cellvalues, testatol)
    end
end

# Regression test: L2 projection on a non-conforming grid must reproduce a function
# that lies in the hanging-node constrained ansatz space exactly. This guards the
# condensation of the mass matrix and the projection rhs in the L2Projector
# (assembling only the upper triangle before condensation, or applying the
# solution-style `apply!` to the rhs, both silently corrupt the projection).
@testset "L2 projection on non-conforming grids" begin
    @testset "$geometry" for geometry in (Quadrilateral, Hexahedron)
        dim = getrefdim(geometry)
        grid = generate_grid(geometry, ntuple(_ -> 2, dim))
        forest = ForestBWG(grid, 3)
        Ferrite.refine!(forest, [1])
        Ferrite.balanceforest!(forest)
        ncgrid = Ferrite.creategrid(forest)
        # The refined grid must contain hanging nodes for this test to be meaningful
        @test !isempty(ncgrid.conformity_info)

        ip = geometric_interpolation(geometry)
        qr = QuadratureRule{getrefshape(ip)}(2)
        cv = CellValues(qr, ip, ip)

        proj = L2Projector(ncgrid)
        add!(proj, collect(1:getncells(ncgrid)), ip; qr_rhs = qr)
        close!(proj)

        # u(x) = x is componentwise linear, hence in the constrained ansatz space
        qpdata = map(1:getncells(ncgrid)) do cellid
            coords = getcoordinates(ncgrid, cellid)
            reinit!(cv, coords)
            [spatial_coordinate(cv, qp, coords) for qp in 1:getnquadpoints(cv)]
        end
        projected = project(proj, qpdata, qr)

        for cell in CellIterator(proj.dh)
            for (dof, x) in zip(celldofs(cell), getcoordinates(cell))
                @test projected[dof] ≈ x atol = 1.0e-12
            end
        end
    end
end
