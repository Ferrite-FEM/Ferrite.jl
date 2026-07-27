# Imports for parallel (isolated) test execution:
using Test, LinearAlgebra
include(joinpath(@__DIR__, "test_utils.jl"))
include(joinpath(@__DIR__, "interpolation_test_utils.jl"))

using Ferrite: reference_shape_value, reference_shape_gradient, OrderedSet

@testset "Hermite{RefLine, 3}" begin
    ip = Hermite{RefLine, 3}()

    test_interpolation_properties(ip)

    ξL, ξR = Vec((-1.0,)), Vec((1.0,))
    N(j, ξ) = reference_shape_value(ip, ξ, j)
    dN(j, ξ) = reference_shape_gradient(ip, ξ, j)[1]

    # The dof functionals are (N(-1), N'(-1), N(1), N'(1)) instead of point evaluations,
    # so the nodal interpolation tests in test_interpolations.jl do not apply.
    @testset "generalized Kronecker property" begin
        for j in 1:4
            @test N(j, ξL) ≈ (j == 1 ? 1.0 : 0.0) atol = 1.0e-14
            @test dN(j, ξL) ≈ (j == 2 ? 1.0 : 0.0) atol = 1.0e-14
            @test N(j, ξR) ≈ (j == 3 ? 1.0 : 0.0) atol = 1.0e-14
            @test dN(j, ξR) ≈ (j == 4 ? 1.0 : 0.0) atol = 1.0e-14
        end
    end

    @testset "cubic reproduction in reference coordinates" begin
        for (f, df) in ((x -> 1.0, x -> 0.0), (x -> x, x -> 1.0), (x -> x^2, x -> 2x), (x -> x^3, x -> 3x^2))
            for ξv in (-0.7, 0.13, 0.9)
                ξ = Vec((ξv,))
                interp = f(-1.0) * N(1, ξ) + df(-1.0) * N(2, ξ) + f(1.0) * N(3, ξ) + df(1.0) * N(4, ξ)
                @test interp ≈ f(ξv) atol = 1.0e-13
            end
        end
    end

    # Nonuniform 2-cell grid: nodes at x = 0, 1, 3 (element lengths 1 and 2). Two different
    # element lengths make the tests below sensitive to the Jacobian scaling of the
    # derivative basis functions in the HermiteMapping.
    grid2 = Grid([Line((1, 2)), Line((2, 3))], [Node(Vec((0.0,))), Node(Vec((1.0,))), Node(Vec((3.0,)))])
    dh2 = DofHandler(grid2)
    add!(dh2, :w, ip)
    close!(dh2)

    @testset "dof distribution" begin
        @test ndofs(dh2) == 6
        cd1, cd2 = celldofs(dh2, 1), celldofs(dh2, 2)
        @test cd1[3:4] == cd2[1:2] # shared (value, derivative) pair at the middle vertex
        @test length(unique(vcat(cd1, cd2))) == 6
    end

    # Exact interpolant of w(x) = x^3: value dofs x³, derivative dofs 3x² (physical derivative)
    w_ex(x) = x^3
    dw_ex(x) = 3x^2
    u2 = zeros(ndofs(dh2))
    for cellid in 1:2
        cd = celldofs(dh2, cellid)
        xs = [x[1] for x in getcoordinates(grid2, cellid)]
        u2[cd[1]] = w_ex(xs[1]); u2[cd[2]] = dw_ex(xs[1])
        u2[cd[3]] = w_ex(xs[2]); u2[cd[4]] = dw_ex(xs[2])
    end

    @testset "mapping exactness on nonuniform mesh" begin
        cv = CellValues(QuadratureRule{RefLine}(3), ip; update_hessians = true)
        for cellid in 1:2
            coords = getcoordinates(grid2, cellid)
            reinit!(cv, coords) # cell-less reinit must work (mapping_needs_cell = false)
            ue = u2[celldofs(dh2, cellid)]
            for qp in 1:getnquadpoints(cv)
                x = spatial_coordinate(cv, qp, coords)[1]
                @test function_value(cv, qp, ue) ≈ w_ex(x) atol = 1.0e-12
                @test function_gradient(cv, qp, ue)[1] ≈ dw_ex(x) atol = 1.0e-12
                @test function_hessian(cv, qp, ue)[1, 1] ≈ 6x atol = 1.0e-11
            end
        end
    end

    @testset "per-DiffOrder mapping consistency" begin
        qr = QuadratureRule{RefLine}(2)
        coords = getcoordinates(grid2, 2)
        cv0 = CellValues(qr, ip; update_gradients = false, update_detJdV = false)
        cv1 = CellValues(qr, ip)
        cv2 = CellValues(qr, ip; update_hessians = true)
        reinit!(cv0, coords); reinit!(cv1, coords); reinit!(cv2, coords)
        for qp in 1:2, j in 1:4
            @test shape_value(cv0, qp, j) ≈ shape_value(cv2, qp, j)
            @test shape_value(cv1, qp, j) ≈ shape_value(cv2, qp, j)
            @test shape_gradient(cv1, qp, j) ≈ shape_gradient(cv2, qp, j)
        end
    end

    @testset "C1 continuity at shared vertex" begin
        urand = rand(ndofs(dh2))
        cv_r = CellValues(QuadratureRule{RefLine}([1.0], [Vec((1.0,))]), ip)
        cv_l = CellValues(QuadratureRule{RefLine}([1.0], [Vec((-1.0,))]), ip)
        reinit!(cv_r, getcoordinates(grid2, 1))
        reinit!(cv_l, getcoordinates(grid2, 2))
        @test function_value(cv_r, 1, urand[celldofs(dh2, 1)]) ≈ function_value(cv_l, 1, urand[celldofs(dh2, 2)])
        @test function_gradient(cv_r, 1, urand[celldofs(dh2, 1)]) ≈ function_gradient(cv_l, 1, urand[celldofs(dh2, 2)])
    end

    @testset "Float32 and inference" begin
        cvf = CellValues(Float32, QuadratureRule{RefLine}(2), ip; update_hessians = true)
        reinit!(cvf, [Vec((0.0f0,)), Vec((2.0f0,))])
        uef = rand(Float32, 4)
        @test (@inferred function_value(cvf, 1, uef)) isa Float32
        @test (@inferred function_gradient(cvf, 1, uef)) isa Vec{1, Float32}
        @test (@inferred function_hessian(cvf, 1, uef)) isa Tensor{2, 1, Float32}
    end

    @testset "FacetValues" begin
        fv = FacetValues(FacetQuadratureRule{RefLine}(1), ip)
        reinit!(fv, getcoordinates(grid2, 1), 2) # right vertex of cell 1
        @test shape_value(fv, 1, 3) ≈ 1.0
        @test shape_value(fv, 1, 1) ≈ 0.0 atol = 1.0e-14
    end

    @testset "MultiFieldCellValues" begin
        qr = QuadratureRule{RefLine}(2)
        coords = getcoordinates(grid2, 2)
        cmv = MultiFieldCellValues(qr, (w = ip, u = Lagrange{RefLine, 1}()))
        reinit!(cmv, coords)
        cv = CellValues(qr, ip)
        reinit!(cv, coords)
        for qp in 1:2, j in 1:4
            @test shape_value(cmv.w, qp, j) ≈ shape_value(cv, qp, j)
            @test shape_gradient(cmv.w, qp, j) ≈ shape_gradient(cv, qp, j)
        end
    end

    @testset "evaluate_at_grid_nodes and PointEvalHandler" begin
        @test evaluate_at_grid_nodes(dh2, u2, :w) ≈ [0.0, 1.0, 27.0]
        ph = PointEvalHandler(grid2, [Vec((0.3,)), Vec((1.77,))])
        @test evaluate_at_points(ph, dh2, u2, :w) ≈ [0.3^3, 1.77^3]
    end

    @testset "VTK export" begin
        mktempdir() do dir
            fname = joinpath(dir, "hermite")
            VTKGridFile(fname, dh2) do vtk
                write_solution(vtk, dh2, u2, "")
            end
            @test isfile(fname * ".vtu")
        end
    end

    @testset "Dirichlet on value and derivative dofs" begin
        grid = generate_grid(Line, (2,), Vec((0.0,)), Vec((1.0,)))
        dh = DofHandler(grid); add!(dh, :w, ip); close!(dh)
        # default kind = :value constrains only the value dofs
        ch = ConstraintHandler(dh)
        add!(ch, Dirichlet(:w, union(getfacetset(grid, "left"), getfacetset(grid, "right")), x -> x[1] + 42.0))
        close!(ch); update!(ch, 0.0)
        left_dof = celldofs(dh, 1)[1]
        right_dof = celldofs(dh, 2)[3]
        @test sort(ch.prescribed_dofs) == sort([left_dof, right_dof])
        @test ch.inhomogeneities[findfirst(==(left_dof), ch.prescribed_dofs)] ≈ 42.0
        @test ch.inhomogeneities[findfirst(==(right_dof), ch.prescribed_dofs)] ≈ 43.0
        # kind = :derivative constrains only the derivative dofs, and f prescribes du/dx
        ch2 = ConstraintHandler(dh)
        add!(ch2, Dirichlet(:w, getfacetset(grid, "left"), Returns(0.0)))
        add!(ch2, Dirichlet(:w, union(getfacetset(grid, "left"), getfacetset(grid, "right")), (x, t) -> x[1] + t; kind = :derivative))
        close!(ch2); update!(ch2, 0.0)
        left_deriv = celldofs(dh, 1)[2]
        right_deriv = celldofs(dh, 2)[4]
        @test sort(ch2.prescribed_dofs) == sort([left_dof, left_deriv, right_deriv])
        a = zeros(ndofs(dh)); apply!(a, ch2)
        @test a[left_dof] ≈ 0.0
        @test a[left_deriv] ≈ 0.0
        @test a[right_deriv] ≈ 1.0
        update!(ch2, 2.0) # derivative BCs are update!-able in time
        apply!(a, ch2)
        @test a[left_deriv] ≈ 2.0
        @test a[right_deriv] ≈ 3.0
        # requesting a dof kind the interpolation does not have errors
        @test_throws ErrorException add!(ConstraintHandler(dh), Dirichlet(:w, getfacetset(grid, "left"), Returns(0.0); kind = :nope))
        dh_lag = DofHandler(grid); add!(dh_lag, :u, Lagrange{RefLine, 1}()); close!(dh_lag)
        @test_throws ErrorException add!(ConstraintHandler(dh_lag), Dirichlet(:u, getfacetset(grid, "left"), Returns(0.0); kind = :derivative))
    end

    @testset "unsupported usage errors" begin
        @test_throws ArgumentError Hermite{RefLine, 3}()^2
        @test_throws ArgumentError CellValues(QuadratureRule{RefLine}(2), ip, Lagrange{RefLine, 2}()^1) # curved
        @test_throws ArgumentError CellValues(QuadratureRule{RefLine}(2), ip, Lagrange{RefLine, 1}()^2) # embedded
        @test_throws ErrorException apply_analytical!(zeros(ndofs(dh2)), dh2, :w, x -> 0.0)
        ch = ConstraintHandler(dh2)
        @test_throws ArgumentError add!(ch, Dirichlet(:w, Set([1]), Returns(0.0))) # nodeset
        @test_throws ArgumentError add!(ch, PeriodicDirichlet(:w, Ferrite.PeriodicFacetPair[]))
    end

    @testset "cantilever beam vs analytic solution" begin
        function solve_cantilever(n; L, EI, q = 0.0, P = 0.0, M = 0.0, nonuniform = false)
            grid = if nonuniform
                xs = L .* range(0.0, 1.0; length = n + 1) .^ 2
                Grid(
                    [Line((i, i + 1)) for i in 1:n], [Node(Vec((x,))) for x in xs];
                    facetsets = Dict("left" => OrderedSet([FacetIndex(1, 1)]))
                )
            else
                generate_grid(Line, (n,), Vec((0.0,)), Vec((L,)))
            end
            dh = DofHandler(grid); add!(dh, :w, ip); close!(dh)
            ch = ConstraintHandler(dh)
            add!(ch, Dirichlet(:w, getfacetset(grid, "left"), Returns(0.0)))
            add!(ch, Dirichlet(:w, getfacetset(grid, "left"), Returns(0.0); kind = :derivative))
            close!(ch)
            cv = CellValues(QuadratureRule{RefLine}(2), ip; update_hessians = true)
            K = allocate_matrix(dh)
            f = zeros(ndofs(dh))
            assembler = start_assemble(K, f)
            Ke = zeros(4, 4); fe = zeros(4)
            for cell in CellIterator(dh)
                reinit!(cv, cell)
                fill!(Ke, 0.0); fill!(fe, 0.0)
                for qp in 1:getnquadpoints(cv)
                    dΩ = getdetJdV(cv, qp)
                    for i in 1:4
                        fe[i] += q * shape_value(cv, qp, i) * dΩ
                        κi = shape_hessian(cv, qp, i)
                        for j in 1:4
                            Ke[i, j] += EI * (κi ⊡ shape_hessian(cv, qp, j)) * dΩ
                        end
                    end
                end
                assemble!(assembler, celldofs(cell), Ke, fe)
            end
            tip = celldofs(dh, getncells(grid))
            f[tip[3]] += P # point load -> value dof
            f[tip[4]] += M # point moment -> derivative dof
            apply!(K, f, ch)
            u = K \ f
            return u[tip[3]], u[tip[4]]
        end
        L, EI = 2.0, 3.0
        for n in (1, 5), nonuniform in (false, true)
            n == 1 && nonuniform && continue
            for (q, P, M) in ((0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (2.0, 3.0, 0.5))
                wt, θt = solve_cantilever(n; L, EI, q, P, M, nonuniform)
                # The cubic Hermite solution is nodally exact for Euler-Bernoulli beams
                @test wt ≈ q * L^4 / (8EI) + P * L^3 / (3EI) + M * L^2 / (2EI)
                @test θt ≈ q * L^3 / (6EI) + P * L^2 / (2EI) + M * L / EI
            end
        end
    end
end
