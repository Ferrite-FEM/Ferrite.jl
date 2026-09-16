using Ferrite, Test, LinearAlgebra
using Ferrite:
    TensorProductEvaluator, lexicographic_numbering, lexicographic_dofmap,
    quadrature_point_data, read_dof_values!, distribute_local_to_global!,
    evaluate_gradients!, integrate_gradients!, get_gradient, submit_gradient!

@testset "matrix-free operator evaluation" begin
    # A small distorted (non-affine) hexahedral grid such that the Jacobian varies within
    # the cells.
    function distorted_grid(n)
        grid = generate_grid(Hexahedron, (n, n, n))
        transform_coordinates!(grid, x -> x + Vec(0.03 * sinpi(x[2]) * x[3], 0.02 * sinpi(x[3]), 0.04 * sinpi(x[1]) * x[2]))
        return grid
    end

    @testset "reference gradient evaluation ($(nc == 1 ? "scalar" : "vector"))" for nc in (1, 3)
        ip_scalar = Lagrange{RefHexahedron, 2}()
        ip = nc == 1 ? ip_scalar : ip_scalar^3
        qr1d = QuadratureRule{RefLine}(3)
        ev = TensorProductEvaluator(ip, qr1d)
        @test Ferrite.getnquadpoints(ev) == 27
        @test Ferrite.getnbasefunctions(ev) == 27 * nc
        ## Random local dof values, gathered through the lexicographic permutation
        lex = lexicographic_numbering(ip)
        ue = rand(getnbasefunctions(ip))
        read_dof_values!(ev, ue, invperm(lex))
        evaluate_gradients!(ev)
        ## Compare with direct evaluation of the interpolation in the 3D tensor product
        ## quadrature points (first coordinate fastest)
        p1d = Ferrite.getpoints(qr1d)
        nq = length(p1d)
        for q3 in 1:nq, q2 in 1:nq, q1 in 1:nq
            q = q1 + nq * (q2 - 1) + nq * nq * (q3 - 1)
            ξ = Vec(p1d[q1][1], p1d[q2][1], p1d[q3][1])
            ĝ = sum(ue[i] * Ferrite.reference_shape_gradient(ip, ξ, i) for i in 1:getnbasefunctions(ip))
            @test get_gradient(ev, q) ≈ ĝ
        end
    end

    @testset "heat operator vs assembled matrix" begin
        grid = distorted_grid(3)
        ip = Lagrange{RefHexahedron, 2}()
        qr1d = QuadratureRule{RefLine}(3)
        qr = QuadratureRule{RefHexahedron}(3)
        dh = close!(add!(DofHandler(grid), :u, ip))
        κ(x) = 1.3 + 0.9 * sinpi(x[1]) * x[2]
        ## Matrix-free operator pieces
        ev = TensorProductEvaluator(ip, qr1d)
        dofmap = lexicographic_dofmap(dh, ip)
        Dq = quadrature_point_data(grid, qr) do x, J, w
            Jinv = inv(J)
            return det(J) * w * κ(x) * dott(Jinv)
        end
        ## Assembled reference
        cv = CellValues(qr, ip)
        K = allocate_matrix(dh)
        assembler = start_assemble(K)
        Ke = zeros(getnbasefunctions(cv), getnbasefunctions(cv))
        for cell in CellIterator(dh)
            reinit!(cv, cell)
            fill!(Ke, 0)
            for q in 1:getnquadpoints(cv)
                dΩ = getdetJdV(cv, q) * κ(spatial_coordinate(cv, q, getcoordinates(cell)))
                for i in 1:size(Ke, 1), j in 1:size(Ke, 2)
                    Ke[i, j] += (shape_gradient(cv, q, i) ⋅ shape_gradient(cv, q, j)) * dΩ
                end
            end
            assemble!(assembler, celldofs(cell), Ke)
        end
        ## Compare operator application
        x = rand(ndofs(dh))
        y = zeros(ndofs(dh))
        for e in 1:getncells(grid)
            dofs = view(dofmap, :, e)
            read_dof_values!(ev, x, dofs)
            evaluate_gradients!(ev)
            for q in 1:Ferrite.getnquadpoints(ev)
                submit_gradient!(ev, Dq[q, e] ⋅ get_gradient(ev, q), q)
            end
            integrate_gradients!(ev)
            distribute_local_to_global!(y, ev, dofs)
        end
        @test y ≈ K * x
    end

    @testset "elasticity operator vs assembled matrix" begin
        grid = distorted_grid(2)
        ipv = Lagrange{RefHexahedron, 2}()^3
        qr1d = QuadratureRule{RefLine}(3)
        qr = QuadratureRule{RefHexahedron}(3)
        dh = close!(add!(DofHandler(grid), :u, ipv))
        λ(x) = 2.0 + x[1]
        μ(x) = 1.0 + 0.5 * sinpi(x[3])
        ## Matrix-free operator pieces
        ev = TensorProductEvaluator(ipv, qr1d)
        dofmap = lexicographic_dofmap(dh, ipv)
        data = quadrature_point_data(grid, qr) do x, J, w
            return (Jinv = inv(J), λw = det(J) * w * λ(x), μw = det(J) * w * μ(x))
        end
        ## Assembled reference
        cv = CellValues(qr, ipv)
        K = allocate_matrix(dh)
        assembler = start_assemble(K)
        Ke = zeros(getnbasefunctions(cv), getnbasefunctions(cv))
        for cell in CellIterator(dh)
            reinit!(cv, cell)
            fill!(Ke, 0)
            for q in 1:getnquadpoints(cv)
                x_q = spatial_coordinate(cv, q, getcoordinates(cell))
                dΩ = getdetJdV(cv, q)
                for i in 1:size(Ke, 1)
                    εi = shape_symmetric_gradient(cv, q, i)
                    for j in 1:size(Ke, 2)
                        εj = shape_symmetric_gradient(cv, q, j)
                        Ke[i, j] += (λ(x_q) * tr(εi) * tr(εj) + 2 * μ(x_q) * (εi ⊡ εj)) * dΩ
                    end
                end
            end
            assemble!(assembler, celldofs(cell), Ke)
        end
        ## Compare operator application
        x = rand(ndofs(dh))
        y = zeros(ndofs(dh))
        for e in 1:getncells(grid)
            dofs = view(dofmap, :, e)
            read_dof_values!(ev, x, dofs)
            evaluate_gradients!(ev)
            for q in 1:Ferrite.getnquadpoints(ev)
                d = data[q, e]
                ε = symmetric(get_gradient(ev, q) ⋅ d.Jinv)
                σw = d.λw * tr(ε) * one(ε) + 2 * d.μw * ε
                submit_gradient!(ev, σw ⋅ d.Jinv', q)
            end
            integrate_gradients!(ev)
            distribute_local_to_global!(y, ev, dofs)
        end
        @test y ≈ K * x
    end
end
