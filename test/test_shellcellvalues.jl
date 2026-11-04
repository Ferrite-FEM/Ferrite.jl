# Imports for parallel (isolated) test execution:
using LinearAlgebra
include(joinpath(@__DIR__, "test_utils.jl"))

@testset "ShellCellValues" begin
    @testset "Non-embedded: ip=$scalar_interpol" for (scalar_interpol, quad_rule) in (
            (Lagrange{RefLine, 2}(), QuadratureRule{RefLine}(2)),
            (Lagrange{RefQuadrilateral, 1}(), QuadratureRule{RefQuadrilateral}(2)),
            (Lagrange{RefQuadrilateral, 2}(), QuadratureRule{RefQuadrilateral}(2)),
            (Lagrange{RefTriangle, 2}(), QuadratureRule{RefTriangle}(2)),
            (Lagrange{RefHexahedron, 2}(), QuadratureRule{RefHexahedron}(2)),
            (Serendipity{RefQuadrilateral, 2}(), QuadratureRule{RefQuadrilateral}(2)),
            (Lagrange{RefTetrahedron, 2}(), QuadratureRule{RefTetrahedron}(2)),
        )
        for func_interpol in (scalar_interpol, VectorizedInterpolation(scalar_interpol)), update_local_hessians in (false, true)
            geom_interpol = scalar_interpol
            n_basefunc_base = getnbasefunctions(scalar_interpol)
            update_hessians = update_local_hessians && Ferrite.getorder(func_interpol) > 1
            cv = CellValues(quad_rule, func_interpol, geom_interpol; update_hessians)
            scv = ShellCellValues(quad_rule, func_interpol, geom_interpol; update_hessians, update_local_hessians)
            @test getnbasefunctions(scv) == getnbasefunctions(cv)

            # Non-linear geometry
            coords = [x + 0.05 * rand(x) for x in valid_coordinates_and_normals(func_interpol)[1]]
            reinit!(cv, coords)
            reinit!(scv, coords)
            ue = rand(getnbasefunctions(func_interpol))
            for i in 1:getnquadpoints(scv)
                # Behaves as CellValues
                @test getdetJdV(scv, i) ≈ getdetJdV(cv, i)
                @test spatial_coordinate(scv, i, coords) ≈ spatial_coordinate(cv, i, coords)
                @test function_value(scv, i, ue) ≈ function_value(cv, i, ue)
                @test function_gradient(scv, i, ue) ≈ function_gradient(cv, i, ue)
                if update_hessians
                    @test Ferrite.function_hessian(scv, i, ue) ≈ Ferrite.function_hessian(cv, i, ue)
                end

                # The local frame is the Gram-Schmidt frame of the jacobian, and a rotation for non-embedded elements,
                # such that ∇ₛu = ∇u⋅E and ∇ₛ∇ₛu = Eᵀ⋅∇∇u⋅E
                ξqp = Ferrite.getpoints(quad_rule)[i]
                J = Tensors.gradient(ξ -> spatial_coordinate(geom_interpol, ξ, coords), ξqp)
                E = Ferrite.getlocalframe(scv, i)
                @test E ≈ Ferrite.gram_schmidt_frame(J)
                @test E' ⋅ E ≈ one(E)
                for j in 1:getnbasefunctions(scv)
                    @test shape_local_gradient(scv, i, j) ≈ shape_gradient(cv, i, j) ⋅ E
                end
                @test function_local_gradient(scv, i, ue) ≈ function_gradient(cv, i, ue) ⋅ E
                if update_hessians
                    Hqp = Ferrite.function_hessian(cv, i, ue)
                    Hqp_s = func_interpol isa Ferrite.ScalarInterpolation ? E' ⋅ Hqp ⋅ E : Hqp ⊡ otimesu(E, E)
                    @test function_local_hessian(scv, i, ue) ≈ Hqp_s
                end

                # Compared with AD of the function wrt. the coordinates in the local frame, s
                if update_local_hessians && scalar_interpol === Lagrange{RefQuadrilateral, 2}()
                    xqp = spatial_coordinate(scv, i, coords)
                    Hqp_s, Gqp_s, Vqp = local_frame_function_derivatives(func_interpol, coords, xqp, J, ue)
                    @test function_value(scv, i, ue) ≈ Vqp
                    @test function_local_gradient(scv, i, ue) ≈ Gqp_s
                    @test function_local_hessian(scv, i, ue) ≈ Hqp_s
                end
            end
        end
    end

    @testset "Embedded: $ip_base in $(sdim)D, vdim = $vdim" for (ip_base, sdim) in (
                (Lagrange{RefLine, 2}(), 2),
                (Lagrange{RefLine, 2}(), 3),
                (Lagrange{RefQuadrilateral, 2}(), 3),
                (Lagrange{RefTriangle, 2}(), 3),
            ), vdim in (0, 2)
        rdim = Ferrite.getrefdim(ip_base)
        ip = vdim > 0 ? ip_base^vdim : ip_base
        qr = QuadratureRule{Ferrite.getrefshape(ip_base)}(2)
        scv = ShellCellValues(qr, ip, ip_base^sdim; update_local_hessians = true)
        # Curved geometry: Reference coordinates lifted to sdim, a quadratic bump in the normal direction,
        # a small random perturbation, and a rotation.
        R = rotmat(sdim)
        coords = map(Ferrite.reference_coordinates(ip_base)) do ξ
            bump = 0.2 * (ξ ⋅ ξ)
            x = Vec{sdim}(d -> d <= rdim ? ξ[d] : (d == rdim + 1 ? bump : 0.0))
            return R ⋅ (x + 0.02 * rand(Vec{sdim}))
        end
        reinit!(scv, coords)
        ue = rand(getnbasefunctions(ip))
        for i in 1:getnquadpoints(scv)
            J = Ferrite.getjacobian(Ferrite.calculate_mapping(Ferrite.get_geo_mapping(scv), i, coords))
            E = Ferrite.getlocalframe(scv, i)
            @test E ≈ Ferrite.gram_schmidt_frame(J)
            @test E' ⋅ E ≈ one(Tensor{2, rdim})
            if rdim == 2 && sdim == 3 # Frame is tangent to the surface
                n = J[:, 1] × J[:, 2]
                @test norm(n ⋅ E) < 1.0e-12 * norm(n)
            end
            # Consistency with the tangential gradients
            for j in 1:getnbasefunctions(scv)
                @test shape_local_gradient(scv, i, j) ≈ shape_gradient(scv, i, j) ⋅ E
            end
            # Compare with AD of the function wrt. the local frame coordinates
            xqp = spatial_coordinate(scv, i, coords)
            Hqp_s, Gqp_s, Vqp = local_frame_function_derivatives(ip, coords, xqp, J, ue)
            @test function_value(scv, i, ue) ≈ Vqp
            @test function_local_gradient(scv, i, ue) ≈ Gqp_s
            @test function_local_hessian(scv, i, ue) ≈ Hqp_s
        end
    end

    @testset "Options and errors" begin
        ip = Lagrange{RefQuadrilateral, 2}()
        qr = QuadratureRule{RefQuadrilateral}(2)
        x = [xi + 0.1 * rand(typeof(xi)) for xi in Ferrite.reference_coordinates(ip)]

        # Type stable construction, and default geometric interpolation
        scv = @inferred ShellCellValues(qr, ip, ip; update_local_hessians = Val(true))
        @test getfield(scv, :E) isa Vector{Tensor{2, 2, Float64, 4}}
        @test Ferrite.function_local_difforder(scv) == 2
        @inferred ShellCellValues(qr, ip; update_gradients = Val(false), update_detJdV = Val(false))
        reinit!(scv, x)
        reinit!(scv, x)
        @test (@allocated reinit!(scv, x)) == 0

        # Local gradients only (default), also without spatial gradients
        scv_grad = ShellCellValues(qr, ip, ip)
        @test Ferrite.function_local_difforder(scv_grad) == 1
        @test Ferrite.shape_local_hessian_type(scv_grad) === nothing
        scv_nograd = ShellCellValues(qr, ip, ip; update_gradients = false)
        @test Ferrite.shape_gradient_type(scv_nograd) === nothing
        reinit!(scv_grad, x)
        reinit!(scv_nograd, x)
        for i in 1:getnquadpoints(scv), j in 1:getnbasefunctions(scv)
            @test Ferrite.getlocalframe(scv_nograd, i) ≈ Ferrite.getlocalframe(scv, i)
            @test shape_local_gradient(scv_grad, i, j) ≈ shape_local_gradient(scv, i, j)
            @test shape_local_gradient(scv_nograd, i, j) ≈ shape_local_gradient(scv, i, j)
        end

        # Multiple fields
        smv = ShellCellValues(qr, (u = ip^2, p = ip, q = ip), ip; update_local_hessians = true)
        @test propertynames(smv) == (:u, :p, :q)
        scv_u = ShellCellValues(qr, ip^2, ip; update_local_hessians = true)
        reinit!(smv, x)
        reinit!(scv_u, x)
        ue = rand(getnbasefunctions(smv.u))
        pe = rand(getnbasefunctions(smv.p))
        for i in 1:getnquadpoints(smv)
            @test Ferrite.getlocalframe(smv, i) ≈ Ferrite.getlocalframe(scv, i)
            @test getdetJdV(smv, i) ≈ getdetJdV(scv, i)
            @test function_value(smv.p, i, pe) ≈ function_value(scv, i, pe)
            @test function_gradient(smv.p, i, pe) ≈ function_gradient(scv, i, pe)
            @test function_local_gradient(smv.p, i, pe) ≈ function_local_gradient(scv, i, pe)
            @test function_local_hessian(smv.q, i, pe) ≈ function_local_hessian(scv, i, pe)
            @test function_gradient(smv.u, i, ue) ≈ function_gradient(scv_u, i, ue)
            @test function_local_gradient(smv.u, i, ue) ≈ function_local_gradient(scv_u, i, ue)
            @test function_local_hessian(smv.u, i, ue) ≈ function_local_hessian(scv_u, i, ue)
        end
        @test smv.p === smv.q # Equal interpolations are aliased
        @test smv.u isa Ferrite.FunctionValues
        @test_throws ArgumentError getnbasefunctions(smv)
        @test_throws ArgumentError shape_value(smv, 1, 1)
        @test_throws ArgumentError shape_local_gradient(smv, 1, 1)
        @test_throws ArgumentError function_value(smv, 1, ue)
        @test_throws ArgumentError function_local_hessian(smv, 1, ue)
        reinit!(smv, x)
        @test (@allocated reinit!(smv, x)) == 0

        # Copies
        scv_copy = Ferrite.task_local_copy(scv)
        @test getfield(scv_copy, :E) == getfield(scv, :E)
        @test getfield(scv_copy, :E) !== getfield(scv, :E)
        @test only(Ferrite.get_fun_values(scv_copy)).dNds !== only(Ferrite.get_fun_values(scv)).dNds
        @test Ferrite.task_local_copy(smv).u.dNds == smv.u.dNds

        # Show
        @test occursin("ShellCellValues", sprint(show, MIME"text/plain"(), scv))
        @test occursin("u: ", sprint(show, MIME"text/plain"(), smv))

        # Errors
        ip_rt = RaviartThomas{RefQuadrilateral, 1}()
        @test_throws ArgumentError ShellCellValues(qr, ip_rt)
        @test_throws ArgumentError reinit!(scv, x[1:(end - 1)])
        @test_throws ArgumentError reinit!(scv, [Vec{3}((xi..., 0.0)) for xi in x]) # sdim mismatch
    end
end
