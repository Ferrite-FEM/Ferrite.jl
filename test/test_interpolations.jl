# Imports for parallel (isolated) test execution:
using LinearAlgebra
using ForwardDiff
include(joinpath(@__DIR__, "test_utils.jl"))
include(joinpath(@__DIR__, "interpolation_test_utils.jl"))

using Ferrite: reference_shape_value, reference_shape_gradient

@testset "interpolations" begin
    @testset "Correctness of $interpolation" for interpolation in (
            Lagrange{RefLine, 1}(),
            Lagrange{RefLine, 2}(),
            Lagrange{RefQuadrilateral, 1}(),
            Lagrange{RefQuadrilateral, 2}(),
            Lagrange{RefQuadrilateral, 3}(),
            Lagrange{RefTriangle, 1}(),
            Lagrange{RefTriangle, 2}(),
            Lagrange{RefTriangle, 3}(),
            Lagrange{RefTriangle, 4}(),
            Lagrange{RefTriangle, 5}(),
            Lagrange{RefHexahedron, 1}(),
            Lagrange{RefHexahedron, 2}(),
            Serendipity{RefQuadrilateral, 2}(),
            Serendipity{RefHexahedron, 2}(),
            Lagrange{RefTetrahedron, 1}(),
            Lagrange{RefTetrahedron, 2}(),
            Lagrange{RefPrism, 1}(),
            Lagrange{RefPrism, 2}(),
            Lagrange{RefPyramid, 1}(),
            Lagrange{RefPyramid, 2}(),
            #
            DiscontinuousLagrange{RefLine, 0}(),
            DiscontinuousLagrange{RefQuadrilateral, 0}(),
            DiscontinuousLagrange{RefHexahedron, 0}(),
            DiscontinuousLagrange{RefTriangle, 0}(),
            DiscontinuousLagrange{RefTetrahedron, 0}(),
            DiscontinuousLagrange{RefLine, 1}(),
            DiscontinuousLagrange{RefQuadrilateral, 1}(),
            DiscontinuousLagrange{RefHexahedron, 1}(),
            DiscontinuousLagrange{RefTriangle, 1}(),
            DiscontinuousLagrange{RefTetrahedron, 1}(),
            DiscontinuousLagrange{RefPrism, 1}(),
            DiscontinuousLagrange{RefPyramid, 1}(),
            #
            BubbleEnrichedLagrange{RefTriangle, 1}(),
            #
            CrouzeixRaviart{RefTriangle, 1}(),
            CrouzeixRaviart{RefTetrahedron, 1}(),
            RannacherTurek{RefQuadrilateral, 1}(),
            RannacherTurek{RefHexahedron, 1}(),
        )
        # Standard test all base interpolations must fulfill
        test_interpolation_properties(interpolation)

        ref_dim = Ferrite.getrefdim(interpolation)
        ref_shape = Ferrite.getrefshape(interpolation)
        func_order = Ferrite.getorder(interpolation)

        # Note that not every element formulation exists for every order and dimension.
        if applicable(Ferrite.getlowerorder, interpolation)
            @test isa(Ferrite.getlowerorder(interpolation), Interpolation{ref_shape, func_order - 1})
        end

        n_basefuncs = getnbasefunctions(interpolation)
        coords = Ferrite.reference_coordinates(interpolation)
        @test length(coords) == n_basefuncs

        @testset "Value Type $value_type" for value_type in (Float32, Float64)
            @testset let x = Vec{ref_dim, value_type}(sample_random_point(ref_shape))
                # Check gradient evaluation
                f(ξ) = [reference_shape_value(interpolation, Vec{ref_dim}(ξ), i) for i in 1:n_basefuncs]
                @test vec(ForwardDiff.jacobian(f, Array(x))') ≈
                    reinterpret(value_type, [reference_shape_gradient(interpolation, x, i) for i in 1:n_basefuncs])
                # Check partition of unity at random point.
                @test sum([reference_shape_value(interpolation, x, i) for i in 1:n_basefuncs]) ≈ 1.0
                # Check if the important functions are consistent
                @test_throws ArgumentError reference_shape_value(interpolation, x, n_basefuncs + 1)
                # Idempotency test
                @test reference_shape_value(interpolation, x, n_basefuncs) == reference_shape_value(interpolation, x, n_basefuncs)

                # Check for evaluation type correctness of interpolation
                for dof in 1:n_basefuncs
                    @test (@inferred reference_shape_value(interpolation, x, dof)) isa value_type
                    @test (@inferred reference_shape_gradient(interpolation, x, dof)) isa Vec{ref_dim, value_type}
                end
            end
        end

        # Check for Kronecker delta property of interpolation
        @testset "Kronecker delta property of dof $dof" for dof in 1:n_basefuncs
            for k in 1:n_basefuncs
                N_dof = reference_shape_value(interpolation, coords[dof], k)
                if k == dof
                    @test N_dof ≈ 1.0
                else
                    factor = interpolation isa Lagrange{RefQuadrilateral, 3} ? 200 : 4
                    @test N_dof ≈ 0.0 atol = factor * eps(typeof(N_dof))
                end
            end
        end

        # Test that facedof_indices(...) return in counter clockwise order (viewing from the outside)
        if interpolation isa Lagrange
            function __outward_normal(coords::Vector{<:Vec{1}}, nodes)
                n = coords[nodes[1]]
                return n / norm(n)
            end
            function __outward_normal(coords::Vector{<:Vec{2}}, nodes)
                p1 = coords[nodes[1]]
                p2 = coords[nodes[2]]
                n = Vec{2}((p2[2] - p1[2], - p2[1] + p1[1]))
                return n / norm(n)
            end
            function __outward_normal(coords::Vector{<:Vec{3}}, nodes)
                p1 = coords[nodes[1]]
                p2 = coords[nodes[2]]
                p3 = coords[nodes[3]]
                n = (p3 - p2) × (p1 - p2)
                return n / norm(n)
            end
            normals = reference_normals(getrefshape(interpolation))
            for (facetnodes, normal) in zip(Ferrite.facetdof_indices(interpolation), normals)
                @test __outward_normal(coords, facetnodes) ≈ normal
            end
        end

        # regression for https://github.com/Ferrite-FEM/Ferrite.jl/issues/520
        interpolation_type = typeof(interpolation).name.wrapper
        if func_order > 1 && interpolation_type != Ferrite.Serendipity
            first_order = interpolation_type{ref_shape, 1}()
            for (highorderface, firstorderface) in zip(Ferrite.facedof_indices(interpolation), Ferrite.facedof_indices(first_order))
                for (h_node, f_node) in zip(highorderface, firstorderface)
                    @test h_node == f_node
                end
            end
            if ref_dim > 2
                for (highorderedge, firstorderedge) in zip(Ferrite.edgedof_indices(interpolation), Ferrite.edgedof_indices(first_order))
                    for (h_node, f_node) in zip(highorderedge, firstorderedge)
                        @test h_node == f_node
                    end
                end
            end
        end

        @testset "VectorizedInterpolation" begin
            v_interpolation_1 = interpolation^2
            v_interpolation_2 = (d = 2; interpolation^d)
            @test getnbasefunctions(v_interpolation_1) ==
                getnbasefunctions(v_interpolation_2) ==
                getnbasefunctions(interpolation) * 2
            # pretty printing
            @test repr("text/plain", v_interpolation_1) == repr(v_interpolation_1.ip) * "^2"

            # Check for evaluation type correctness of vectorized interpolation
            v_interpolation_3 = interpolation^ref_dim

            @testset "Value Type $value_type" for value_type in (Float32, Float64)
                x = Vec{ref_dim, value_type}(sample_random_point(getrefshape(v_interpolation_1)))
                @testset "vectorized case of return type correctness of dof $dof" for dof in 1:n_basefuncs
                    @test @inferred(reference_shape_value(v_interpolation_1, x, dof)) isa Vec{2, value_type}
                    @test @inferred(reference_shape_gradient(v_interpolation_3, x, dof)) isa Tensor{2, ref_dim, value_type}
                end
            end

            if applicable(Ferrite.getlowerorder, interpolation)
                @test isa(Ferrite.getlowerorder(v_interpolation_1), Interpolation{ref_shape, func_order - 1})
                @test isa(Ferrite.getlowerorder(v_interpolation_2), Interpolation{ref_shape, func_order - 1})
                @test isa(Ferrite.getlowerorder(v_interpolation_3), Interpolation{ref_shape, func_order - 1})
            end
        end

        @testset "TensorizedInterpolation" begin
            @testset "TB = $TB" for TB in (Tensor{2, ref_dim}, SymmetricTensor{2, ref_dim})
                t_interpolation = TensorizedInterpolation{TB}(interpolation)
                nc = Tensors.n_components(TB)
                @test Ferrite.n_components(t_interpolation) == nc
                @test getnbasefunctions(t_interpolation) == nc * getnbasefunctions(interpolation)
                # eltype parameters in TB are stripped
                @test TensorizedInterpolation{TB{Float32}}(interpolation) === t_interpolation
                # pretty printing
                @test repr("text/plain", t_interpolation) ==
                    "TensorizedInterpolation{$TB}(" * repr("text/plain", interpolation) * ")"

                @testset "Value Type $value_type" for value_type in (Float32, Float64)
                    x = Vec{ref_dim, value_type}(sample_random_point(ref_shape))
                    GradT = Tensors.regular_if_possible(MixedTensor3{ref_dim, ref_dim, ref_dim, value_type})
                    for dof in 1:getnbasefunctions(t_interpolation)
                        base_dof, comp = fldmod1(dof, nc)
                        N = @inferred(reference_shape_value(t_interpolation, x, dof))
                        @test N isa TB{value_type}
                        dNdξ = @inferred(reference_shape_gradient(t_interpolation, x, dof))
                        @test dNdξ isa GradT
                        # One-hot in data component `comp`, scaled by the scalar base function
                        Nbase = reference_shape_value(interpolation, x, base_dof)
                        @test N.data[comp] ≈ Nbase
                        @test sum(abs, N.data) ≈ abs(Nbase)
                        # Analytic derivatives match the scalar base ones
                        dNdξ_base, Nbase2 = Ferrite.reference_shape_gradient_and_value(interpolation, x, base_dof)
                        E = Ferrite._tensorized_basis(TB, comp, one(value_type))
                        @test dNdξ ≈ otimes(E, dNdξ_base)
                        d2, d1, d0 = Ferrite.reference_shape_hessian_gradient_and_value(t_interpolation, x, dof)
                        h_base, g_base, v_base = Ferrite.reference_shape_hessian_gradient_and_value(interpolation, x, base_dof)
                        @test d2 ≈ otimes(E, h_base)
                        @test d1 ≈ dNdξ
                        @test d0 ≈ N
                    end
                end

                if applicable(Ferrite.getlowerorder, interpolation)
                    @test Ferrite.getlowerorder(t_interpolation) ===
                        TensorizedInterpolation{TB}(Ferrite.getlowerorder(interpolation))
                end
            end
        end
    end

    @testset "Discontinuous interpolations" begin
        @test Ferrite.reference_coordinates(DiscontinuousLagrange{RefTriangle, 0}()) ≈ [Vec{2, Float64}((1 / 3, 1 / 3))]
        @test Ferrite.reference_coordinates(DiscontinuousLagrange{RefQuadrilateral, 0}()) ≈ [Vec{2, Float64}((0, 0))]
        @test Ferrite.reference_coordinates(DiscontinuousLagrange{RefTetrahedron, 0}()) ≈ [Vec{3, Float64}((1 / 4, 1 / 4, 1 / 4))]
        @test Ferrite.reference_coordinates(DiscontinuousLagrange{RefHexahedron, 0}()) ≈ [Vec{3, Float64}((0, 0, 0))]
    end

    @testset "Correctness of AD of embedded interpolations" begin
        ips = Lagrange{RefQuadrilateral, 2}()
        vdim = 3
        ipv = ips^vdim
        ξ = rand(Vec{2, Float64})
        for ipv_ind in 1:getnbasefunctions(ipv)
            ips_ind, v_ind = fldmod1(ipv_ind, vdim)
            H, G, V = Ferrite.reference_shape_hessian_gradient_and_value(ipv, ξ, ipv_ind)
            h, g, v = Ferrite.reference_shape_hessian_gradient_and_value(ips, ξ, ips_ind)
            @test h ≈ H[v_ind, :, :]
            @test g ≈ G[v_ind, :]
            @test v ≈ V[v_ind]
        end
    end

    @testset "Errors for entitydof_indices on VectorizedInterpolations" begin
        for ip in (
                Lagrange{RefQuadrilateral, 2}()^2,
                TensorizedInterpolation{SymmetricTensor{2, 2}}(Lagrange{RefQuadrilateral, 2}()),
            )
            @test_throws ArgumentError Ferrite.vertexdof_indices(ip)
            @test_throws ArgumentError Ferrite.edgedof_indices(ip)
            @test_throws ArgumentError Ferrite.facedof_indices(ip)
            @test_throws ArgumentError Ferrite.facetdof_indices(ip)

            @test_throws ArgumentError Ferrite.edgedof_interior_indices(ip)
            @test_throws ArgumentError Ferrite.facedof_interior_indices(ip)
            @test_throws ArgumentError Ferrite.volumedof_interior_indices(ip)
            @test_throws ArgumentError Ferrite.facetdof_interior_indices(ip)
        end
    end

    @testset "TensorizedInterpolation constructor" begin
        ip = Lagrange{RefTriangle, 1}()
        # Unsupported value types
        @test_throws MethodError TensorizedInterpolation{Vec{2}}(ip)
        @test_throws MethodError TensorizedInterpolation{Tensor{4, 2}}(ip)
        @test_throws MethodError TensorizedInterpolation{SymmetricTensor{4, 2}}(ip)
        # Tensor dimension must be specified
        @test_throws MethodError TensorizedInterpolation{Tensor{2}}(ip)
        @test_throws MethodError TensorizedInterpolation{SymmetricTensor{2}}(ip)
        # Unsupported tensor dimension
        @test_throws ArgumentError TensorizedInterpolation{Tensor{2, 4}}(ip)
        # The full-parameter constructor cannot bypass normalization/validation
        @test_throws MethodError TensorizedInterpolation{SymmetricTensor{2, 2, Float32}, RefTriangle, 1, typeof(ip)}(ip)
        @test_throws MethodError TensorizedInterpolation{Vec{2}, RefTriangle, 1, typeof(ip)}(ip)
    end
end # testset
