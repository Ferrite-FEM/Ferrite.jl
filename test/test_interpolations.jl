@testset "interpolations" begin

@testset "Value Type $value_type" for value_type in ( Float32, )
    @testset "$interpolation" for interpolation in (
                        Lagrange{RefLine, 1, value_type}(),
                        Lagrange{RefLine, 2, value_type}(),
                        Lagrange{RefQuadrilateral, 1, value_type}(),
                        Lagrange{RefQuadrilateral, 2, value_type}(),
                        Lagrange{RefQuadrilateral, 3, value_type}(),
                        Lagrange{RefTriangle, 1, value_type}(),
                        Lagrange{RefTriangle, 2, value_type}(),
                        Lagrange{RefTriangle, 3, value_type}(),
                        Lagrange{RefTriangle, 4, value_type}(),
                        Lagrange{RefTriangle, 5, value_type}(),
                        Lagrange{RefHexahedron, 1, value_type}(),
                        Lagrange{RefHexahedron, 2, value_type}(),
                        Serendipity{RefQuadrilateral, 2, value_type}(),
                        Serendipity{RefHexahedron, 2, value_type}(),
                        Lagrange{RefTetrahedron, 1, value_type}(),
                        Lagrange{RefTetrahedron, 2, value_type}(),
                        Lagrange{RefPrism, 1, value_type}(),
                        Lagrange{RefPrism, 2, value_type}(),
                        Lagrange{RefPyramid, 1, value_type}(),
                        Lagrange{RefPyramid, 2, value_type}(),
                        #
                        DiscontinuousLagrange{RefLine, 0, value_type}(),
                        DiscontinuousLagrange{RefQuadrilateral, 0, value_type}(),
                        DiscontinuousLagrange{RefHexahedron, 0, value_type}(),
                        DiscontinuousLagrange{RefTriangle, 0, value_type}(),
                        DiscontinuousLagrange{RefTetrahedron, 0, value_type}(),
                        DiscontinuousLagrange{RefLine, 1, value_type}(),
                        DiscontinuousLagrange{RefQuadrilateral, 1, value_type}(),
                        DiscontinuousLagrange{RefHexahedron, 1, value_type}(),
                        DiscontinuousLagrange{RefTriangle, 1, value_type}(),
                        DiscontinuousLagrange{RefTetrahedron, 1, value_type}(),
                        #
                        BubbleEnrichedLagrange{RefTriangle, 1, value_type}(),
                        #
                        CrouzeixRaviart{RefTriangle, 1, value_type}(),
        )

        # Test of utility functions
        ref_dim = Ferrite.getdim(interpolation)
        ref_shape = Ferrite.getrefshape(interpolation)
        func_order = Ferrite.getorder(interpolation)
        @test typeof(interpolation) <: Interpolation{ref_shape,func_order}

        # Note that not every element formulation exists for every order and dimension.
        applicable(Ferrite.getlowerorder, interpolation) && @test typeof(Ferrite.getlowerorder(interpolation)) <: Interpolation{ref_shape,func_order-1}

        # Check partition of unity at random point.
        n_basefuncs = getnbasefunctions(interpolation)
        x = rand(Tensor{1, ref_dim, value_type, ref_dim})
        f = (x) -> [shape_value(interpolation, Tensor{1, ref_dim}(x), i) for i in 1:n_basefuncs]
        @test vec(ForwardDiff.jacobian(f, Array(x))') ≈
            reinterpret(value_type, [shape_gradient(interpolation, x, i) for i in 1:n_basefuncs])
        @test sum([shape_value(interpolation, x, i) for i in 1:n_basefuncs]) ≈ value_type(1.0)

        # Check if the important functions are consistent
        coords = Ferrite.reference_coordinates(interpolation)
        @test length(coords) == n_basefuncs
        @test shape_value(interpolation, x, n_basefuncs) == shape_value(interpolation, x, n_basefuncs)
        @test_throws BoundsError shape_value(interpolation, x, n_basefuncs+1)

        # Test whether we have for each entity corresponding dof indices (possibly empty)
        @test length(Ferrite.vertexdof_indices(interpolation)) == Ferrite.nvertices(interpolation)
        if ref_dim > 1
            @test length(Ferrite.facedof_indices(interpolation)) == Ferrite.nfaces(interpolation)
            @test length(Ferrite.facedof_interior_indices(interpolation)) == Ferrite.nfaces(interpolation)
        elseif ref_dim > 2
            @test length(Ferrite.edgedof_indices(interpolation)) == Ferrite.nedges(interpolation)
            @test length(Ferrite.edgedof_interior_indices(interpolation)) == Ferrite.nedges(interpolation)
        end
        # We have at least as many edge/face dofs as we have edge/face interior dofs
        if ref_dim > 1
            @test all(length.(Ferrite.facedof_interior_indices(interpolation)) .<= length.(Ferrite.facedof_indices(interpolation)))
        elseif ref_dim > 2
            @test all(length.(Ferrite.edgedof_interior_indices(interpolation)) .<= length.(Ferrite.edgedof_indices(interpolation)))
        end
        # The total number of dofs must match the number of base functions
        totaldofs = sum(length.(Ferrite.vertexdof_indices(interpolation));init=0)
        if ref_dim > 1
            totaldofs += sum(length.(Ferrite.facedof_interior_indices(interpolation));init=0)
        end
        if ref_dim > 2
            totaldofs += sum(length.(Ferrite.edgedof_interior_indices(interpolation));init=0) 
        end
        totaldofs += length(Ferrite.celldof_interior_indices(interpolation))
        @test totaldofs == n_basefuncs

        # The dof indices are valid.
        @test all([all(0 .< i .<= n_basefuncs) for i ∈ Ferrite.vertexdof_indices(interpolation)])
        if ref_dim > 1
            @test all([all(0 .< i .<= n_basefuncs) for i ∈ Ferrite.facedof_indices(interpolation)])
            @test all([all(0 .< i .<= n_basefuncs) for i ∈ Ferrite.facedof_interior_indices(interpolation)])
        elseif ref_dim > 2
            @test all([all(0 .< i .<= n_basefuncs) for i ∈ Ferrite.edgedof_indices(interpolation)])
            @test all([all(0 .< i .<= n_basefuncs) for i ∈ Ferrite.edgedof_interior_indices(interpolation)])
        end
        @test all([all(0 .< i .<= n_basefuncs) for i ∈ Ferrite.celldof_interior_indices(interpolation)])

        # Check for evaluation type stability of interpolation
        @testset "type stability dof $dof" for dof in 1:n_basefuncs
            @test eltype(shape_value(interpolation, x, dof)) == value_type
            @test eltype(shape_gradient(interpolation, x, dof)) == value_type
        end

        # Check for dirac delta property of interpolation
        @testset "dirac delta property of dof $dof" for dof in 1:n_basefuncs
            for k in 1:n_basefuncs
                N_dof = shape_value(interpolation, coords[dof], k)
                if k == dof
                    @test N_dof ≈ 1.0
                else
                    factor = interpolation isa Lagrange{RefQuadrilateral, 3} ? 250 : 5
                    @test N_dof ≈ 0.0 atol = factor * eps(value_type)
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
            for (facenodes, normal) in zip(Ferrite.facedof_indices(interpolation), reference_normals(interpolation))
                @test __outward_normal(coords, facenodes) ≈ normal
            end
        end

        # regression for https://github.com/Ferrite-FEM/Ferrite.jl/issues/520
        interpolation_type = typeof(interpolation).name.wrapper
        if func_order > 1 && interpolation_type != Ferrite.Serendipity
            first_order = interpolation_type{ref_shape,1}()
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

        # VectorizedInterpolation
        v_interpolation_1 = interpolation^2
        v_interpolation_2 = (d = 2; interpolation^d)
        @test getnbasefunctions(v_interpolation_1) == getnbasefunctions(v_interpolation_2) ==
            getnbasefunctions(interpolation) * 2
        # pretty printing
        @test repr("text/plain", v_interpolation_1) == repr(v_interpolation_1.ip) * "^2"
        
        # Check for evaluation type stability of vectorized interpolation
        v_interpolation_3 = interpolation^ref_dim
        @testset "vectorized case stability of dof $dof" for dof in 1:n_basefuncs
            @test eltype(shape_value(v_interpolation_1, x, dof)) == value_type
            @test eltype(shape_gradient(v_interpolation_3, x, dof)) == value_type # Gradient currently only works for 
        end
    end

    @test Ferrite.reference_coordinates(DiscontinuousLagrange{RefTriangle,0,value_type}()) ≈ [Vec{2,value_type}((1/3,1/3))]
    @test Ferrite.reference_coordinates(DiscontinuousLagrange{RefQuadrilateral,0,value_type}()) ≈ [Vec{2,value_type}((0,0))]
    @test Ferrite.reference_coordinates(DiscontinuousLagrange{RefTetrahedron,0,value_type}()) ≈ [Vec{3,value_type}((1/4,1/4,1/4))]
    @test Ferrite.reference_coordinates(DiscontinuousLagrange{RefHexahedron,0,value_type}()) ≈ [Vec{3,value_type}((0,0,0))]

    # Test discontinuous interpolations related functions
    d_ip = DiscontinuousLagrange{RefQuadrilateral,1,value_type}()
    d_ip_t = DiscontinuousLagrange{RefQuadrilateral,1,value_type}

    ip = Lagrange{RefQuadrilateral,1,value_type}()
    ip_t = Lagrange{RefQuadrilateral,1,value_type}

    @test Ferrite.is_discontinuous(ip) == false
    @test Ferrite.is_discontinuous(ip_t) == false
    @test Ferrite.is_discontinuous(d_ip) == true
    @test Ferrite.is_discontinuous(d_ip_t) == true
end

end # testset
