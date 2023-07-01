#----------------------------------------------------------------------#
# Assembly functionality benchmarks
#----------------------------------------------------------------------#
SUITE["assembly"] = BenchmarkGroup()

# Permute over common combinations for some commonly required local matrices.
SUITE["assembly"]["common-local"] = BenchmarkGroup()
COMMON_LOCAL_ASSEMBLY = SUITE["assembly"]["common-local"]
for spatial_dim ∈ 1:3
    ξ_dummy = Vec{spatial_dim}(ntuple(x->0.0, spatial_dim))
    COMMON_LOCAL_ASSEMBLY["spatial-dim",spatial_dim] = BenchmarkGroup()
    for geo_type ∈ FerriteBenchmarkHelper.geo_types_for_spatial_dim(spatial_dim)
        COMMON_LOCAL_ASSEMBLY["spatial-dim",spatial_dim][string(geo_type)] = BenchmarkGroup()

        grid = generate_grid(geo_type, tuple(repeat([1], spatial_dim)...));
        topology = ExclusiveTopology(grid)
        ref_type = FerriteBenchmarkHelper.default_refshape(geo_type)
        ip_geo = Ferrite.default_interpolation(geo_type)

        # Nodal interpolation tests
        for order ∈ 1:2, ip_type ∈ [Lagrange, Serendipity]
            ip = ip_type{spatial_dim, ref_type, order}()

            # Skip over elements which are not implemented
            !applicable(Ferrite.value, ip, 1, ξ_dummy) && continue

            qr = QuadratureRule{spatial_dim, ref_type}(2*order-1)

            # Currently we just benchmark nodal Lagrange bases.
            COMMON_LOCAL_ASSEMBLY["spatial-dim",spatial_dim][string(geo_type)][string(ip_type),string(order)] = BenchmarkGroup()
            LAGRANGE_SUITE = COMMON_LOCAL_ASSEMBLY["spatial-dim",spatial_dim][string(geo_type)][string(ip_type),string(order)]
            LAGRANGE_SUITE["fe-values"] = BenchmarkGroup()
            LAGRANGE_SUITE["ritz-galerkin"] = BenchmarkGroup()
            LAGRANGE_SUITE["petrov-galerkin"] = BenchmarkGroup()

            # Note: at the time of writing this PR the ctor makes the heavy lifting and caches important values.
            LAGRANGE_SUITE["fe-values"]["scalar"] = @benchmarkable CellScalarValues($qr, $ip, $ip_geo);
            LAGRANGE_SUITE["fe-values"]["vector"] = @benchmarkable CellVectorValues($qr, $ip, $ip_geo);

            csv = CellScalarValues(qr, ip, ip_geo);
            csv2 = CellScalarValues(qr, ip, ip_geo);

            cvv = CellVectorValues(qr, ip, ip_geo);
            cvv2 = CellVectorValues(qr, ip, ip_geo);

            # Scalar shape φ and test ψ: ∫ φ ψ
            LAGRANGE_SUITE["ritz-galerkin"]["mass"] = @benchmarkable FerriteAssemblyHelper._generalized_ritz_galerkin_assemble_local_matrix($grid, $csv, shape_value, shape_value, *)
            LAGRANGE_SUITE["petrov-galerkin"]["mass"] = @benchmarkable FerriteAssemblyHelper._generalized_petrov_galerkin_assemble_local_matrix($grid, $csv, shape_value, $csv2, shape_value, *)
            # Vectorial shape φ and test ψ: ∫ φ ⋅ ψ
            LAGRANGE_SUITE["ritz-galerkin"]["vector-mass"] = @benchmarkable FerriteAssemblyHelper._generalized_ritz_galerkin_assemble_local_matrix($grid, $cvv, shape_value, shape_value, ⋅)
            LAGRANGE_SUITE["petrov-galerkin"]["vector-mass"] = @benchmarkable FerriteAssemblyHelper._generalized_petrov_galerkin_assemble_local_matrix($grid, $cvv, shape_value, $cvv2, shape_value, ⋅)
            # Scalar shape φ and test ψ: ∫ ∇φ ⋅ ∇ψ
            LAGRANGE_SUITE["ritz-galerkin"]["Laplace"] = @benchmarkable FerriteAssemblyHelper._generalized_ritz_galerkin_assemble_local_matrix($grid, $csv, shape_gradient, shape_gradient, ⋅)
            LAGRANGE_SUITE["petrov-galerkin"]["Laplace"] = @benchmarkable FerriteAssemblyHelper._generalized_petrov_galerkin_assemble_local_matrix($grid, $csv, shape_gradient, $csv2, shape_gradient, ⋅)
            # Vectorial shape φ and test ψ: ∫ ∇φ : ∇ψ
            LAGRANGE_SUITE["ritz-galerkin"]["vector-Laplace"] = @benchmarkable FerriteAssemblyHelper._generalized_ritz_galerkin_assemble_local_matrix($grid, $cvv, shape_gradient, shape_gradient, ⊡)
            LAGRANGE_SUITE["petrov-galerkin"]["vector-Laplace"] = @benchmarkable FerriteAssemblyHelper._generalized_petrov_galerkin_assemble_local_matrix($grid, $cvv, shape_gradient, $cvv2, shape_gradient, ⊡)
            # Vectorial shape φ and scalar test ψ: ∫ (∇ ⋅ φ) ψ
            LAGRANGE_SUITE["petrov-galerkin"]["pressure-velocity"] = @benchmarkable FerriteAssemblyHelper._generalized_petrov_galerkin_assemble_local_matrix($grid, $cvv, shape_divergence, $csv, shape_value, *)

            if spatial_dim > 1
                qr_face = QuadratureRule{spatial_dim-1, ref_type}(2*order-1)
                fsv = FaceScalarValues(qr_face, ip, ip_geo);
                fsv2 = FaceScalarValues(qr_face, ip, ip_geo);

                LAGRANGE_SUITE["ritz-galerkin"]["face-flux"] = @benchmarkable FerriteAssemblyHelper._generalized_ritz_galerkin_assemble_local_matrix($grid, $fsv, shape_gradient, shape_value, *)
                LAGRANGE_SUITE["petrov-galerkin"]["face-flux"] = @benchmarkable FerriteAssemblyHelper._generalized_petrov_galerkin_assemble_local_matrix($grid, $fsv, shape_gradient, $fsv2, shape_value, *)
                ip = DiscontinuousLagrange{ref_type, order}()
                isv = InterfaceValues(grid, qr_face, ip, ip_geo; geom_interpol_b = ip_geo);
                isv2 = InterfaceValues(grid, qr_face, ip, ip_geo; geom_interpol_b = ip_geo);
                LAGRANGE_SUITE["ritz-galerkin"]["interface-{grad}⋅[[val]]"] = @benchmarkable FerriteAssemblyHelper._generalized_ritz_galerkin_assemble_local_matrix($grid, $topology, $isv, shape_gradient_average, shape_value_jump, ⋅)
                LAGRANGE_SUITE["petrov-galerkin"]["interface-{grad}⋅[[val]]"] = @benchmarkable FerriteAssemblyHelper._generalized_petrov_galerkin_assemble_local_matrix($grid, $topology, $isv, shape_gradient_average, $isv2, shape_value_jump, ⋅)
    
                LAGRANGE_SUITE["ritz-galerkin"]["interface-interior-penalty"] = @benchmarkable FerriteAssemblyHelper._generalized_ritz_galerkin_assemble_local_matrix($grid, $topology, $isv, shape_value_jump, shape_value_jump, ⋅)
                LAGRANGE_SUITE["petrov-galerkin"]["interface-interior-penalty"] = @benchmarkable FerriteAssemblyHelper._generalized_petrov_galerkin_assemble_local_matrix($grid, $topology, $isv, shape_value_jump, $isv2, shape_value_jump, ⋅)
    
            end
        end
    end
end
