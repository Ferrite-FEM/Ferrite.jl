#----------------------------------------------------------------------#
# Assembly functionality benchmarks
#----------------------------------------------------------------------#
SUITE["assembly"] = BenchmarkGroup()

# Permute over common combinations for some commonly required local matrices.
SUITE["assembly"]["common-local"] = BenchmarkGroup()
COMMON_LOCAL_ASSEMBLY = SUITE["assembly"]["common-local"]
for spatial_dim in 1:3
    ξ_dummy = Vec{spatial_dim}(ntuple(x -> 0.0, spatial_dim))
    COMMON_LOCAL_ASSEMBLY["spatial-dim", spatial_dim] = BenchmarkGroup()
    for geo_type in FerriteBenchmarkHelper.geo_types_for_spatial_dim(spatial_dim)
        # The quadratic *geometry* variants only change the Jacobian mapping, which measured a
        # median 1.06x against the straight-sided cell across every kernel below. The function
        # space is already swept through `order`, so they are skipped here (`mesh` still
        # benchmarks generating them).
        geo_type in (QuadraticLine, QuadraticTriangle, QuadraticQuadrilateral) && continue
        COMMON_LOCAL_ASSEMBLY["spatial-dim", spatial_dim][string(geo_type)] = BenchmarkGroup()

        grid = generate_grid(geo_type, tuple(repeat([2], spatial_dim)...))
        topology = ExclusiveTopology(grid)
        ip_geo = Ferrite.geometric_interpolation(geo_type)
        ref_type = FerriteBenchmarkHelper.getrefshape(geo_type)

        # Nodal interpolation tests
        for order in 1:2, ip_type in [Lagrange, Serendipity]
            ip_type == Serendipity && (order < 2 || ref_type ∉ (RefQuadrilateral, RefHexahedron)) && continue
            ip = ip_type{ref_type, order}()
            ip_vectorized = ip^spatial_dim

            # Skip over elements which are not implemented
            !applicable(Ferrite.reference_shape_value, ip, ξ_dummy, 1) && continue

            qr = QuadratureRule{ref_type}(2 * order - 1)

            # Currently we just benchmark nodal Lagrange bases.
            COMMON_LOCAL_ASSEMBLY["spatial-dim", spatial_dim][string(geo_type)][string(ip_type), string(order)] = BenchmarkGroup()
            LAGRANGE_SUITE = COMMON_LOCAL_ASSEMBLY["spatial-dim", spatial_dim][string(geo_type)][string(ip_type), string(order)]
            LAGRANGE_SUITE["fe-values"] = BenchmarkGroup()
            LAGRANGE_SUITE["ritz-galerkin"] = BenchmarkGroup()
            LAGRANGE_SUITE["petrov-galerkin"] = BenchmarkGroup()

            # Note: at the time of writing this PR the ctor makes the heavy lifting and caches important values.
            LAGRANGE_SUITE["fe-values"]["scalar"] = @benchmarkable CellValues($qr, $ip, $ip_geo)
            LAGRANGE_SUITE["fe-values"]["vector"] = @benchmarkable CellValues($qr, $ip_vectorized, $ip_geo)

            csv = CellValues(qr, ip, ip_geo)
            cvv = CellValues(qr, ip_vectorized, ip_geo)

            # Only the Petrov-Galerkin benchmarks that actually pair two *different* function
            # spaces are kept. The rest passed the same values object twice, which measured a
            # median 1.20x against the Ritz-Galerkin form of the same kernel and shares its
            # inner loop, so a regression in one shows up in the other.

            # Scalar shape φ and test ψ: ∫ φ ψ
            LAGRANGE_SUITE["ritz-galerkin"]["mass"] = @benchmarkable FerriteAssemblyHelper._generalized_ritz_galerkin_assemble_local_matrix($grid, $csv, shape_value, shape_value, *)
            # Vectorial shape φ and test ψ: ∫ φ ⋅ ψ
            LAGRANGE_SUITE["ritz-galerkin"]["vector-mass"] = @benchmarkable FerriteAssemblyHelper._generalized_ritz_galerkin_assemble_local_matrix($grid, $cvv, shape_value, shape_value, ⋅)
            # Scalar shape φ and test ψ: ∫ ∇φ ⋅ ∇ψ
            LAGRANGE_SUITE["ritz-galerkin"]["Laplace"] = @benchmarkable FerriteAssemblyHelper._generalized_ritz_galerkin_assemble_local_matrix($grid, $csv, shape_gradient, shape_gradient, ⋅)
            # Vectorial shape φ and test ψ: ∫ ∇φ : ∇ψ
            LAGRANGE_SUITE["ritz-galerkin"]["vector-Laplace"] = @benchmarkable FerriteAssemblyHelper._generalized_ritz_galerkin_assemble_local_matrix($grid, $cvv, shape_gradient, shape_gradient, ⊡)
            # Vectorial shape φ and scalar test ψ: ∫ (∇ ⋅ φ) ψ
            LAGRANGE_SUITE["petrov-galerkin"]["pressure-velocity"] = @benchmarkable FerriteAssemblyHelper._generalized_petrov_galerkin_assemble_local_matrix($grid, $cvv, shape_divergence, $csv, shape_value, *)

            if spatial_dim > 1
                qr_facet = FacetQuadratureRule{ref_type}(2 * order - 1)
                fsv = FacetValues(qr_facet, ip, ip_geo)

                LAGRANGE_SUITE["ritz-galerkin"]["face-flux"] = @benchmarkable FerriteAssemblyHelper._generalized_ritz_galerkin_assemble_local_matrix($grid, $fsv, shape_gradient, shape_value, *)

                ip = DiscontinuousLagrange{ref_type, order}()
                isv = InterfaceValues(qr_facet, ip, ip_geo)
                dh = DofHandler(grid)
                add!(dh, :u, ip)
                close!(dh)

                LAGRANGE_SUITE["ritz-galerkin"]["interface-{grad}⋅[[val]]"] = @benchmarkable FerriteAssemblyHelper._generalized_ritz_galerkin_assemble_interfaces($dh, $isv, shape_gradient_average, shape_value_jump, ⋅)

                LAGRANGE_SUITE["ritz-galerkin"]["interface-interior-penalty"] = @benchmarkable FerriteAssemblyHelper._generalized_ritz_galerkin_assemble_interfaces($dh, $isv, shape_value_jump, shape_value_jump, ⋅)

            end
        end
    end
end
