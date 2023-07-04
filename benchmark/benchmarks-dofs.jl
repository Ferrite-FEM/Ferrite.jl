
#----------------------------------------------------------------------#
# Benchmarks around the dof management
#----------------------------------------------------------------------#
SUITE["dof-management"] = BenchmarkGroup()
SUITE["dof-management"]["numbering"] = BenchmarkGroup()
# !!! NOTE close! must wrapped into a custom function, because consecutive calls to close!, since the dofs are already distributed.
NUMBERING_SUITE = SUITE["dof-management"]["numbering"]
for spatial_dim ∈ [3]# 1:3
    NUMBERING_SUITE["spatial-dim",spatial_dim] = BenchmarkGroup()
    for geo_type ∈ FerriteBenchmarkHelper.geo_types_for_spatial_dim(spatial_dim)
        NUMBERING_SUITE["spatial-dim",spatial_dim][string(geo_type)] = BenchmarkGroup()

        ref_type = FerriteBenchmarkHelper.getrefshape(geo_type)

        for grid_size ∈ [2]#[3, 6, 9] #multiple grid sized to estimate computational complexity...
            NUMBERING_SUITE["spatial-dim",spatial_dim][string(geo_type)]["grid-size-",grid_size] = BenchmarkGroup()
            NUMBERING_SUITE["spatial-dim",spatial_dim][string(geo_type)]["grid-size-",grid_size] = BenchmarkGroup()

            grid = generate_grid(geo_type, ntuple(x->grid_size, spatial_dim));

            for field_dim ∈ [3]#1:3
                NUMBERING_SUITE["spatial-dim",spatial_dim][string(geo_type)]["grid-size-",grid_size]["field-dim-", field_dim] = BenchmarkGroup()
                NUMBERING_FIELD_DIM_SUITE = NUMBERING_SUITE["spatial-dim",spatial_dim][string(geo_type)]["grid-size-",grid_size]["field-dim-", field_dim]
                # Lagrange tests
                for order ∈ 1:2
                    ip = Lagrange{ref_type, order}()

                    # Skip over elements which are not implemented
                    ξ_dummy = Vec{spatial_dim}(ntuple(x->0.0, spatial_dim))
                    !applicable(Ferrite.shape_value, ip, ξ_dummy, 1) && continue

                    NUMBERING_FIELD_DIM_SUITE["Lagrange",order] = BenchmarkGroup()
                    LAGRANGE_SUITE = NUMBERING_FIELD_DIM_SUITE["Lagrange",order]
                    order2 = max(order-1, 1)
                    ip2 = Lagrange{ref_type, order2}()

                    LAGRANGE_SUITE["DofHandler"] = BenchmarkGroup()

                    close_helper = function(grid, ip)
                        dh = DofHandler(grid)
                        push!(dh, :u, field_dim, ip)
                        close!(dh)
                    end
                    LAGRANGE_SUITE["DofHandler"]["one-field"] = @benchmarkable $close_helper($grid, $ip)

                    close_helper = function(grid, ip, ip2)
                        dh = DofHandler(grid)
                        push!(dh, :u, field_dim, ip)
                        push!(dh, :p, 1, ip2)
                        close!(dh)
                    end
                    LAGRANGE_SUITE["DofHandler"]["two-fields"] = @benchmarkable $close_helper($grid, $ip, $ip2)

                    close_helper = function(grid)
                        dh = DofHandler(grid)
                        sdh = SubDofHandler(dh, Set(1:Int(round(getncells(grid)/2))))
                        add!(sdh, :u, ip^field_dim)
                        close!(dh)
                    end
                    LAGRANGE_SUITE["DofHandler"]["one-field-subdomain"] = @benchmarkable $close_helper($grid)

                    close_helper = function(grid)
                        dh = DofHandler(grid)
                        sdh = SubDofHandler(dh, Set(1:Int(round(getncells(grid)/2))))
                        add!(sdh, :u, ip^field_dim)
                        add!(sdh, :p, ip2)
                        close!(dh)
                    end
                    LAGRANGE_SUITE["DofHandler"]["two-fields-subdomain"] = @benchmarkable $close_helper($grid)
                end
            end
        end
    end
end
