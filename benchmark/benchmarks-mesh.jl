
#----------------------------------------------------------------------#
# Benchmarks for mesh functionality within Ferrite
#----------------------------------------------------------------------#
SUITE["mesh"] = BenchmarkGroup()

# Generator benchmarks
SUITE["mesh"]["generator"] = BenchmarkGroup()

# Structured hyperrectangle generators
SUITE["mesh"]["generator"]["hyperrectangle"] = BenchmarkGroup()
HYPERRECTANGLE_GENERATOR = SUITE["mesh"]["generator"]["hyperrectangle"]
for spatial_dim ∈ 1:3
    HYPERRECTANGLE_GENERATOR["spatial-dim",spatial_dim] = BenchmarkGroup()
    for geo_type ∈ FerriteBenchmarkHelper.geo_types_for_spatial_dim(spatial_dim)
        HYPERRECTANGLE_GENERATOR["spatial-dim",spatial_dim][string(geo_type)] = @benchmarkable generate_grid($geo_type, $(ntuple(x->4, spatial_dim)));
    end
end

# TODO AMR performance
# TODO topology performance
