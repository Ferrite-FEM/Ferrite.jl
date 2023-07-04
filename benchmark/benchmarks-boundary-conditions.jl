#----------------------------------------------------------------------#
# Boundary condition benchmarks
#----------------------------------------------------------------------#
SUITE["boundary-conditions"] = BenchmarkGroup()

SUITE["boundary-conditions"]["Dirichlet"] = BenchmarkGroup()
DIRICHLET_SUITE = SUITE["boundary-conditions"]["Dirichlet"]
# span artificial scope...
for spatial_dim ∈ [2]
    # Benchmark application on global system
    DIRICHLET_SUITE["global"] = BenchmarkGroup()

    geo_type = Quadrilateral
    grid = generate_grid(geo_type, ntuple(x->2, spatial_dim));
    ref_type = geo_type.super.parameters[1]
    ip_geo = Ferrite.default_interpolation(geo_type)
    order = 2

    # assemble a mass matrix to apply BCs on (because its cheap)
    ip = Lagrange{ref_type, order}()
    qr = QuadratureRule{ref_type}(2*order-1)
    cellvalues = CellValues(qr, ip, ip_geo);
    dh = DofHandler(grid)
    push!(dh, :u, 1, ip)
    close!(dh);

    ch = ConstraintHandler(dh);
    ∂Ω = union(getfaceset.((grid, ), ["left"])...);
    dbc = Dirichlet(:u, ∂Ω, (x, t) -> 0)
    add!(ch, dbc);
    close!(ch);

    # Non-symmetric application
    M, f = FerriteAssemblyHelper._assemble_mass(dh, cellvalues, false);
    DIRICHLET_SUITE["global"]["apply!(M,f,APPLY_TRANSPOSE)"] = @benchmarkable apply!($M, $f, $ch; strategy=$(Ferrite.APPLY_TRANSPOSE));
    DIRICHLET_SUITE["global"]["apply!(M,f,APPLY_INPLACE)"] = @benchmarkable apply!($M, $f, $ch; strategy=$(Ferrite.APPLY_INPLACE));
    # Symmetric application
    M, f = FerriteAssemblyHelper._assemble_mass(dh, cellvalues, true);
    DIRICHLET_SUITE["global"]["apply!(M_sym,f,APPLY_TRANSPOSE)"] = @benchmarkable apply!($M, $f, $ch; strategy=$(Ferrite.APPLY_TRANSPOSE));
    DIRICHLET_SUITE["global"]["apply!(M_sym,f,APPLY_INPLACE)"] = @benchmarkable apply!($M, $f, $ch; strategy=$(Ferrite.APPLY_INPLACE));

    DIRICHLET_SUITE["global"]["apply!(f)"] = @benchmarkable apply!($f, $ch);
    DIRICHLET_SUITE["global"]["apply_zero!(f)"] = @benchmarkable apply!($f, $ch);
end
