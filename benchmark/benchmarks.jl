using BenchmarkTools
using Ferrite

module FerriteBenchmarkHelper

using Ferrite

function geo_types_for_spatial_dim(spatial_dim)
    spatial_dim == 1 && return [Line, QuadraticLine]
    spatial_dim == 2 && return [Triangle, QuadraticTriangle, Quadrilateral, QuadraticQuadrilateral]
    spatial_dim == 3 && return [Tetrahedron, Hexahedron] # Quadratic* not yet functional in 3D. 3D triangle missing. Embedded also missing.
end

default_refshape(t::Type{C}) where {C <: Ferrite.AbstractCell} = typeof(Ferrite.default_interpolation(t)).parameters[2]

end

const SUITE = BenchmarkGroup()

#----------------------------------------------------------------------#
# Benchmarks for mesh functionality within Ferrite
#----------------------------------------------------------------------#
SUITE["mesh"] = BenchmarkGroup()

# Generator benchmarks
SUITE["mesh"]["generator"] = BenchmarkGroup()

# Strucutred hyperrectangle generators
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

#----------------------------------------------------------------------#
# Benchmarks around the dof management
#----------------------------------------------------------------------#
SUITE["dof-management"] = BenchmarkGroup()
SUITE["dof-management"]["numbering"] = BenchmarkGroup()
# !!! NOTE close! must wrapped into a custom function, because consecutive calls to close!, since the dofs are already distributed.
NUMBERING_SUITE = SUITE["dof-management"]["numbering"]
for spatial_dim ∈ 1:3
    NUMBERING_SUITE["spatial-dim",spatial_dim] = BenchmarkGroup()
    for geo_type ∈ FerriteBenchmarkHelper.geo_types_for_spatial_dim(spatial_dim)
        NUMBERING_SUITE["spatial-dim",spatial_dim][string(geo_type)] = BenchmarkGroup()

        ref_type = FerriteBenchmarkHelper.default_refshape(geo_type)

        for grid_size ∈ [3, 6, 9] #multiple grid sized to estimate computational complexity...
            NUMBERING_SUITE["spatial-dim",spatial_dim][string(geo_type)]["grid-size-",grid_size] = BenchmarkGroup()
            NUMBERING_SUITE["spatial-dim",spatial_dim][string(geo_type)]["grid-size-",grid_size] = BenchmarkGroup()

            grid = generate_grid(geo_type, ntuple(x->grid_size, spatial_dim));

            for field_dim ∈ 1:3
                NUMBERING_SUITE["spatial-dim",spatial_dim][string(geo_type)]["grid-size-",grid_size]["field-dim-", field_dim] = BenchmarkGroup()
                NUMBERING_FIELD_DIM_SUITE = NUMBERING_SUITE["spatial-dim",spatial_dim][string(geo_type)]["grid-size-",grid_size]["field-dim-", field_dim]
                # Lagrange tests
                for order ∈ 1:2
                    # higher order Lagrange on hex not working yet...
                    order > 1 && geo_type ∈ [Hexahedron] && continue

                    NUMBERING_FIELD_DIM_SUITE["Lagrange",order] = BenchmarkGroup()
                    LAGRANGE_SUITE = NUMBERING_FIELD_DIM_SUITE["Lagrange",order]
                    ip = Lagrange{spatial_dim, ref_type, order}()
                    order2 = max(order-1, 1)
                    ip2 = Lagrange{spatial_dim, ref_type, order2}()

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


                    LAGRANGE_SUITE["MixedDofHandler"] = BenchmarkGroup()
                    f1 = Field(:u, ip, field_dim)
                    f2 = Field(:p, ip2, 1)

                    close_helper = function(grid, f1)
                        dh = MixedDofHandler(grid)
                        push!(dh, FieldHandler([f1], Set(1:getncells(grid))))
                        close!(dh)
                    end
                    LAGRANGE_SUITE["MixedDofHandler"]["one-field"] = @benchmarkable $close_helper($grid, $f1)

                    close_helper = function(grid, f1)
                        dh = MixedDofHandler(grid)
                        push!(dh, FieldHandler([f1], Set(1:Int(round(getncells(grid)/2)))))
                        close!(dh)
                    end
                    LAGRANGE_SUITE["MixedDofHandler"]["one-field-subdomain"] = @benchmarkable $close_helper($grid, $f1)

                    close_helper = function(grid, f1, f2)
                        dh = MixedDofHandler(grid)
                        push!(dh, FieldHandler([f1, f2], Set(1:getncells(grid))))
                        close!(dh)
                    end
                    LAGRANGE_SUITE["MixedDofHandler"]["two-fields"] = @benchmarkable $close_helper($grid, $f1, $f2)

                    close_helper = function(grid, f1, f2)
                        dh = MixedDofHandler(grid)
                        push!(dh, FieldHandler([f1, f2], Set(1:Int(round(getncells(grid)/2)))))
                        close!(dh)
                    end
                    LAGRANGE_SUITE["MixedDofHandler"]["two-fields-subdomain"] = @benchmarkable $close_helper($grid, $f1, $f2)
                end
            end
        end
    end
end

#----------------------------------------------------------------------#
# Assembly functionality benchmarks
#----------------------------------------------------------------------#
SUITE["assembly"] = BenchmarkGroup()

module FerriteAssemblyHelper

using Ferrite

# Minimal Ritz-Galerkin type local assembly loop.
function _generalized_ritz_galerkin_assemble_local_matrix(grid::Ferrite.AbstractGrid, cellvalues::CellValues{dim,T,refshape}, f_shape, f_test, op) where {dim,T,refshape}
    n_basefuncs = getnbasefunctions(cellvalues)

    Ke = zeros(n_basefuncs, n_basefuncs)

    X = getcoordinates(grid, 1)
    reinit!(cellvalues, X)

    for q_point in 1:getnquadpoints(cellvalues)
        dΩ = getdetJdV(cellvalues, q_point)
        for i in 1:n_basefuncs
            test = f_test(cellvalues, q_point, i)
            for j in 1:n_basefuncs
                shape = f_shape(cellvalues, q_point, j)
                Ke[i, j] += op(test, shape) * dΩ
            end
        end
    end

    Ke
end

function _generalized_ritz_galerkin_assemble_local_matrix(grid::Ferrite.AbstractGrid, facevalues::FaceValues{dim,T,refshape}, f_shape, f_test, op) where {dim,T,refshape}
    n_basefuncs = getnbasefunctions(facevalues)

    f = zeros(n_basefuncs)

    X = getcoordinates(grid, 1)
    for face in 1:nfaces(getcells(grid)[1])
        reinit!(facevalues, X, face)

        for q_point in 1:getnquadpoints(facevalues)
            n = getnormal(facevalues, q_point)
            dΓ = getdetJdV(facevalues, q_point)
            for i in 1:n_basefuncs
                test = f_test(facevalues, q_point, i)
                for j in 1:n_basefuncs
                    shape = f_shape(facevalues, q_point, j)
                    f[i] += op(test, shape) ⋅ n * dΓ
                end
            end
        end
    end

    f
end

# Minimal Petrov-Galerkin type local assembly loop. We assume that both function spaces share the same integration rule. Test is applied from the left.
function _generalized_petrov_galerkin_assemble_local_matrix(grid::Ferrite.AbstractGrid, cellvalues_shape::CellValues{dim,T,refshape}, f_shape, cellvalues_test::CellValues{dim,T,refshape}, f_test, op) where {dim,T,refshape}
    n_basefuncs_shape = getnbasefunctions(cellvalues_shape)
    n_basefuncs_test = getnbasefunctions(cellvalues_test)

    Ke = zeros(n_basefuncs_test, n_basefuncs_shape)

    #implicit assumption: Same geometry!
    X_shape = zeros(Vec{dim,Float64}, Ferrite.getngeobasefunctions(cellvalues_shape))
    getcoordinates!(X_shape, grid, 1)
    reinit!(cellvalues_shape, X_shape)

    X_test = zeros(Vec{dim,Float64}, Ferrite.getngeobasefunctions(cellvalues_test))
    getcoordinates!(X_test, grid, 1)
    reinit!(cellvalues_test, X_test)

    for q_point in 1:getnquadpoints(cellvalues_test) #assume same quadrature rule
        dΩ = getdetJdV(cellvalues_test, q_point)
        for i in 1:n_basefuncs_test
            test = f_test(cellvalues_test, q_point, i)
            for j in 1:n_basefuncs_shape
                shape = f_shape(cellvalues_shape, q_point, j)
                Ke[i, j] += op(test, shape) * dΩ
            end
        end
    end

    Ke
end

function _generalized_petrov_galerkin_assemble_local_matrix(grid::Ferrite.AbstractGrid, facevalues_shape::FaceValues{dim,T,refshape}, f_shape, facevalues_test::FaceValues{dim,T,refshape}, f_test, op) where {dim,T,refshape}
    n_basefuncs_shape = getnbasefunctions(facevalues_shape)
    n_basefuncs_test = getnbasefunctions(facevalues_test)

    f = zeros(n_basefuncs_test)

    X_shape = getcoordinates(grid, 1)
    X_test = getcoordinates(grid, 1)
    for face in 1:nfaces(getcells(grid)[1])
        reinit!(facevalues_shape, X_shape, face)
        reinit!(facevalues_test, X_test, face)

        for q_point in 1:getnquadpoints(facevalues_shape)
            n = getnormal(facevalues_test, q_point)
            dΓ = getdetJdV(facevalues_test, q_point)
            for i in 1:n_basefuncs_test
                test = f_test(facevalues_test, q_point, i)
                for j in 1:n_basefuncs_shape
                    shape = f_shape(facevalues_shape, q_point, j)
                    f[i] += op(test, shape) ⋅ n * dΓ
                end
            end
        end
    end

    f
end

function _assemble_mass(dh, cellvalues, sym)
    n_basefuncs = getnbasefunctions(cellvalues)
    Me = zeros(n_basefuncs, n_basefuncs)
    fe = zeros(n_basefuncs)

    M = sym ? create_symmetric_sparsity_pattern(dh) : create_sparsity_pattern(dh);
    f = zeros(ndofs(dh))

    assembler = start_assemble(M, f);
    @inbounds for cell in CellIterator(dh)
        fill!(Me, 0)

        reinit!(cellvalues, cell)

        for q_point in 1:getnquadpoints(cellvalues)
            dΩ = getdetJdV(cellvalues, q_point)

            for i in 1:n_basefuncs
                φ  = shape_value(cellvalues, q_point, i)
                fe[i] += φ * dΩ
                for j in 1:n_basefuncs
                    ψ  = shape_value(cellvalues, q_point, j)
                    Me[i, j] += (φ * ψ) * dΩ
                end
            end
        end

        assemble!(assembler, celldofs(cell), Me, fe)
    end

    return M, f
end

end

# Permute over common combinations for some commonly required local matrices.
SUITE["assembly"]["common-local"] = BenchmarkGroup()
COMMON_LOCAL_ASSEMBLY = SUITE["assembly"]["common-local"]
for spatial_dim ∈ 1:3
    ξ_dummy = Vec{spatial_dim}(ntuple(x->0.0, spatial_dim))
    COMMON_LOCAL_ASSEMBLY["spatial-dim",spatial_dim] = BenchmarkGroup()
    for geo_type ∈ FerriteBenchmarkHelper.geo_types_for_spatial_dim(spatial_dim)
        COMMON_LOCAL_ASSEMBLY["spatial-dim",spatial_dim][string(geo_type)] = BenchmarkGroup()

        grid = generate_grid(geo_type, tuple(repeat([1], spatial_dim)...));
        ref_type = FerriteBenchmarkHelper.default_refshape(geo_type)
        ip_geo = Ferrite.default_interpolation(geo_type)

        # Nodal interpolation tests
        for order ∈ 1:2, ip_type ∈ [Lagrange, Serendipity]
            ip = ip_type{spatial_dim, ref_type, order}()

            # Skip over elements which are not implemented
            !applicable(Ferrite.value, ip, 1, ξ_dummy) && continue

            qr = QuadratureRule{spatial_dim, ref_type}(2*order-1)

            csv = CellScalarValues(qr, ip, ip_geo);
            csv2 = CellScalarValues(qr, ip, ip_geo);

            cvv = CellVectorValues(qr, ip, ip_geo);
            cvv2 = CellVectorValues(qr, ip, ip_geo);

            # Currenlty we just benchmark nodal Lagrange bases.
            COMMON_LOCAL_ASSEMBLY["spatial-dim",spatial_dim][string(geo_type)][string(ip_type),string(order)] = BenchmarkGroup()
            LAGRANGE_SUITE = COMMON_LOCAL_ASSEMBLY["spatial-dim",spatial_dim][string(geo_type)][string(ip_type),string(order)]
            LAGRANGE_SUITE["ritz-galerkin"] = BenchmarkGroup()
            LAGRANGE_SUITE["petrov-galerkin"] = BenchmarkGroup()

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
            end
        end
    end
end

SUITE["assembly"]["Dirichlet"] = BenchmarkGroup()
DIRICHLET_SUITE = SUITE["assembly"]["Dirichlet"]
# span artifical scope...
for spatial_dim ∈ [2]
    # Benchmark application on global system
    DIRICHLET_SUITE["global"] = BenchmarkGroup()

    geo_type = Quadrilateral
    grid = generate_grid(geo_type, tuple([spatial_dim, spatial_dim]));
    ref_type = FerriteBenchmarkHelper.default_refshape(geo_type)
    ip_geo = Ferrite.default_interpolation(geo_type)
    order = 2

    # assemble a mass matrix to apply BCs on (because its cheap)
    ip = Lagrange{spatial_dim, ref_type, order}()
    qr = QuadratureRule{spatial_dim, ref_type}(2*order-1)
    cellvalues = CellScalarValues(qr, ip, ip_geo);
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