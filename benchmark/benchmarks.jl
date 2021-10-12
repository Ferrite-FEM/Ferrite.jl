using BenchmarkTools
using Ferrite

module FerriteBenchmarkHelper

using Ferrite

function geo_types_for_spatial_dim(dim)
    dim == 1 && return [Line, QuadraticLine]
    dim == 2 && return [Triangle, QuadraticTriangle, Quadrilateral, QuadraticQuadrilateral]
    dim == 3 && return [Tetrahedron, Hexahedron] # Quadratic* not yet functional in 3D. 3D triangle missing. Embedded also missing.
end

# TODO refactor into grid
# Quadratic* not yet functional in 3D
default_refshape(::Union{Type{Line}, Type{Line2D}, Type{Line3D}, Type{QuadraticLine}, Type{Quadrilateral}, Type{Quadrilateral3D}, Type{QuadraticQuadrilateral}, Type{Hexahedron}}) = RefCube
default_refshape(::Union{Type{Triangle}, Type{QuadraticTriangle}, Type{Tetrahedron}}) = RefTetrahedron

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
        HYPERRECTANGLE_GENERATOR["spatial-dim",spatial_dim][string(geo_type)] = @benchmarkable generate_grid($geo_type, $(tuple(repeat([4], spatial_dim)...)));
    end
end

# TODO AMR performance
# TODO topology performance

#----------------------------------------------------------------------#
# Benchmarks around the dof management
#----------------------------------------------------------------------#
SUITE["dof-management"] = BenchmarkGroup()
SUITE["dof-management"]["numbering"] = BenchmarkGroup()
NUMBERING_SUITE = SUITE["dof-management"]["numbering"]
for spatial_dim ∈ 1:3
    NUMBERING_SUITE["spatial-dim",string(spatial_dim)] = BenchmarkGroup()
    for geo_type ∈ FerriteBenchmarkHelper.geo_types_for_spatial_dim(spatial_dim)
        NUMBERING_SUITE["spatial-dim",string(spatial_dim)][string(geo_type)] = BenchmarkGroup()

        ref_type = FerriteBenchmarkHelper.default_refshape(geo_type)

        for grid_size ∈ [3, 6, 12] #multiple grid sized to estimate computational complexity...
            NUMBERING_SUITE["spatial-dim",string(spatial_dim)][string(geo_type)]["grid-size-",grid_size] = BenchmarkGroup()
            NUMBERING_SUITE["spatial-dim",string(spatial_dim)][string(geo_type)]["grid-size-",grid_size] = BenchmarkGroup()

            grid = generate_grid(geo_type, tuple(repeat([grid_size], spatial_dim)...));

            for field_dim ∈ 1:3
                NUMBERING_SUITE["spatial-dim",string(spatial_dim)][string(geo_type)]["grid-size-",grid_size]["field-dim-", field_dim] = BenchmarkGroup()
                NUMBERING_FIELD_DIM_SUITE = NUMBERING_SUITE["spatial-dim",string(spatial_dim)][string(geo_type)]["grid-size-",grid_size]["field-dim-", field_dim]
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

module FerriteAssemblyBenchmarks

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

end

# Permute over common combinations for some commonly required local matrices.
SUITE["assembly"]["common-local"] = BenchmarkGroup()
COMMON_LOCAL_ASSEMBLY = SUITE["assembly"]["common-local"]
# dim(topo) = dim(geo)
for dim ∈ 1:3
    COMMON_LOCAL_ASSEMBLY["dim",string(dim)] = BenchmarkGroup()
    for geo_type ∈ FerriteBenchmarkHelper.geo_types_for_spatial_dim(dim)
        COMMON_LOCAL_ASSEMBLY["dim",string(dim)][string(geo_type)] = BenchmarkGroup()

        grid = generate_grid(geo_type, tuple(repeat([1], dim)...));
        ref_type = FerriteBenchmarkHelper.default_refshape(geo_type)
        ip_geo = Ferrite.default_interpolation(geo_type)

        for order ∈ 1:2
            # higher order Lagrange on hex not working yet...
            order > 1 && geo_type ∈ [Hexahedron] && continue

            # Currenlty we just benchmark nodal Lagrange bases.
            COMMON_LOCAL_ASSEMBLY["dim",string(dim)][string(geo_type)]["Lagrange",string(order)] = BenchmarkGroup()
            LAGRANGE_SUITE = COMMON_LOCAL_ASSEMBLY["dim",string(dim)][string(geo_type)]["Lagrange",string(order)]
            LAGRANGE_SUITE["ritz-galerkin"] = BenchmarkGroup()
            LAGRANGE_SUITE["petrov-galerkin"] = BenchmarkGroup()

            ip = Lagrange{dim, ref_type, order}()
            qr = QuadratureRule{dim, ref_type}(2*order-1)

            csv = CellScalarValues(qr, ip, ip_geo);
            csv2 = CellScalarValues(qr, ip, ip_geo);

            cvv = CellVectorValues(qr, ip, ip_geo);
            cvv2 = CellVectorValues(qr, ip, ip_geo);

            # Scalar shape φ and test ψ: ∫ φ ψ
            LAGRANGE_SUITE["ritz-galerkin"]["mass"] = @benchmarkable FerriteAssemblyBenchmarks._generalized_ritz_galerkin_assemble_local_matrix($grid, $csv, shape_value, shape_value, *)
            LAGRANGE_SUITE["petrov-galerkin"]["mass"] = @benchmarkable FerriteAssemblyBenchmarks._generalized_petrov_galerkin_assemble_local_matrix($grid, $csv, shape_value, $csv2, shape_value, *)
            # Vectorial shape φ and test ψ: ∫ φ ⋅ ψ
            LAGRANGE_SUITE["ritz-galerkin"]["vector-mass"] = @benchmarkable FerriteAssemblyBenchmarks._generalized_ritz_galerkin_assemble_local_matrix($grid, $cvv, shape_value, shape_value, ⋅)
            LAGRANGE_SUITE["petrov-galerkin"]["vector-mass"] = @benchmarkable FerriteAssemblyBenchmarks._generalized_petrov_galerkin_assemble_local_matrix($grid, $cvv, shape_value, $cvv2, shape_value, ⋅)
            # Scalar shape φ and test ψ: ∫ ∇φ ⋅ ∇ψ
            LAGRANGE_SUITE["ritz-galerkin"]["Laplace"] = @benchmarkable FerriteAssemblyBenchmarks._generalized_ritz_galerkin_assemble_local_matrix($grid, $csv, shape_gradient, shape_gradient, ⋅)
            LAGRANGE_SUITE["petrov-galerkin"]["Laplace"] = @benchmarkable FerriteAssemblyBenchmarks._generalized_petrov_galerkin_assemble_local_matrix($grid, $csv, shape_gradient, $csv2, shape_gradient, ⋅)
            # Vectorial shape φ and test ψ: ∫ ∇φ : ∇ψ
            LAGRANGE_SUITE["ritz-galerkin"]["vector-Laplace"] = @benchmarkable FerriteAssemblyBenchmarks._generalized_ritz_galerkin_assemble_local_matrix($grid, $cvv, shape_gradient, shape_gradient, ⊡)
            LAGRANGE_SUITE["petrov-galerkin"]["vector-Laplace"] = @benchmarkable FerriteAssemblyBenchmarks._generalized_petrov_galerkin_assemble_local_matrix($grid, $cvv, shape_gradient, $cvv2, shape_gradient, ⊡)
            # Vectorial shape φ and scalar test ψ: ∫ (∇ ⋅ φ) ψ
            LAGRANGE_SUITE["petrov-galerkin"]["pressure-velocity"] = @benchmarkable FerriteAssemblyBenchmarks._generalized_petrov_galerkin_assemble_local_matrix($grid, $cvv, shape_divergence, $csv, shape_value, *)

            if dim > 1
                qr_face = QuadratureRule{dim-1, ref_type}(2*order-1)
                fsv = FaceScalarValues(qr_face, ip, ip_geo);
                fsv2 = FaceScalarValues(qr_face, ip, ip_geo);

                LAGRANGE_SUITE["ritz-galerkin"]["face-flux"] = @benchmarkable FerriteAssemblyBenchmarks._generalized_ritz_galerkin_assemble_local_matrix($grid, $fsv, shape_gradient, shape_value, *)
                LAGRANGE_SUITE["petrov-galerkin"]["face-flux"] = @benchmarkable FerriteAssemblyBenchmarks._generalized_petrov_galerkin_assemble_local_matrix($grid, $fsv, shape_gradient, $fsv2, shape_value, *)
            end
        end
    end
end

SUITE["assembly"]["Dirichlet"] = BenchmarkGroup()