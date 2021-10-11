using BenchmarkTools
using Ferrite

module FerriteBenchmarkHelper

using Ferrite

function geo_types_for_dim(dim)
    dim == 1 && return [Line, QuadraticLine]
    dim == 2 && return [Triangle, QuadraticTriangle, Quadrilateral, QuadraticQuadrilateral]
    dim == 3 && return [Tetrahedron, Hexahedron] # Quadratic* not yet functional in 3D
end

function ref_type_for_geo_type(geo_type)
    # Quadratic* not yet functional in 3D
    geo_type ∈ [Line, QuadraticLine, Quadrilateral, QuadraticQuadrilateral, Hexahedron] && return RefCube
    geo_type ∈ [Triangle, QuadraticTriangle, Tetrahedron] && return RefTetrahedron
end

end

const SUITE = BenchmarkGroup()

#----------------------------------------------------------------------#
# Benchmarks for mesh functionality within Ferrite
#----------------------------------------------------------------------#
SUITE["mesh"] = BenchmarkGroup()

#----------------------------------------------------------------------#
# Benchmarks around the dof management
#----------------------------------------------------------------------#
SUITE["dof-management"] = BenchmarkGroup()

# for cell in CellIterator(grid)
#     coords = getcoordinates(cell) # get the coordinates
#     dofs = celldofs(cell)         # get the dofs for this cell
#     reinit!(cv, cell)             # reinit! the FE-base with a CellIterator
# end

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

    X = zeros(Vec{dim,Float64}, Ferrite.getngeobasefunctions(cellvalues))
    getcoordinates!(X, grid, 1)
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

# Minimal Petrov-Galerkin type local assembly loop. We assume that both function spaces share the same integration rule. Test is applied from the left.
function _generalized_petrov_galerkin_assemble_local_matrix(grid::Ferrite.AbstractGrid, cellvalues_shape::CellValues{dim,T,refshape}, f_shape, cellvalues_test::CellValues{dim,T,refshape}, f_test, op) where {dim,T,refshape}
    n_basefuncs_shape = getnbasefunctions(cellvalues_shape)
    n_basefuncs_test = getnbasefunctions(cellvalues_test)

    Ke = zeros(n_basefuncs_test, n_basefuncs_shape)

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

end

# Permute over common combinations for some commonly required local matrices.
SUITE["assembly"]["common-local"] = BenchmarkGroup()
# dim(topo) = dim(geo)
for dim ∈ 1:3
    SUITE["assembly"]["common-local"]["dim",string(dim)] = BenchmarkGroup()
    for geo_type ∈ FerriteBenchmarkHelper.geo_types_for_dim(dim)
        SUITE["assembly"]["common-local"]["dim",string(dim)][string(geo_type)] = BenchmarkGroup()

        grid = generate_grid(geo_type, tuple(repeat([1], dim)...));
        ref_type = FerriteBenchmarkHelper.ref_type_for_geo_type(geo_type)
        ip_geo = Ferrite.default_interpolation(geo_type)

        for order ∈ 1:2
            # second order Lagrange on hex not working yet...
            order > 1 && geo_type ∈ [Hexahedron] && continue

            SUITE["assembly"]["common-local"]["dim",string(dim)][string(geo_type)]
            # Currenlty we just benchmark nodal Lagrange bases.
            SUITE["assembly"]["common-local"]["dim",string(dim)][string(geo_type)]["lagrange",string(order)] = BenchmarkGroup()
            SUITE["assembly"]["common-local"]["dim",string(dim)][string(geo_type)]["lagrange",string(order)]["ritz-galerkin"] = BenchmarkGroup()
            SUITE["assembly"]["common-local"]["dim",string(dim)][string(geo_type)]["lagrange",string(order)]["petrov-galerkin"] = BenchmarkGroup()

            ip = Lagrange{dim, ref_type, order}()
            qr = QuadratureRule{dim, ref_type}(2*order-1)

            csv = CellScalarValues(qr, ip, ip_geo);
            csv2 = CellScalarValues(qr, ip, ip_geo);

            cvv = CellVectorValues(qr, ip, ip_geo);
            cvv2 = CellVectorValues(qr, ip, ip_geo);

            SUITE["assembly"]["common-local"]["dim",string(dim)][string(geo_type)]["lagrange",string(order)]["ritz-galerkin"]["mass"] = @benchmarkable FerriteAssemblyBenchmarks._generalized_ritz_galerkin_assemble_local_matrix($grid, $csv, shape_value, shape_value, *)
            SUITE["assembly"]["common-local"]["dim",string(dim)][string(geo_type)]["lagrange",string(order)]["ritz-galerkin"]["vector-mass"] = @benchmarkable FerriteAssemblyBenchmarks._generalized_ritz_galerkin_assemble_local_matrix($grid, $cvv, shape_value, shape_value, ⋅)
            SUITE["assembly"]["common-local"]["dim",string(dim)][string(geo_type)]["lagrange",string(order)]["ritz-galerkin"]["laplace"] = @benchmarkable FerriteAssemblyBenchmarks._generalized_ritz_galerkin_assemble_local_matrix($grid, $csv, shape_gradient, shape_gradient, ⋅)
            SUITE["assembly"]["common-local"]["dim",string(dim)][string(geo_type)]["lagrange",string(order)]["ritz-galerkin"]["vector-laplace"] = @benchmarkable FerriteAssemblyBenchmarks._generalized_ritz_galerkin_assemble_local_matrix($grid, $cvv, shape_gradient, shape_gradient, ⊡)

            SUITE["assembly"]["common-local"]["dim",string(dim)][string(geo_type)]["lagrange",string(order)]["petrov-galerkin"]["mass"] = @benchmarkable FerriteAssemblyBenchmarks._generalized_petrov_galerkin_assemble_local_matrix($grid, $csv, shape_value, $csv2, shape_value, *)
            SUITE["assembly"]["common-local"]["dim",string(dim)][string(geo_type)]["lagrange",string(order)]["petrov-galerkin"]["vector-mass"] = @benchmarkable FerriteAssemblyBenchmarks._generalized_petrov_galerkin_assemble_local_matrix($grid, $cvv, shape_value, $cvv2, shape_value, ⋅)
            SUITE["assembly"]["common-local"]["dim",string(dim)][string(geo_type)]["lagrange",string(order)]["petrov-galerkin"]["laplace"] = @benchmarkable FerriteAssemblyBenchmarks._generalized_petrov_galerkin_assemble_local_matrix($grid, $csv, shape_gradient, $csv2, shape_gradient, ⋅)
            SUITE["assembly"]["common-local"]["dim",string(dim)][string(geo_type)]["lagrange",string(order)]["petrov-galerkin"]["vector-laplace"] = @benchmarkable FerriteAssemblyBenchmarks._generalized_petrov_galerkin_assemble_local_matrix($grid, $cvv, shape_gradient, $cvv2, shape_gradient, ⊡)
        end
    end
end
# # dim(topo) = dim(geo)-1
# for dim ∈ 2:3
#     for geo_type ∈ FerriteBenchmarkHelper.geo_types_for_dim(dim)
#         grid = generate_grid(geo_type, tuple(repeat([2], dim)...));
#     end
# end

SUITE["assembly"]["local-to-global"] = BenchmarkGroup()
SUITE["assembly"]["dirichlet"] = BenchmarkGroup()