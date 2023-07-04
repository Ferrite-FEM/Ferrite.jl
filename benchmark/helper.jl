module FerriteBenchmarkHelper

using Ferrite

function geo_types_for_spatial_dim(spatial_dim)
    spatial_dim == 1 && return [Line, QuadraticLine]
    spatial_dim == 2 && return [Triangle, QuadraticTriangle, Quadrilateral, QuadraticQuadrilateral]
    spatial_dim == 3 && return [Tetrahedron, Hexahedron] # Quadratic* not yet functional in 3D. 3D triangle missing. Embedded also missing.
end

end


module FerriteAssemblyHelper

using Ferrite

# Minimal Ritz-Galerkin type local assembly loop.
function _generalized_ritz_galerkin_assemble_local_matrix(grid::Ferrite.AbstractGrid, cellvalues::CellValues{<: Ferrite.InterpolationByDim{dim}}, f_shape, f_test, op) where {dim}
    n_basefuncs = getnbasefunctions(cellvalues)

    Ke = zeros(n_basefuncs, n_basefuncs)

    X = get_cell_coordinates(grid, 1)
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

function _generalized_ritz_galerkin_assemble_local_matrix(grid::Ferrite.AbstractGrid, facevalues::FaceValues{<: Ferrite.InterpolationByDim{dim}}, f_shape, f_test, op) where {dim}
    n_basefuncs = getnbasefunctions(facevalues)

    f = zeros(n_basefuncs)

    X = get_cell_coordinates(grid, 1)
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
function _generalized_petrov_galerkin_assemble_local_matrix(grid::Ferrite.AbstractGrid, cellvalues_shape::CellValues{<: Ferrite.InterpolationByDim{dim}}, f_shape, cellvalues_test::CellValues{<: Ferrite.InterpolationByDim{dim}}, f_test, op) where {dim}
    n_basefuncs_shape = getnbasefunctions(cellvalues_shape)
    n_basefuncs_test = getnbasefunctions(cellvalues_test)
    Ke = zeros(n_basefuncs_test, n_basefuncs_shape)

    #implicit assumption: Same geometry!
    X_shape = zeros(Vec{dim,Float64}, Ferrite.getngeobasefunctions(cellvalues_shape))
    get_cell_coordinates!(X_shape, grid, 1)
    reinit!(cellvalues_shape, X_shape)

    X_test = zeros(Vec{dim,Float64}, Ferrite.getngeobasefunctions(cellvalues_test))
    get_cell_coordinates!(X_test, grid, 1)
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

function _generalized_petrov_galerkin_assemble_local_matrix(grid::Ferrite.AbstractGrid, facevalues_shape::FaceValues{<: Ferrite.InterpolationByDim{dim}}, f_shape, facevalues_test::FaceValues{<: Ferrite.InterpolationByDim{dim}}, f_test, op) where {dim}
    n_basefuncs_shape = getnbasefunctions(facevalues_shape)
    n_basefuncs_test = getnbasefunctions(facevalues_test)

    f = zeros(n_basefuncs_test)

    X_shape = get_cell_coordinates(grid, 1)
    X_test = get_cell_coordinates(grid, 1)
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
