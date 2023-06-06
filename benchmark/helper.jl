module FerriteBenchmarkHelper

using Ferrite

function geo_types_for_spatial_dim(spatial_dim)
    spatial_dim == 1 && return [Line, QuadraticLine]
    spatial_dim == 2 && return [Triangle, QuadraticTriangle, Quadrilateral, QuadraticQuadrilateral]
    spatial_dim == 3 && return [Tetrahedron, Hexahedron] # Quadratic* not yet functional in 3D. 3D triangle missing. Embedded also missing.
end

getrefshape(::Type{T}) where {refshape, T <: Ferrite.AbstractCell{refshape}} = refshape

end


module FerriteAssemblyHelper

using Ferrite

# Minimal Ritz-Galerkin type local assembly loop.
function _generalized_ritz_galerkin_assemble_local_matrix(grid::Ferrite.AbstractGrid, cellvalues::CellValues, f_shape, f_test, op)
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

function _generalized_ritz_galerkin_assemble_local_matrix(grid::Ferrite.AbstractGrid, facevalues::FaceValues, f_shape, f_test, op)
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

function _generalized_ritz_galerkin_assemble_interfaces(dh::Ferrite.AbstractDofHandler, interfacevalues::InterfaceValues, f_shape, f_test, op)
    n_basefuncs = getnbasefunctions(interfacevalues)

    K = zeros(ndofs(dh), ndofs(dh))

    for ic in InterfaceIterator(dh)
        reinit!(interfacevalues, ic)
        for q_point in 1:getnquadpoints(interfacevalues)
            dΓ = getdetJdV(interfacevalues, q_point)
            for i in 1:n_basefuncs
                test = f_test(interfacevalues, q_point, i)
                f_test == shape_value_jump && (test *= getnormal(interfacevalues, q_point))
                for j in 1:n_basefuncs
                    shape = f_shape(interfacevalues, q_point, j)
                    f_shape == shape_value_jump && (shape *= getnormal(interfacevalues, q_point))
                    K[interfacedofs(ic)[i], interfacedofs(ic)[j]] += op(test, shape) * dΓ
                end
            end
        end
    end

    K
end

# Minimal Petrov-Galerkin type local assembly loop. We assume that both function spaces share the same integration rule. Test is applied from the left.
function _generalized_petrov_galerkin_assemble_local_matrix(grid::Ferrite.AbstractGrid, cellvalues_shape::CellValues, f_shape, cellvalues_test::CellValues, f_test, op)
    n_basefuncs_shape = getnbasefunctions(cellvalues_shape)
    n_basefuncs_test = getnbasefunctions(cellvalues_test)
    Ke = zeros(n_basefuncs_test, n_basefuncs_shape)

    #implicit assumption: Same geometry!
    X_shape = zeros(Ferrite.get_coordinate_type(grid), Ferrite.getngeobasefunctions(cellvalues_shape))
    getcoordinates!(X_shape, grid, 1)
    reinit!(cellvalues_shape, X_shape)

    X_test = zeros(Ferrite.get_coordinate_type(grid), Ferrite.getngeobasefunctions(cellvalues_test))
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

function _generalized_petrov_galerkin_assemble_local_matrix(grid::Ferrite.AbstractGrid, facevalues_shape::FaceValues, f_shape, facevalues_test::FaceValues, f_test, op)
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

function _generalized_petrov_galerkin_assemble_interfaces(dh::Ferrite.AbstractDofHandler, interfacevalues_shape::InterfaceValues, f_shape, interfacevalues_test::InterfaceValues, f_test, op)
    n_basefuncs_shape = getnbasefunctions(interfacevalues_shape)
    n_basefuncs_test = getnbasefunctions(interfacevalues_test)

    K = zeros(ndofs(dh), ndofs(dh))

    for ic in InterfaceIterator(dh)
        reinit!(interfacevalues_shape, ic)
        reinit!(interfacevalues_test, ic)
        for q_point in 1:getnquadpoints(interfacevalues_shape)
            dΓ = getdetJdV(interfacevalues_test, q_point)
            for i in 1:n_basefuncs_test
                test = f_test(interfacevalues_test, q_point, i)
                f_test == shape_value_jump && (test *= getnormal(interfacevalues_test, q_point))
                for j in 1:n_basefuncs_shape
                    shape = f_shape(interfacevalues_shape, q_point, j)
                    f_shape == shape_value_jump && (shape *= getnormal(interfacevalues_shape, q_point))
                    K[interfacedofs(ic)[i], interfacedofs(ic)[j]] += op(test, shape) * dΓ
                end
            end
        end
    end

    K
end

function _assemble_mass(dh, cellvalues, sym)
    n_basefuncs = getnbasefunctions(cellvalues)
    Me = zeros(n_basefuncs, n_basefuncs)
    fe = zeros(n_basefuncs)

    M = sym ? create_symmetric_sparsity_pattern(dh) : create_matrix(dh);
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
