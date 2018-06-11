

using Revise
using JuAFEM
using Tensors
using TimerOutputs



function solve1()
    reset_timer!()
    corner1 = Vec{3}((0.0, 0.0, 0.0))
    corner2 = Vec{3}((10.0, 1.0, 1.0))
    grid = generate_grid(Tetrahedron, (60, 60, 12), corner1, corner2)
    addnodeset!(grid, "nodes", x -> x[1] == 0)

    # Interpolations and values
    interpolation_space = Lagrange{3, RefTetrahedron, 1}()
    quadrature_rule = QuadratureRule{3, RefTetrahedron}(1)
    cellvalues = CellVectorValues(quadrature_rule, interpolation_space)
    facevalues = FaceVectorValues(QuadratureRule{2, RefTetrahedron}(1), interpolation_space)

    # DofHandler
    # @timeit "dofhandler" begin
        dh = DofHandler(grid)
        push!(dh, :u, 3) # Add a displacement field
        close!(dh)
    # end

    # K = create_symmetric_sparsity_pattern(dh); # assemble only upper half since it is symmetric

    # Boundaryconditions
    @timeit "node bc" begin
        @timeit "ch" dbcs = ConstraintHandler(dh)
        # Add a homogenoush boundary condition on the "clamped" edge
        @timeit "Dirichlet" dbc = Dirichlet(:u, getnodeset(grid, "nodes"), (x,t) -> [0.0, 0.0, 0.0], collect(1:3))
        @timeit "add!" add!(dbcs, dbc)
        # @code_warntype close!(dbcs)
        @timeit "close!" close!(dbcs)
        # @show length(dbcs)
        t = 0.0
        @timeit "update!" update!(dbcs, t)
    end

    # # Create the stiffness tensor
    # E = 200e9
    # ν = 0.3
    # λ = E*ν / ((1+ν) * (1 - 2ν))
    # μ = E / (2(1+ν))
    # δ(i,j) = i == j ? 1.0 : 0.0
    # g(i,j,k,l) = λ*δ(i,j)*δ(k,l) + μ*(δ(i,k)*δ(j,l) + δ(i,l)*δ(j,k))
    # C = SymmetricTensor{4, 3}(g)

    # @timeit "assemble" begin
    #     K, f = doassemble(cellvalues, K, grid, dh, C)
    # end
    # # @timeit "solve" K\f
    print_timer()
end
function solve2()
    reset_timer!()
    corner1 = Vec{3}((0.0, 0.0, 0.0))
    corner2 = Vec{3}((10.0, 1.0, 1.0))
    grid = generate_grid(Tetrahedron, (60, 60, 12), corner1, corner2)
    addnodeset!(grid, "nodes", x -> x[1] == 0)

    # Interpolations and values
    interpolation_space = Lagrange{3, RefTetrahedron, 1}()
    quadrature_rule = QuadratureRule{3, RefTetrahedron}(1)
    cellvalues = CellVectorValues(quadrature_rule, interpolation_space)
    facevalues = FaceVectorValues(QuadratureRule{2, RefTetrahedron}(1), interpolation_space)

    # DofHandler
    # @timeit "dofhandler" begin
        dh = DofHandler(grid)
        push!(dh, :u, 3) # Add a displacement field
        close!(dh)
    # end

    # K = create_symmetric_sparsity_pattern(dh); # assemble only upper half since it is symmetric


    @timeit "face bc" begin
        @timeit "ch" dbcs_face = ConstraintHandler(dh)
        # Add a homogenoush boundary condition on the "clamped" edge
        @timeit "Dirichlet" dbc_face = Dirichlet(:u, getfaceset(grid, "left"), (x,t) -> [0.0, 0.0, 0.0], collect(1:3))
        @timeit "add!" add!(dbcs_face, dbc_face)
        # @code_warntype close!(dbcs_face)
        @timeit "close!" close!(dbcs_face)
        t = 0.0
        @timeit "update!" update!(dbcs_face, t)
    end

    # # Create the stiffness tensor
    # E = 200e9
    # ν = 0.3
    # λ = E*ν / ((1+ν) * (1 - 2ν))
    # μ = E / (2(1+ν))
    # δ(i,j) = i == j ? 1.0 : 0.0
    # g(i,j,k,l) = λ*δ(i,j)*δ(k,l) + μ*(δ(i,k)*δ(j,l) + δ(i,l)*δ(j,k))
    # C = SymmetricTensor{4, 3}(g)

    # @timeit "assemble" begin
    #     K, f = doassemble(cellvalues, K, grid, dh, C)
    # end
    # # @timeit "solve" K\f
    print_timer()
end


gc(); gc()
solve1()
gc(); gc()
solve2()

function doassemble{dim}(cellvalues::CellVectorValues{dim}, K::Symmetric, grid::Grid, dh::DofHandler, C::SymmetricTensor{4, dim})


    f = zeros(ndofs(dh))
    assembler = start_assemble(K, f)

    n_basefuncs = getnbasefunctions(cellvalues)

    fe = zeros(n_basefuncs) # Local force vector
    Ke = Symmetric(zeros(n_basefuncs, n_basefuncs), :U) # Local stiffness mastrix

    t = Vec{3}((0.0, 1e8, 0.0)) # Traction vector
    b = Vec{3}((0.0, 0.0, 0.0)) # Body force
    ɛ = [zero(SymmetricTensor{2, dim}) for i in 1:n_basefuncs]
    @inbounds for (cellcount, cell) in enumerate(CellIterator(dh))
        fill!(Ke.data, 0)
        fill!(fe, 0)

        reinit!(cellvalues, cell)
        for q_point in 1:getnquadpoints(cellvalues)
            for i in 1:n_basefuncs
                ɛ[i] = symmetric(shape_gradient(cellvalues, q_point, i))
            end
            dΩ = getdetJdV(cellvalues, q_point)
            for i in 1:n_basefuncs
                δu = shape_value(cellvalues, q_point, i)
                fe[i] += (δu ⋅ b) * dΩ
                ɛC = ɛ[i] ⊡ C
                for j in i:n_basefuncs # assemble only upper half
                    Ke.data[i, j] += (ɛC ⊡ ɛ[j]) * dΩ # can only assign to parent of the Symmetric wrapper
                end
            end
        end
        global_dofs = celldofs(cell)
        assemble!(assembler, global_dofs, fe, Ke)
    end
    return K, f
end;
