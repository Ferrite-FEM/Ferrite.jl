using JuAFEM
using Tensors
using TimerOutputs
using UnicodePlots
const to = TimerOutput();

const USE_HEX = true
const dim = 3

function create_grid(n)
    geoshape = USE_HEX ? Hexahedron : Tetrahedron
    corner1 = Vec{dim}((0.0, 0.0, 0.0))
    corner2 = Vec{dim}((10.0, 1.0, 1.0))
    grid = generate_grid(geoshape, (n, n, n), corner1, corner2);
    # Extract the left boundary
    addnodeset!(grid, "clamped", x -> norm(x[1]) ≈ 0.0);
    cell_colors, final_colors = JuAFEM.create_coloring(grid)
    return grid, final_colors
end

function create_values()
    refshape = USE_HEX ? RefCube    : RefTetrahedron
    order = USE_HEX ? 2 : 1
    # Interpolations and values
    interpolation_space = Lagrange{dim, refshape, 1}()
    quadrature_rule = QuadratureRule{dim, refshape}(order)
    face_quadrature_rule = QuadratureRule{dim-1, refshape}(order)
    cellvalues = [CellVectorValues(quadrature_rule, interpolation_space) for i in 1:Threads.nthreads()];
    facevalues = [FaceVectorValues(face_quadrature_rule, interpolation_space) for i in 1:Threads.nthreads()];
    return cellvalues, facevalues
end

# DofHandler
function create_dofhandler(grid)
    dh = DofHandler(grid)
    push!(dh, :u, dim) # Add a displacement field
    close!(dh)
end


function create_boundary_conditions(dh)
    # Boundaryconditions
    dbc = DirichletBoundaryConditions(dh)
    # Add a homogenoush boundary condition on the "clamped" edge
    add!(dbc, :u, getnodeset(grid, "clamped"), (x,t) -> [0.0, 0.0, 0.0], collect(1:dim))
    close!(dbc)
    t = 0.0
    update!(dbc, t)
end

# Create the stiffness tensor
function create_stiffness()
    E = 200e9
    ν = 0.3
    λ = E*ν / ((1+ν) * (1 - 2ν))
    μ = E / (2(1+ν))
    δ(i,j) = i == j ? 1.0 : 0.0
    g(i,j,k,l) = λ*δ(i,j)*δ(k,l) + μ*(δ(i,k)*δ(j,l) + δ(i,l)*δ(j,k))
    C = SymmetricTensor{4, dim}(g);
    return C
end

immutable ScratchValues{T, CV <: CellValues, FV <: FaceValues, TT <: AbstractTensor, dim, Ti}
    Ke::Matrix{T}
    fe::Vector{T}
    cellvalues::CV
    facevalues::FV
    global_dofs::Vector{Int}
    ɛ::Vector{TT}
    coordinates::Vector{Vec{dim, T}}
    assembler::JuAFEM.AssemblerSparsityPattern{T, Ti}
end

function doassemble{CV <: CellValues, FV <: FaceValues}(cellvalues::Vector{CV}, facevalues::Vector{FV},
                         K::SparseMatrixCSC, colors, grid::Grid, dh::DofHandler, C::SymmetricTensor{4, dim})

    f = zeros(ndofs(dh))
    assemblers = [start_assemble(K, f) for i in 1:Threads.nthreads()]

    n_basefuncs = getnbasefunctions(cellvalues[1])
    global_dofs = [zeros(Int, ndofs_per_cell(dh)) for i in 1:Threads.nthreads()]

    fes = [zeros(n_basefuncs) for i in 1:Threads.nthreads()] # Local force vector
    Kes = [zeros(n_basefuncs, n_basefuncs) for i in 1:Threads.nthreads()]

    ɛs = [[zero(SymmetricTensor{2, dim}) for i in 1:n_basefuncs] for i in 1:Threads.nthreads()]

    coordinates = [[zero(Vec{dim}) for i in 1:length(grid.cells[1].nodes)] for i in 1:Threads.nthreads()]

    scratches = [ScratchValues(Kes[i], fes[i], cellvalues[i], facevalues[i], global_dofs[i], ɛs[i], coordinates[i], assemblers[i]) for i in 1:Threads.nthreads()]

    t = Vec{3}((0.0, 1e8, 0.0)) # Traction vector
    b = Vec{3}((0.0, 0.0, 0.0)) # Body force

    for color in colors
        Threads.@threads for i in 1:length(color)
            assemble_cell!(scratches[Threads.threadid()], color[i], K, grid, dh, C, t, b)
        end
    end

    return K, f
end

function assemble_cell!{dim}(scratch::ScratchValues, cell::Int, K::SparseMatrixCSC, grid::Grid, dh::DofHandler, C::SymmetricTensor{4, dim},
                            t, b)

    Ke, fe, cellvalues, facevalues, global_dofs, ɛ, coordinates, assembler =
         scratch.Ke, scratch.fe, scratch.cellvalues, scratch.facevalues,
         scratch.global_dofs, scratch.ɛ, scratch.coordinates, scratch.assembler

    fill!(Ke, 0)
    fill!(fe, 0)

    n_basefuncs = getnbasefunctions(cellvalues)

    # Fill upp the coordinates
    nodeids = grid.cells[cell].nodes
    for j in 1:length(coordinates)
        coordinates[j] = grid.nodes[nodeids[j]].x
    end

    reinit!(cellvalues, coordinates)
    for q_point in 1:getnquadpoints(cellvalues)
        for i in 1:n_basefuncs
            ɛ[i] = symmetric(shape_gradient(cellvalues, q_point, i))
        end
        dΩ = getdetJdV(cellvalues, q_point)
        for i in 1:n_basefuncs
            δu = shape_value(cellvalues, q_point, i)
            fe[i] += (δu ⋅ b) * dΩ
            ɛC = ɛ[i] ⊡ C
            for j in 1:n_basefuncs
                Ke[i, j] += (ɛC ⊡ ɛ[j]) * dΩ
            end
        end
    end

    for face in 1:JuAFEM.n_faces_per_cell(grid)
        if onboundary(grid.cells[cell], face) && (cell, face) ∈ getfaceset(grid, "right")
            reinit!(facevalues, coordinates, face)
            for q_point in 1:getnquadpoints(facevalues)
                dΓ = getdetJdV(facevalues, q_point)
                for i in 1:n_basefuncs
                    δu = shape_value(facevalues, q_point, i)
                    fe[i] += (δu ⋅ t) * dΓ
                end
            end
        end
    end
    celldofs!(global_dofs, dh, cell)
    assemble!(assembler, fe, Ke, global_dofs)
end

"""
grid, colors = create_grid(40);
dh = create_dofhandler(grid);
dbc = create_boundary_conditions(dh);
cellvalues, facevalues = create_values();

K = create_sparsity_pattern(dh);
C = create_stiffness();
@time K, f = doassemble(cellvalues, facevalues, K, colors, grid, dh, C);
"""