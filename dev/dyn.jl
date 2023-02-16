
using Ferrite, SparseArrays

function create_example_2d_grid()
    grid = generate_grid(Quadrilateral, (10, 10), Vec{2}((0.0, 0.0)), Vec{2}((10.0, 10.0)))
    colors_workstream = create_coloring(grid; alg=ColoringAlgorithm.WorkStream)
    colors_greedy = create_coloring(grid; alg=ColoringAlgorithm.Greedy)
    vtk_grid("colored", grid) do vtk
        vtk_cell_data_colors(vtk, colors_workstream, "workstream-coloring")
        vtk_cell_data_colors(vtk, colors_greedy, "greedy-coloring")
    end
end

create_example_2d_grid();

# ![](coloring.png)
#
# *Figure 1*: Element coloring using the "workstream"-algorithm (left) and the "greedy"-
# algorithm (right).

# ## Cantilever beam in 3D with threaded assembly
# We will now look at an example where we assemble the stiffness matrix using multiple
# threads. We set up a simple grid and create a coloring, then create a DofHandler,
# and define the material stiffness

# #### Grid for the beam
function create_colored_cantilever_grid(celltype, n)
    grid = generate_grid(celltype, (10*n, n, n), Vec{3}((0.0, 0.0, 0.0)), Vec{3}((10.0, 1.0, 1.0)))
    colors = create_coloring(grid)
    return grid, colors
end;

# #### DofHandler
function create_dofhandler(grid::Grid{dim}) where {dim}
    dh = DofHandler(grid)
    push!(dh, :u, dim) # Add a displacement field
    close!(dh)
end;

# ### Stiffness tensor for linear elasticity
function create_stiffness(::Val{dim}) where {dim}
    E = 200e9
    ν = 0.3
    λ = E*ν / ((1+ν) * (1 - 2ν))
    μ = E / (2(1+ν))
    δ(i,j) = i == j ? 1.0 : 0.0
    g(i,j,k,l) = λ*δ(i,j)*δ(k,l) + μ*(δ(i,k)*δ(j,l) + δ(i,l)*δ(j,k))
    C = SymmetricTensor{4, dim}(g);
    return C
end;

# ## Threaded data structures
#
# ScratchValues is a thread-local collection of data that each thread needs to own,
# since we need to be able to mutate the data in the threads independently
struct ScratchValues{T, CV <: CellValues, dim, Ti}
    Ke::Matrix{T}
    fe::Vector{T}
    cellvalues::CV
    global_dofs::Vector{Int}
    coordinates::Vector{Vec{dim, T}}
    assembler::Ferrite.AssemblerSparsityPattern{T, Ti}
end;

# Each thread need its own CellValues
function create_values(refshape, dim, order::Int)
    ## Interpolations and values
    interpolation_space = Lagrange{dim, refshape, 1}()
    quadrature_rule = QuadratureRule{dim, refshape}(order)
    cellvalues = [CellVectorValues(quadrature_rule, interpolation_space) for i in 1:Threads.nthreads()];
    return cellvalues
end;

# Create a `ScratchValues` for each thread with the thread local data
function create_scratchvalues(K, f, dh::DofHandler{dim}) where {dim}
    nthreads = Threads.nthreads()
    assemblers = [start_assemble(K, f) for i in 1:nthreads]
    cellvalues = create_values(RefCube, dim, 2)

    n_basefuncs = getnbasefunctions(cellvalues[1])
    global_dofs = [zeros(Int, ndofs_per_cell(dh)) for i in 1:nthreads]

    fes = [zeros(n_basefuncs) for i in 1:nthreads] # Local force vector
    Kes = [zeros(n_basefuncs, n_basefuncs) for i in 1:nthreads]

    coordinates = [[zero(Vec{dim}) for i in 1:length(dh.grid.cells[1].nodes)] for i in 1:nthreads]

    return [ScratchValues(Kes[i], fes[i], cellvalues[i], global_dofs[i], coordinates[i], assemblers[i]) for i in 1:nthreads]
end;

# ## Threaded assemble

# The assembly function loops over each color and does a threaded assembly for that color
function doassemble!(scratches, colors, dh::DofHandler, C::SymmetricTensor{4}, b::Vec)

    for color in colors
        ## Each color is safe to assemble threaded
        Threads.@threads :static for i in eachindex(color)
            assemble_cell!(scratches[Threads.threadid()], color[i], dh, C, b)
        end
    end

end

# The cell assembly function is written the same way as if it was a single threaded example.
# The only difference is that we unpack the variables from our `scratch`.
function assemble_cell!(scratch::ScratchValues, cell::Int, dh::DofHandler, C::SymmetricTensor{4}, b::Vec)

    ## Unpack our stuff from the scratch
    Ke, fe, cellvalues, global_dofs, coordinates, assembler =
         scratch.Ke, scratch.fe, scratch.cellvalues, 
         scratch.global_dofs, scratch.coordinates, scratch.assembler

    fill!(Ke, 0)
    fill!(fe, 0)
    getcoordinates!(coordinates, dh.grid, cell)
    reinit!(cellvalues, coordinates)
    element_routine!(Ke, fe, cellvalues, C, b)

    celldofs!(global_dofs, dh, cell)
    assemble!(assembler, global_dofs, fe, Ke)
end;

function element_routine!(Ke, fe, cellvalues, C, b)
    n_basefuncs = getnbasefunctions(cellvalues)
    for q_point in 1:getnquadpoints(cellvalues)
        dΩ = getdetJdV(cellvalues, q_point)
        for i in 1:n_basefuncs
            δu = shape_value(cellvalues, q_point, i)
            δɛ = symmetric(shape_gradient(cellvalues, q_point, i))
            fe[i] += (δu ⋅ b) * dΩ
            ɛC = δɛ ⊡ C
            for j in 1:n_basefuncs
                ɛ = symmetric(shape_gradient(cellvalues, q_point, j))
                Ke[i, j] += (ɛC ⊡ ɛ) * dΩ
            end
        end
    end
end

function run_assemble(n = 20)

    grid, colors = create_colored_cantilever_grid(Hexahedron, n);
    dh = create_dofhandler(grid);

    K = create_sparsity_pattern(dh);
    C = create_stiffness(Val{3}());
    b = Vec{3}((0.0, 0.0, 1.0)) # Body force

    f = zeros(ndofs(dh))

    scratches = create_scratchvalues(K, f, dh);
    doassemble!(scratches, colors, dh, C, b); # compilation

    scratches = create_scratchvalues(K, f, dh);
    b = @elapsed @time doassemble!(scratches, colors, dh, C, b);
    return b
end

open("dyn.log"; append=true) do fid
    write(fid, "$(Threads.nthreads()), $(run_assemble())\n")
end