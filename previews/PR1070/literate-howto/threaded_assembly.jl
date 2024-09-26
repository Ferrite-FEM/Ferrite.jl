# ```@meta
# Draft = false
# ```
# # [Multi-threaded assembly](@id tutorial-threaded-assembly)
#
#-
#md # !!! tip
#md #     This example is also available as a Jupyter notebook:
#md #     [`threaded_assembly.ipynb`](@__NBVIEWER_ROOT_URL__/howto/threaded_assembly.ipynb).
#-

# ## Introduction
#
# In this howto we will explore how to use task based multithreading (shared memory
# parallelism) to speed up the analysis. Some parts of a finite element simulation are
# trivially parallelizable such as the computation of the local element contributions since
# each element can be processed independently. However, two things need to be considered in
# order to parallelize safely:
#
#  - **Modification of shared data**: Although the contributions from all the elements can
#    be computed independently, eventually they need to be assembled into the global
#    matrix and vector. Letting each task assemble their own contribution would lead to
#    race conditions since elements share degrees of freedom with each other. There are
#    various ways to remedy this, for example:
#     - **Locking**: By using a lock around the call to `assemble!` we can ensure that only
#       one task assembles at a time. This is simple to implement but can lead to lock
#       contention and thus poor performance. Another drawback is that the results will not
#       be deterministic since floating point operations are neither associative nor
#       commutative.
#     - **Assembler task**: By using a designated task for the assembling we (obviously)
#       ensure that only a single task assembles. The worker tasks (the tasks computing the
#       element contributions) would then hand off their results to the assemly task. This
#       can be a useful approach if computing the element contributions is much slower than
#       the assembly -- otherwise the assembler task can't keep up with the worker tasks.
#       There might also be some extra overhead because of task switching in the scheduler.
#       The problem with non-deterministic results still remains.
#     - **Grid coloring**: By "coloring" the grid such that, within each color, no two
#       elements share degrees of freedom, we can safely assemble each color in parallel.
#       Even if concurrently running tasks will write to the global matrix and vector they
#       will not write to the same memory locations. Note also that this procedure gives
#       predictable results because for a memory location which, for example, a "red",
#       a "blue", and a "green" element will contribute to we will always add the red first,
#       then the blue, and finally the green.
#  - **Scratch data**: In order to speed up the computation of the element contributions we
#    typically pre-allocate some data structures that can be reused for every element. Such
#    scratch data include, for example, the local matrix and vector, and the CellValues.
#    Each task need their own copy of the scratch data since they will be modified for each
#    element.

# ## Grid coloring
#
# Ferrite include functionality to color the grid with the [`create_coloring`](@ref)
# function. Here we create a simple 2D grid, color it, and export the colors to a VTK file
# to visualize the result (see *Figure 1*.). Note that no cells with the same color has any
# shared nodes (dofs). This means that it is safe to assemble in parallel as long as we only
# assemble one color at a time.
#
# There are two coloring algorithms implemented: the "workstream" algorithm (from Turcksin
# et al. [Turcksin2016](@cite)) and a "greedy" algorithm. For this structured grid the
# greedy algorithm uses fewer colors, but both algorithms result in colors that contain
# roughly the same number of elements. The workstream algorithm is the default one since it
# in general results in more balanced colors. For unstructured grids the greedy algorithm
# can result in colors with very few elements, for example.

using Ferrite, SparseArrays

function create_example_2d_grid()
    grid = generate_grid(Quadrilateral, (10, 10), Vec{2}((0.0, 0.0)), Vec{2}((10.0, 10.0)))
    colors_workstream = create_coloring(grid; alg = ColoringAlgorithm.WorkStream)
    colors_greedy = create_coloring(grid; alg = ColoringAlgorithm.Greedy)
    VTKGridFile("colored", grid) do vtk
        Ferrite.write_cell_colors(vtk, grid, colors_workstream, "workstream-coloring")
        Ferrite.write_cell_colors(vtk, grid, colors_greedy, "greedy-coloring")
    end
end

create_example_2d_grid()

# ![](coloring.png)
#
# *Figure 1*: Element coloring using the "workstream"-algorithm (left) and the "greedy"-
# algorithm (right).

# ## Multithreaded assembly of a cantilever beam in 3D
#
# We will now look at an example where we assemble the stiffness matrix and right hand side
# using multiple threads. The problem setup is a cantilever beam in 3D with a linear elastic
# material behavior. For this exercise we only focus on the multithreading and are not
# bothered with boundary conditions. For more details refer to the [tutorial on linear
# elasticity](../tutorials/linear_elasticity.md).

# ### Setup
#
# We define the element routine, material stiffness, grid and DofHandler just like in the
# [tutorial on linear elasticity](../tutorials/linear_elasticity.md) without discussing it
# further here.

## Element routine
function assemble_cell!(Ke::Matrix, fe::Vector, cellvalues::CellValues, C::SymmetricTensor, b::Vec)
    fill!(Ke, 0)
    fill!(fe, 0)
    for q_point in 1:getnquadpoints(cellvalues)
        dΩ = getdetJdV(cellvalues, q_point)
        for i in 1:getnbasefunctions(cellvalues)
            δui = shape_value(cellvalues, q_point, i)
            fe[i] += (δui ⋅ b) * dΩ
            ∇δui = shape_symmetric_gradient(cellvalues, q_point, i)
            for j in 1:getnbasefunctions(cellvalues)
                ∇uj = shape_symmetric_gradient(cellvalues, q_point, j)
                Ke[i, j] += (∇δui ⊡ C ⊡ ∇uj) * dΩ
            end
        end
    end
    return Ke, fe
end

## Material stiffness
function create_material_stiffness()
    E = 200.0e9
    ν = 0.3
    λ = E * ν / ((1 + ν) * (1 - 2ν))
    μ = E / (2(1 + ν))
    δ(i, j) = i == j ? 1.0 : 0.0
    C = SymmetricTensor{4, 3}() do i, j, k, l
        return λ * δ(i, j) * δ(k, l) + μ * (δ(i, k) * δ(j, l) + δ(i, l) * δ(j, k))
    end
    return C
end

## Grid and grid coloring
function create_cantilever_grid(n::Int)
    xmin = Vec{3}((0.0, 0.0, 0.0))
    xmax = Vec{3}((10.0, 1.0, 1.0))
    grid = generate_grid(Hexahedron, (10 * n, n, n), xmin, xmax)
    colors = create_coloring(grid)
    return grid, colors
end

## DofHandler with displacement field u
function create_dofhandler(grid::Grid, interpolation::VectorInterpolation)
    dh = DofHandler(grid)
    add!(dh, :u, interpolation)
    close!(dh)
    return dh
end
nothing # hide

# ### Task local scratch data
#
# We group everything that needs to be duplicated for each task in the struct
# `ScratchData`:
#  - `cell_cache::CellCache`: contain buffers for coordinates and (global) dofs which will
#    be `reinit!`ed for each cell.
#  - `cellvalues::CellValues`: the cell values which will be `reinit!`ed for each cell using
#    the `cell_cache`
#  - `Ke::Matrix`: the local matrix
#  - `fe::Vector`: the local vector
#  - `assembler`: the assembler (which needs to be duplicated because it contains buffers
#    that are modified during the call to `assemble!`)
struct ScratchData{CC, CV, T, A}
    cell_cache::CC
    cellvalues::CV
    Ke::Matrix{T}
    fe::Vector{T}
    assembler::A
end

# This constructor will be called within each task to create a independent `ScratchData`
# object. For `cell_cache`, `Ke`, and `fe` we simply call the constructors to allocate
# independent objects. For `cellvalues` we use `copy` which Ferrite defines for this
# purpose. Finally, for the assembler we call `start_assemble` to create a new assembler but
# note that we set `fillzero = false` because we don't want to risk that a task that starts
# a bit later will zero out data that another task have already assembled.
function Ferrite.task_local(scratch::ScratchData)
    ScratchData(
        task_local(scratch.cell_cache), task_local(scratch.cellvalues),
        task_local(scratch.Ke), task_local(scratch.fe),
        task_local(scratch.assembler)
    )
end
nothing # hide

# ### Global assembly routine

# Finally we define the global assemble routine, which is where the parallelization happens.
# The main difference from all previous `assemble_global!` functions is that we now have an
# outer loop over the colors, and then the inner loop over the cells in each color, which
# can be parallelized.
#
# For the scheduling of parallel tasks we use the
# [OhMyThreads.jl](https://github.com/JuliaFolds2/OhMyThreads.jl) package. OhMyThreads
# provides a macro based and a functional API. Here we use the macro based API because it is
# slightly more convenient when using task local values since they can be defined with the
# `@local` macro.
#
# !!! note "Schedulers and load balancing"
#     OhMyThreads provides a number of different
#     [schedulers](https://juliafolds2.github.io/OhMyThreads.jl/stable/refs/api/#Schedulers).
#     In this example we use the `DynamicScheduler` (which is the default one). The
#     `DynamicScheduler` will spawn `ntasks` tasks where each task will process a chunk of
#     (roughly) equal number of cells (i.e. `length(color) ÷ ntasks`). This should be a good
#     choice for this example because we expect all cells to take the same time to process
#     and we don't need any load balancing.
#
#     For a different problem setup where some cells might take longer to process (perhaps
#     they experience plastic deformation and we need to solve a local problem) we might
#     benefit from load balancing. The `DynamicScheduler` can be used also for load
#     balancing by specifiying `nchunks` or `chunksize`. However, the `DynamicScheduler`
#     will always spawn `nchunks` tasks which can become costly since we are allocating
#     scratch data for every task. To limit the number of tasks, while allowing for more
#     than `ntasks` chunks, we can use the `GreedyScheduler` *with chunking*. For example,
#     `scheduler = OhMyThreads.GreedyScheduler(; ntasks = ntasks, nchunks = 10 * ntasks)`
#     will split the work into `10 * ntasks` chunks and spawn `ntasks` tasks to process
#     them. Refer to the [OhMyThreads
#     documentation](https://juliafolds2.github.io/OhMyThreads.jl/stable/) for details.

using OhMyThreads, TaskLocalValues

function assemble_global!(
        K::SparseMatrixCSC, f::Vector, dh::DofHandler, colors,
        cellvalues_template::CellValues; ntasks = Threads.nthreads()
    )
    ## Body force and material stiffness
    b = Vec{3}((0.0, 0.0, -1.0))
    C = create_material_stiffness()
    ## Scratch data
    scratch_template = ScratchData(
        CellCache(dh), cellvalues_template,
        zeros(ndofs_per_cell(dh), ndofs_per_cell(dh)), zeros(ndofs_per_cell(dh)),
        start_assemble(K, f)
    )
    ## Loop over the colors
    for color in colors
        ## Dynamic scheduler spawning `ntasks` tasks where each task will process a chunk of
        ## (roughly) equal number of cells (`length(color) ÷ ntasks`).
        scheduler = OhMyThreads.DynamicScheduler(; ntasks)
        ## Parallelize the loop over the cells in this color
        OhMyThreads.@tasks for cellidx in color
            ## Tell the @tasks loop to use the scheduler defined above
            @set scheduler = scheduler
            ## Obtain a task local scratch and unpack it
            @local scratch = task_local(scratch_template)
            (; cell_cache, cellvalues, Ke, fe, assembler) = scratch
            ## Reinitialize the cell cache and then the cellvalues
            reinit!(cell_cache, cellidx)
            reinit!(cellvalues, cell_cache)
            ## Compute the local contribution of the cell
            assemble_cell!(Ke, fe, cellvalues, C, b)
            ## Assemble local contribution
            assemble!(assembler, celldofs(cell_cache), Ke, fe)
        end
    end
    return K, f
end
nothing # hide

# !!! details "OhMyThreads functional API: OhMyThreads.tforeach"
#     The `OhMyThreads.@tasks` block above corresponds to a call to `OhMyThreads.tforeach`.
#     Using the functional API directly would look like below. The main difference is that
#     we need to manually create a `TaskLocalValue` for the scratch data.
#     ```julia
#     # using TaskLocalValues
#     scratches = TaskLocalValue() do
#         task_local(scratch_template)
#     end
#     OhMyThreads.tforeach(color; scheduler) do cellidx
#         # Obtain a task local scratch and unpack it
#         scratch = scratches[]
#         (; cell_cache, cellvalues, Ke, fe, assembler) = scratch
#         # Reinitialize the cell cache and then the cellvalues
#         reinit!(cell_cache, cellidx)
#         reinit!(cellvalues, cell_cache)
#         # Compute the local contribution of the cell
#         assemble_cell!(Ke, fe, cellvalues, C, b)
#         # Assemble local contribution
#         assemble!(assembler, celldofs(cell_cache), Ke, fe)
#     end
#     ```

# We define the main function to setup everything and then time the call to
# `assemble_global!`.

function main(; n = 20, ntasks = Threads.nthreads())
    ## Interpolation, quadrature and cellvalues
    interpolation = Lagrange{RefHexahedron, 1}()^3
    quadrature = QuadratureRule{RefHexahedron}(2)
    cellvalues = CellValues(quadrature, interpolation)
    ## Grid, colors and DofHandler
    grid, colors = create_cantilever_grid(n)
    dh = create_dofhandler(grid, interpolation)
    ## Global matrix and vector
    K = allocate_matrix(dh)
    f = zeros(ndofs(dh))
    ## Compile it
    assemble_global!(K, f, dh, colors, cellvalues; ntasks = ntasks)
    ## Time it
    @time assemble_global!(K, f, dh, colors, cellvalues; ntasks = ntasks)
    return norm(K.nzval), norm(f) #src
    return
end
nothing # hide

# On a machine with 4 cores, starting julia with `--threads=auto`, we obtain the following
# timings:
# ```julia
# main(; ntasks = 1) # 1.970784 seconds (902 allocations: 816.172 KiB)
# main(; ntasks = 2) # 1.025065 seconds (1.64 k allocations: 1.564 MiB)
# main(; ntasks = 3) # 0.700423 seconds (2.38 k allocations: 2.332 MiB)
# main(; ntasks = 4) # 0.548356 seconds (3.12 k allocations: 3.099 MiB)
# ```

using Test                           #src
nK1, nf1 = main(; n = 5, ntasks = 1) #src
nK2, nf2 = main(; n = 5, ntasks = 2) #src
nK4, nf4 = main(; n = 5, ntasks = 4) #src
@test nK1 == nK2 == nK4              #src
@test nf1 == nf2 == nf4              #src

#md # ## [Plain program](@id threaded_assembly-plain-program)
#md #
#md # Here follows a version of the program without any comments.
#md # The file is also available here: [`threaded_assembly.jl`](threaded_assembly.jl).
#md #
#md # ```julia
#md # @__CODE__
#md # ```
