# # [Multithreaded assembly](@id howto-threaded-assembly)
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
#       element contributions) would then hand off their results to the assembly task. This
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
#     - **Atomic accumulation**: By accumulating the values into the global matrix and
#       vector with atomic additions we can let all tasks assemble concurrently without
#       any partitioning of the cells: even if two tasks add to the same memory location
#       at the same time the result is correct. Similar to the locking approach the
#       results are not deterministic since the order of the additions depend on the task
#       scheduling. See [the last section](@ref howto-threaded-assembly-atomic) of this
#       howto for more details.
#  - **Scratch data**: In order to speed up the computation of the element contributions we
#    typically pre-allocate some data structures that can be reused for every element. Such
#    scratch data include, for example, the local matrix and vector, and the CellValues.
#    Each task need their own copy of the scratch data since they will be modified for each
#    element.

# ## Grid coloring
#
# Ferrite include functionality to color the grid with the [`create_coloring`](@ref)
# function. Here we create a simple 2D grid, color it, and export the colors to a VTK file
# to visualize the result (see [*Figure 1*](@ref howto-threaded-assembly-figure-1)). Note that
# no cells with the same color has any
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
    return
end

create_example_2d_grid()

# ![](coloring-light.png)
# ![](coloring-dark.png)
#
# [[*Figure 1*](@ref howto-threaded-assembly-figure-1)](@id howto-threaded-assembly-figure-1):
# Element coloring using the "workstream"-algorithm (left) and the "greedy"-
# algorithm (right). The swatches below each grid are the colors the algorithm used.

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
# Some of the data used during assembly is modified for each cell, and each task therefore
# needs its own instance of it:
#  - `cell_cache::CellCache`: contain buffers for coordinates and (global) dofs which will
#    be `reinit!`ed for each cell.
#  - `cellvalues::CellValues`: the cell values which will be `reinit!`ed for each cell using
#    the `cell_cache`
#  - `Ke::Matrix`: the local matrix
#  - `fe::Vector`: the local vector
#  - `assembler`: the assembler (which needs to be duplicated because it contains buffers
#    that are modified during the call to `assemble!`)
#
# We create all of these outside of the parallel loop, just like for serial assembly, and
# then each task creates its own instances from these using [`task_local_copy`](@ref).
# `task_local_copy` is similar to `copy`, but only the data that is modified during
# assembly ("scratch data") is duplicated: for example, the duplicated `CellValues` get
# their own internal buffers, and the duplicated assembler gets its own internal buffers
# but still assembles into the same global matrix `K` and vector `f`. The task local data
# is created with the `@local` macro in the parallel loop below. Note that the right hand
# side of `@local x = ...` is evaluated in the scope outside of the loop, so `@local
# assembler = task_local_copy(assembler)` creates a task local copy of the `assembler`
# defined outside of the loop.

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
# Since the parallel loop is nested inside of the loop over the colors, task local values
# created with `@local` directly would be recreated for every color. To avoid this we
# create the scratch data for all tasks once, before the loop over the colors, and then
# use `@local` together with `@task_index` to give each task its own instance. `@task_index`
# is the index of the task, i.e. an integer in `1:ntasks`.
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
#     balancing by specifying `nchunks` or `chunksize`. However, the `DynamicScheduler`
#     will always spawn one task per chunk (`nchunks` and `ntasks` are aliases for the
#     `DynamicScheduler`), which can become costly since we are allocating scratch data for
#     every task. Note also that `@task_index` then goes up to the number of chunks, so the
#     scratch data must be created for that many tasks. To limit the number of tasks, while
#     allowing for more than `ntasks` chunks, we can use the `GreedyScheduler` *with
#     chunking*, for which `@task_index` is always an integer in `1:ntasks`. For example,
#     `scheduler = OhMyThreads.GreedyScheduler(; ntasks = ntasks, nchunks = 10 * ntasks)`
#     will split the work into `10 * ntasks` chunks and spawn `ntasks` tasks to process
#     them. Refer to the [OhMyThreads
#     documentation](https://juliafolds2.github.io/OhMyThreads.jl/stable/) for details.

using OhMyThreads

function assemble_global!(
        K::SparseMatrixCSC, f::Vector, dh::DofHandler, colors,
        cellvalues::CellValues; ntasks = Threads.nthreads()
    )
    ## Body force and material stiffness
    b = Vec{3}((0.0, 0.0, -1.0))
    C = create_material_stiffness()
    ## Create the assembler. Note that `start_assemble` zeroes out existing data in K and f,
    ## but since this is only done once, before the parallel loop, it is safe.
    assembler = start_assemble(K, f)
    ## Cell cache and local matrix and vector
    cell_cache = CellCache(dh)
    n = ndofs_per_cell(dh)
    Ke = zeros(n, n)
    fe = zeros(n)
    ## Scratch data for each task, created once and reused for all colors
    scratches = map(1:ntasks) do _
        return (;
            cell_cache = task_local_copy(cell_cache),
            cellvalues = task_local_copy(cellvalues),
            Ke = task_local_copy(Ke), fe = task_local_copy(fe),
            assembler = task_local_copy(assembler),
        )
    end
    ## Loop over the colors
    for color in colors
        ## Dynamic scheduler spawning `ntasks` tasks where each task will process a chunk of
        ## (roughly) equal number of cells (`length(color) ÷ ntasks`).
        scheduler = OhMyThreads.DynamicScheduler(; ntasks)
        ## Parallelize the loop over the cells in this color
        OhMyThreads.@tasks for cellidx in color
            ## Tell the @tasks loop to use the scheduler defined above
            @set scheduler = scheduler
            ## Obtain the scratch data for this task and unpack it. Note that `local` is
            ## required since the variables shadow variables outside of the loop.
            @local scratch = scratches[@task_index]
            local (; cell_cache, cellvalues, Ke, fe, assembler) = scratch
            ## Reinitialize the cell cache and then the cellvalues
            reinit!(cell_cache, cellidx)
            reinit!(cellvalues, cell_cache)
            fill!(Ke, 0)
            fill!(fe, 0)
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
#     the function is wrapped in `OhMyThreads.WithTaskIndex` to obtain the task index as
#     the first argument.
#     ```julia
#     OhMyThreads.tforeach(
#         OhMyThreads.WithTaskIndex() do taskindex, cellidx
#             # Obtain the scratch data for this task and unpack it. Note that `local` is
#             # required here: without it the assignment would overwrite the variables with
#             # the same names outside of the closure, which are shared by all tasks.
#             local (; cell_cache, cellvalues, Ke, fe, assembler) = scratches[taskindex]
#             # Reinitialize the cell cache and then the cellvalues
#             reinit!(cell_cache, cellidx)
#             reinit!(cellvalues, cell_cache)
#             fill!(Ke, 0)
#             fill!(fe, 0)
#             # Compute the local contribution of the cell
#             assemble_cell!(Ke, fe, cellvalues, C, b)
#             # Assemble local contribution
#             assemble!(assembler, celldofs(cell_cache), Ke, fe)
#         end,
#         color; scheduler
#     )
#     ```

# ### [Assembly without coloring: Atomic accumulation](@id howto-threaded-assembly-atomic)
#
# The grid coloring above ensures that no two concurrently running tasks write to the
# same entries of `K` and `f`. An alternative is to allow concurrent writes, but make the
# additions atomic: passing `atomic = true` to `start_assemble` returns an assembler
# where the accumulation into the global matrix and vector uses atomic instructions.
# Compared to the coloring approach this has some advantages:
#  - No coloring of the grid is needed. The coloring computation itself takes time (often
#    comparable to one assembly) and for some grids/couplings it can be hard to obtain a
#    good coloring.
#  - The cells can be processed in their natural order, in a single parallel loop. This
#    gives better memory locality (the cells of one color are by construction spread out
#    over the grid) which can make assembly faster even when comparing at the same number
#    of tasks.
# and some drawbacks:
#  - Atomic additions are more expensive than regular ones, which cost a few percent in
#    the serial limit for a typical assembly loop (more if the element routine is very
#    cheap relative to the scatter into the global matrix).
#  - The result is not deterministic: the order in which contributions are added to a
#    given entry of `K` and `f` depends on the task scheduling, and since floating point
#    addition is not associative the result changes slightly between runs (this is the
#    same drawback as the locking and assembler task approaches have, whereas the
#    coloring approach is deterministic).
#
# Note that each task still needs its own assembler (the assembler wraps buffers that are
# used during `assemble!`). Calling `task_local_copy` on an atomic assembler returns a new
# atomic assembler. The global assembly routine is like before, but with an atomic
# assembler and a single loop over all cells. Since there is no outer loop over colors the
# task local data can be created directly with `@local` and `task_local_copy`:

function assemble_global_atomic!(
        K::SparseMatrixCSC, f::Vector, dh::DofHandler,
        cellvalues::CellValues; ntasks = Threads.nthreads()
    )
    ## Body force and material stiffness
    b = Vec{3}((0.0, 0.0, -1.0))
    C = create_material_stiffness()
    ## Create an atomic assembler (this also zeroes out existing data in K and f)
    assembler = start_assemble(K, f; atomic = true)
    ## Cell cache and local matrix and vector
    cell_cache = CellCache(dh)
    n = ndofs_per_cell(dh)
    Ke = zeros(n, n)
    fe = zeros(n)
    scheduler = OhMyThreads.DynamicScheduler(; ntasks)
    ## Parallelize the loop over all cells
    OhMyThreads.@tasks for cellidx in 1:getncells(dh.grid)
        ## Tell the @tasks loop to use the scheduler defined above
        @set scheduler = scheduler
        ## Task local scratch data (with an atomic assembler), created once for each task
        @local begin
            cell_cache = task_local_copy(cell_cache)
            cellvalues = task_local_copy(cellvalues)
            Ke = task_local_copy(Ke)
            fe = task_local_copy(fe)
            assembler = task_local_copy(assembler)
        end
        ## Reinitialize the cell cache and then the cellvalues
        reinit!(cell_cache, cellidx)
        reinit!(cellvalues, cell_cache)
        fill!(Ke, 0)
        fill!(fe, 0)
        ## Compute the local contribution of the cell
        assemble_cell!(Ke, fe, cellvalues, C, b)
        ## Assemble local contribution
        assemble!(assembler, celldofs(cell_cache), Ke, fe)
    end
    return K, f
end
nothing # hide

# We define the main function to setup everything and then time the calls to
# `assemble_global!` and `assemble_global_atomic!`.

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
    ## Compile and time the colored version
    assemble_global!(K, f, dh, colors, cellvalues; ntasks = ntasks)
    @time assemble_global!(K, f, dh, colors, cellvalues; ntasks = ntasks)
    nK, nf = norm(K.nzval), norm(f) #src
    ## Compile and time the atomic version
    assemble_global_atomic!(K, f, dh, cellvalues; ntasks = ntasks)
    @time assemble_global_atomic!(K, f, dh, cellvalues; ntasks = ntasks)
    return nK, nf, norm(K.nzval), norm(f) #src
    return
end
nothing # hide

# On a machine with 10 (performance) cores, starting julia with `--threads=auto`, we
# obtain the following timings, where the first `@time` is the colored version and the
# second the atomic version:
# ```julia
# main(; ntasks = 1) # 0.820127 seconds (596 allocations: 688.125 KiB)
#                    # 0.734474 seconds (42 allocations: 43.156 KiB)
# main(; ntasks = 2) # 0.414206 seconds (1.75 k allocations: 1.398 MiB)
#                    # 0.364351 seconds (114 allocations: 89.625 KiB)
# main(; ntasks = 4) # 0.211400 seconds (3.12 k allocations: 2.759 MiB)
#                    # 0.188947 seconds (200 allocations: 176.703 KiB)
# main(; ntasks = 8) # 0.112129 seconds (5.88 k allocations: 5.483 MiB)
#                    # 0.096397 seconds (372 allocations: 350.859 KiB)
# ```
# Both versions scale well with the number of tasks. In this example the atomic version
# is even a bit faster than the colored version: the atomic overhead in the accumulation
# is more than compensated for by processing the cells in their natural order (and, for
# the timings above, the atomic version doesn't pay for the grid coloring itself, which
# for this grid takes about 0.3 seconds).

using Test                                               #src
nK1, nf1, aK1, af1 = main(; n = 5, ntasks = 1)           #src
nK2, nf2, aK2, af2 = main(; n = 5, ntasks = 2)           #src
nK4, nf4, aK4, af4 = main(; n = 5, ntasks = 4)           #src
## Coloring gives deterministic results                  #src
@test nK1 == nK2 == nK4                                  #src
@test nf1 == nf2 == nf4                                  #src
## Atomic accumulation is exact up to summation order    #src
@test aK1 ≈ nK1                                          #src
@test af1 ≈ nf1                                          #src
@test aK2 ≈ nK1                                          #src
@test af2 ≈ nf1                                          #src
@test aK4 ≈ nK1                                          #src
@test af4 ≈ nf1                                          #src

#md # ## [Plain program](@id threaded_assembly-plain-program)
#md #
#md # Here follows a version of the program without any comments.
#md # The file is also available here: [`threaded_assembly.jl`](threaded_assembly.jl).
#md #
#md # ```julia
#md # @__CODE__
#md # ```
