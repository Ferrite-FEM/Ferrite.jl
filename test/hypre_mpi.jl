# Run with e.g.
#   mpiexecjl --project=docs -np 4 julia test/hypre_mpi.jl 1000
using Ferrite, MPI, HYPRE, Metis, TimerOutputs

# Initialize MPI and HYPRE
MPI.Init()
HYPRE.Init()

const comm = MPI.COMM_WORLD
const root = 0 + 1
const rank = MPI.Comm_rank(comm) + 1
const comm_size = MPI.Comm_size(comm)

# No changes from serial solve
function assemble_element!(Ke::Matrix, fe::Vector, cellvalues::CellValues)
    n_basefuncs = getnbasefunctions(cellvalues)
    fill!(Ke, 0)
    fill!(fe, 0)
    for q_point in 1:getnquadpoints(cellvalues)
        dΩ = getdetJdV(cellvalues, q_point)
        for i in 1:n_basefuncs
            δu = shape_value(cellvalues, q_point, i)
            ∇δu = shape_gradient(cellvalues, q_point, i)
            fe[i] += δu * dΩ
            for j in 1:n_basefuncs
                ∇u = shape_gradient(cellvalues, q_point, j)
                Ke[i, j] += (∇δu ⋅ ∇u) * dΩ
            end
        end
    end
    return Ke, fe
end

# No changes from serial solve other than looping over owned cells
function assemble_global(cellvalues::CellValues, A::HYPREMatrix, b::HYPREVector, dh::DofHandler, ch::ConstraintHandler)
    n_basefuncs = getnbasefunctions(cellvalues)
    Ke = zeros(n_basefuncs, n_basefuncs)
    fe = zeros(n_basefuncs)
    assembler = start_assemble(A, b)
    for cell in CellIterator(dh, getcellset(dh.grid, "proc-$(rank)"))
        reinit!(cellvalues, cell)
        assemble_element!(Ke, fe, cellvalues)
        apply_assemble!(assembler, ch, celldofs(cell), Ke, fe)
    end
    # TODO: Should maybe be finish_assemble! (with !)
    finish_assemble(assembler)
    return A, b
end

# Partition the grid using Metis.jl
function partition_grid!(grid)
    # TODO: Can this be done on all ranks? Not sure if Metis is deterministic.
    if rank == root
        cell_connectivity = Ferrite.create_incidence_matrix(grid)
        parts = Metis.partition(cell_connectivity, comm_size)
    else
        parts = Vector{Cint}(undef, getncells(grid))
    end
    MPI.Bcast!(parts, comm)

    # Create the cell sets based on the Metis partition
    sets = [Set{Int}() for _ in 1:comm_size]
    for (cell_id, part_id) in pairs(parts)
        push!(sets[part_id], cell_id)
    end
    for p in 1:comm_size
        addcellset!(grid, "proc-$p", sets[p])
    end
    return grid
end

function main(n)

    reset_timer!()

    # FE Values
    ip = Lagrange{RefQuadrilateral, 1}()
    qr = QuadratureRule{RefQuadrilateral}(2)
    cellvalues = CellValues(qr, ip)

    # Create the grid
    @timeit "Generate grid" grid = generate_grid(Quadrilateral, (n, n))

    # Partition the mesh
    @timeit "Partition grid" partition_grid!(grid)

    # Create the DofHandler
    @timeit "Create DofHandler" begin
        dh = DofHandler(grid)
        add!(dh, :u, ip)
        close!(dh)
    end

    # Renumber dofs by part
    @timeit "Renumber DoFs by processor" begin
        all = Set{Int}()
        sets = [Set{Int}() for _ in 1:comm_size]
        cc = CellCache(dh)
        for p in 1:comm_size
            set = sets[p]
            for cell_id in getcellset(grid, "proc-$p")
                reinit!(cc, cell_id)
                union!(set, cc.dofs)
            end
            setdiff!(set, all)
            union!(all, set)
        end
        iperm = Int[]
        rank_dof_ranges = UnitRange{Int}[]
        for set in sets
            push!(rank_dof_ranges, (length(iperm) + 1):(length(iperm) + length(set)))
            append!(iperm, sort!(collect(set)))
        end
        perm = invperm(iperm)
        renumber!(dh, perm)
        rank_dof_range = rank_dof_ranges[rank]
    end


    # ConstraintHandler
    @timeit "Create ConstraintHandler" begin
        ch = ConstraintHandler(dh)
        ∂Ω = union(
            getfacetset(grid, "left"),
            getfacetset(grid, "right"),
            getfacetset(grid, "top"),
            getfacetset(grid, "bottom"),
        )
        dbc = Dirichlet(:u, ∂Ω, (x, t) -> 0)
        add!(ch, dbc)
        close!(ch)
    end


    # Set up HYPRE arrays
    ilower, iupper = extrema(rank_dof_range)
    A = HYPREMatrix(comm, ilower, iupper)
    b = HYPREVector(comm, ilower, iupper)

    # Assemble
    @timeit "Assembly ($(length(getcellset(grid, "proc-$(rank)"))) of $(getncells(grid)) elements)" begin
        assemble_global(cellvalues, A, b, dh, ch)
    end

    # Set up solver and solve
    @timeit "HYPRE setup and solve" begin
        precond = HYPRE.BoomerAMG()
        solver = HYPRE.PCG(; Precond = precond)
        xh = HYPRE.solve(solver, A, b)
    end

    # Copy solution from HYPRE to Julia
    @timeit "Collect solution to root for VTK output" begin
        x = Vector{Float64}(undef, length(rank_dof_range))
        copy!(x, xh)

        # Collect to root rank
        if rank == root
            X = Vector{Float64}(undef, ndofs(dh))
            counts = length.(rank_dof_ranges)
            MPI.Gatherv!(x, VBuffer(X, counts), comm)
        else
            MPI.Gatherv!(x, nothing, comm)
        end
    end

    # Exporting to VTK
    if rank == root
        @timeit "VTK export" begin
            VTKGridFile("heat_equation", dh) do vtk
                write_solution(vtk, dh, X)
            end
        end
    end

    # Print the timer on root proc
    rank == root && print_timer()

    return
end

# Run it!
if abspath(PROGRAM_FILE) == @__FILE__
    n = parse(Int, get(ARGS, 1, "100"))
    main(n)
    main(n)
end
