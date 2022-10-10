# # Distributed Assembly of Heat Equation
#
# ## Introduction
#
# Now we want to solve the heat problem in parallel. To be specific, this example shows
# how to utilize process parallelism to assemble finite element matrices in parallel.
# This example presumes that the reader is familiar with solving the heat problem in
# serial with Ferrite.jl, as presented in [the first example](@ref heat_example).
#
#-
# ## Commented Program
#
# Now we solve the problem in Ferrite. What follows is a program spliced with comments.
#md # The full program, without comments, can be found in the next [section](@ref heat_equation-plain-program).
#
# First we load Ferrite, and some other packages we need
using Ferrite, MPI
using IterativeSolvers #, HYPRE
using PartitionedArrays #src

# Launch MPI
MPI.Init()

# We start generating a simple grid with 20x20 quadrilateral elements
# and distribute it across our processors using `generate_distributed_grid`. 
dgrid = generate_distributed_grid(Quadrilateral, (20, 20));

# ### Trial and test functions
# Nothing changes here.
dim = 2
ip = Lagrange{dim, RefCube, 1}()
ip_geo = Lagrange{dim, RefCube, 1}()
qr = QuadratureRule{dim, RefCube}(3)
cellvalues = CellScalarValues(qr, ip, ip_geo);

# ### Degrees of freedom
# To handle the dofs correctly we now utilize the `DistributedDofHandle` 
# instead of the `DofHandler`. For the user the interface is the same.
dh = DistributedDofHandler(dgrid)
push!(dh, :u, 1, ip)
close!(dh);

# ### Boundary conditions
# Nothing has to be changed here either.
ch = ConstraintHandler(dh);
∂Ω = union(getfaceset.((getlocalgrid(dgrid), ), ["left", "right", "top", "bottom"])...);
dbc = Dirichlet(:u, ∂Ω, (x, t) -> 0)
add!(ch, dbc);
close!(ch)
update!(ch, 0.0);

# ### Assembling the linear system
# Assembling the system works also mostly analogue.
function doassemble(cellvalues::CellScalarValues{dim}, dh::DistributedDofHandler) where {dim}
    n_basefuncs = getnbasefunctions(cellvalues)
    Ke = zeros(n_basefuncs, n_basefuncs)
    fe = zeros(n_basefuncs)

    # --------------------- Distributed assembly --------------------
    # The synchronization with the global sparse matrix is handled by 
    # an assembler again. You can choose from different backends, which
    # are described in the docs and will be expaned over time. This call
    # may trigger a large amount of communication.
    # NOTE: At the time of writing the only backend available is a COO 
    #       assembly via PartitionedArrays.jl .
    assembler = PartitionedArraysCOOAssembler{Float64}(dh)

    # For the local assembly nothing changes
    for cell in CellIterator(dh)
        fill!(Ke, 0)
        fill!(fe, 0)

        reinit!(cellvalues, cell)
        coords = getcoordinates(cell)
                
        for q_point in 1:getnquadpoints(cellvalues)
            dΩ = getdetJdV(cellvalues, q_point)
            
            for i in 1:n_basefuncs
                v  = shape_value(cellvalues, q_point, i)
                ∇v = shape_gradient(cellvalues, q_point, i)
                # Manufactured solution of Π cos(xᵢπ)
                x = spatial_coordinate(cellvalues, q_point, coords)
                fe[i] += (π/2)^2 * dim * prod(cos, x*π/2) * v * dΩ

                for j in 1:n_basefuncs
                    ∇u = shape_gradient(cellvalues, q_point, j)
                    Ke[i, j] += (∇v ⋅ ∇u) * dΩ
                end
            end
        end

        # Note that this call should be communication-free!
        Ferrite.assemble!(assembler, celldofs(cell), fe, Ke)
    end

    # Finally, for the `PartitionedArraysCOOAssembler` we have to call
    # `end_assemble` to construct the global sparse matrix and the global
    # right hand side vector.
    return end_assemble(assembler)
end
#md nothing # hide

# ### Solution of the system
# Again, we assemble our problem and apply the constraints as needed.
K, f = doassemble(cellvalues, dh);
apply!(K, f, ch)

# To compute the solution we utilize conjugate gradients because at the time of writing 
# this is the only available scalable working solver.
# Additional note: At the moment of writing this we have no good preconditioners for PSparseMatrix in Julia, 
# partly due to unimplemented multiplication operators for the matrix data type.
u = cg(K, f)

#FIXME #src
# Compute the solution with HYPRE (needs the hotfix in https://github.com/fredrikekre/HYPRE.jl/pull/4 to function partially) #src
# u_ = HYPRE.solve(                         #src
#     HYPRE.PCG(                            #src
#         global_comm(dgrid);               #src
#         Precond = HYPRE.BoomerAMG()       #src
#     ),                                    #src
#     HYPRE.HYPREMatrix(K),                 #src
#     HYPRE.HYPREVector(f)                  #src
# )

# Convert back to PartitionedArrays vector #src
# u = PVector(0.0, K.cols) #src
# copy!(u, u_) #src
# PartitionedArrays.assemble!(u) #src

# ### Exporting via PVTK
# To visualize the result we export the grid and our field `u`
# to a VTK-file, which can be viewed in e.g. [ParaView](https://www.paraview.org/).
vtk_grid("heat_equation_distributed", dh) do vtk
    vtk_point_data(vtk, dh, u)
    # For debugging purposes it can be helpful to enrich 
    # the visualization with some meta  information about 
    # the grid and its partitioning
    vtk_shared_vertices(vtk, dgrid)
    vtk_partitioning(vtk, dgrid)
end

## Test the result against the manufactured solution                    #src
using Test                                                              #src
for cell in CellIterator(dh)                                            #src
    reinit!(cellvalues, cell)                                           #src
    n_basefuncs = getnbasefunctions(cellvalues)                         #src
    coords = getcoordinates(cell)                                       #src
    map_parts(local_view(u, u.rows)) do u_local                         #src
        uₑ = u_local[celldofs(cell)]                                    #src
        for q_point in 1:getnquadpoints(cellvalues)                     #src
            x = spatial_coordinate(cellvalues, q_point, coords)         #src
            for i in 1:n_basefuncs                                      #src
                uₐₙₐ    = prod(cos, x*π/2)                              #src
                uₐₚₚᵣₒₓ = function_value(cellvalues, q_point, uₑ)       #src
                @test isapprox(uₐₙₐ, uₐₚₚᵣₒₓ; atol=1e-1)                #src
            end                                                         #src
        end                                                             #src
    end                                                                 #src
end                                                                     #src

# Finally, we gracefully shutdown MPI
MPI.Finalize()

#md # ## [Plain program](@id distributed-assembly-plain-program)
#md #
#md # Here follows a version of the program without any comments.
#md # The file is also available here: [`distributed_assembly.jl`](distributed_assembly.jl).
#md #
#md # ```julia
#md # @__CODE__
#md # ```
