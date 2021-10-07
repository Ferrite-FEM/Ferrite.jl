# # Postprocessing
#
# ![](heat_square_fluxes.png)


# ## Introduction
#
# After running a simulation, we usually want to visualize the results in different ways.
# The `L2Projector` and the `PointEvalHandler` build a pipeline for doing so. With the `L2Projector`,
# integration point quantities can be projected to the nodes. The `PointEvalHandler` enables evaluation of
# the finite element approximated function in any coordinate in the domain. Thus with the combination of both functionalities,
# both nodal quantities and integration point quantities can be evaluated in any coordinate, allowing for example
# cut-planes through 3D structures or cut-lines through 2D-structures.
#
# This example continues from the Heat equation example, where the temperature field was
# determined on a square domain. In this example, we first compute the heat flux in each
# integration point (based on the solved temperature field) and then we do an L2-projection
# of the fluxes to the nodes of the mesh. By doing this, we can more easily visualize
# integration points quantities. Finally, we visualize the temperature field and the heat fluxes along a cut-line.
#
# The L2-projection is defined as follows: Find projection ``q(\boldsymbol{x}) \in L_2(\Omega)`` such that
# ```math
# \int v q \ \mathrm{d}\Omega = \int v d \ \mathrm{d}\Omega \quad \forall v \in L_2(\Omega),
# ```
# where ``d`` is the quadrature data to project. Since the flux is a vector the projection function
# will be solved with multiple right hand sides, e.g. with ``d = q_x`` and ``d = q_y`` for this 2D problem.
#
# Ferrite has functionality for doing much of this automatically, as displayed in the code below.
# In particular [`L2Projector`](@ref) for assembling the left hand side, and
# [`project`](@ref) for assembling the right hand sides and solving for the projection.

# ## Implementation
#
# Start by simply running the Heat equation example to solve the problem
include("heat_equation.jl");


# Next we define a function that computes the heat flux for each integration point in the domain.
# Fourier's law is adopted, where the conductivity tensor is assumed to be isotropic with unit
# conductivity ``\lambda = 1 ⇒ q = - \nabla u``, where ``u`` is the temperature.
function compute_heat_fluxes(cellvalues::CellScalarValues{dim,T}, dh::DofHandler, a) where {dim,T}

    n = getnbasefunctions(cellvalues)
    cell_dofs = zeros(Int, n)
    nqp = getnquadpoints(cellvalues)

    ## Allocate storage for the fluxes to store
    q = [Vec{2,T}[] for _ in 1:getncells(dh.grid)]

    for (cell_num, cell) in enumerate(CellIterator(dh))
        q_cell = q[cell_num]
        celldofs!(cell_dofs, dh, cell_num)
        aᵉ = a[cell_dofs]
        reinit!(cellvalues, cell)

        for q_point in 1:nqp
            q_qp = - function_gradient(cellvalues, q_point, aᵉ)
            push!(q_cell, q_qp)
        end
    end
    return q
end
#md nothing # hide

# Now call the function to get all the fluxes.
q_gp = compute_heat_fluxes(cellvalues, dh, u);

# Next, create an [`L2Projector`](@ref) using the same interpolation as was used to approximate the
# temperature field. On instantiation, the projector assembles the coefficient matrix `M` and
# computes the Cholesky factorization of it. By doing so, the projector can be reused without
# having to invert `M` every time.
projector = L2Projector(ip, grid);

# Project the integration point values to the nodal values
q_nodes = project(projector, q_gp, qr);


# ## Exporting to VTK
# To visualize the heat flux, we export the projected field `q_nodes`
# to a VTK-file, which can be viewed in e.g. [ParaView](https://www.paraview.org/).
vtk_grid("heat_equation_flux", grid) do vtk
    vtk_point_data(vtk, q_nodes, "q")
end;

# ## Point Evaluation
# ![](heat_square_pointevaluation.png)

# Consider a cut-line through the domain, like the black line in the figure above.
# We will evaluate the temperature and the heat flux distribution along a horizontal line.
points = [Vec((x, 0.75)) for x in range(-1.0, 1.0, length=101)];

# First, we need to generate a `PointEvalHandler`
ph = PointEvalHandler(dh, points);

# After the L2-Projection, the heat fluxes are stored in `q_nodes` in nodal order. We can extract the heat fluxes in the `points` as follows:
q_points = Ferrite.get_point_values(ph, q_nodes);

# We can also extract the field values, here the temperature, right away from the result vector of the simulation, that is stored in `u`. Opposed to the heat flux vector obtained from the `L2Projection`, the values are stored in the order of the degrees of freedom. 
# Therefore, we additionally give the field name which we want to extract from the dof-vector.
# Notice that for using this function, the `PointEvalHandler` must always be constructed with the same `DofHandler` 
# which was used for computing the dof-vector.
u_points = Ferrite.get_point_values(ph, u, :u);

# Now, we can plot the temperature and flux values with the help of any plotting library, e.g. Plots.jl. 
# To do this, we need to import the package:
import Plots

# Firstly, we are going to plot the temperature values along the given line.
Plots.plot(getindex.(points,1), u_points, label="Temperature", xlabel="X-Coordinate", ylabel = "Temperature")

# Secondly, the horizontal heat flux (i.e. the first component of the heat flux vector) is plotted.
Plots.plot(getindex.(points,1), getindex.(q_points,1),label="Flux", legend=:topleft, xlabel = "X-Coordinate", ylabel = "Heat flux")

#md # ## [Plain Program](@id postprocessing-plain-program)
#md #
#md # Below follows a version of the program without any comments.
#md # The file is also available here: [postprocessing.jl](postprocessing.jl)
#md #
#md # ```julia
#md # @__CODE__
#md # ```
