# # Point Evaluation
#


# ## Introduction
#
# For postprocessing puposes we often want to evaluate quantities in arbitrary points of the domain, e.g.
# for plotting them along lines or planes through the domain. The `PointEvalHandler` (in combination with the `L2Projector`) 
# allows to extract field values in arbitrary points.
# 
# This example continues from the L2-projection example, where heat fluxes were interpolated from
# integration points to nodes. 
#

# ## Implementation
#
# Start by simply running the L2-projection example to solve the problem, compute the heat fluxes and project them to nodes.
include("l2_projection.jl");

# We will evaluate heat flux distribution along the x-axis of the domain.
points = [Vec((x, 0.5)) for x in range(-1.0, 1.0, length=101)];

# First, we need to generate a `PointEvalHandler`
ph = PointEvalHandler(dh, points);

# We can extract the heat fluxes, which after the L2-Projection are stored in `q_nodes` in nodal order, in the given points.
q_points = Ferrite.get_point_values(ph, q_nodes);

# We can also extract the field values, here the temperature, right away from the result vector of the simulation that is stored in `u` in the order of the degrees of freedom. 
# Therefor, we additionally give the field name which we want to extract from the dof-vector.
# Notice that for using this function, the `PointEvalHandler` should always be constructed with the same `DofHandler` 
# which was used for computing the dof-vector.
u_points = Ferrite.get_point_values(ph, u, :u);

# Now, we can plot the temperature and flux values with the help of any plotting library, e.g. Plots.jl. 
# To do this, we need to import the package:
import Plots

# Firstly, we are going to plot the temperature values across the x-axis
Plots.plot(getindex.(points_upper,1), u_points, label="Temperature")

# Secondly, the first index of the flux value at each node is plotted. 
Plots.plot(getindex.(points_upper,1), getindex.(q_points,1),label="Flux")

#md # ## [Plain Program](@id point_evaluation-plain-program)
#md #
#md # Below follows a version of the program without any comments.
#md # The file is also available here: [point_evaluation.jl](point_evaluation.jl)
#md #
#md # ```julia
#md # @__CODE__
#md # ```
