# # L2-projection
#
# ![](heat_square_fluxes.png)


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
points_upper = [Vec((x, 0.0)) for x in range(-1.0, 1.0, length=101)]

# First, we need to generate a `PointEvalHandler`
ph = PointEvalHandler(dh, [ip], points)

# We can extract the heat fluxes, which after the L2-Projection are stored in `q_nodes` in nodal order, in the given points.
q_points = Ferrite.get_point_values(ph, q_nodes)

# We can also extract the field values, here the temperature, right away from the result vector of the simulation that is stored in `u` in the order of the degrees of freedom. 
# Therefor, we additionally give the field name which we want to extract from the dof-vector.
# Notice that for using this function, the `PointEvalHandler` should always be constructed with the same `DofHandler` 
# which was used for computing the dof-vector.
u_points = Ferrite.get_point_values(ph, u, :u)

# plot([p.data[1] for p in points], [v.data[1] for v in vals])

#md # ## [Plain Program](@id l2_projection-plain-program)
#md #
#md # Below follows a version of the program without any comments.
#md # The file is also available here: [l2_projection.jl](l2_projection.jl)
#md #
#md # ```julia
#md # @__CODE__
#md # ```
