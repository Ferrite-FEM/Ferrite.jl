# # Stokes flow
#
# **Keywords**: *periodic boundary conditions, multiple fields*
#-
#md # !!! tip
#md #     This example is also available as a Jupyter notebook:
#md #     [`stokes-flow.ipynb`](@__NBVIEWER_ROOT_URL__/examples/stokes-flow.ipynb).
#-
#
# ![](stokes-flow.png)
# *Figure 1*: Left: Computational domain ``\Omega`` with boundaries ``\Gamma_1``,
# ``\Gamma_3`` (periodic boundary conditions) and ``\Gamma_2``, ``\Gamma_4`` (homogeneous
# Dirichlet boundary conditions). Right: Magnitude of the resulting velocity field.

# ## Introduction and problem formulation
#
# This example is a translation of the [step-45 example from
# deal.ii](https://www.dealii.org/current/doxygen/deal.II/step_45.html) which solves Stokes
# flow on a quarter circle. In particular it shows how to use periodic boundary conditions
# and how to solve a problem with multiple unknown fields. For the mesh generation we use
# [`Gmsh.jl`](https://github.com/JuliaFEM/Gmsh.jl) and then use
# [`FerriteGmsh.jl`](https://github.com/Ferrite-FEM/FerriteGmsh.jl) to import the mesh into
# Ferrite's format.
#
# The strong form of Stokes flow with velocity ``\boldsymbol{u}`` and pressure ``p`` can be
# written as follows:
# ```math
# \begin{align*}
# -\Delta \boldsymbol{u} + \nabla p &= \bigl(\exp(-100||\boldsymbol{x} - (0.75, 0.1)||^2), 0\bigr) =:
# \boldsymbol{b} \quad \forall \boldsymbol{x} \in \Omega,\\
# -\boldsymbol{\nabla} \cdot \boldsymbol{u} &= 0 \quad \forall \boldsymbol{x} \in \Omega,
# \end{align*}
# ```
# where the domain is defined as ``\Omega = \{\boldsymbol{x} \in (0, 1)^2:
# \ ||\boldsymbol{x}|| \in (0.5, 1)\}``, see *Figure 1*. For the velocity we use periodic
# boundary conditions on the inlet ``\Gamma_1`` and outlet ``\Gamma_3``:
# ```math
# \begin{align*}
# u_x(0,\nu) &= -u_y(\nu, 0) \quad & \nu\ \in\ [0.5, 1],\\
# u_y(0,\nu) &= u_x(\nu, 0) \quad & \nu\ \in\ [0.5, 1],
# \end{align*}
# ```
# and homogeneous Dirichlet boundary conditions for ``\Gamma_2`` and ``\Gamma_4``:
# ```math
# \boldsymbol{u} = \boldsymbol{0} \quad \forall \boldsymbol{x}\ \in\
# \Gamma_2 \cup \Gamma_4 := \{ \boldsymbol{x}:\ ||\boldsymbol{x}|| \in \{0.5, 1\}\}.
# ```
# With this formulation and boundary conditions for ``\boldsymbol{u}`` the pressure will
# only be determined up to a constant. We will add an additional constraint which fixes this
# constant. In particular, we will enforce that the mean value of the pressure on the
# boundary is 0, see [deal.ii
# step-11](https://www.dealii.org/current/doxygen/deal.II/step_11.html) for some more
# discussion around this.
#
# The corresponding weak form reads as follows: Find ``(\boldsymbol{u}, p) \in \mathbb{U}
# \times \mathrm{L}_2`` s.t.
# ```math
# \begin{align*}
# \int_\Omega \Bigl[[\delta\boldsymbol{u}\otimes\boldsymbol{\nabla}]:[\boldsymbol{u}\otimes\boldsymbol{\nabla}] -
# (\boldsymbol{\nabla}\cdot\delta\boldsymbol{u})\ p\ \Bigr] \mathrm{d}\Omega &=
# \int_\Omega \delta\boldsymbol{u} \cdot \boldsymbol{b}\ \mathrm{d}\Omega \quad \forall
# \delta \boldsymbol{u} \in \mathbb{U},\\
# \int_\Omega - (\boldsymbol{\nabla}\cdot\boldsymbol{u})\ \delta p\ \mathrm{d}\Omega &= 0
# \quad \forall \delta p \in \mathrm{L}_2,
# \end{align*}
# ```
# where ``\mathbb{U}`` is a suitable function space, that, in particular, enforces the
# Dirichlet boundary conditions, and the periodicity constraints.


# ## Commented program
#
# What follows is a program spliced with comments.
#md # The full program, without comments, can be found in the next
#md # [section](@ref stokes-flow-plain-program).

using Ferrite, FerriteGmsh, Gmsh, Tensors, LinearAlgebra
using Test #src


# ### Geometry and mesh generation with `Gmsh.jl`
#
# In the `setup_grid` function below we use the
# [`Gmsh.jl`](https://github.com/JuliaFEM/Gmsh.jl) package for setting up the geometry and
# performing the meshing. We will not discuss this part in much detail but refer to the
# [Gmsh API documentation](https://gmsh.info/doc/texinfo/gmsh.html#Gmsh-API) instead. The
# most important thing to note is the mesh periodicity constraint that is applied between
# the "inlet" and "outlet" parts using `gmsh.model.set_periodic`. This is necessary to later
# on apply a periodicity constraint for the approximated velocity field.

function setup_grid(h=0.05)
    ## Initialize gmsh
    Gmsh.initialize()
    gmsh.option.set_number("General.Verbosity", 2)

    ## Add the points
    o = gmsh.model.geo.add_point(0.0, 0.0, 0.0, h)
    p1 = gmsh.model.geo.add_point(0.5, 0.0, 0.0, h)
    p2 = gmsh.model.geo.add_point(1.0, 0.0, 0.0, h)
    p3 = gmsh.model.geo.add_point(0.0, 1.0, 0.0, h)
    p4 = gmsh.model.geo.add_point(0.0, 0.5, 0.0, h)

    ## Add the lines
    l1 = gmsh.model.geo.add_line(p1, p2)
    l2 = gmsh.model.geo.add_circle_arc(p2, o, p3)
    l3 = gmsh.model.geo.add_line(p3, p4)
    l4 = gmsh.model.geo.add_circle_arc(p4, o, p1)

    ## Create the closed curve loop and the surface
    loop = gmsh.model.geo.add_curve_loop([l1, l2, l3, l4])
    surf = gmsh.model.geo.add_plane_surface([loop])

    ## Synchronize the model
    gmsh.model.geo.synchronize()

    ## Create the physical domains
    gmsh.model.add_physical_group(1, [l1], -1, "Γ1")
    gmsh.model.add_physical_group(1, [l2], -1, "Γ2")
    gmsh.model.add_physical_group(1, [l3], -1, "Γ3")
    gmsh.model.add_physical_group(1, [l4], -1, "Γ4")
    gmsh.model.add_physical_group(2, [surf])

    ## Add the periodicity constraint using 4x4 affine transformation matrix,
    ## see https://en.wikipedia.org/wiki/Transformation_matrix#Affine_transformations
    transformation_matrix = zeros(4, 4)
    transformation_matrix[1, 2] = 1  # -sin(-pi/2)
    transformation_matrix[2, 1] = -1 #  cos(-pi/2)
    transformation_matrix[3, 3] = 1
    transformation_matrix[4, 4] = 1
    transformation_matrix = vec(transformation_matrix')
    gmsh.model.mesh.set_periodic(1, [l1], [l3], transformation_matrix)

    ## Generate a 2D mesh
    gmsh.model.mesh.generate(2)

    ## Save the mesh, and read back in as a Ferrite Grid
    grid = mktempdir() do dir
        path = joinpath(dir, "mesh.msh")
        gmsh.write(path)
        togrid(path)
    end

    ## Finalize the Gmsh library
    Gmsh.finalize()

    return grid
end
#md nothing #hide

# ### Degrees of freedom and boundary conditions
#
# Stokes flow is a saddle point problem, and, just like the example with incompressible
# elasticity, we need our formulation to fulfill the [LBB
# condition](https://en.wikipedia.org/wiki/Ladyzhenskaya%E2%80%93Babu%C5%A1ka%E2%80%93Brezzi_condition).
# Just like in that example we will use a quadratic approximation for the velocity field,
# and a linear approximation for the pressure. The `setup_dofs` function creates the
# `DofHandler`, and adds the two fields: a vector field `:u` with interpolation `ipu`, and a
# scalar field `:p` with interpolation `ipp`.

function setup_dofs(grid, ipu, ipp)
    dh = DofHandler(grid)
    push!(dh, :u, 2, ipu)
    push!(dh, :p, 1, ipp)
    close!(dh)
    return dh
end
#md nothing #hide

# Next we create the FE values. Note that we use linear geometric interpolation (`ipg`) for
# both the velocity and pressure, this is because our grid contains linear triangles.

function setup_fevalues(ipu, ipp, ipg)
    qr = QuadratureRule{2,RefTetrahedron}(2)
    cvu = CellVectorValues(qr, ipu, ipg)
    cvp = CellScalarValues(qr, ipp, ipg)
    return cvu, cvp
end
#md nothing #hide

# Now it is time to setup the `ConstraintHandler` and add our boundary conditions. This is
# perhaps the most interesting section in this example, and deserves some attention.
#
# Since the periodicity constraint for this example is between two boundaries which are not
# parallel to each other we need to i) compute the mapping between each mirror face and the
# corresponding image face (on the element level) and ii) describe the dof relation between
# dofs on these two faces. In Ferrite this is done by defining a transformation of entities
# on the mirror boundary such that they line up with the matching entities on the image
# boundary. In this example we consider the outlet ``\Gamma_3`` to be the mirror, and the
# inlet ``\Gamma_1`` to be the image. The necessary transformation to apply then becomes a
# rotation of ``-\pi/2`` radians around the out-of-plane axis. We set up the rotation matrix
# `R`, and then compute the mapping between mirror and image faces using
# [`collect_periodic_faces`](@ref) where the rotation is applied to the coordinates. In the
# next step we construct the constraint using the [`PeriodicDirichlet`](@ref) constructor.
# We pass the constructor the computed mapping, and also the rotation matrix. This matrix is
# used to rotate the dofs on the mirror surface such that we properly constrain
# ``\boldsymbol{u}_x``-dofs on the mirror to ``-\boldsymbol{u}_y``-dofs on the image, and
# ``\boldsymbol{u}_y``-dofs on the mirror to ``\boldsymbol{u}_x``-dofs on the image.
#
# For the remaining part of the boundary we add a homogeneous Dirichlet boundary condition
# on both components of the velocity field. This is done using the [`Dirichlet`](@ref)
# constructor, which we have discussed in other tutorials.
#
# As discussed in the introduction, we will add a constraint to the mean value of the
# pressure. We do this by constructing an [`AffineConstraint`](@ref) for the pressure dofs
# on the boundary:
# ```math
# 0 = \int_{\partial\Omega} p\ \mathrm{d}\Gamma \approx \sum_{i=1}^{D} p_i \quad \Rightarrow
# \quad p_1 = \sum_{i=2}^{D} -p_i,
# ```
# where ``\{p_i: i \in 1..D\}`` are the pressure dofs on the boundary.

function setup_constraints(dh)
    ch = ConstraintHandler(dh)
    ## Periodic BC
    R = rotation_tensor(-pi/2)
    periodic_faces = collect_periodic_faces(dh.grid, "Γ3", "Γ1", x -> R ⋅ x)
    periodic = PeriodicDirichlet(:u, periodic_faces, R, [1, 2])
    add!(ch, periodic)
    ## Dirichlet BC
    Γ24 = union(getfaceset(dh.grid, "Γ2"), getfaceset(dh.grid, "Γ4"))
    dbc = Dirichlet(:u, Γ24, (x, t) -> [0, 0], [1, 2])
    add!(ch, dbc)
    ## Mean value constraint: Ensure that \sum
    ch_mean = ConstraintHandler(dh)
    ∂Ω = union(getfaceset(dh.grid, "Γ1"), getfaceset(dh.grid, "Γ2"),
               getfaceset(dh.grid, "Γ3"), getfaceset(dh.grid, "Γ4"))
    add!(ch_mean, Dirichlet(:p, ∂Ω, (x, t) -> 0))
    close!(ch_mean)
    mean_value_constraint = AffineConstraint(
        pop!(ch_mean.prescribed_dofs),
        [d => -1. for d in ch_mean.prescribed_dofs],
        0.0
    )
    add!(ch, mean_value_constraint)
    ## Finalize
    close!(ch)
    update!(ch, 0)
    return ch
end
#md nothing #hide

# ### Global and local assembly
#
# Assembly of the global system is also something that we have seen in many previous
# tutorials. One interesting thing to note here is that, since we have two unknown fields,
# we use the [`dof_range`](@ref) function to make sure we assemble the element contributions
# to the correct block of the local stiffness matrix `ke`.

function assemble_system!(K, f, dh, cvu, cvp)
    assembler = start_assemble(K, f)
    ke = zeros(ndofs_per_cell(dh), ndofs_per_cell(dh))
    fe = zeros(ndofs_per_cell(dh))
    range_u = dof_range(dh, :u)
    ndofs_u = length(range_u)
    range_p = dof_range(dh, :p)
    ndofs_p = length(range_p)
    ϕᵤ = Vector{Vec{2,Float64}}(undef, ndofs_u)
    ∇ϕᵤ = Vector{Tensor{2,2,Float64,4}}(undef, ndofs_u)
    divϕᵤ = Vector{Float64}(undef, ndofs_u)
    ϕₚ = Vector{Float64}(undef, ndofs_p)
    for cell in CellIterator(dh)
        reinit!(cvu, cell)
        reinit!(cvp, cell)
        ke .= 0
        fe .= 0
        for qp in 1:getnquadpoints(cvu)
            dΩ = getdetJdV(cvu, qp)
            for i in 1:ndofs_u
                ϕᵤ[i] = shape_value(cvu, qp, i)
                ∇ϕᵤ[i] = shape_gradient(cvu, qp, i)
                divϕᵤ[i] = shape_divergence(cvu, qp, i)
            end
            for i in 1:ndofs_p
                ϕₚ[i] = shape_value(cvp, qp, i)
            end
            ## u-u
            for (i, I) in pairs(range_u), (j, J) in pairs(range_u)
                ke[I, J] += ( ∇ϕᵤ[i] ⊡ ∇ϕᵤ[j] ) * dΩ
            end
            ## u-p
            for (i, I) in pairs(range_u), (j, J) in pairs(range_p)
                ke[I, J] += ( - divϕᵤ[i] * ϕₚ[j] ) * dΩ
            end
            ## p-u
            for (i, I) in pairs(range_p), (j, J) in pairs(range_u)
                ke[I, J] += ( - divϕᵤ[j] * ϕₚ[i] ) * dΩ
            end
            ## rhs
            for (i, I) in pairs(range_u)
                x = spatial_coordinate(cvu, qp, getcoordinates(cell))
                b = exp(-100 * norm(x - Vec{2}((0.75, 0.1)))^2)
                bv = Vec{2}((b, 0.0))
                fe[I] += (ϕᵤ[i] ⋅ bv) * dΩ
            end
        end
        assemble!(assembler, celldofs(cell), ke, fe)
    end
    return K, f
end
#md nothing #hide

# ### Running the simulation
#
# We now have all the puzzle pieces, and just need to define the main function, which puts
# them all together.

function main()
    ## Grid
    h = 0.05 # approximate element size
    grid = setup_grid(h)
    ## Interpolations
    ipu = Lagrange{2,RefTetrahedron,2}() # quadratic
    ipp = Lagrange{2,RefTetrahedron,1}() # linear
    ## Dofs
    dh = setup_dofs(grid, ipu, ipp)
    ## Boundary conditions
    ch = setup_constraints(dh)
    ## FE values
    ipg = Lagrange{2,RefTetrahedron,1}() # linear geometric interpolation
    cvu, cvp = setup_fevalues(ipu, ipp, ipg)
    ## Global tangent matrix and rhs
    K = create_sparsity_pattern(dh, ch)
    f = zeros(ndofs(dh))
    ## Assemble system
    assemble_system!(K, f, dh, cvu, cvp)
    ## Apply boundary conditions and solve
    apply!(K, f, ch)
    u = K \ f
    apply!(u, ch)
    ## Export the solution
    vtk_grid("stokes-flow", grid) do vtk
        vtk_point_data(vtk, dh, u)
    end
    @test norm(u) ≈ 0.32278604074418793 #src
    return
end
#md nothing #hide

# Run it!

main()

# The resulting magnitude of the velocity field is visualized in *Figure 1*.

#md # ## [Plain program](@id stokes-flow-plain-program)
#md #
#md # Here follows a version of the program without any comments.
#md # The file is also available here: [`stokes-flow.jl`](stokes-flow.jl).
#md #
#md # ```julia
#md # @__CODE__
#md # ```
