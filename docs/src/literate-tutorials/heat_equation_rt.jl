# # [Heat equation (Mixed, RaviartThomas)](@id tutorial-heat-equation-rt)
# Note, there are a lot to consider here it seems like. Good refs,
# ```
# @book{Gatica2014,
# title = {A Simple Introduction to the Mixed Finite Element Method: Theory and Applications},
# ISBN = {9783319036953},
# ISSN = {2191-8201},
# url = {http://dx.doi.org/10.1007/978-3-319-03695-3},
# DOI = {10.1007/978-3-319-03695-3},
# journal = {SpringerBriefs in Mathematics},
# publisher = {Springer International Publishing},
# author = {Gatica,  Gabriel N.},
# year = {2014}
# }
# See also,
# @book{Boffi2013,
#   title = {Mixed Finite Element Methods and Applications},
#   ISBN = {9783642365195},
#   ISSN = {0179-3632},
#   url = {http://dx.doi.org/10.1007/978-3-642-36519-5},
#   DOI = {10.1007/978-3-642-36519-5},
#   journal = {Springer Series in Computational Mathematics},
#   publisher = {Springer Berlin Heidelberg},
#   author = {Boffi,  Daniele and Brezzi,  Franco and Fortin,  Michel},
#   year = {2013}
# }
# for a(n even) more comprehensive book.
# ```
#
# ## Theory
# We start with the strong form of the heat equation: Find the temperature, $u(\boldsymbol{x})$, and heat flux, $\boldsymbol{q}(x)$,
# such that
# ```math
# \begin{align*}
# \boldsymbol{\nabla}\cdot \boldsymbol{q} &= h(\boldsymbol{x}), \quad \forall \boldsymbol{x} \in \Omega \\
# \boldsymbol{q}(\boldsymbol{x}) &= - k\ \boldsymbol{\nabla} u(\boldsymbol{x}), \quad \forall \boldsymbol{x} \in \Omega \\
# \boldsymbol{q}(\boldsymbol{x})\cdot \boldsymbol{n}(\boldsymbol{x}) &= q_n, \quad \forall \boldsymbol{x} \in \Gamma_\mathrm{N}\\
# u(\boldsymbol{x}) &= u_\mathrm{D}, \quad \forall \boldsymbol{x} \in \Gamma_\mathrm{D}
# \end{align*}
# ```
#
# From this strong form, we can formulate the weak form as a mixed formulation.
# Find $u \in \mathbb{U}$ and $\boldsymbol{q}\in\mathbb{Q}$ such that
# ```math
# \begin{align*}
# \int_{\Omega} \delta u [\boldsymbol{\nabla} \cdot \boldsymbol{q}]\ \mathrm{d}\Omega &= - \int_\Omega \delta u h\ \mathrm{d}\Omega, \quad \forall\ \delta u \in \delta\mathbb{U} \\
# %\int_{\Omega} \boldsymbol{\delta q} \cdot \boldsymbol{q}\ \mathrm{d}\Omega &= \int_\Omega \boldsymbol{\delta q} \cdot [k\ \boldsymbol{\nabla} u]\ \mathrm{d}\Omega \\
# \int_{\Omega} \boldsymbol{\delta q} \cdot \boldsymbol{q}\ \mathrm{d}\Omega + \int_{\Omega} [\boldsymbol{\nabla} \cdot \boldsymbol{\delta q}] k u \ \mathrm{d}\Omega &=
# \int_\Gamma \boldsymbol{\delta q} \cdot \boldsymbol{n} k\ u\ \mathrm{d}\Omega, \quad \forall\ \boldsymbol{\delta q} \in \delta\mathbb{Q}
# \end{align*}
# ```
# where we have the function spaces
# * $\mathbb{U} = \delta\mathbb{U} = L^2$
# * $\mathbb{Q} = \lbrace \boldsymbol{q} \in H(\mathrm{div}) \text{such that} \boldsymbol{q}\cdot\boldsymbol{n} = q_\mathrm{n} \text{ on } \Gamma_\mathrm{D}\rbrace$
# * $\mathbb{Q} = \lbrace \boldsymbol{q} \in H(\mathrm{div}) \text{such that} \boldsymbol{q}\cdot\boldsymbol{n} = 0 \text{ on } \Gamma_\mathrm{D}\rbrace$
#
# A stable choice of finite element spaces for this problem on grid with triangles is using
# * `DiscontinuousLagrange{RefTriangle, k-1}` for approximating $L^2$
# * `BrezziDouglasMarini{RefTriangle, k}` for approximating $H(\mathrm{div})$
# following [fenics](https://fenicsproject.org/olddocs/dolfin/1.4.0/python/demo/documented/mixed-poisson/python/documentation.html).
# For further details, see Boffi2013.
# We will also see what happens if we instead use `Lagrange` elements which are a subspace of $H^1$ instead of $H(\mathrm{div})$ elements.
#
# ## Commented Program
#
# Now we solve the problem in Ferrite. What follows is a program spliced with comments.
#
# First we load Ferrite,
using Ferrite
# And define our grid, representing a channel with a central part having a lower
# conductivity, $k$, than the surrounding.
function create_grid(ny::Int)
    width = 10.0
    length = 40.0
    center_width = 5.0
    center_length = 20.0
    upper_right = Vec((length / 2, width / 2))
    grid = generate_grid(Triangle, (round(Int, ny * length / width), ny), -upper_right, upper_right);
    addcellset!(grid, "center", x -> abs(x[1]) < center_width/2 && abs(x[2]) < center_length / 2)
    addcellset!(grid, "around", setdiff(1:getncells(grid), getcellset(grid, "center")))
    return grid
end

grid = create_grid(10)

# ### Trial and test functions
# A `CellValues` facilitates the process of evaluating values and gradients of
# test and trial functions (among other things). To define
# this we need to specify an interpolation space for the shape functions.
# We use Lagrange functions
# based on the two-dimensional reference quadrilateral. We also define a quadrature rule based on
# the same reference element. We combine the interpolation and the quadrature rule
# to a `CellValues` object.
ip_geo = geometric_interpolation(getcelltype(grid))
ipu = DiscontinuousLagrange{RefTriangle, 1}() # Why does it "explode" for 2nd order ipu?
ipq = RaviartThomas{2,RefTriangle, 1}()
qr = QuadratureRule{RefTriangle}(2)
cellvalues = (u=CellValues(qr, ipu, ip_geo), q=CellValues(qr, ipq, ip_geo))

# ### Degrees of freedom
# Next we need to define a `DofHandler`, which will take care of numbering
# and distribution of degrees of freedom for our approximated fields.
# We create the `DofHandler` and then add a single scalar field called `:u` based on
# our interpolation `ip` defined above.
# Lastly we `close!` the `DofHandler`, it is now that the dofs are distributed
# for all the elements.
dh = DofHandler(grid)
add!(dh, :u, ipu)
add!(dh, :q, ipq)
close!(dh);

# Now that we have distributed all our dofs we can create our tangent matrix,
# using `create_sparsity_pattern`. This function returns a sparse matrix
# with the correct entries stored.
K = allocate_matrix(dh)

# ### Boundary conditions
# In Ferrite constraints like Dirichlet boundary conditions
# are handled by a `ConstraintHandler`.
ch = ConstraintHandler(dh);

# Next we need to add constraints to `ch`. For this problem we define
# homogeneous Dirichlet boundary conditions on the whole boundary, i.e.
# the `union` of all the boundary facet sets.
∂Ω = union(
    getfacetset(grid, "left"),
    getfacetset(grid, "right"),
    getfacetset(grid, "top"),
    getfacetset(grid, "bottom"),
);

# Now we are set up to define our constraint. We specify which field
# the condition is for, and our combined face set `∂Ω`. The last
# argument is a function of the form $f(\textbf{x})$ or $f(\textbf{x}, t)$,
# where $\textbf{x}$ is the spatial coordinate and
# $t$ the current time, and returns the prescribed value. Since the boundary condition in
# this case do not depend on time we define our function as $f(\textbf{x}) = 0$, i.e.
# no matter what $\textbf{x}$ we return $0$. When we have
# specified our constraint we `add!` it to `ch`.
dbc = Dirichlet(:u, ∂Ω, (x, t) -> 0)
add!(ch, dbc);

# Finally we also need to `close!` our constraint handler. When we call `close!`
# the dofs corresponding to our constraints are calculated and stored
# in our `ch` object.
close!(ch)

# Note that if one or more of the constraints are time dependent we would use
# [`update!`](@ref) to recompute prescribed values in each new timestep.

# ### Assembling the linear system
#
# Now we have all the pieces needed to assemble the linear system, $K u = f$.
# Assembling of the global system is done by looping over all the elements in order to
# compute the element contributions ``K_e`` and ``f_e``, which are then assembled to the
# appropriate place in the global ``K`` and ``f``.
#
# #### Element assembly
# We define the function `assemble_element!` (see below) which computes the contribution for
# an element. The function takes pre-allocated `ke` and `fe` (they are allocated once and
# then reused for all elements) so we first need to make sure that they are all zeroes at
# the start of the function by using `fill!`. Then we loop over all the quadrature points,
# and for each quadrature point we loop over all the (local) shape functions. We need the
# value and gradient of the test function, `δu` and also the gradient of the trial function
# `u`. We get all of these from `cellvalues`.
#
# !!! note "Notation"
#     Comparing with the brief finite element introduction in [Introduction to FEM](@ref),
#     the variables `δu`, `∇δu` and `∇u` are actually $\phi_i(\textbf{x}_q)$, $\nabla
#     \phi_i(\textbf{x}_q)$ and $\nabla \phi_j(\textbf{x}_q)$, i.e. the evaluation of the
#     trial and test functions in the quadrature point ``\textbf{x}_q``. However, to
#     underline the strong parallel between the weak form and the implementation, this
#     example uses the symbols appearing in the weak form.

function assemble_element!(Ke::Matrix, fe::Vector, cv::NamedTuple, dr::NamedTuple)
    cvu = cv[:u]
    cvq = cv[:q]
    dru = dr[:u]
    drq = dr[:q]
    ## Loop over quadrature points
    for q_point in 1:getnquadpoints(cvu)
        ## Get the quadrature weight
        dΩ = getdetJdV(cvu, q_point)
        ## Loop over test shape functions
        for (iu, Iu) in pairs(dru)
            δu  = shape_value(cvu, q_point, iu)
            ∇δu = shape_gradient(cvu, q_point, iu)
            ## Add contribution to fe
            fe[Iu] -= δu * dΩ
            ## Loop over trial shape functions
            for (jq, Jq) in pairs(drq)
                q = shape_value(cvq, q_point, jq)
                ## Add contribution to Ke
                Ke[Iu, Jq] += (∇δu ⋅ q) * dΩ
            end
        end
        for (iq, Iq) in pairs(drq)
            δq = shape_value(cvq, q_point, iq)
            for (ju, Ju) in pairs(dru)
                ∇u = shape_gradient(cvu, q_point, ju)
                Ke[Iq, Ju] += (δq ⋅ ∇u) * dΩ
            end
            for (jq, Jq) in pairs(drq)
                q = shape_value(cvq, q_point, jq)
                Ke[Iq, Jq] += (δq ⋅ q) * dΩ
            end
        end
    end
    return Ke, fe
end
#md nothing # hide

# #### Global assembly
# We define the function `assemble_global` to loop over the elements and do the global
# assembly. The function takes our `cellvalues`, the sparse matrix `K`, and our DofHandler
# as input arguments and returns the assembled global stiffness matrix, and the assembled
# global force vector. We start by allocating `Ke`, `fe`, and the global force vector `f`.
# We also create an assembler by using `start_assemble`. The assembler lets us assemble into
# `K` and `f` efficiently. We then start the loop over all the elements. In each loop
# iteration we reinitialize `cellvalues` (to update derivatives of shape functions etc.),
# compute the element contribution with `assemble_element!`, and then assemble into the
# global `K` and `f` with `assemble!`.
#
# !!! note "Notation"
#     Comparing again with [Introduction to FEM](@ref), `f` and `u` correspond to
#     $\underline{\hat{f}}$ and $\underline{\hat{u}}$, since they represent the discretized
#     versions. However, through the code we use `f` and `u` instead to reflect the strong
#     connection between the weak form and the Ferrite implementation.

function assemble_global(cellvalues, K::SparseMatrixCSC, dh::DofHandler)
    grid = dh.grid
    ## Allocate the element stiffness matrix and element force vector
    dofranges = (u = dof_range(dh, :u), q = dof_range(dh, :q))
    ncelldofs = ndofs_per_cell(dh)
    Ke = zeros(ncelldofs, ncelldofs)
    fe = zeros(ncelldofs)
    ## Allocate global force vector f
    f = zeros(ndofs(dh))
    ## Create an assembler
    assembler = start_assemble(K, f)
    x = copy(getcoordinates(grid, 1))
    dofs = copy(celldofs(dh, 1))
    ## Loop over all cels
    for cellnr in 1:getncells(grid)
        ## Reinitialize cellvalues for this cell
        cell = getcells(grid, cellnr)
        getcoordinates!(x, grid, cell)
        celldofs!(dofs, dh, cellnr)
        reinit!(cellvalues[:u], cell, x)
        reinit!(cellvalues[:q], cell, x)
        ## Reset to 0
        fill!(Ke, 0)
        fill!(fe, 0)
        ## Compute element contribution
        assemble_element!(Ke, fe, cellvalues, dofranges)
        ## Assemble Ke and fe into K and f
        assemble!(assembler, dofs, Ke, fe)
    end
    return K, f
end
#md nothing # hide

# ### Solution of the system
# The last step is to solve the system. First we call `assemble_global`
# to obtain the global stiffness matrix `K` and force vector `f`.
K, f = assemble_global(cellvalues, K, dh);

# To account for the boundary conditions we use the `apply!` function.
# This modifies elements in `K` and `f` respectively, such that
# we can get the correct solution vector `u` by using `\`.
apply!(K, f, ch)
u = K \ f;

# ### Exporting to VTK
# To visualize the result we export the grid and our field `u`
# to a VTK-file, which can be viewed in e.g. [ParaView](https://www.paraview.org/).
u_nodes = evaluate_at_grid_nodes(dh, u, :u)
∂Ω_cells = zeros(Int, getncells(grid))
for (cellnr, _) in ∂Ω
    ∂Ω_cells[cellnr] = 1
end
VTKGridFile("heat_equation_rt", dh) do vtk
    write_node_data(vtk, u_nodes, "u")
    write_cell_data(vtk, ∂Ω_cells, "dO")
end

@show norm(u_nodes)/length(u_nodes)

# ## Postprocess the total flux
function calculate_flux(dh, dΩ, ip, a)
    grid = dh.grid
    qr = FacetQuadratureRule{RefTriangle}(4)
    ip_geo = geometric_interpolation(getcelltype(grid))
    fv = FacetValues(qr, ip, ip_geo)

    dofrange = dof_range(dh, :q)
    flux = 0.0
    dofs = celldofs(dh, 1)
    ae = zeros(length(dofs))
    x = getcoordinates(grid, 1)
    for (cellnr, facenr) in dΩ
        getcoordinates!(x, grid, cellnr)
        cell = getcells(grid, cellnr)
        celldofs!(dofs, dh, cellnr)
        map!(i->a[i], ae, dofs)
        reinit!(fv, cell, x, facenr)
        for q_point in 1:getnquadpoints(fv)
            dΓ = getdetJdV(fv, q_point)
            n = getnormal(fv, q_point)
            q = function_value(fv, q_point, ae, dofrange)
            flux += (q ⋅ n)*dΓ
        end
    end
    return flux
end

function calculate_flux_lag(dh, dΩ, ip, a)
    grid = dh.grid
    qr = FacetQuadratureRule{RefTriangle}(4)
    ip_geo = geometric_interpolation(getcelltype(grid))
    fv = FacetValues(qr, ip, ip_geo)
    dofrange = dof_range(dh, :u)
    flux = 0.0
    dofs = celldofs(dh, 1)
    ae = zeros(length(dofs))
    x = getcoordinates(grid, 1)
    for (cellnr, facenr) in dΩ
        getcoordinates!(x, grid, cellnr)
        cell = getcells(grid, cellnr)
        celldofs!(dofs, dh, cellnr)
        map!(i->a[i], ae, dofs)
        reinit!(fv, cell, x, facenr)
        for q_point in 1:getnquadpoints(fv)
            dΓ = getdetJdV(fv, q_point)
            n = getnormal(fv, q_point)
            q = function_gradient(fv, q_point, ae, dofrange)
            flux -= (q ⋅ n)*dΓ
        end
    end
    return flux
end

flux = calculate_flux(dh, ∂Ω, ipq, u)
flux_lag = calculate_flux_lag(dh, ∂Ω, ipu, u)
@show flux, flux_lag


function get_Ke(dh, cellvalues; cellnr=1)
    dofranges = (u = dof_range(dh, :u), q = dof_range(dh, :q))
    ncelldofs = ndofs_per_cell(dh)
    Ke = zeros(ncelldofs, ncelldofs)
    fe = zeros(ncelldofs)
    x = getcoordinates(grid, cellnr)
    cell = getcells(grid, cellnr)
    reinit!(cellvalues[:u], cell, x)
    reinit!(cellvalues[:q], cell, x)

    ## Reset to 0
    fill!(Ke, 0)
    fill!(fe, 0)
    ## Compute element contribution
    assemble_element!(Ke, fe, cellvalues, dofranges)
    return Ke
end
#md # ## [Plain program](@id heat_equation-plain-program)
#md #
#md # Here follows a version of the program without any comments.
#md # The file is also available here: [`heat_equation.jl`](heat_equation.jl).
#md #
#md # ```julia
#md # @__CODE__
#md # ```
