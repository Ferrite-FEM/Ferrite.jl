#=
# Maxwell Discretizations: The Good, The Bad, and The Ugly
This tutorial is based on Jay Gopalakrishnan (Portland State University)
[Maxwell Discretizations: The Good, The Bad & The Ugly](https://web.pdx.edu/~gjay/pub/MaxwellGoodBadUgly.html)
from the graduate course MTH 653: Advanced Numerical Analysis (Spring 2019)

The purpose of the tutorial is to demonstrate how `Nedelec` vector interpolations will converge to the correct
solution for a Maxwell problem, when vectorized `Lagrange` interpolations converge to an incorrect solution.

## Loading packages
We start by adding the required packages for this tutorial
=#
import Pkg
# `Pkg.add(;url = ...)`
using Ferrite, Tensors, ForwardDiff
using Gmsh, FerriteGmsh
using FerriteTriangulation: Triangulation, SubTriangulation
import CairoMakie as Plt
import GeometryBasics as GB

using FerriteAssembly

#=
## Introduction
Specifically, we will consider the problem to determine the vector-valued field
``\boldsymbol{E}``, such that
```math
\begin{align*}
\mathrm{curl}(\mathrm{curl}(\boldsymbol{E})) &= 0\quad \text{in }\Omega \\
\mathrm{div}(\boldsymbol{E}) &= 0\quad \text{in }\Omega \\
\boldsymbol{E}\cdot \boldsymbol{t} &= g\quad \text{on }\Gamma
\end{align*}
```
where ``\boldsymbol{t}`` is the normalized tangential vector along
the boundary, ``\Gamma``, of the domain, ``\Omega``.
The rotated L-shaped domain is located such that the sharp internal corner
is at the origin.
=#
fig = Plt.Figure()
ax = Plt.Axis(fig[1, 1]; xlabel = "x", ylabel = "y")
points = [(0, 0), (1, 0), (1, 1), (-1, 1), (-1, -1), (0, -1), (0, 0)]
Plt.lines!(ax, first.(points), last.(points))
fig #hide

#=
## Theoretical background
We have the following partial integration rules, where ``\boldsymbol{n}`` is the outward pointing normal vector,
 following Monk (2003) [Monk2003; Eqs. (3.47) and (3.51)](@cite),
```math
\begin{align*}
\int_\Gamma \left[\boldsymbol{n}\times\boldsymbol{u}\right]\cdot\boldsymbol{\phi}\ \mathrm{d}\Gamma &=
\int_\Omega \left[\nabla \times \boldsymbol{u}\right]\cdot \boldsymbol{\phi}\ \mathrm{d}\Omega -
\int_\Omega \boldsymbol{u}\cdot\left[ \nabla \times \boldsymbol{\phi} \right]\ \mathrm{d}\Omega, \quad \boldsymbol{u} \in H(\mathrm{curl}),\ \boldsymbol{\phi} \in (\mathcal{C}^1)^3\\
\int_\Gamma \left[\boldsymbol{n}\times\boldsymbol{u}\right]\cdot\left[\left[\boldsymbol{n}\times\boldsymbol{\phi}\right]\times\boldsymbol{n}\right] \mathrm{d}\Gamma &=
\int_\Omega \left[\nabla \times \boldsymbol{u}\right]\cdot \boldsymbol{\phi}\ \mathrm{d}\Omega -
\int_\Omega \boldsymbol{u}\cdot\left[ \nabla \times \boldsymbol{\phi} \right]\ \mathrm{d}\Omega, \quad \boldsymbol{u},\ \boldsymbol{\phi}\in H(\mathrm{curl}),
\end{align*}
```
respectively. Note that [Monk2003; Eq. (3.27)](@cite), requiring that ``\boldsymbol{u} \in (\mathcal{C}^1)^3``,
is a special case of [Monk2003; Eq. (3.47)](@cite) as ``(\mathcal{C}^1)^3 \subset H(\mathrm{curl})``.

We remark that in 2D for ``\boldsymbol{u}`` pointing out of the plane, ``\boldsymbol{u} = u_3 \boldsymbol{e}_3``,
and ``\boldsymbol{\phi}`` in the plane, ``\boldsymbol{\phi} \cdot \boldsymbol{e}_3 = 0``, we have
```math
\left[\boldsymbol{n}\times\boldsymbol{u}\right]\cdot\left[\left[\boldsymbol{n}\times\boldsymbol{\phi}\right]\times\boldsymbol{n}\right] = -u_3 \boldsymbol{\phi} \cdot \boldsymbol{t}
```
where ``\boldsymbol{t} = [-n_2,\ n_1]`` is the counter-clockwise tangent vector.

=#
#=
## Exact solution
In this example, we choose an exact solution, ``\boldsymbol{E}_\mathrm{exact}(\boldsymbol{x})``,
that fullfills the differential equations. We then use ``\boldsymbol{E}_\mathrm{exact}`` to
find ``g`` to insert into the Dirichlet boundary conditions. Specifically, we choose
```math
\boldsymbol{E}_\mathrm{exact}(\boldsymbol{x}) = \mathrm{grad}(r^{2/3}\sin(2\theta/3))
```
where ``r`` and ``\theta`` are the polar coordinates. Where ``\theta`` is defined as the positive
angle measured from the ``x``-axis.

The first PDE is directly satisfied by using the gradient to define the vector-valued solution,
since ``\mathrm{curl}(\mathrm{grad}(u)) = 0`` holds for any field ``u``. Furthermore,
along the lines, ``\theta = 0`` and ``\theta = 3\pi/2``, ``\sin(2\theta/3) = 0``, such that the
``\boldsymbol{E}_\mathrm{exact} \cdot \boldsymbol{t} = 0``. Consequently, even if we have a
singularity at ``\boldsymbol{x} = \boldsymbol{0}``, this doesn't enter the boundary conditions.
Finally, due to the singularity, the components of ``\boldsymbol{E}_\mathrm{exact}`` are not in
``H^1(\Omega)``.
**TODO:** *Explain why ``\mathrm{div}(\boldsymbol{E})=0`` is fullfilled, use divergence theorem?*

## Lagrange interpolation
Following the notes in the linked example, the lagrange problem becomes to solve
```math
\begin{align*}
\int_\Omega \mathrm{curl}(\delta \boldsymbol{E}) \cdot \mathrm{curl}(\boldsymbol{E})\ \mathrm{d}\Omega
+ \int_\Omega \mathrm{div}(\delta \boldsymbol{E}) \mathrm{div}(\boldsymbol{E})\ \mathrm{d}\Omega &= 0 \quad \forall\ \delta\boldsymbol{E} \in H^1 \\
\boldsymbol{E}\cdot\boldsymbol{t} &= g\quad \text{on }\Gamma
\end{align*}
```

## Nedelec interpolation
Using a Lagrange multiplier, ``\phi``, to weakly enforce the divergence equation,
we obtain the following problem:
Find ``\boldsymbol{E}\in H(\mathrm{div})`` and ``\phi\in H_0^1``, such that
```math
\begin{align*}
\int_\Gamma \left[\left[\boldsymbol{n}\times\delta\boldsymbol{E}\right]\times\boldsymbol{n}\right]\cdot
\left[\boldsymbol{n}\cdot\mathrm{curl}(\boldsymbol{E})\right]\ \mathrm{d}\Gamma +
\int_\Omega \mathrm{curl}(\delta\boldsymbol{E})\cdot\mathrm{curl}(\boldsymbol{E})\ \mathrm{d}\Omega +
\int_\Omega \delta\boldsymbol{E}\cdot\mathrm{grad}(\phi)\ \mathrm{d}\Omega &= 0\\
\int_\Omega \mathrm{grad}(\delta\phi) \cdot \boldsymbol{E}\ \mathrm{d}\Omega &= 0 \\
\boldsymbol{E}\cdot \boldsymbol{t} &= g\; \text{on }\Gamma
\end{align*}
```
for all ``\delta\boldsymbol{E}\in H_0(\mathrm{curl})`` and ``\delta\phi\in H_0^1``.
=#

# We then use `FerriteGmsh.jl` to create the grid
function setup_grid(h = 0.2; origin_refinement = 1)
    # Initialize gmsh
    Gmsh.initialize()
    gmsh.option.set_number("General.Verbosity", 2)

    # Add the points, finer grid at the discontinuity
    o = gmsh.model.geo.add_point(0.0, 0.0, 0.0, h / origin_refinement)
    p1 = gmsh.model.geo.add_point(1.0, 0.0, 0.0, h)
    p2 = gmsh.model.geo.add_point(1.0, 1.0, 0.0, h)
    p3 = gmsh.model.geo.add_point(-1.0, 1.0, 0.0, h)
    p4 = gmsh.model.geo.add_point(-1.0, -1.0, 0.0, h)
    p5 = gmsh.model.geo.add_point(0.0, -1.0, 0.0, h)

    pts = [o, p1, p2, p3, p4, p5, o]
    # Add the lines
    lines = [gmsh.model.geo.add_line(pts[i - 1], pts[i]) for i in 2:length(pts)]

    # Create the closed curve loop and the surface
    loop = gmsh.model.geo.add_curve_loop(lines)
    gmsh.model.geo.add_plane_surface([loop])

    # Synchronize the model
    gmsh.model.geo.synchronize()

    # Generate a 2D mesh
    gmsh.model.mesh.generate(2)

    # Save the mesh, and read back in as a Ferrite Grid
    grid = mktempdir() do dir
        path = joinpath(dir, "mesh.msh")
        gmsh.write(path)
        togrid(path)
    end

    # Finalize the Gmsh library
    Gmsh.finalize()

    # Add boundary parts
    addfacetset!(grid, "vertical_facets", x -> abs((x[1] - 1) * x[1] * (x[1] + 1)) ≤ 1.0e-6)
    addfacetset!(grid, "horizontal_facets", x -> abs((x[2] - 1) * x[2] * (x[2] + 1)) ≤ 1.0e-6)

    return grid
end

# And prepare some functions to process and plot the data on the grid
# by using `FerriteTriangulation.jl`
function _create_data!(f, data, grid, a, cvs, subtria::SubTriangulation)
    c1 = first(subtria.faces)[1]
    x = copy(getcoordinates(grid, c1))
    dofs = copy(celldofs(dh, c1))
    ae = zeros(eltype(a), length(dofs))
    for (i, (cellnr, facenr)) in enumerate(subtria.faces)
        cv = cvs[facenr]
        getcoordinates!(x, grid, cellnr)
        reinit!(cv, getcells(grid, cellnr), x)
        celldofs!(dofs, dh, cellnr)
        copyto!(ae, view(a, dofs))
        node_idxs = subtria.face_nodes[i]:(subtria.face_nodes[i + 1] - 1)
        for q_point in 1:getnquadpoints(cv)
            data[node_idxs[q_point]] = f(function_value(cv, q_point, ae))
        end
    end
    return
end

"""
    create_data(tr::Triangulation, grid::AbstractGrid, a::Vector{<:Number}, ::NTuple{N, <:Interpolation};
        f = identity)

Create scalar data by evaluating `f(function_value(...))` at each triangulation node in the `grid`.
"""
function create_data(tr::Triangulation, grid, a, ips; f = identity)
    data = zeros(length(tr.nodes))
    for (ip, subtria) in zip(ips, tr.sub_triangulation)
        cvs = [CellValues(cr, ip, geometric_interpolation(getcelltype(subtria.sdh))) for cr in subtria.rules]
        _create_data!(f, data, grid, a, cvs, subtria)
    end
    return data
end

grid = setup_grid(0.00390625; origin_refinement = 1)

#
ip = DiscontinuousLagrange{RefTriangle, 1}()^2
dh = close!(add!(DofHandler(grid), :u, ip))

function analytical_potential(x::Vec{2}) # Analytical potential to be differentiated
    Δθ = -3π / 4 # Rotate discontinuous line to 4th quadrant
    xp = rotate(x, Δθ)
    r = sqrt(x ⋅ x + eps())
    θ = r ≤ 1.0e-6 ? zero(eltype(x)) : (atan(xp[2], xp[1]) - Δθ)
    return r^(2 // 3) * sin(2θ / 3)
end

a = zeros(ndofs(dh))

apply_analytical!(a, dh, :u, x -> gradient(analytical_potential, x))

tr = Triangulation(dh, 2)
data = create_data(tr, grid, a, (ip,); f = first)

fig = Plt.Figure()
ax = Plt.Axis(fig[1, 1]; aspect = Plt.DataAspect())

nodes = [GB.Point(x.data) for x in tr.nodes]
m = Plt.mesh!(
    ax, nodes, reshape(tr.triangles, :); color = data,
    colormap = Plt.Makie.wong_colors(),
    interpolate = true,
    colorrange = (-2.0, 0.0),
)
Plt.Colorbar(fig[1, 2], m)
#=
```julia
for i in 2:length(tr.tri_edges)
    Plt.lines!(view(nodes, view(tr.edges, tr.tri_edges[i-1]:(tr.tri_edges[i]-1))); color=:black)
end
```
=#

# Error calculator
mutable struct L2Error{F}
    l2error::Float64
    volume::Float64
    const exact_fun::F
end

function FerriteAssembly.integrate_cell!(vals::L2Error{F}, state, ae, material, cv, cellbuffer) where {F}
    for q_point in 1:getnquadpoints(cv)
        Eh = function_value(cv, q_point, ae)
        x = spatial_coordinate(cv, q_point, getcoordinates(cellbuffer))
        dΩ = getdetJdV(cv, q_point)
        vals.l2error += norm(Eh - vals.exact_fun(x))^2 * dΩ
        vals.volume += dΩ
    end
    return
end

# ## Lagrange solution
struct LagrangeMaterial end
function FerriteAssembly.element_routine!(Ke, re, s, ae, ::LagrangeMaterial, cv, cellbuffer)
    for q_point in 1:getnquadpoints(cv)
        dΩ = getdetJdV(cv, q_point)
        for i in 1:getnbasefunctions(cv)
            div_δNi = shape_divergence(cv, q_point, i)
            curl_δNi = shape_curl(cv, q_point, i)
            for j in 1:getnbasefunctions(cv)
                div_Nj = shape_divergence(cv, q_point, j)
                curl_Nj = shape_curl(cv, q_point, j)
                Ke[i, j] += (curl_δNi ⋅ curl_Nj + div_δNi * div_Nj) * dΩ
            end
        end
    end
    return
end

function solve_lagrange(dh, ip)
    ch = ConstraintHandler(dh)
    add!(ch, Dirichlet(:E, getfacetset(grid, "horizontal_facets"), (x, _) -> gradient(analytical_potential, x)[2], [2]))
    add!(ch, Dirichlet(:E, getfacetset(grid, "vertical_facets"), (x, _) -> gradient(analytical_potential, x)[1], [1]))
    close!(ch)

    cv = CellValues(QuadratureRule{RefTriangle}(1), ip)
    K = allocate_matrix(dh)
    f = zeros(ndofs(dh))
    db = setup_domainbuffer(DomainSpec(dh, LagrangeMaterial(), cv))
    as = start_assemble(K, f)
    work!(as, db)
    apply!(K, f, ch)
    a = K \ f
    l2_vals = L2Error(0.0, 0.0, x -> gradient(analytical_potential, x))
    work!(Integrator(l2_vals), db; a)
    return a, sqrt(l2_vals.l2error) / l2_vals.volume
end

ip = Lagrange{RefTriangle, 1}()^2
dh = close!(add!(DofHandler(grid), :E, ip))
a_lagrange, e_lagrange = solve_lagrange(dh, ip)

function lagrange_error(h::Float64)
    grid = setup_grid(h; origin_refinement = 1)
    ip = Lagrange{RefTriangle, 1}()^2
    dh = close!(add!(DofHandler(grid), :E, ip))
    _, e = solve_lagrange(dh, ip)
    return e
end

#=
```julia
mesh_sizes = (1/2) .^(3:8)
lagrange_errors = Float64[]
for h in mesh_sizes
    println("h = $h")
    e = @time lagrange_error(h)
    push!(lagrange_errors, e)
end
```
=#

tr = Triangulation(dh, 2)
data = create_data(tr, grid, a_lagrange, (ip,); f = first)

ax = Plt.Axis(fig[2, 1]; aspect = Plt.DataAspect())

nodes = [GB.Point(x.data) for x in tr.nodes]
m = Plt.mesh!(
    ax, nodes, reshape(tr.triangles, :); color = data,
    colormap = Plt.Makie.wong_colors(),
    interpolate = true,
    colorrange = (-2.0, 0.0),
)
Plt.Colorbar(fig[2, 2], m)
fig
