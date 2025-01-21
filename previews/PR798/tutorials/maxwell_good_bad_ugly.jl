using Ferrite, Tensors, ForwardDiff
using Gmsh, FerriteGmsh
using FerriteTriangulation: Triangulation, SubTriangulation
import CairoMakie as Plt
import GeometryBasics as GB

using FerriteAssembly

fig = Plt.Figure()
ax = Plt.Axis(fig[1, 1]; xlabel = "x", ylabel = "y")
points = [(0, 0), (1, 0), (1, 1), (-1, 1), (-1, -1), (0, -1), (0, 0)]
Plt.lines!(ax, first.(points), last.(points))
fig #hide

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
    top = ExclusiveTopology(grid)
    addboundaryfacetset!(grid, top, "vertical_facets", x -> abs((x[1] - 1) * x[1] * (x[1] + 1)) ≤ 1.0e-6)
    addboundaryfacetset!(grid, top, "horizontal_facets", x -> abs((x[2] - 1) * x[2] * (x[2] + 1)) ≤ 1.0e-6)
    bfacets = union(getfacetset(grid, "vertical_facets"), getfacetset(grid, "horizontal_facets"))
    addfacetset!(grid, "boundary_facets", bfacets)
    return grid
end

function _create_data!(f, data, a, cvs, subtria::SubTriangulation, dr::UnitRange)
    sdh = subtria.sdh
    grid = sdh.dh.grid
    c1 = first(subtria.faces)[1]
    x = copy(getcoordinates(grid, c1))
    dofs = copy(celldofs(sdh, c1))
    ae = zeros(eltype(a), length(dofs))
    for (i, (cellnr, facenr)) in enumerate(subtria.faces)
        cv = cvs[facenr]
        getcoordinates!(x, grid, cellnr)
        reinit!(cv, getcells(grid, cellnr), x)
        celldofs!(dofs, sdh, cellnr)
        copyto!(ae, view(a, dofs))
        node_idxs = subtria.face_nodes[i]:(subtria.face_nodes[i + 1] - 1)
        for q_point in 1:getnquadpoints(cv)
            data[node_idxs[q_point]] = f(function_value(cv, q_point, ae, dr))
        end
    end
    return
end

"""
    create_data(tr::Triangulation, grid::AbstractGrid, a::Vector{<:Number}, ::NTuple{N, <:Interpolation};
        f = identity)

Create scalar data by evaluating `f(function_value(...))` at each triangulation node in the `grid`.
"""
function create_data(tr::Triangulation, fieldname::Symbol, a; f = identity)
    data = zeros(length(tr.nodes))
    dh = first(tr.sub_triangulation).sdh.dh
    if length(a) != ndofs(dh)
        display(dh)
        println((dh = ndofs(dh), a = length(a)))
        error("dof vector length not matching number of dofs in triangulation dh")
    end

    for subtria in tr.sub_triangulation
        sdh = subtria.sdh
        ip = Ferrite.getfieldinterpolation(sdh, fieldname)
        cvs = [CellValues(cr, ip, geometric_interpolation(getcelltype(subtria.sdh))) for cr in subtria.rules]
        _create_data!(f, data, a, cvs, subtria, dof_range(sdh, fieldname))
    end
    return data
end

mesh_size = 0.01
grid = setup_grid(mesh_size; origin_refinement = 1)

dh_ana = close!(add!(DofHandler(grid), :u, DiscontinuousLagrange{RefTriangle, 1}()^2))

function analytical_potential(x::Vec{2}) # Analytical potential to be differentiated
    Δθ = -3π / 4 # Rotate discontinuous line to 4th quadrant
    xp = rotate(x, Δθ)
    r = sqrt(x ⋅ x + eps())
    θ = r ≤ 1.0e-6 ? zero(eltype(x)) : (atan(xp[2], xp[1]) - Δθ)
    return r^(2 // 3) * sin(2θ / 3)
end
analytical_solution(x::Vec{2}) = gradient(analytical_potential, x)

a_ana = zeros(ndofs(dh_ana))

apply_analytical!(a_ana, dh_ana, :u, analytical_solution);

mutable struct L2Error{F}
    l2error::Float64
    volume::Float64
    const exact_fun::F
end

function FerriteAssembly.integrate_cell!(vals::L2Error{F}, state, ae, material, cv::CellValues, cellbuffer) where {F}
    for q_point in 1:getnquadpoints(cv)
        Eh = function_value(cv, q_point, ae)
        x = spatial_coordinate(cv, q_point, getcoordinates(cellbuffer))
        dΩ = getdetJdV(cv, q_point)
        vals.l2error += norm(Eh - vals.exact_fun(x))^2 * dΩ
        vals.volume += dΩ
    end
    return
end
function FerriteAssembly.integrate_cell!(vals::L2Error{F}, state, ae, material, cv::NamedTuple, cellbuffer) where {F}
    for q_point in 1:getnquadpoints(cv.E)
        Eh = function_value(cv.E, q_point, ae, dof_range(cellbuffer, :E))
        x = spatial_coordinate(cv.E, q_point, getcoordinates(cellbuffer))
        dΩ = getdetJdV(cv.E, q_point)
        vals.l2error += norm(Eh - vals.exact_fun(x))^2 * dΩ
        vals.volume += dΩ
    end
    return
end;

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

function solve_lagrange(dh)
    ip = Ferrite.getfieldinterpolation(dh, Ferrite.find_field(dh, :E))
    ch = ConstraintHandler(dh)
    add!(ch, Dirichlet(:E, getfacetset(dh.grid, "horizontal_facets"), (x, _) -> gradient(analytical_potential, x)[2], [2]))
    add!(ch, Dirichlet(:E, getfacetset(dh.grid, "vertical_facets"), (x, _) -> gradient(analytical_potential, x)[1], [1]))
    close!(ch)

    cv = CellValues(QuadratureRule{RefTriangle}(1), ip)
    K = allocate_matrix(dh)
    f = zeros(ndofs(dh))
    db = setup_domainbuffer(DomainSpec(dh, LagrangeMaterial(), cv))
    as = start_assemble(K, f)
    work!(as, db)
    apply!(K, f, ch)
    a = K \ f
    l2_vals = L2Error(0.0, 0.0, analytical_solution)
    work!(Integrator(l2_vals), db; a)
    return a, sqrt(l2_vals.l2error) / l2_vals.volume
end

dh_lagrange = close!(add!(DofHandler(grid), :E, Lagrange{RefTriangle, 1}()^2))
a_lagrange, e_lagrange = solve_lagrange(dh_lagrange)

function lagrange_error(grid)
    ip = Lagrange{RefTriangle, 1}()^2
    dh = close!(add!(DofHandler(grid), :E, ip))
    _, e = solve_lagrange(dh)
    return e
end;

struct NedelecMaterial end
function FerriteAssembly.element_residual!(re, s, ae, ::NedelecMaterial, cv, cellbuffer)
    for q_point in 1:getnquadpoints(cv.E)
        dΩ = getdetJdV(cv.E, q_point)
        E = function_value(cv.E, q_point, ae, dof_range(cellbuffer, :E))
        curlE = function_curl(cv.E, q_point, ae, dof_range(cellbuffer, :E))
        ∇ϕ = function_gradient(cv.ϕ, q_point, ae, dof_range(cellbuffer, :ϕ))
        for (i, I) in pairs(dof_range(cellbuffer, :E))
            δNE = shape_value(cv.E, q_point, i)
            curl_δNE = shape_curl(cv.E, q_point, i)
            re[I] += (curl_δNE ⋅ curlE + δNE ⋅ ∇ϕ) * dΩ
        end
        for (i, I) in pairs(dof_range(cellbuffer, :ϕ))
            gradδNϕ = shape_gradient(cv.ϕ, q_point, i)
            re[I] += (gradδNϕ ⋅ E) * dΩ
        end
    end
    return
end

function solve_nedelec(dh)
    ipE = Ferrite.getfieldinterpolation(dh, Ferrite.find_field(dh, :E))
    ipϕ = Ferrite.getfieldinterpolation(dh, Ferrite.find_field(dh, :ϕ))
    CT = getcelltype(dh.grid)

    ch = ConstraintHandler(dh)
    add!(ch, WeakDirichlet(:E, getfacetset(dh.grid, "boundary_facets"), (x, _, n) -> analytical_solution(x) × n))
    add!(ch, Dirichlet(:ϕ, getfacetset(dh.grid, "boundary_facets"), Returns(0.0)))
    close!(ch)

    qr = QuadratureRule{RefTriangle}(1)
    ipg = geometric_interpolation(CT)
    cv = (E = CellValues(qr, ipE, ipg), ϕ = CellValues(qr, ipϕ, ipg))
    K = allocate_matrix(dh)
    f = zeros(ndofs(dh))
    db = setup_domainbuffer(DomainSpec(dh, NedelecMaterial(), cv); autodiffbuffer = true)
    as = start_assemble(K, f)
    a = zeros(ndofs(dh))
    work!(as, db; a)
    apply!(K, f, ch)
    a .= K \ f
    l2_vals = L2Error(0.0, 0.0, analytical_solution)
    work!(Integrator(l2_vals), db; a)
    return a, sqrt(l2_vals.l2error) / l2_vals.volume
end

ipE = Nedelec{2, RefTriangle, 1}()
ipϕ = Lagrange{RefTriangle, 1}()
dh_nedelec = DofHandler(grid)
add!(dh_nedelec, :E, ipE)
add!(dh_nedelec, :ϕ, ipϕ)
close!(dh_nedelec)

a_nedelec, e_nedelec = solve_nedelec(dh_nedelec)

function nedelec_error(grid)
    ipE = Nedelec{2, RefTriangle, 1}()
    ipϕ = Lagrange{RefTriangle, 1}()
    dh = DofHandler(grid)
    add!(dh, :E, ipE)
    add!(dh, :ϕ, ipϕ)
    close!(dh)
    _, e = solve_nedelec(dh)
    return e
end

function calculate_errors(mesh_sizes)
    lagrange_errors = zeros(length(mesh_sizes))
    nedelec_errors = similar(lagrange_errors)
    for (i, h) in enumerate(mesh_sizes)
        grid = setup_grid(h; origin_refinement = 1)
        lagrange_errors[i] = lagrange_error(grid)
        nedelec_errors[i] = nedelec_error(grid)
    end
    return lagrange_errors, nedelec_errors
end

mesh_sizes = 0.1 * ((1 / 2) .^ (0:5));
lagrange_errors = [0.10849541129807588, 0.09262531863237256, 0.08372918040381809, 0.07939314534131932, 0.07639600032795617, 0.07579072269391056]; #hide
nedelec_errors = [0.02246208564658041, 0.014303181582571767, 0.00906062850047998, 0.0057408487615066535, 0.003616459776120822, 0.0022876386764458592]; #hide

function plot_field(fig_part, dh, fieldname, dofvec, name::String; plot_edges = false, meshkwargs...)
    tr = Triangulation(dh, 2)
    data = create_data(tr, fieldname, dofvec; f = (x -> x[1]))

    ax = Plt.Axis(fig_part; aspect = Plt.DataAspect(), xlabel = "x₁", ylabel = "x₂", title = name)

    nodes = [GB.Point(x.data) for x in tr.nodes]
    m = Plt.mesh!(
        ax, nodes, reshape(tr.triangles, :); color = data,
        colormap = Plt.Makie.wong_colors(),
        interpolate = true,
        meshkwargs...
    )
    if plot_edges
        for i in 2:length(tr.tri_edges)
            Plt.lines!(ax, view(nodes, view(tr.edges, tr.tri_edges[i - 1]:(tr.tri_edges[i] - 1))); color = :black)
        end
    end
    return m
end

fig = let
    fig = Plt.Figure(size = (1000, 600))
    m_ana = plot_field(fig[1, 1], dh_ana, :u, a_ana, "Analytical"; plot_edges = false, colorrange = (-2, 0))
    m_lag = plot_field(fig[1, 2], dh_lagrange, :E, a_lagrange, "Lagrange (h = $mesh_size)"; plot_edges = false, colorrange = (-2, 0))
    m_ned = plot_field(fig[1, 3], dh_nedelec, :E, a_nedelec, "Nedelec (h = $mesh_size)"; plot_edges = false, colorrange = (-2, 0))
    Plt.Colorbar(fig[1, 4], m_ned; label = "E₁")

    ax = Plt.Axis(
        fig[2, 1:2];
        xscale = log10, yscale = log10,
        xlabel = "mesh size, h", ylabel = "error"
    )
    Plt.lines!(ax, mesh_sizes, lagrange_errors; label = "Lagrange")
    Plt.lines!(ax, mesh_sizes, nedelec_errors; label = "Nedelec")
    Plt.axislegend(ax; position = :rb)

    fig
end;

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl
