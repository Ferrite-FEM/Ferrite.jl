using Pkg
Pkg.activate(@__DIR__) # Ensure correct environment

include(joinpath(@__DIR__, "..", "test", "integration", "convergence_test_utils.jl"))

using Ferrite, Tensors
using Test
using .ConvergenceTestHelper:
    get_geometry, get_num_elements, get_quadrature_order,
    setup_poisson_problem, solve, check_and_compute_convergence_norms

using Ferrite: getrefdim, getorder
import CairoMakie as Plt

function get_convergence_rate(ip::Interpolation)
    @show ip
    nel_base = ceil(Int, 11 / getorder(ip))
    nels = nel_base * (2 .^ (1:4))
    CT = get_geometry(ip) # Cell type
    ip_geo = geometric_interpolation(CT)

    L2_norms = Float64[]
    H1_norms = Float64[]
    for nel in nels
        grid = generate_grid(CT, ntuple(x -> nel, getrefdim(CT)))
        qr_order = get_quadrature_order(ip)
        qr = QuadratureRule{getrefshape(ip)}(qr_order)
        dh, ch, cellvalues = setup_poisson_problem(grid, ip, ip_geo, qr)
        u = solve(dh, ch, cellvalues)
        L2_i, H1_i, _ = check_and_compute_convergence_norms(dh, u, cellvalues, Inf)
        push!(L2_norms, L2_i)
        push!(H1_norms, H1_i)
    end
    return (; nels, L2_norms, H1_norms)
end

function plot_convergence_rates(data::Vector{<:Pair})
    fig = Plt.Figure(; size = (1000, 400))
    Plt.Label(fig[0, :], "Convergence for Lagrange{RefTriangle, p}")
    ax_L2 = Plt.Axis(fig[1, 1]; yscale = log2, xscale = log2, ylabel = L"$L_2$ norm", xlabel = L"$N$ (number of elements)")
    ax_H1 = Plt.Axis(fig[1, 2]; yscale = log2, xscale = log2, ylabel = L"$H^1$ seminorm", xlabel = L"$N$ (number of elements)")
    for (ip, ip_results) in data
        (; nels, L2_norms, H1_norms) = ip_results
        h_rel = nels[1] ./ collect(nels)
        label = string(typeof(ip))
        label = "p = $(getorder(ip))"
        order = getorder(ip)
        Plt.scatter!(ax_L2, nels, L2_norms; label)
        Plt.lines!(ax_L2, nels, L2_norms[1] * h_rel .^ (order + 1); color = :black, linestyle = :dash)
        Plt.scatter!(ax_H1, nels, H1_norms; label)
        Plt.lines!(ax_H1, nels, H1_norms[1] * h_rel .^ order; color = :black, linestyle = :dash)
    end
    Plt.Legend(fig[1, 3], ax_L2)
    return fig
end

results = [
    ip => get_convergence_rate(ip) for ip in [
            Lagrange{RefTriangle, 1}(), Lagrange{RefTriangle, 2}(), Lagrange{RefTriangle, 3}(),
            Lagrange{RefTriangle, 4}(), Lagrange{RefTriangle, 5}(),
        ]
]

fig = plot_convergence_rates(results)
Plt.save(joinpath(@__DIR__, "convergence_rates.pdf"), fig)
fig
