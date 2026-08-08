using Ferrite

L = 1.0  # beam length
EI = 1.0 # bending stiffness
q = 1.0  # distributed load
P = 1.0  # tip point load
M̄ = 1.0; # tip moment

grid = generate_grid(Line, (4,), Vec((0.0,)), Vec((L,)));

ip = Hermite{RefLine, 3}();

qr = QuadratureRule{RefLine}(2);

cellvalues = CellValues(qr, ip; update_hessians = true);

dh = DofHandler(grid)
add!(dh, :w, ip)
close!(dh);

ch = ConstraintHandler(dh)
add!(ch, Dirichlet(:w, getfacetset(grid, "left"), (x, t) -> 0.0));

add!(ch, Dirichlet(:w, getfacetset(grid, "left"), (x, t) -> 0.0; kind = :derivative))
close!(ch);

function assemble_global!(K, f, cellvalues, dh, EI, q)
    n_basefuncs = getnbasefunctions(cellvalues)
    Ke = zeros(n_basefuncs, n_basefuncs)
    fe = zeros(n_basefuncs)
    assembler = start_assemble(K, f)
    for cell in CellIterator(dh)
        reinit!(cellvalues, cell)
        fill!(Ke, 0)
        fill!(fe, 0)
        for q_point in 1:getnquadpoints(cellvalues)
            dΩ = getdetJdV(cellvalues, q_point)
            for i in 1:n_basefuncs
                δw = shape_value(cellvalues, q_point, i)
                δw′′ = shape_hessian(cellvalues, q_point, i)
                fe[i] += q * δw * dΩ
                for j in 1:n_basefuncs
                    w′′ = shape_hessian(cellvalues, q_point, j)
                    Ke[i, j] += EI * (δw′′ ⊡ w′′) * dΩ
                end
            end
        end
        assemble!(assembler, celldofs(cell), Ke, fe)
    end
    return K, f
end

K = allocate_matrix(dh)
f = zeros(ndofs(dh))
assemble_global!(K, f, cellvalues, dh, EI, q);

tip_dofs = celldofs(dh, getncells(grid))
f[tip_dofs[3]] += P
f[tip_dofs[4]] += M̄;

apply!(K, f, ch)
u = K \ f;

ξs = [Vec((ξ,)) for ξ in range(-1.0, 1.0; length = 21)]
qr_plot = QuadratureRule{RefLine}(zeros(length(ξs)), ξs)
cv_plot = CellValues(qr_plot, ip; update_hessians = true, update_detJdV = false)

x_fe = Float64[]; w_fe = Float64[]; M_fe = Float64[]
for cell in CellIterator(dh)
    reinit!(cv_plot, cell)
    ue = u[celldofs(cell)]
    for q_point in 1:getnquadpoints(cv_plot)
        push!(x_fe, spatial_coordinate(cv_plot, q_point, getcoordinates(cell))[1])
        push!(w_fe, function_value(cv_plot, q_point, ue))
        push!(M_fe, EI * function_hessian(cv_plot, q_point, ue)[1, 1])
    end
end

w_ana(x) = (q * x^2 * (x^2 - 4L * x + 6L^2) / 24 + P * x^2 * (3L - x) / 6 + M̄ * x^2 / 2) / EI
M_ana(x) = q * (L - x)^2 / 2 + P * (L - x) + M̄;

import Plots

node_x = range(0.0, L; length = getncells(grid) + 1)
# Transparent background and gray foreground render well on both the light and the dark
# documentation theme
theme = (background_color = :transparent, foreground_color = :gray)
plt_w = Plots.plot(x_fe, w_fe; label = "FE solution", color = 1, linewidth = 2, theme...)
Plots.plot!(plt_w, w_ana; xlims = (0, L), label = "analytic", color = :gray, linestyle = :dash)
Plots.scatter!(plt_w, node_x, evaluate_at_grid_nodes(dh, u, :w); label = "nodal values", color = 1)
Plots.plot!(plt_w; xlabel = "x", ylabel = "deflection w", legend = :topleft)

plt_M = Plots.plot(x_fe, M_fe; label = "FE solution", color = 1, linewidth = 2, theme...)
Plots.plot!(plt_M, M_ana; xlims = (0, L), label = "analytic", color = :gray, linestyle = :dash)
Plots.plot!(plt_M; xlabel = "x", ylabel = "bending moment M", legend = :topright)

using Test                                                       #hide
@test u[tip_dofs[3]] ≈ (q * L^4 / 8 + P * L^3 / 3 + M̄ * L^2 / 2) / EI #hide
@test u[tip_dofs[4]] ≈ (q * L^3 / 6 + P * L^2 / 2 + M̄ * L) / EI      #hide
nothing                                                          #hide

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl
