using Ferrite, SparseArrays, LinearAlgebra

grid = generate_grid(Triangle, (40, 40), Vec(0.0, 0.0), Vec(1.0, 1.0));

addcellset!(grid, "barrier", x -> 0.4 ≤ x[1] ≤ 0.6 && (x[2] ≤ 0.4 || x[2] ≥ 0.6))

k = fill(1.0, getncells(grid))            # matrix permeability
k[collect(getcellset(grid, "barrier"))] .= 1.0e-4; # barrier permeability

ip_q = RaviartThomas{RefTriangle, 1}()
ip_p = DiscontinuousLagrange{RefTriangle, 0}()
ip_geo = Lagrange{RefTriangle, 1}()

qr = QuadratureRule{RefTriangle}(2)
cv_q = CellValues(qr, ip_q, ip_geo)
cv_p = CellValues(qr, ip_p, ip_geo)

facet_qr = FacetQuadratureRule{RefTriangle}(2)
fv_q = FacetValues(facet_qr, ip_q, ip_geo);

dh = DofHandler(grid)
add!(dh, :q, ip_q)
add!(dh, :p, ip_p)
close!(dh);

ch = ConstraintHandler(dh)
add!(ch, ProjectedDirichlet(:q, union(getfacetset(grid, "top"), getfacetset(grid, "bottom")), (x, t, n) -> 0.0))
close!(ch);

function assemble_darcy!(K, dh, cv_q, cv_p, k)
    range_q = dof_range(dh, :q)
    range_p = dof_range(dh, :p)
    n_dofs = ndofs_per_cell(dh)
    Ke = zeros(n_dofs, n_dofs)
    assembler = start_assemble(K)
    for cell in CellIterator(dh)
        reinit!(cv_q, cell)
        reinit!(cv_p, cell)
        fill!(Ke, 0.0)
        kᵉ = k[cellid(cell)]
        for qp in 1:getnquadpoints(cv_q)
            dΩ = getdetJdV(cv_q, qp)
            # A and Bᵀ blocks (test function δq)
            for (i, I) in pairs(range_q)
                δNq = shape_value(cv_q, qp, i)
                div_δNq = shape_divergence(cv_q, qp, i)
                for (j, J) in pairs(range_q)
                    Nq = shape_value(cv_q, qp, j)
                    Ke[I, J] += (δNq ⋅ Nq) / kᵉ * dΩ
                end
                for (j, J) in pairs(range_p)
                    Np = shape_value(cv_p, qp, j)
                    Ke[I, J] -= div_δNq * Np * dΩ
                end
            end
            # B block (test function δp)
            for (i, I) in pairs(range_p)
                δNp = shape_value(cv_p, qp, i)
                for (j, J) in pairs(range_q)
                    div_Nq = shape_divergence(cv_q, qp, j)
                    Ke[I, J] -= δNp * div_Nq * dΩ
                end
            end
        end
        assemble!(assembler, celldofs(cell), Ke)
    end
    return K
end

function assemble_pressure_bc!(f, dh, fv_q, facetset, p_D)
    range_q = dof_range(dh, :q)
    fe = zeros(ndofs_per_cell(dh))
    for facet in FacetIterator(dh, facetset)
        reinit!(fv_q, facet)
        fill!(fe, 0.0)
        for qp in 1:getnquadpoints(fv_q)
            dΓ = getdetJdV(fv_q, qp)
            n = getnormal(fv_q, qp)
            for (i, I) in pairs(range_q)
                δNq = shape_value(fv_q, qp, i)
                fe[I] -= (δNq ⋅ n) * p_D * dΓ
            end
        end
        assemble!(f, celldofs(facet), fe)
    end
    return f
end

K = allocate_matrix(dh)
f = zeros(ndofs(dh))
assemble_darcy!(K, dh, cv_q, cv_p, k)
assemble_pressure_bc!(f, dh, fv_q, getfacetset(grid, "left"), 1.0)

apply!(K, f, ch)
a = K \ f;

p_cells = [a[celldofs(dh, cellid)[dof_range(dh, :p)[1]]] for cellid in 1:getncells(grid)]

function collect_qp_fluxes(dh, cv_q, a)
    qp_fluxes = [
        [zero(Vec{2}) for _ in 1:getnquadpoints(cv_q)]
            for _ in 1:getncells(dh.grid)
    ]
    for cell in CellIterator(dh)
        reinit!(cv_q, cell)
        aᵉ = a[celldofs(cell)][dof_range(dh, :q)]
        for qp in 1:getnquadpoints(cv_q)
            qp_fluxes[cellid(cell)][qp] = function_value(cv_q, qp, aᵉ)
        end
    end
    return qp_fluxes
end
qp_fluxes = collect_qp_fluxes(dh, cv_q, a)

proj = L2Projector(Lagrange{RefTriangle, 1}(), grid)
flux_projected = project(proj, qp_fluxes, qr)

VTKGridFile("darcy_flow", dh) do vtk
    write_cell_data(vtk, p_cells, "p")
    write_projection(vtk, proj, flux_projected, "q")
    Ferrite.write_cellset(vtk, grid, "barrier")
end;

function cell_imbalances(dh, cv_q, a)
    imbalances = zeros(getncells(dh.grid))
    for cell in CellIterator(dh)
        reinit!(cv_q, cell)
        aᵉ = a[celldofs(cell)][dof_range(dh, :q)]
        for qp in 1:getnquadpoints(cv_q)
            imbalances[cellid(cell)] += function_divergence(cv_q, qp, aᵉ) * getdetJdV(cv_q, qp)
        end
    end
    return imbalances
end
maximum(abs, cell_imbalances(dh, cv_q, a))

function boundary_flux(dh, fv_q, facetset, a)
    Q = 0.0
    for facet in FacetIterator(dh, facetset)
        reinit!(fv_q, facet)
        aᵉ = a[celldofs(facet)][dof_range(dh, :q)]
        for qp in 1:getnquadpoints(fv_q)
            Q += (function_value(fv_q, qp, aᵉ) ⋅ getnormal(fv_q, qp)) * getdetJdV(fv_q, qp)
        end
    end
    return Q
end
Q_in = boundary_flux(dh, fv_q, getfacetset(grid, "left"), a)
Q_out = boundary_flux(dh, fv_q, getfacetset(grid, "right"), a)
Q_in + Q_out

function solve_primal(grid, k)
    ip = Lagrange{RefTriangle, 1}()
    cv = CellValues(QuadratureRule{RefTriangle}(2), ip)
    dh = DofHandler(grid)
    add!(dh, :p, ip)
    close!(dh)
    ch = ConstraintHandler(dh)
    add!(ch, Dirichlet(:p, getfacetset(grid, "left"), Returns(1.0)))
    add!(ch, Dirichlet(:p, getfacetset(grid, "right"), Returns(0.0)))
    close!(ch)
    K = allocate_matrix(dh)
    f = zeros(ndofs(dh))
    assembler = start_assemble(K)
    Ke = zeros(ndofs_per_cell(dh), ndofs_per_cell(dh))
    for cell in CellIterator(dh)
        reinit!(cv, cell)
        fill!(Ke, 0.0)
        for qp in 1:getnquadpoints(cv)
            dΩ = getdetJdV(cv, qp)
            for i in 1:getnbasefunctions(cv)
                ∇δNp = shape_gradient(cv, qp, i)
                for j in 1:getnbasefunctions(cv)
                    Ke[i, j] += k[cellid(cell)] * (∇δNp ⋅ shape_gradient(cv, qp, j)) * dΩ
                end
            end
        end
        assemble!(assembler, celldofs(cell), Ke)
    end
    apply!(K, f, ch)
    return dh, K \ f
end
dh_primal, p_primal = solve_primal(grid, k);

function boundary_flux_primal(dh, grid, k, facetset, p)
    ip = Lagrange{RefTriangle, 1}()
    fv = FacetValues(FacetQuadratureRule{RefTriangle}(2), ip)
    Q = 0.0
    for facet in FacetIterator(dh, facetset)
        reinit!(fv, facet)
        pᵉ = p[celldofs(facet)]
        for qp in 1:getnquadpoints(fv)
            q_vec = -k[cellid(facet)] * function_gradient(fv, qp, pᵉ)
            Q += (q_vec ⋅ getnormal(fv, qp)) * getdetJdV(fv, qp)
        end
    end
    return Q
end
Q_in_primal = boundary_flux_primal(dh_primal, grid, k, getfacetset(grid, "left"), p_primal)
Q_out_primal = boundary_flux_primal(dh_primal, grid, k, getfacetset(grid, "right"), p_primal)

using Printf
@printf("mixed:  Q_in = %10.6f, Q_out = %10.6f, imbalance = %9.2e\n", Q_in, Q_out, (Q_in + Q_out) / abs(Q_in))
@printf("primal: Q_in = %10.6f, Q_out = %10.6f, imbalance = %9.2e\n", Q_in_primal, Q_out_primal, (Q_in_primal + Q_out_primal) / abs(Q_in_primal))

function flux_jump_mixed(dh, a, topology)
    iv = InterfaceValues(FacetQuadratureRule{RefTriangle}(2), RaviartThomas{RefTriangle, 1}(), Lagrange{RefTriangle, 1}())
    range_q = dof_range(dh, :q)
    J = 0.0
    for ic in InterfaceIterator(dh, topology)
        reinit!(iv, ic)
        aᵉ = vcat(a[celldofs(ic.a)][range_q], a[celldofs(ic.b)][range_q])
        for qp in 1:getnquadpoints(iv)
            jump = function_value_jump(iv, qp, aᵉ) ⋅ getnormal(iv, qp)
            J += abs(jump) * getdetJdV(iv, qp)
        end
    end
    return J
end

function flux_jump_primal(dh, p, k, topology)
    iv = InterfaceValues(FacetQuadratureRule{RefTriangle}(2), Lagrange{RefTriangle, 1}())
    J = 0.0
    for ic in InterfaceIterator(dh, topology)
        reinit!(iv, ic)
        pᵉ = vcat(p[celldofs(ic.a)], p[celldofs(ic.b)])
        for qp in 1:getnquadpoints(iv)
            n = getnormal(iv, qp)
            q_here = -k[cellid(ic.a)] * function_gradient(iv, qp, pᵉ; here = true)
            q_there = -k[cellid(ic.b)] * function_gradient(iv, qp, pᵉ; here = false)
            J += abs((q_here - q_there) ⋅ n) * getdetJdV(iv, qp)
        end
    end
    return J
end

topology = ExclusiveTopology(grid)
J_mixed = flux_jump_mixed(dh, a, topology)
J_primal = flux_jump_primal(dh_primal, p_primal, k, topology)
@printf("normal-flux jump, mixed:  %9.2e\n", J_mixed / abs(Q_in))
@printf("normal-flux jump, primal: %9.2e\n", J_primal / abs(Q_in_primal))

using Test                                                            #hide
@test maximum(abs, cell_imbalances(dh, cv_q, a)) < 1.0e-12            #hide
@test abs(Q_in + Q_out) / abs(Q_in) < 1.0e-12                         #hide
@test J_mixed / abs(Q_in) < 1.0e-12                                   #hide
@test J_primal / abs(Q_in_primal) > 1.0                               #hide
@test 1.0e-5 < abs(Q_in_primal + Q_out_primal) / abs(Q_in_primal) < 1.0e-3 #hide
@test Q_in < 0 < Q_out                                                #hide
@test Q_out ≈ 0.3830975 atol = 1.0e-4                                 #hide
nothing                                                               #hide

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl
