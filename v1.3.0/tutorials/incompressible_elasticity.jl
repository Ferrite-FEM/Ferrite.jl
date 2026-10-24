using Ferrite, Tensors
using BlockArrays, SparseArrays, LinearAlgebra

function create_cook_grid(nx, ny)
    corners = [
        Vec{2}((0.0, 0.0)),
        Vec{2}((48.0, 44.0)),
        Vec{2}((48.0, 60.0)),
        Vec{2}((0.0, 44.0)),
    ]
    grid = generate_grid(Triangle, (nx, ny), corners)
    # facesets for boundary conditions
    addfacetset!(grid, "clamped", x -> norm(x[1]) ≈ 0.0)
    addfacetset!(grid, "traction", x -> norm(x[1]) ≈ 48.0)
    return grid
end;

function create_values(interpolation_u, interpolation_p)
    # quadrature rules
    qr = QuadratureRule{RefTriangle}(3)
    facet_qr = FacetQuadratureRule{RefTriangle}(3)

    # cell and FacetValues for u
    cellvalues_u = CellValues(qr, interpolation_u)
    facetvalues_u = FacetValues(facet_qr, interpolation_u)

    # cellvalues for p
    cellvalues_p = CellValues(qr, interpolation_p)

    return cellvalues_u, cellvalues_p, facetvalues_u
end;

function create_dofhandler(grid, ipu, ipp)
    dh = DofHandler(grid)
    add!(dh, :u, ipu) # displacement
    add!(dh, :p, ipp) # pressure
    close!(dh)
    return dh
end;

function create_bc(dh)
    dbc = ConstraintHandler(dh)
    add!(dbc, Dirichlet(:u, getfacetset(dh.grid, "clamped"), x -> zero(x), [1, 2]))
    close!(dbc)
    return dbc
end;

struct LinearElasticity{T}
    G::T
    K::T
end

function doassemble(
        cellvalues_u::CellValues,
        cellvalues_p::CellValues,
        facetvalues_u::FacetValues,
        K::SparseMatrixCSC, grid::Grid, dh::DofHandler, mp::LinearElasticity
    )
    f = zeros(ndofs(dh))
    assembler = start_assemble(K, f)
    nu = getnbasefunctions(cellvalues_u)
    np = getnbasefunctions(cellvalues_p)

    fe = BlockedArray(zeros(nu + np), [nu, np]) # local force vector
    ke = BlockedArray(zeros(nu + np, nu + np), [nu, np], [nu, np]) # local stiffness matrix

    # traction vector
    t = Vec{2}((0.0, 1 / 16))

    for cell in CellIterator(dh)
        fill!(ke, 0)
        fill!(fe, 0)
        assemble_up!(ke, fe, cell, cellvalues_u, cellvalues_p, facetvalues_u, grid, mp, t)
        assemble!(assembler, celldofs(cell), ke, fe)
    end

    return K, f
end;

function dev_3d(t::SymmetricTensor{2, 2, T}) where {T}
    # Given 2d and 3d tensors, t2 and t3, where the out-of-plane components for t3 are zero,
    # we have t2 ⊡ t2 == t3 ⊡ t3, but dev(t2) ⊡ dev(t2) != dev(t3) ⊡ dev(t3), so we have to
    # expand the tensor before calling `dev` to get the correct value in the element routine.
    return dev(SymmetricTensor{2, 3}((i, j) -> (i ≤ 2 && j ≤ 2) ? t[i, j] : zero(T)))
end

function assemble_up!(Ke, fe, cell, cellvalues_u, cellvalues_p, facetvalues_u, grid, mp, t)

    n_basefuncs_u = getnbasefunctions(cellvalues_u)
    n_basefuncs_p = getnbasefunctions(cellvalues_p)
    u▄, p▄ = 1, 2
    reinit!(cellvalues_u, cell)
    reinit!(cellvalues_p, cell)

    # We only assemble lower half triangle of the stiffness matrix and then symmetrize it.
    for q_point in 1:getnquadpoints(cellvalues_u)
        dΩ = getdetJdV(cellvalues_u, q_point)
        for i in 1:n_basefuncs_u
            ɛdev_i = dev_3d(symmetric(shape_gradient(cellvalues_u, q_point, i)))
            for j in 1:i
                ɛdev_j = dev_3d(symmetric(shape_gradient(cellvalues_u, q_point, j)))
                Ke[BlockIndex((u▄, u▄), (i, j))] += 2 * mp.G * ɛdev_i ⊡ ɛdev_j * dΩ
            end
        end

        for i in 1:n_basefuncs_p
            δp = shape_value(cellvalues_p, q_point, i)
            for j in 1:n_basefuncs_u
                divδu = shape_divergence(cellvalues_u, q_point, j)
                Ke[BlockIndex((p▄, u▄), (i, j))] += -δp * divδu * dΩ
            end
            for j in 1:i
                p = shape_value(cellvalues_p, q_point, j)
                Ke[BlockIndex((p▄, p▄), (i, j))] += - 1 / mp.K * δp * p * dΩ
            end

        end
    end

    symmetrize_lower!(Ke)

    # We integrate the Neumann boundary using the FacetValues.
    # We loop over all the facets in the cell, then check if the facet
    # is in our `"traction"` facetset.
    for facet in 1:nfacets(cell)
        if (cellid(cell), facet) ∈ getfacetset(grid, "traction")
            reinit!(facetvalues_u, cell, facet)
            for q_point in 1:getnquadpoints(facetvalues_u)
                dΓ = getdetJdV(facetvalues_u, q_point)
                for i in 1:n_basefuncs_u
                    δu = shape_value(facetvalues_u, q_point, i)
                    fe[i] += (δu ⋅ t) * dΓ
                end
            end
        end
    end
    return
end

function symmetrize_lower!(Ke)
    for i in 1:size(Ke, 1)
        for j in (i + 1):size(Ke, 1)
            Ke[i, j] = Ke[j, i]
        end
    end
    return
end;

function compute_stresses(
        cellvalues_u::CellValues, cellvalues_p::CellValues,
        dh::DofHandler, mp::LinearElasticity, a::Vector
    )
    ae = zeros(ndofs_per_cell(dh)) # local solution vector
    u_range = dof_range(dh, :u)    # local range of dofs corresponding to u
    p_range = dof_range(dh, :p)    # local range of dofs corresponding to p
    # Allocate storage for the stresses
    σ = zeros(SymmetricTensor{2, 3}, getncells(dh.grid))
    # Loop over the cells and compute the cell-average stress
    for cc in CellIterator(dh)
        # Update cellvalues
        reinit!(cellvalues_u, cc)
        reinit!(cellvalues_p, cc)
        # Extract the cell local part of the solution
        for (i, I) in pairs(celldofs(cc))
            ae[i] = a[I]
        end
        # Loop over the quadrature points
        σΩi = zero(SymmetricTensor{2, 3}) # stress integrated over the cell
        Ωi = 0.0                          # cell volume (area)
        for qp in 1:getnquadpoints(cellvalues_u)
            dΩ = getdetJdV(cellvalues_u, qp)
            # Evaluate the strain and the pressure
            ε = function_symmetric_gradient(cellvalues_u, qp, ae, u_range)
            p = function_value(cellvalues_p, qp, ae, p_range)
            # Expand strain to 3D
            ε3D = SymmetricTensor{2, 3}((i, j) -> i < 3 && j < 3 ? ε[i, j] : 0.0)
            # Compute the stress in this quadrature point
            σqp = 2 * mp.G * dev(ε3D) - one(ε3D) * p
            σΩi += σqp * dΩ
            Ωi += dΩ
        end
        # Store the value
        σ[cellid(cc)] = σΩi / Ωi
    end
    return σ
end;

function solve(ν, interpolation_u, interpolation_p)
    # material
    Emod = 1.0
    Gmod = Emod / 2(1 + ν)
    Kmod = Emod * ν / (3 * (1 - 2ν))
    mp = LinearElasticity(Gmod, Kmod)

    # Grid, dofhandler, boundary condition
    n = 50
    grid = create_cook_grid(n, n)
    dh = create_dofhandler(grid, interpolation_u, interpolation_p)
    dbc = create_bc(dh)

    # CellValues
    cellvalues_u, cellvalues_p, facetvalues_u = create_values(interpolation_u, interpolation_p)

    # Assembly and solve
    K = allocate_matrix(dh)
    K, f = doassemble(cellvalues_u, cellvalues_p, facetvalues_u, K, grid, dh, mp)
    apply!(K, f, dbc)
    u = K \ f

    # Compute the stress
    σ = compute_stresses(cellvalues_u, cellvalues_p, dh, mp, u)
    σvM = map(x -> √(3 / 2 * dev(x) ⊡ dev(x)), σ) # von Mise effective stress

    # Export the solution and the stress
    filename = "cook_" *
        (interpolation_u == Lagrange{RefTriangle, 1}()^2 ? "linear" : "quadratic") *
        "_linear"

    VTKGridFile(filename, grid) do vtk
        write_solution(vtk, dh, u)
        for i in 1:3, j in 1:3
            σij = [x[i, j] for x in σ]
            write_cell_data(vtk, σij, "sigma_$(i)$(j)")
        end
        write_cell_data(vtk, σvM, "sigma von Mises")
    end
    return u
end

linear_p = Lagrange{RefTriangle, 1}()
linear_u = Lagrange{RefTriangle, 1}()^2
quadratic_u = Lagrange{RefTriangle, 2}()^2

u1 = solve(0.5, linear_u, linear_p);
u2 = solve(0.5, quadratic_u, linear_p);

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl
