using Ferrite, FerriteMeshParser, WriteVTK, Downloads

function setup_grid()
    gridfile = "case1.inp"
    isfile(gridfile) || Downloads.download(Ferrite.asset_url(gridfile), gridfile)
    grid = get_ferrite_grid(gridfile)
    addfacetset!(grid, "left", x -> x[1] ≈ -1.0)
    addfacetset!(grid, "right", x -> x[1] ≈ 1.0)
    return grid
end

base_grid = setup_grid()
forest = ForestBWG(base_grid, 20)

const Emod = 200.0e3 # Young's modulus [MPa]
const ν = 0.3        # Poisson's ratio [-]
const Cmat = let
    C_voigt = Emod / (1 - ν^2) * [1.0 ν 0.0; ν 1.0 0.0; 0.0 0.0 (1 - ν) / 2]
    fromvoigt(SymmetricTensor{4, 2}, C_voigt)
end

traction(x) = Vec{2}((1.0e3, 0.0))

function assemble_cell!(ke, cellvalues, C)
    for q_point in 1:getnquadpoints(cellvalues)
        dΩ = getdetJdV(cellvalues, q_point)
        for i in 1:getnbasefunctions(cellvalues)
            ∇Nᵢ = shape_gradient(cellvalues, q_point, i)
            for j in 1:getnbasefunctions(cellvalues)
                ∇ˢʸᵐNⱼ = shape_symmetric_gradient(cellvalues, q_point, j)
                ke[i, j] += (∇Nᵢ ⊡ C ⊡ ∇ˢʸᵐNⱼ) * dΩ
            end
        end
    end
    return ke
end

function assemble_global!(K, dh, cellvalues, C)
    n_basefuncs = getnbasefunctions(cellvalues)
    ke = zeros(n_basefuncs, n_basefuncs)
    assembler = start_assemble(K)
    for cell in CellIterator(dh)
        reinit!(cellvalues, cell)
        fill!(ke, 0.0)
        assemble_cell!(ke, cellvalues, C)
        assemble!(assembler, celldofs(cell), ke)
    end
    return K
end

function assemble_external_forces!(f_ext, dh, facetset, facetvalues, prescribed_traction)
    fe_ext = zeros(getnbasefunctions(facetvalues))
    for facet in FacetIterator(dh, facetset)
        reinit!(facetvalues, facet)
        fill!(fe_ext, 0.0)
        coords = getcoordinates(facet)
        for qp in 1:getnquadpoints(facetvalues)
            x = spatial_coordinate(facetvalues, qp, coords)
            tₚ = prescribed_traction(x)
            dΓ = getdetJdV(facetvalues, qp)
            for i in 1:getnbasefunctions(facetvalues)
                Nᵢ = shape_value(facetvalues, qp, i)
                fe_ext[i] += tₚ ⋅ Nᵢ * dΓ
            end
        end
        assemble!(f_ext, celldofs(facet), fe_ext)
    end
    return f_ext
end

function solve(grid, C)
    dim = 2
    ip = Lagrange{RefQuadrilateral, 1}()^dim
    qr = QuadratureRule{RefQuadrilateral}(2)
    qr_facet = FacetQuadratureRule{RefQuadrilateral}(2)
    cellvalues = CellValues(qr, ip)
    facetvalues = FacetValues(qr_facet, ip)

    dh = DofHandler(grid)
    add!(dh, :u, ip)
    close!(dh)

    ch = ConstraintHandler(dh)
    add!(ch, Dirichlet(:u, getfacetset(grid, "left"), (x, t) -> zero(Vec{2})))
    add!(ch, ConformityConstraint(:u))
    close!(ch)

    K = allocate_matrix(dh, ch)
    f = zeros(ndofs(dh))
    assemble_global!(K, dh, cellvalues, C)
    assemble_external_forces!(f, dh, getfacetset(grid, "right"), facetvalues, traction)
    apply!(K, f, ch)
    u = K \ f
    apply!(u, ch)
    return u, dh, cellvalues
end

function vonmises_stress(grid, dh, u, cv, C)
    σvM = zeros(getncells(grid))
    for cell in CellIterator(dh)
        reinit!(cv, cell)
        ue = u[celldofs(cell)]
        s = 0.0
        for qp in 1:getnquadpoints(cv)
            σ = C ⊡ function_symmetric_gradient(cv, qp, ue)
            s += sqrt(σ[1, 1]^2 - σ[1, 1] * σ[2, 2] + σ[2, 2]^2 + 3 * σ[1, 2]^2)
        end
        σvM[cellid(cell)] = s / getnquadpoints(cv)
    end
    return σvM
end

function estimate_error(forest, grid, dh, u, C)
    nc = getncells(grid)
    cells = getcells(grid)
    X = [get_node_coordinate(n) for n in getnodes(grid)]

    ip = Lagrange{RefQuadrilateral, 1}()^2
    qr_facet = FacetQuadratureRule{RefQuadrilateral}(2)
    fv = FacetValues(qr_facet, ip)      # boundary residuals
    iv = InterfaceValues(qr_facet, ip)  # interior traction jumps

    # The interior facet interfaces of the refined forest: conforming pairs and, for
    # each hanging interface, one (fine subfacet, coarse facet) pair per fine subfacet.
    skeleton = Ferrite.facetskeleton(forest)

    error_arr = zeros(nc)

    # Facet diameter from the facet's node coordinates.
    function facet_diameter(fi::FacetIndex)
        fnodes = Ferrite.facets(cells[fi[1]])[fi[2]]
        return maximum(norm(X[n1] - X[n2]) for n1 in fnodes, n2 in fnodes)
    end

    # Squared traction jump over the interface `(fiA, fiB)`, integrated over the `here`
    # facet `fiA` and split evenly between the two cells. For hanging interfaces `fiA`
    # is the fine subfacet and `fiB` the coarse facet.
    function add_jump!(fiA::FacetIndex, fiB::FacetIndex)
        cA, fA = fiA[1], fiA[2]
        cB, fB = fiB[1], fiB[2]
        coordsA = getcoordinates(grid, cA)
        coordsB = getcoordinates(grid, cB)
        trans = Ferrite.AffineInterfaceTransformation(cells[cA], coordsA, fA, cells[cB], coordsB, fB)
        reinit!(iv, cells[cA], coordsA, fA, cells[cB], coordsB, fB, trans)
        ue = u[vcat(celldofs(dh, cA), celldofs(dh, cB))]
        hF = facet_diameter(fiA)
        s = 0.0
        for qp in 1:getnquadpoints(iv)
            n = getnormal(iv, qp)
            σA = C ⊡ function_symmetric_gradient(iv, qp, ue; here = true)
            σB = C ⊡ function_symmetric_gradient(iv, qp, ue; here = false)
            jump = (σA - σB) ⋅ n
            s += (jump ⋅ jump) * getdetJdV(iv, qp)
        end
        contrib = hF * s
        error_arr[cA] += 0.5 * contrib
        error_arr[cB] += 0.5 * contrib
        return nothing
    end

    # Boundary residual over the domain-boundary facet `fi` with prescribed traction `g`.
    function add_boundary!(fi::FacetIndex, g)
        cA, fA = fi[1], fi[2]
        coordsA = getcoordinates(grid, cA)
        reinit!(fv, coordsA, fA)
        ueA = u[celldofs(dh, cA)]
        hF = facet_diameter(fi)
        s = 0.0
        for qp in 1:getnquadpoints(fv)
            x = spatial_coordinate(fv, qp, coordsA)
            n = getnormal(fv, qp)
            r = g(x) - (C ⊡ function_symmetric_gradient(fv, qp, ueA)) ⋅ n
            s += (r ⋅ r) * getdetJdV(fv, qp)
        end
        error_arr[cA] += hF * s
        return nothing
    end

    # Interior interfaces: integrate the jump from the (fine) `here` side against the
    # neighbouring cell, collecting the covered facets along the way.
    covered = Set{FacetIndex}()
    for (fiA, fiB) in skeleton
        push!(covered, fiA)
        push!(covered, fiB)
        add_jump!(fiA, fiB)
    end

    # Domain-boundary facets are exactly the ones the skeleton does not cover:
    # Dirichlet facets carry no residual, the Neumann facets the applied traction,
    # all other boundary facets are traction-free.
    dirichlet_facets = getfacetset(grid, "left")
    neumann_facets = getfacetset(grid, "right")
    zero_traction(x) = zero(Vec{2})
    for c in 1:nc, f in 1:Ferrite.nfacets(cells[c])
        fi = FacetIndex(c, f)
        (fi in covered || fi in dirichlet_facets) && continue
        g = fi in neumann_facets ? traction : zero_traction
        add_boundary!(fi, g)
    end
    return error_arr
end

function dorfler_mark(error_arr, θ)
    cells_to_refine = Int[]
    sizehint!(cells_to_refine, length(error_arr))
    total = sum(error_arr)
    total > 0 || return cells_to_refine, total
    perm = sortperm(error_arr; rev = true)
    target = θ * total
    acc = 0.0
    for idx in perm
        push!(cells_to_refine, idx)
        acc += error_arr[idx]
        acc >= target && break
    end
    return cells_to_refine, total
end

function solve_adaptive(initial_forest; nsteps = 4, θ = 0.3)
    forest = deepcopy(initial_forest)
    pvd = paraview_collection("elasticity_amr")
    for i in 1:nsteps
        # Materialize the forest into a NonConformingGrid and solve.
        grid = creategrid(forest)
        u, dh, cv = solve(grid, Cmat)

        # Estimate the error and mark cells with Dörfler marking.
        error_arr = estimate_error(forest, grid, dh, u, Cmat)
        cells_to_refine, total = dorfler_mark(error_arr, θ)
        @info "AMR step $i: $(getncells(grid)) cells, $(length(cells_to_refine)) marked, total error = $total"

        # Export displacement, von Mises stress and the cell-wise error indicator.
        VTKGridFile("elasticity_amr-$i", dh) do vtk
            write_solution(vtk, dh, u)
            write_cell_data(vtk, vonmises_stress(grid, dh, u, cv, Cmat), "von Mises [MPa]")
            write_cell_data(vtk, error_arr, "error indicator")
            pvd[i] = vtk
        end

        isempty(cells_to_refine) && break

        # Refine the marked cells and restore 2:1 balance across the forest.
        refine!(forest, cells_to_refine)
        balanceforest!(forest)
    end
    vtk_save(pvd)
    return forest
end

solve_adaptive(forest);

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl
