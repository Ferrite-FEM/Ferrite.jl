# # [Adaptive Linear Elasticity](@id tutorial-elasticity-adaptivity)
#
# ![Adaptively refined L-shape mesh, concentrated at the re-entrant corner.](elasticity_adaptivity.png)
#
# ## Introduction
#
# This tutorial demonstrates adaptive mesh refinement (AMR) for 2D linear
# elasticity using Ferrite's `ForestBWG` — a p4est-style forest-of-octrees data
# structure. In contrast to the [heat equation AMR tutorial](@ref tutorial-heat-adaptivity),
# the mesh here is *imported from an Abaqus input file* with
# [FerriteMeshParser](https://github.com/Ferrite-FEM/FerriteMeshParser.jl) rather
# than generated, showing that the adaptive machinery works on arbitrary
# quadrilateral meshes.
#
# The domain `case1.inp` is the classic **L-shape**: the square $[-1,1]^2$ with the
# upper-right quadrant $[0,1]\times[0,1]$ removed, leaving a re-entrant corner at the
# origin $(0,0)$. That corner produces a stress singularity when the body is loaded —
# exactly the kind of localized feature that adaptive refinement resolves efficiently.
#
# We clamp the left edge, pull the (lower) right edge with a horizontal traction,
# and drive refinement with a facet-jump (Kelly-type) error estimator combined with
# Dörfler (bulk) marking. Hanging nodes introduced by refinement are made conforming
# with a `ConformityConstraint`.
#
# ## Commented Program
#
# First we load the required packages.
using Ferrite, FerriteMeshParser, Tensors, SparseArrays, WriteVTK, Downloads

# ### Mesh import
# We read the Abaqus mesh with FerriteMeshParser's `get_ferrite_grid`. The file
# uses `CPS4R` (4-node plane-stress quadrilaterals), which are mapped to Ferrite
# `Quadrilateral` cells — exactly the cell type required by `ForestBWG`. The mesh
# is fetched from the Ferrite asset storage if it is not available locally.
#
# The input file does not define any node or element sets, so we attach the
# boundary facet sets we need geometrically: the straight left edge $x=-1$ and the
# straight lower-right edge $x=+1$. These sets are carried through refinement by
# `creategrid`, so they remain valid on every adapted mesh.
function setup_grid()
    gridfile = "case1.inp"
    isfile(gridfile) || Downloads.download(Ferrite.asset_url(gridfile), gridfile)
    grid = get_ferrite_grid(gridfile)
    addfacetset!(grid, "left", x -> x[1] ≈ -1.0)
    addfacetset!(grid, "right", x -> x[1] ≈ 1.0)
    return grid
end

# We wrap the imported grid in a `ForestBWG` that allows up to 7 levels of
# refinement. Each original quadrilateral becomes the root of an octree and is
# subdivided in reference space, with physical coordinates obtained by bilinear
# interpolation of the original element corners.
base_grid = setup_grid()
forest = ForestBWG(base_grid, 20)

# ### Material behavior
# We use plane-stress linear isotropic elasticity (matching the `CPS4R` element
# type). Starting from Young's modulus $E$ and Poisson's ratio $\nu$, the
# plane-stress stiffness in Voigt form is converted to a 4th-order tensor.
const Emod = 200.0e3 # Young's modulus [MPa]
const ν = 0.3        # Poisson's ratio [-]
const Cmat = let
    C_voigt = Emod / (1 - ν^2) * [1.0 ν 0.0; ν 1.0 0.0; 0.0 0.0 (1 - ν) / 2]
    fromvoigt(SymmetricTensor{4, 2}, C_voigt)
end

# The applied traction on the right edge (a horizontal pull).
traction(x) = Vec{2}((1.0e3, 0.0))

# ### Element assembly
# Standard small-strain elasticity stiffness, $k_{ij} = \int_K \nabla N_i : \mathsf{C} : \nabla^\mathrm{sym} N_j \, \mathrm{d}\Omega$.
function assemble_cell!(ke, cellvalues, C)
    fill!(ke, 0.0)
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

# ### Global assembly
function assemble_global!(K, dh, cellvalues, C)
    n_basefuncs = getnbasefunctions(cellvalues)
    ke = zeros(n_basefuncs, n_basefuncs)
    assembler = start_assemble(K)
    for cell in CellIterator(dh)
        reinit!(cellvalues, cell)
        assemble_cell!(ke, cellvalues, C)
        assemble!(assembler, celldofs(cell), ke)
    end
    return K
end

# ### External (Neumann) forces
# The traction is integrated over the loaded facet set.
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

# ### Solve on a single grid
# Given a (non-conforming) grid we set up the FE problem with vector-valued
# bilinear quadrilateral elements: clamp the left edge, apply the traction on the
# right edge, and add a `ConformityConstraint` so the displacement is continuous
# across hanging nodes.
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
    add!(ch, Dirichlet(:u, getfacetset(grid, "left"), (x, t) -> 0.0, 1))
    add!(ch, Dirichlet(:u, getfacetset(grid, "left"), (x, t) -> 0.0, 2))
    add!(ch, ConformityConstraint(:u))
    close!(ch)

    K = allocate_matrix(dh, ch)
    f = zeros(ndofs(dh))
    assemble_global!(K, dh, cellvalues, C)
    assemble_external_forces!(f, dh, getfacetset(grid, "right"), facetvalues, traction)
    apply!(K, f, ch)
    u = K \ f
    apply!(u, ch)
    return u, dh, ch, cellvalues, qr
end

# ### Facet-jump (Kelly-type) error estimator
# A recovery-based (Zienkiewicz-Zhu) estimator is a poor fit here: it assumes flux
# superconvergence, which fails at the hanging-node interfaces created by octree
# refinement, so the indicator lights up along refinement fronts and *spreads*
# refinement instead of localizing it (the total estimate then grows with each step).
#
# Instead we use a facet-jump (Kelly-type) estimator: the inter-element jump of the
# normal stress is the natural residual of an elasticity solution and vanishes as the
# mesh resolves the field. Per cell,
# ```math
#  \eta_K^2 = \tfrac{1}{2}\sum_{F \subset \partial K \setminus \partial\Omega} h_F \int_F \|[\![\sigma_h\cdot n]\!]\|^2 \, \mathrm{d}\Gamma
#           + \sum_{F \subset \partial K \cap \Gamma_N} h_F \int_F \|g - \sigma_h\cdot n\|^2 \, \mathrm{d}\Gamma ,
# ```
# where ``[\![\sigma_h\cdot n]\!]`` is the traction jump across the interior facet ``F``
# and the second sum is the boundary residual on the Neumann boundary ``\Gamma_N`` with
# prescribed traction ``g`` (``g = 0`` on traction-free surfaces). Facets on the
# Dirichlet boundary do not contribute.
#
# Both integrals are evaluated with facet quadrature: the owning side is evaluated
# with `FacetValues`, and the neighbouring cell's stress at the same physical
# quadrature point is obtained by inverse mapping (`Ferrite.find_local_coordinate`)
# followed by a `PointValues` evaluation. This treats conforming, across-tree and
# hanging (coarse↔fine) facets uniformly — for a hanging facet the integration runs
# over the fine subfacet and the neighbour is the coarse cell.
#
# !!! note "Why not `ExclusiveTopology`?"
#     The facet neighbours must be the *true* faces of the refined forest, including
#     coarse↔fine (hanging) and across-tree faces. `ExclusiveTopology` only knows the
#     macro (root) mesh and would give a wrong estimator. The hanging (coarse↔fine)
#     adjacency comes directly from the topological records in
#     `grid.conformity_info.hanging_facets` — each record pairs the coarse facet with
#     its covering fine sub-facets, which is exactly the facet-pair list a jump
#     estimator needs. Conforming faces (also across trees) are read off the
#     materialized grid: `creategrid` merges shared nodes across trees, so they appear
#     as shared node-pairs.
function estimate_error(grid, dh, u, C)
    nc = getncells(grid)
    cells = getcells(grid)
    X = [get_node_coordinate(n) for n in getnodes(grid)]

    ip = Lagrange{RefQuadrilateral, 1}()^2
    ip_geo = Lagrange{RefQuadrilateral, 1}()
    fv = FacetValues(FacetQuadratureRule{RefQuadrilateral}(2), ip)
    ## The neighbour side is evaluated at the owning side's quadrature points,
    ## which are arbitrary points in the neighbour's reference cell: inverse-map
    ## them with `find_local_coordinate` and evaluate with `PointValues`.
    pv = PointValues(ip)
    finder = Ferrite.NewtonLineSearchPointFinder()
    function stress_at(c, x)
        coords = getcoordinates(grid, c)
        converged, ξ = Ferrite.find_local_coordinate(ip_geo, coords, x, finder)
        @assert converged
        reinit!(pv, coords, ξ)
        return C ⊡ function_symmetric_gradient(pv, u[celldofs(dh, c)])
    end

    ## Map every facet (as a sorted node-id pair) to the `FacetIndex`(es) that own it.
    ## This resolves the *conforming* adjacency (hanging interfaces come from the
    ## `hanging_facets` records below). TODO: the node-pair key is hard-coded for 2D —
    ## in 3D a facet is a 4-node face; a conforming-adjacency service on `ForestBWG`
    ## would generalize this.
    facet_owner = Dict{Tuple{Int, Int}, Vector{FacetIndex}}()
    for c in 1:nc
        for (f, fnodes) in enumerate(Ferrite.facets(cells[c]))
            key = minmax(fnodes...)
            push!(get!(() -> FacetIndex[], facet_owner, key), FacetIndex(c, f))
        end
    end

    error_arr = zeros(nc)

    ## Squared traction jump over the facet `fi` against neighbour cell `cB`,
    ## split evenly between the two cells. For hanging facets `fi` is the fine
    ## subfacet and `cB` the coarse cell.
    function add_jump!(fi::FacetIndex, cB::Int, na::Int, nb::Int)
        cA, fA = fi[1], fi[2]
        coordsA = getcoordinates(grid, cA)
        reinit!(fv, coordsA, fA)
        ueA = u[celldofs(dh, cA)]
        hF = norm(X[nb] - X[na])
        s = 0.0
        for qp in 1:getnquadpoints(fv)
            x = spatial_coordinate(fv, qp, coordsA)
            n = getnormal(fv, qp)
            jump = (C ⊡ function_symmetric_gradient(fv, qp, ueA) - stress_at(cB, x)) ⋅ n
            s += (jump ⋅ jump) * getdetJdV(fv, qp)
        end
        contrib = hF * s
        error_arr[cA] += 0.5 * contrib
        error_arr[cB] += 0.5 * contrib
        return nothing
    end

    ## Boundary residual over the domain-boundary facet `fi` with prescribed traction `g`.
    function add_boundary!(fi::FacetIndex, na::Int, nb::Int, g)
        cA, fA = fi[1], fi[2]
        coordsA = getcoordinates(grid, cA)
        reinit!(fv, coordsA, fA)
        ueA = u[celldofs(dh, cA)]
        hF = norm(X[nb] - X[na])
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

    ## Hanging interfaces straight from the topological records: integrate the jump
    ## over each fine sub-facet against the coarse neighbour cell, and remember both
    ## sides' node-pair keys so the conforming/boundary loop below skips them.
    hanging_keys = Set{Tuple{Int, Int}}()
    for rec in grid.conformity_info.hanging_facets
        push!(hanging_keys, minmax(Ferrite.facets(cells[rec.coarse[1]])[rec.coarse[2]]...))
        for ff in rec.fine
            fnodes = Ferrite.facets(cells[ff[1]])[ff[2]]
            push!(hanging_keys, minmax(fnodes...))
            add_jump!(ff, rec.coarse[1], fnodes[1], fnodes[2])
        end
    end

    ## Dirichlet facets carry no boundary residual; the Neumann facets the applied
    ## traction; all other boundary facets are traction-free.
    dirichlet_facets = getfacetset(grid, "left")
    neumann_facets = getfacetset(grid, "right")
    zero_traction(x) = zero(Vec{2})

    for (key, owners) in facet_owner
        key in hanging_keys && continue
        a, b = key
        if length(owners) == 2
            ## Conforming facet (possibly across trees): both cells share this facet.
            add_jump!(owners[1], owners[2][1], a, b)
        else
            ## Domain-boundary facet: Neumann/free-surface boundary residual.
            owners[1] in dirichlet_facets && continue
            g = owners[1] in neumann_facets ? traction : zero_traction
            add_boundary!(owners[1], a, b, g)
        end
    end
    return error_arr
end

# ### Dörfler marking
# Sort cells by decreasing error and mark the smallest set whose cumulative error
# accounts for a fraction $\theta$ of the total.
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

# ### Adaptive solve loop
# Repeat: materialize the forest, solve, estimate, mark, refine, and enforce 2:1
# balance. `nsteps` is kept comfortably below the forest's maximum refinement level
# `b` (set when constructing the `ForestBWG`): a cell may not be refined past level `b`,
# so the corner — refined at almost every step — must not reach it.
function solve_adaptive(initial_forest; nsteps = 4, θ = 0.3)
    forest = deepcopy(initial_forest)
    pvd = paraview_collection("elasticity_amr")
    for i in 1:nsteps
        ## Materialize the forest into a NonConformingGrid and solve.
        grid = Ferrite.creategrid(forest)
        u, dh, ch, cv, qr = solve(grid, Cmat)

        ## Estimate the error and mark cells with Dörfler marking.
        error_arr = estimate_error(grid, dh, u, Cmat)
        cells_to_refine, total = dorfler_mark(error_arr, θ)
        @info "AMR step $i: $(getncells(grid)) cells, $(length(cells_to_refine)) marked, total error = $total"

        ## Export displacement and the cell-wise error indicator.
        VTKGridFile("elasticity_amr-$i", dh) do vtk
            write_solution(vtk, dh, u)
            write_cell_data(vtk, error_arr, "error indicator")
            pvd[i] = vtk
        end

        isempty(cells_to_refine) && break

        ## Refine the marked cells and restore 2:1 balance across the forest.
        Ferrite.refine!(forest, cells_to_refine)
        Ferrite.balanceforest!(forest)
    end
    vtk_save(pvd)
    return forest
end

# ### Run
solve_adaptive(forest)
