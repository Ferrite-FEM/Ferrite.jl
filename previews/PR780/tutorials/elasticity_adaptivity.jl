using Ferrite, FerriteMeshParser, Tensors, SparseArrays, WriteVTK, Downloads

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

function estimate_error(grid, dh, u, cv, C)
    nc = getncells(grid)
    cells = getcells(grid)
    X = [get_node_coordinate(n) for n in getnodes(grid)]

    # Cell-averaged stress σ̄_K.
    σ̄ = Vector{SymmetricTensor{2, 2, Float64, 3}}(undef, nc)
    for cell in CellIterator(dh)
        reinit!(cv, cell)
        s = zero(SymmetricTensor{2, 2})
        vol = 0.0
        for q_point in 1:getnquadpoints(cv)
            dΩ = getdetJdV(cv, q_point)
            s += (C ⊡ function_symmetric_gradient(cv, q_point, u, celldofs(cell))) * dΩ
            vol += dΩ
        end
        σ̄[cellid(cell)] = s / vol
    end

    # Map every facet (as a sorted node-id pair) to the cell(s) that own it.
    # TODO: this facet enumeration is hard-coded for 2D linear quadrilaterals — a facet
    # is the node pair `(nids[f], nids[f+1])`. It does not generalize: in 3D a facet is a
    # 4-node face, and higher-order geometric ansätze put extra nodes on each facet, so
    # the "sorted node tuple" key would differ. This ownership/adjacency query (the true
    # coarse↔fine and cross-tree facet neighbours of the refined forest) should instead
    # be provided by `ForestBWG` itself, which knows the leaf topology for any dimension
    # and ansatz, rather than being reconstructed here from the materialized grid.
    facet_owner = Dict{Tuple{Int, Int}, Vector{Int}}()
    for c in 1:nc
        nids = Ferrite.get_node_ids(cells[c])
        nf = length(nids)
        for f in 1:nf
            key = minmax(nids[f], nids[mod1(f + 1, nf)])
            push!(get!(() -> Int[], facet_owner, key), c)
        end
    end

    # Accumulate the squared traction jump over each interior facet.
    error_arr = zeros(nc)
    hang = grid.conformity_info
    function add_jump!(cA, cB, na, nb)
        L = norm(X[nb] - X[na])
        t = (X[nb] - X[na]) / L
        n = Vec{2}((t[2], -t[1]))                      # facet normal
        jump = (σ̄[cA] - σ̄[cB]) ⋅ n
        contrib = L^2 * (jump ⋅ jump)                  # h_F * |F| * ‖[[σ·n]]‖²
        error_arr[cA] += 0.5 * contrib
        return error_arr[cB] += 0.5 * contrib
    end
    for (key, owners) in facet_owner
        if length(owners) == 2
            # Conforming face (possibly across trees): both cells share this facet.
            add_jump!(owners[1], owners[2], key[1], key[2])
        elseif length(owners) == 1
            # A single owner is either a domain-boundary facet or the fine side of a
            # hanging face. In the latter case one endpoint is a hanging node whose
            # master pair spans the coarse facet → look up the coarse cell.
            a, b = key
            for (hn, oth) in ((a, b), (b, a))
                if haskey(hang, hn) && length(hang[hn]) == 2 && oth in hang[hn]
                    coarse_key = minmax(hang[hn][1], hang[hn][2])
                    haskey(facet_owner, coarse_key) && add_jump!(owners[1], facet_owner[coarse_key][1], a, b)
                    break
                end
            end
        end
    end
    return error_arr
end

function dorfler_mark(error_arr, θ)
    cells_to_refine = Int[]
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
        grid = Ferrite.creategrid(forest)
        u, dh, ch, cv, qr = solve(grid, Cmat)

        # Estimate the error and mark cells with Dörfler marking.
        error_arr = estimate_error(grid, dh, u, cv, Cmat)
        cells_to_refine, total = dorfler_mark(error_arr, θ)
        @info "AMR step $i: $(getncells(grid)) cells, $(length(cells_to_refine)) marked, total error = $total"

        # Export displacement and the cell-wise error indicator.
        VTKGridFile("elasticity_amr-$i", dh) do vtk
            write_solution(vtk, dh, u)
            write_cell_data(vtk, error_arr, "error")
            pvd[i] = vtk
        end

        isempty(cells_to_refine) && break

        # Refine the marked cells and restore 2:1 balance across the forest.
        Ferrite.refine!(forest, cells_to_refine)
        Ferrite.balanceforest!(forest)
    end
    vtk_save(pvd)
    return forest
end

solve_adaptive(forest)

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl
