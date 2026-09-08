# Forests of simplex trees (Burstedde–Holke AMR for triangles and tetrahedra): uniform and
# adaptive refinement, cross-tree node identity, 2:1 balancing, hanging nodes, facet sets,
# the facet skeleton and the conformity constraints end to end. Mirrors
# `src/Adaptivity/simplex_forest.jl`.
using Ferrite, Test
using LinearAlgebra, Random

const AMR = Ferrite.AMR

# --- independent geometric ground truth ---------------------------------------------------
function cell_volume(grid, cell)
    x = [get_node_coordinate(getnodes(grid, n)) for n in cell.nodes]
    if length(x) == 4
        return ((x[2] - x[1]) ⋅ ((x[3] - x[1]) × (x[4] - x[1]))) / 6
    else
        a = x[2] - x[1]
        c = x[3] - x[1]
        return (a[1] * c[2] - a[2] * c[1]) / 2
    end
end

leaf_levels(forest) = [Int(leaf.l) for tree in forest.cells for leaf in tree.leaves]

# closed point-in-simplex test in physical space
function point_in_cell(grid, cell, p; tol = 1.0e-9)
    x = [get_node_coordinate(getnodes(grid, n)) for n in cell.nodes]
    dim = length(p)
    A = hcat([Vector(x[i + 1] - x[1]) for i in 1:dim]...)
    λ = A \ Vector(p - x[1])
    return all(λ .>= -tol) && sum(λ) <= 1 + tol
end

# 2:1 audit on the materialized grid: a cell touching (a corner inside the closure of) a cell
# two or more levels coarser is a violation.
function unbalanced_contacts(grid, levels)
    n = getncells(grid)
    corners = [[get_node_coordinate(getnodes(grid, m)) for m in getcells(grid, i).nodes] for i in 1:n]
    lo = [minimum(hcat(Vector.(c)...), dims = 2) for c in corners]
    hi = [maximum(hcat(Vector.(c)...), dims = 2) for c in corners]
    bad = 0
    for i in 1:n, j in 1:n
        levels[i] - levels[j] >= 2 || continue # i finer, j coarser
        (all(lo[i] .<= hi[j] .+ 1.0e-9) && all(lo[j] .<= hi[i] .+ 1.0e-9)) || continue
        if any(x -> point_in_cell(grid, getcells(grid, j), x), corners[i])
            bad += 1
        end
    end
    return bad
end

edge_pairs(::Val{3}) = ((1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4))
edge_pairs(::Val{2}) = ((1, 2), (1, 3), (2, 3))

# Invariants of every materialized grid: positive orientation, distinct nodes, all nodes
# referenced, hanging nodes interpolate linear functions from non-hanging masters, and the
# strict conformity invariant: a node lying in the closure of a cell it does not belong to is
# a recorded hanging node whose masters form an edge of that cell (no stray nodes).
function check_grid(grid; dim)
    vols = [cell_volume(grid, c) for c in getcells(grid)]
    @test all(v -> v > 0, vols)
    X = [get_node_coordinate(n) for n in getnodes(grid)]
    pts = unique(round.(Vector(x), digits = 10) for x in X)
    @test length(pts) == getnnodes(grid)
    used = falses(getnnodes(grid))
    for c in getcells(grid), n in c.nodes
        used[n] = true
    end
    @test all(used)
    lin(x) = 1.0 + 2.0 * x[1] - 3.0 * x[2] + (dim == 3 ? 0.5 * x[3] : 0.0)
    for (h, ms) in grid.conformity_info
        @test length(ms) == 2
        @test isapprox(lin(X[h]), sum(lin(X[m]) for m in ms) / 2; atol = 1.0e-12)
        @test !any(m -> haskey(grid.conformity_info, m), ms)
    end
    strays = 0
    for c in getcells(grid)
        lo = ntuple(d -> minimum(m -> X[m][d], c.nodes) - 1.0e-9, dim)
        hi = ntuple(d -> maximum(m -> X[m][d], c.nodes) + 1.0e-9, dim)
        for n in 1:getnnodes(grid)
            (n ∈ c.nodes || any(d -> !(lo[d] <= X[n][d] <= hi[d]), 1:dim)) && continue
            point_in_cell(grid, c, X[n]) || continue
            ms = get(grid.conformity_info, n, Int[])
            any(e -> Set((c.nodes[e[1]], c.nodes[e[2]])) == Set(ms), edge_pairs(Val(dim))) || (strays += 1)
        end
    end
    @test strays == 0
    return sum(vols)
end

# Poisson patch test: with the conformity constraints a linear solution is reproduced exactly.
function patch_test(grid, dim)
    refshape = dim == 3 ? RefTetrahedron : RefTriangle
    ip = Lagrange{refshape, 1}()
    cv = CellValues(QuadratureRule{refshape}(2), ip)
    dh = DofHandler(grid)
    add!(dh, :u, ip)
    close!(dh)
    uexact(x) = 1.0 + 2.0 * x[1] - 3.0 * x[2] + (dim == 3 ? 0.5 * x[3] : 0.0)
    ch = ConstraintHandler(dh)
    add!(ch, ConformityConstraint(:u))
    add!(ch, Dirichlet(:u, union(values(Ferrite.getfacetsets(grid))...), uexact))
    close!(ch)
    K = allocate_matrix(dh, ch)
    f = zeros(ndofs(dh))
    assembler = start_assemble(K, f)
    n = getnbasefunctions(cv)
    Ke = zeros(n, n)
    for cell in CellIterator(dh)
        reinit!(cv, cell)
        fill!(Ke, 0)
        for q in 1:getnquadpoints(cv)
            dΩ = getdetJdV(cv, q)
            for i in 1:n, j in 1:n
                Ke[i, j] += (shape_gradient(cv, q, i) ⋅ shape_gradient(cv, q, j)) * dΩ
            end
        end
        assemble!(assembler, celldofs(cell), Ke)
    end
    apply!(K, f, ch)
    u = K \ f
    apply!(u, ch)
    err = 0.0
    for cell in CellIterator(dh)
        dofs = celldofs(cell)
        for (i, x) in enumerate(getcoordinates(cell))
            err = max(err, abs(u[dofs[i]] - uexact(x)))
        end
    end
    return err
end

# The six Kuhn tetrahedra of the unit cube with the paper's corner order (`c_i`, x fastest).
kuhn_nodes() = [Node(Vec{3}((Float64(x), Float64(y), Float64(z)))) for x in 0:1, y in 0:1, z in 0:1][:]
function kuhn_cells()
    c(i) = i + 1
    return [
        Tetrahedron((c(0), c(1), c(5), c(7))), Tetrahedron((c(0), c(1), c(3), c(7))), Tetrahedron((c(0), c(2), c(3), c(7))),
        Tetrahedron((c(0), c(2), c(6), c(7))), Tetrahedron((c(0), c(4), c(6), c(7))), Tetrahedron((c(0), c(4), c(5), c(7))),
    ]
end

function uniformly_refined(grid, l, b = 4)
    forest = ForestBWG(grid, b)
    for i in 1:l
        refine_all!(forest, i)
    end
    return forest
end

@testset "single tree, uniform refinement dim=$dim" for dim in (2, 3)
    grid = if dim == 3
        Grid([Tetrahedron((1, 2, 3, 4))], [Node(Vec{3}((0.0, 0.0, 0.0))), Node(Vec{3}((1.0, 0.0, 0.0))), Node(Vec{3}((0.0, 1.0, 0.0))), Node(Vec{3}((0.0, 0.0, 1.0)))])
    else
        Grid([Triangle((1, 2, 3))], [Node(Vec{2}((0.0, 0.0))), Node(Vec{2}((1.0, 0.0))), Node(Vec{2}((0.0, 1.0)))])
    end
    addfacetset!(grid, "all", x -> true)
    for l in 0:3
        forest = uniformly_refined(grid, l)
        @test getncells(forest) == 2^(dim * l)
        @test length(getcells(forest)) == 2^(dim * l)
        @test Ferrite.get_reference_dimension(forest) == dim
        g = creategrid(forest)
        @test getncells(g) == 2^(dim * l)
        @test getnnodes(g) == (dim == 3 ? binomial(2^l + 3, 3) : binomial(2^l + 2, 2)) # lattice points of the simplex
        @test isempty(g.conformity_info)
        @test check_grid(g; dim) ≈ (dim == 3 ? 1 / 6 : 1 / 2)
        skel = Ferrite.facetskeleton(forest)
        nfaces = (dim + 1) * 2^(dim * l)
        nbnd = (dim + 1) * 2^((dim - 1) * l)
        @test length(skel) == (nfaces - nbnd) ÷ 2
        @test length(Ferrite.facetskeleton(ExclusiveTopology(g), g)) == length(skel) + nbnd # Ferrite's skeleton includes the boundary
        for (a, b) in skel
            @test Set(Ferrite.facets(getcells(g, a[1]))[a[2]]) == Set(Ferrite.facets(getcells(g, b[1]))[b[2]])
        end
        @test length(getfacetset(g, "all")) == nbnd
    end
end

@testset "Kuhn cube: Property 4" begin
    grid = Grid(kuhn_cells(), kuhn_nodes())
    for l in 1:2
        g = creategrid(uniformly_refined(grid, l))
        @test getncells(g) == 6 * 8^l
        @test getnnodes(g) == (2^l + 1)^3 # the six refined trees form the Kuhn triangulation of the refined cube
        @test isempty(g.conformity_info)
        @test check_grid(g; dim = 3) ≈ 1.0
    end
end

@testset "generate_grid, uniform refinement $C" for (dim, C, n) in ((3, Tetrahedron, (2, 2, 2)), (2, Triangle, (3, 3)))
    grid = generate_grid(C, n)
    for l in 1:2
        forest = uniformly_refined(grid, l)
        g = creategrid(forest)
        @test getncells(g) == getncells(grid) * 2^(dim * l)
        @test getnnodes(g) == prod(n .* 2^l .+ 1)
        @test isempty(g.conformity_info)
        @test check_grid(g; dim) ≈ 2.0^dim
        for (name, fs) in Ferrite.getfacetsets(grid)
            @test length(getfacetset(g, name)) == length(fs) * 2^((dim - 1) * l)
        end
        nbnd = sum(length, values(Ferrite.getfacetsets(g)))
        @test length(Ferrite.facetskeleton(ExclusiveTopology(g), g)) == length(Ferrite.facetskeleton(forest)) + nbnd
        @test patch_test(g, dim) < 1.0e-10
    end
end

@testset "adaptive refinement, balance and hanging nodes $C" for (dim, C, n) in ((2, Triangle, (3, 3)), (3, Tetrahedron, (2, 2, 2)))
    grid = generate_grid(C, n)
    rng = MersenneTwister(1234)
    for trial in 1:3
        forest = ForestBWG(grid, 5)
        for step in 1:3
            ncells = getncells(forest)
            refine!(forest, unique(rand(rng, 1:ncells, max(1, ncells ÷ 6))))
            balanceforest!(forest)
            @test AMR._isbalanced(forest)
            g = creategrid(forest)
            levels = leaf_levels(forest)
            @test length(levels) == getncells(g)
            @test unbalanced_contacts(g, levels) == 0
            @test check_grid(g; dim) ≈ 2.0^dim
            @test !isempty(g.conformity_info)
            # transferred facet sets stay on their boundary planes
            for (name, fs) in Ferrite.getfacetsets(g)
                d = name ∈ ("left", "right") ? 1 : name ∈ ("bottom", "top") ? dim : 2
                val = name ∈ ("left", "bottom", "front") ? -1.0 : 1.0
                for fi in fs
                    fnodes = Ferrite.facets(getcells(g, fi[1]))[fi[2]]
                    @test all(m -> get_node_coordinate(getnodes(g, m))[d] ≈ val, fnodes)
                end
            end
            # skeleton: fine side first, the fine facet lies in the coarse cell
            for (a, b) in Ferrite.facetskeleton(forest)
                @test levels[a[1]] >= levels[b[1]]
                cb = getcells(g, b[1])
                @test all(m -> point_in_cell(g, cb, get_node_coordinate(getnodes(g, m))), Ferrite.facets(getcells(g, a[1]))[a[2]])
            end
            @test patch_test(g, dim) < 1.0e-9
        end
        coarsen!(forest, collect(1:getncells(forest)))
        balanceforest!(forest)
        @test check_grid(creategrid(forest); dim) ≈ 2.0^dim
    end
end

# Trees meeting in a single vertex or a single edge only, the adjacency the octree code routes
# through corner/edge transforms and the simplex code through point transforms.
function vertex_only_tets()
    nodes = Node.([Vec{3}((0.0, 0.0, 0.0)), Vec{3}((1.0, 0.0, 0.0)), Vec{3}((0.0, 1.0, 0.0)), Vec{3}((0.0, 0.0, 1.0)), Vec{3}((-1.0, 0.0, 0.0)), Vec{3}((0.0, -1.0, 0.0)), Vec{3}((0.0, 0.0, -1.0))])
    return Grid([Tetrahedron((1, 2, 3, 4)), Tetrahedron((1, 6, 5, 7))], nodes)
end
function edge_only_tets()
    nodes = Node.([Vec{3}((0.0, 0.0, 0.0)), Vec{3}((1.0, 0.0, 0.0)), Vec{3}((0.0, 1.0, 0.0)), Vec{3}((0.0, 0.0, 1.0)), Vec{3}((0.0, -1.0, 0.0)), Vec{3}((0.0, 0.0, -1.0))])
    return Grid([Tetrahedron((1, 2, 3, 4)), Tetrahedron((1, 5, 2, 6))], nodes)
end

@testset "trees sharing only a vertex or an edge" for (name, grid) in (("vertex", vertex_only_tets()), ("edge", edge_only_tets()))
    addfacetset!(grid, "boundary", x -> true)
    for corner in 1:4
        forest = ForestBWG(grid, 6)
        # refine the leaf of tree 1 at root corner `corner` down to level 4, leaving tree 2 coarse
        rootv = AMR.vertex(AMR._simplex_root(Val(3), Int64), corner, 6)
        for l in 0:3
            target = findfirst(o -> o.l == l && rootv ∈ AMR.vertices(o, 6), forest.cells[1].leaves)::Int
            refine!(forest, [target]) # tree 1 comes first, so local and global ids agree
        end
        balanceforest!(forest)
        @test AMR._isbalanced(forest)
        g = creategrid(forest)
        levels = leaf_levels(forest)
        @test unbalanced_contacts(g, levels) == 0
        @test check_grid(g; dim = 3) ≈ 2 / 6
        @test patch_test(g, 3) < 1.0e-9
        # the shared macro vertex has one node
        @test count(x -> norm(get_node_coordinate(x)) < 1.0e-12, getnodes(g)) == 1
    end
end

@testset "maximum depth dim=$dim" for dim in (2, 3)
    grid = dim == 3 ? edge_only_tets() : generate_grid(Triangle, (1, 1))
    b = AMR.DEFAULT_MAXLEVEL[dim]
    forest = ForestBWG(grid, b)
    for _ in 1:b # refine the first leaf of tree 1 all the way down
        refine!(forest, [1])
    end
    @test forest.cells[1].leaves[1].l == b
    refine!(forest, [1]) # a no-op at the maximum level
    @test forest.cells[1].leaves[1].l == b
    keys = [AMR._sortkey(o, b) for o in forest.cells[1].leaves]
    @test issorted(keys) && allunique(keys) && all(k -> k[1] >= 0, keys)
    balanceforest!(forest)
    @test AMR._isbalanced(forest)
    g = creategrid(forest)
    @test check_grid(g; dim) ≈ (dim == 3 ? 2 / 6 : 4.0)
    @test length(Ferrite.facetskeleton(forest)) > 0
end

@testset "coarsening on a balanced forest $C" for (dim, C) in ((2, Triangle), (3, Tetrahedron))
    grid = generate_grid(C, ntuple(_ -> 2, dim))
    forest = ForestBWG(grid, 5)
    for _ in 1:3
        refine!(forest, [1]) # the first leaf stays the corner child at the origin
    end
    balanceforest!(forest)
    n = getncells(forest)
    # a single mark collapses its whole family with `require_all_siblings = false`
    fam = findfirst(o -> o.l == 3, getcells(forest))
    coarsen!(forest, [fam + 1]; require_all_siblings = false)
    @test getncells(forest) == n - (2^dim - 1)
    balanceforest!(forest)
    @test check_grid(creategrid(forest); dim) ≈ 2.0^dim
    # ... but not with the default policy unless all siblings are marked
    n = getncells(forest)
    fam = findfirst(o -> o.l == 2, getcells(forest))
    coarsen!(forest, [fam])
    @test getncells(forest) == n
    coarsen!(forest, collect(fam:(fam + 2^dim - 1)))
    @test getncells(forest) == n - (2^dim - 1)
end

@testset "refine_and_coarsen! conflicts are symmetric" begin
    grid = generate_grid(Triangle, (2, 2))
    forest = ForestBWG(grid, 4)
    refine_all!(forest, 1)
    n = getncells(forest)
    # a refine mark on the first sibling and a coarsen mark on the second, and vice versa
    @test_throws ArgumentError refine_and_coarsen!(forest, [2], [1]; balance = false, require_all_siblings = false)
    @test_throws ArgumentError refine_and_coarsen!(forest, [1], [2]; balance = false, require_all_siblings = false)
    @test getncells(forest) == n
    @test_throws ArgumentError refine!(forest, [n + 1])
    @test_throws ArgumentError coarsen!(forest, [0])
end

@testset "coarsen_octant! guards" begin
    tree = AMR.SimplexTreeBH(Triangle((1, 2, 3)), 4)
    AMR.refine_octant!(tree, tree.leaves[1])
    AMR.refine_octant!(tree, tree.leaves[2]) # an incomplete level-1 family
    @test_throws ArgumentError AMR.coarsen_octant!(tree, tree.leaves[1])
    @test length(tree) == 7
    AMR.coarsen_octant!(tree, tree.leaves[3])
    @test length(tree) == 4
    @test issorted([AMR._sortkey(o, tree.b) for o in tree.leaves])
end

@testset "refine_and_coarsen! round trip $C" for (dim, C) in ((2, Triangle), (3, Tetrahedron))
    grid = generate_grid(C, ntuple(_ -> 2, dim))
    forest = ForestBWG(grid, 4)
    refine_all!(forest, 1)
    n1 = getncells(forest)
    refine_and_coarsen!(forest, Int[], [1, 2]; balance = false)
    @test getncells(forest) == n1 + 2 * (2^dim - 1)
    cells = getcells(forest)
    family = findall(o -> o.l == 2, cells)
    @test length(family) == 2 * 2^dim
    refine_and_coarsen!(forest, family, Int[]; balance = false)
    @test getncells(forest) == n1
    @test all(o -> o.l == 1, getcells(forest))
end

@testset "unbalanced forests are rejected" begin
    grid = generate_grid(Tetrahedron, (1, 1, 1))
    forest = ForestBWG(grid, 5)
    for _ in 1:3
        refine!(forest, [1])
    end
    @test !AMR._isbalanced(forest)
    @test_throws ArgumentError creategrid(forest)
    @test_throws ArgumentError Ferrite.facetskeleton(forest)
    balanceforest!(forest)
    @test AMR._isbalanced(forest)
    @test check_grid(creategrid(forest); dim = 3) ≈ 8.0
end

@testset "constructor errors" begin
    @test_throws ArgumentError ForestBWG(generate_grid(Wedge, (1, 1, 1)), 3)
    grid = Grid([Triangle((1, 2, 3))], [Node(Vec{3}((0.0, 0.0, 0.0))), Node(Vec{3}((1.0, 0.0, 0.0))), Node(Vec{3}((0.0, 1.0, 0.0)))])
    @test_throws ArgumentError ForestBWG(grid, 3) # a 2D cell in a 3D grid
end
