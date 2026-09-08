# Simplex elements of the Burstedde–Holke AMR: the transcribed lookup tables of the paper
# checked against brute-force geometry, the tetrahedral-Morton ordering, the face-neighbour
# algorithm and the corner queries. Mirrors `src/Adaptivity/simplex.jl`.
using Ferrite, Test

const AMR = Ferrite.AMR
using .AMR: SimplexBH, vertices, children, parent, child_id, cube_id, consecutive_index, _sortkey,
    facet_neighbor_face, inside, _bey_child, _push_same_level_neighbors!, _leaf_on_root_face, face,
    _orientation, _root_face_of, _face_index

# The type of the level-`l` simplex at `anchor` with the corner set `verts`, or -1.
function type_of(dim, anchor, verts, l, b)
    for t in 0:(factorial(dim) - 1)
        Set(vertices(SimplexBH(l, anchor, t), b)) == Set(verts) && return t
    end
    return -1
end

# Bey's children straight from eq. (2) of the paper: corner sets built from edge midpoints.
function bey_children_bruteforce(o::SimplexBH{dim}, b) where {dim}
    X = vertices(o, b)
    mid(a, c) = (X[a + 1] .+ X[c + 1]) .÷ 2
    if dim == 3
        x0, x1, x2, x3 = X
        x01, x02, x03, x12, x13, x23 = mid(0, 1), mid(0, 2), mid(0, 3), mid(1, 2), mid(1, 3), mid(2, 3)
        return (
            (x0, x01, x02, x03), (x01, x1, x12, x13), (x02, x12, x2, x23), (x03, x13, x23, x3),
            (x01, x02, x03, x13), (x01, x02, x12, x13), (x02, x03, x13, x23), (x02, x12, x13, x23),
        )
    else
        x0, x1, x2 = X
        x01, x02, x12 = mid(0, 1), mid(0, 2), mid(1, 2)
        return ((x0, x01, x02), (x01, x1, x12), (x02, x12, x2), (x01, x12, x02))
    end
end

intdet(v1, v2, v3) = v1[1] * (v2[2] * v3[3] - v2[3] * v3[2]) - v1[2] * (v2[1] * v3[3] - v2[3] * v3[1]) + v1[3] * (v2[1] * v3[2] - v2[2] * v3[1])
intdet(v1, v2) = v1[1] * v2[2] - v1[2] * v2[1]
function orientation_bruteforce(o::SimplexBH{dim}, b) where {dim}
    X = vertices(o, b)
    d = dim == 3 ? intdet(X[2] .- X[1], X[3] .- X[1], X[4] .- X[1]) : intdet(X[2] .- X[1], X[3] .- X[1])
    return sign(d)
end

# All descendants of the root at level `l`.
function level_set(dim, l, b)
    S = [SimplexBH(0, ntuple(_ -> 0, dim), 0)]
    for _ in 1:l
        S = [c for s in S for c in children(s, b)]
    end
    return S
end

# The tetrahedral Morton index of Definition 13 / eq. (15): the digits `(cube-id, type)` of all
# ancestors, most significant first — computed here independently of the tables.
function tm_index(o::SimplexBH{dim}, b) where {dim}
    chain = SimplexBH{dim, Int}[]
    a = o
    while a.l > 0
        push!(chain, a)
        a = parent(a, b)
    end
    m = 0
    for a in reverse(chain)
        m = m * 2^dim * 8 + cube_id(a, a.l, b) * 8 + a.type
    end
    return m
end

function bruteforce_face_neighbor(o::SimplexBH{dim}, f, b) where {dim}
    fv = Set(face(o, f, b))
    h = 2^(b - o.l)
    for δ in Iterators.product(ntuple(_ -> (-1, 0, 1), dim)...)
        anchor = o.xyz .+ h .* δ
        for t in 0:(factorial(dim) - 1)
            s = SimplexBH(o.l, anchor, t)
            s == o && continue
            if fv ⊆ Set(vertices(s, b))
                return s, findfirst(g -> Set(face(s, g, b)) == fv, 1:(dim + 1))
            end
        end
    end
    return nothing
end

@testset "Bey refinement tables dim=$dim" for dim in (2, 3)
    b = 3
    cubeid_of_tm_child = dim == 3 ? AMR._CHILD_CUBEID3 : AMR._CHILD_CUBEID2 # Table 7
    type_of_tm_child = dim == 3 ? AMR._CHILD_TYPE3 : AMR._CHILD_TYPE2       # Table 8
    σ = dim == 3 ? AMR._SIGMA3 : AMR._SIGMA2                                 # Table 2
    for t in 0:(factorial(dim) - 1)
        P = SimplexBH(1, ntuple(_ -> 4, dim), t) # a level-1 anchor (multiple of h = 4)
        for (i, verts) in enumerate(bey_children_bruteforce(P, b))
            anchor = ntuple(d -> minimum(v -> v[d], verts), dim)
            tt = type_of(dim, anchor, verts, 2, b)
            @test tt >= 0
            c = _bey_child(P, i - 1, b)
            @test c.xyz == anchor              # Algorithm 4.4
            @test c.type == tt                 # Table 1
            @test parent(c, b) == P            # Algorithm 4.3 / Figure 8
            @test child_id(c, b) == σ[t + 1][i] + 1 # Table 6 against Table 2
            @test orientation_bruteforce(c, b) == _orientation(c)
        end
        ch = children(P, b)
        @test allunique(ch)
        for (k, c) in enumerate(ch)
            @test child_id(c, b) == k
            @test cube_id(c, 2, b) == cubeid_of_tm_child[t + 1][k]
            @test c.type == type_of_tm_child[t + 1][k]
            @test consecutive_index(c, b) == 2^dim * consecutive_index(P, b) + (k - 1)
        end
        @test _orientation(P) == orientation_bruteforce(P, b)
        # the corner children are copies of the parent, the inner ones tile the rest
        @test count(c -> c.type == t, ch) >= dim + 1
    end
end

@testset "tetrahedral Morton order dim=$dim" for dim in (2, 3)
    b = dim == 3 ? 3 : 5
    for l in 1:(dim == 3 ? 3 : 5)
        S = level_set(dim, l, b)
        @test length(S) == 2^(dim * l)
        @test allunique(S)
        I = consecutive_index.(S, b)
        @test sort(I) == 0:(2^(dim * l) - 1)               # a bijection onto 0:2^(dim l)-1, eq. (54)
        @test sortperm(I) == sortperm(tm_index.(S, b))     # ordered like the TM-index, eq. (53)
        @test all(s -> inside(s, b), S)
    end
    # the sort key puts ancestors first and keeps every simplex's descendants contiguous
    all_ = vcat([level_set(dim, l, b) for l in 0:2]...)
    sorted = all_[sortperm([_sortkey(s, b) for s in all_])]
    for (i, s) in enumerate(sorted)
        s.l == 2 && continue
        n = 2^(dim * (2 - s.l)) + (s.l == 0 ? 2^dim : 0) # number of descendants present
        @test all(d -> AMR.isancestor(s, d, b), sorted[(i + 1):(i + n)])
    end
    @test parent(SimplexBH(0, ntuple(_ -> 0, dim), 0), b) == SimplexBH(0, ntuple(_ -> 0, dim), 0)
end

@testset "face neighbours dim=$dim" for dim in (2, 3)
    b = 4
    for l in 1:2, s in level_set(dim, l, b), f in 1:(dim + 1)
        nb, f̃ = facet_neighbor_face(s, f, b)                # Algorithm 4.6
        bf = bruteforce_face_neighbor(s, f, b)
        @test bf !== nothing
        @test nb == bf[1]
        @test f̃ == bf[2]
        nb2, f2 = facet_neighbor_face(nb, f̃, b)             # eq. (49)
        @test nb2 == s && f2 == f
        rootface = _root_face_of(s, f, b)
        if !inside(nb, b)                                     # the face lies on the root boundary
            @test rootface != 0
            fv = face(s, f, b)
            @test _leaf_on_root_face(fv, s.l, b) == s
            @test _face_index(s, fv, b) == f
        else
            @test rootface == 0
        end
    end
    # away from the root's diagonal planes, faces of types other than 0 tile the root boundary
    if dim == 3
        types = Set(s.type for s in level_set(3, 2, b) for f in 1:4 if _root_face_of(s, f, b) != 0)
        @test types == Set((0, 1, 2, 4, 5))
    end
end

@testset "same-level neighbours dim=$dim" for dim in (2, 3)
    b = 3
    S = level_set(dim, 2, b)
    for s in S
        P = SimplexBH{dim, Int}[]
        _push_same_level_neighbors!(P, s, b)
        unique!(P)
        vs = Set(vertices(s, b))
        @test Set(P) == Set(t for t in S if t != s && !isempty(vs ∩ Set(vertices(t, b))))
    end
end

@testset "tree construction and errors" begin
    @test_throws ArgumentError AMR.SimplexTreeBH{3}((1, 2, 3), 3)
    @test_throws DomainError AMR.SimplexTreeBH(Tetrahedron((1, 2, 3, 4)), 20)
    tree = AMR.SimplexTreeBH(Triangle((1, 2, 3)), 4)
    @test length(tree) == 1
    @test AMR._nchildren(tree) == 4
    AMR.refine_octant!(tree, tree.leaves[1])
    @test length(tree) == 4
    @test issorted([_sortkey(o, tree.b) for o in tree.leaves])
    AMR.coarsen_octant!(tree, tree.leaves[3])
    @test length(tree) == 1
    @test_throws ArgumentError AMR.coarsen_octant!(tree, tree.leaves[1]) # the root has no family
    str = sprint(show, MIME"text/plain"(), tree.leaves[1])
    @test occursin("SimplexBH{2}", str) && occursin("type = 0", str)
end
