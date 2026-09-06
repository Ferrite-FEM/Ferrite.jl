basis(e, i) = p -> Ferrite.reference_shape_value(e.ip, p, i)
const EDGE_RULE = QuadratureRule{RefLine}(5)
vertices(e) = Ferrite.reference_coordinates(Lagrange{Ferrite.getrefshape(e.ip), 1}())
edges(e) = Ferrite.reference_edges(Ferrite.getrefshape(e.ip))
weighted_edges(e) = Ferrite.getrefshape(e.ip) == RefTriangle && Ferrite.getnbasefunctions(e.ip) == 6
function density(e, f, i, s)
    edge = weighted_edges(e) ? cld(i, 2) : i
    a, b = vertices(e)[collect(edges(e)[edge])]
    d = b - a
    direction = e.kind == :normal ? Vec((d[2], -d[1])) : d
    q = weighted_edges(e) ? (isodd(i) ? 1 - s : s) : 1.0
    return dot(f((1 - s) * a + s * b), direction) * q
end
function functional(e, f, i)
    functional_kind = Ferrite.dof_functionals(e.ip)[i]
    if functional_kind isa PointValue
        return f(Ferrite.reference_coordinates(e.ip)[i])
    elseif functional_kind isa Union{NormalMoment, TangentialMoment}
        return sum(w * density(e, f, i, (p[1] + 1) / 2) / 2 for (w, p) in zip(Ferrite.getweights(EDGE_RULE), Ferrite.getpoints(EDGE_RULE)))
    end
    throw(ArgumentError("No atlas evaluator for $functional_kind"))
end
"""Check the complete duality matrix for each representative example."""
function check()
    count = 0
    for e in CATALOG
        n = Ferrite.getnbasefunctions(e.ip)
        M = [functional(e, basis(e, j), i) for i in 1:n, j in 1:n]
        isapprox(M, Matrix{Float64}(I, n, n); atol = 5.0e-11) || error("Atlas functional mismatch for $(e.slug): $M")
        count += 1
    end
    return count
end
