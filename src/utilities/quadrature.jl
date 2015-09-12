"""
A type that perform Gauss integration
"""
type GaussQuadratureRule
    weights::Vector{Float64}
    points::Vector{Vector{Float64}}
end

weights(qr::GaussQuadratureRule) = qr.weights
points(qr::GaussQuadratureRule) = qr.points

"""
Integrates the function *f* with the given
`GaussQuadratureRule`
"""
function integrate(qr::GaussQuadratureRule, f)
    w = weights(qr)
    p = points(qr)
    I = w[1] * f(p[1])
    for (w, x) in zip(w[2:end], p[2:end])
        I += w * f(x)
    end
    return I
end

get_gaussrule(::Triangle, order::Int) = get_trirule(order)
get_gaussrule(::Square, order::Int) = get_quadrule(order)

"""
Creates a `GaussQuadratureRule` that integrates
functions on a square to the given order.
"""
function make_quadrule(order::Int)
    p, w = gausslegendre(order)
    weights = Array(Float64, order^2)
    points = Array(Vector{Float64}, order^2)
    count = 1
    for i = 1:order, j = 1:order
        points[count] = [p[i], p[j]]
        weights[count] = w[i] * w[j]
        count += 1
    end
    GaussQuadratureRule(weights, points)
end

const quadrules = [make_quadrule(i) for i = 1:5]
function get_quadrule(order::Int)
    if order <= 5
        return quadrules[order]
    else
        return make_quadrule(order)
    end
end


include("gaussquad_tri_table.jl")

function make_trirule(order::Int)
    data = _get_gauss_tridata(order)
    n_points = size(data,1)
    weights = Array(Float64, n_points)
    points = Array(Vector{Float64}, n_points)

    for p in 1:size(data, 1)
        points[p] = [data[p, 1], data[p, 2]]
    end

    weights = 0.5 * data[:, 3]

    GaussQuadratureRule(weights, points)
end

const trirules = GaussQuadratureRule[make_trirule(i) for i = 1:5]
function get_trirule(order::Int)
    if order <= 5
        return trirules[order]
    else
        return make_trirule(order)
    end
end
