using FastGaussQuadrature

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