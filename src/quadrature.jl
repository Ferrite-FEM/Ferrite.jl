include("gaussquad_tri_table.jl")
include("gaussquad_tet_table.jl")

"""
A type that perform quadrature integration
"""
type QuadratureRule{dim, T}
    weights::Vector{T}
    points::Vector{Vec{dim, T}}
end

weights(qr::QuadratureRule) = qr.weights
points(qr::QuadratureRule) = qr.points

function get_gaussrule(::Type{Dim{2}}, ::Triangle, order::Int)
    if order <= 5
        return trirules[order]
    else
        return make_trirule(order)
    end
end

function get_gaussrule(::Type{Dim{1}}, ::Square, order::Int)
    if order <= 5
        return linerules[order]
    else
        return make_linerule(order)
    end
end

function get_gaussrule(::Type{Dim{2}}, ::Square, order::Int)
    if order <= 5
        return quadrules[order]
    else
        return make_quadrule(order)
    end
end

function get_gaussrule(::Type{Dim{3}}, ::Triangle, order::Int)
    if order <= 3
        return tetrules[order]
    else
        return make_tetrule(order)
    end
end

function get_gaussrule(::Type{Dim{3}}, ::Square, order::Int)
    if order <= 5
        return cuberules[order]
    else
        return make_cuberule(order)
    end
end


"""
Creates a `GaussQuadratureRule` that integrates
functions on a cube to the given order.
"""
function make_cuberule(order::Int)
    p, w = gausslegendre(order)
    weights = Array(Float64, order^3)
    points = Array(Vec{3, Float64}, order^3)
    count = 1
    for i = 1:order, j = 1:order, k = 1:order
        points[count] = Vec{3, Float64}((p[i], p[j], p[k]))
        weights[count] = w[i] * w[j] * w[k]
        count += 1
    end
    QuadratureRule(weights, points)
end

const cuberules = [make_cuberule(i) for i = 1:5]


"""
Creates a `QuadratureRule` that integrates
functions on a square to the given order.
"""
function make_quadrule(order::Int)
    p, w = gausslegendre(order)
    weights = Array(Float64, order^2)
    points = Array(Vec{2, Float64}, order^2)
    count = 1
    for i = 1:order, j = 1:order
        points[count] = Vec{2, Float64}((p[i], p[j]))
        weights[count] = w[i] * w[j]
        count += 1
    end
    QuadratureRule(weights, points)
end

const quadrules = [make_quadrule(i) for i = 1:5]


"""
Creates a `QuadratureRule` that integrates
functions on a line to the given order.
"""
function make_linerule(order::Int)
    p, weights = gausslegendre(order)
    points = points = Array(Vec{1, Float64}, order)
    for i = 1:order
        points[i] = Vec{1, Float64}((p[i],))
    end
    QuadratureRule(weights, points)
end

const linerules = [make_linerule(i) for i = 1:5]

function make_trirule(order::Int)
    data = _get_gauss_tridata(order)
    n_points = size(data,1)
    weights = Array(Float64, n_points)
    points = Array(Vec{2, Float64}, n_points)

    for p in 1:size(data, 1)
        points[p] = Vec{2, Float64}((data[p, 1], data[p, 2]))
    end

    weights = 0.5 * data[:, 3]

    QuadratureRule(weights, points)
end


const trirules = [make_trirule(i) for i = 1:5]


function make_tetrule(order::Int)
    data = _get_gauss_tetdata(order)
    n_points = size(data,1)
    weights = Array(Float64, n_points)
    points = Array(Vec{3, Float64}, n_points)

    for p in 1:size(data, 1)
        points[p] = Vec{3, Float64}((data[p, 1], data[p, 2], data[p, 3]))
    end

    weights = data[:, 4]

    QuadratureRule(weights, points)
end

const tetrules = [make_tetrule(i) for i = 1:3]


