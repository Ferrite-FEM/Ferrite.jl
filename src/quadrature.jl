include("quadrature_tables/gaussquad_tri_table.jl")
include("quadrature_tables/gaussquad_tet_table.jl")

import Base.Cartesian: @nloops, @nref, @ntuple, @nexprs

"""
A type that perform quadrature integration
"""
type QuadratureRule{dim, T}
    weights::Vector{T}
    points::Vector{Vec{dim, T}}
end

weights(qr::QuadratureRule) = qr.weights
points(qr::QuadratureRule) = qr.points

# Generate Gauss quadrature rules on cubes by doing an outer product
# over all dimensions
for dim in (1,2,3)
    @eval begin
        function GaussQuadrature(::Type{Dim{$dim}}, ::RefCube, order::Int)
            p, w = gausslegendre(order)
            weights = Vector{Float64}(order^($dim))
            points = Vector{Vec{$dim, Float64}}(order^($dim))
            count = 1
            @nloops $dim i j->(1:order) begin
                t = @ntuple $dim q-> p[$(symbol("i"*"_q"))]
                points[count] = Vec{$dim, Float64}(t)
                weight = 1.0
                @nexprs $dim j->(weight *= w[i_{j}])
                weights[count] = weight
                count += 1
            end
            return QuadratureRule(weights, points)
        end
    end
end

for (dim, table_func) in ((2, _get_gauss_tridata),
                          (3, _get_gauss_tetdata))
    @eval begin
        function GaussQuadrature(::Type{Dim{$dim}}, ::RefTetrahedron, order::Int)
            data = $(table_func)(order)
            n_points = size(data,1)
            weights = Array(Float64, n_points)
            points = Array(Vec{$dim, Float64}, n_points)

            for p in 1:size(data, 1)
                points[p] = Vec{$dim, Float64}(@ntuple $dim i -> data[p, i])
            end
            weights = data[:, $dim + 1]
            QuadratureRule(weights, points)
        end
    end
end
