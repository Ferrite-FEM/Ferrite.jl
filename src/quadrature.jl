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


QuadratureRule{dim}(::Type{Dim{dim}}, shape::RefShape, order::Int) = QuadratureRule(:legendre, Dim{dim}, shape, order)

# Generate Gauss quadrature rules on cubes by doing an outer product
# over all dimensions
QuadratureRule{dim}(::Type{Dim{dim}}, shape::RefShape, order::Int) = QuadratureRule(:legendre, Dim{dim}, shape, order)

for dim in (1,2,3)
    @eval begin
        function QuadratureRule(quad_type::Symbol, ::Type{Dim{$dim}}, ::RefCube, order::Int)
            if quad_type == :legendre
                p, w = gausslegendre(order)
            elseif quad_type == :lobatto
                p, w = gausslobatto(order)
            else
                throw(ArgumentError("unsupported quadrature rule"))
            end
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

for dim in (2, 3)
    @eval begin
        function QuadratureRule(quad_type::Symbol, ::Type{Dim{$dim}}, ::RefTetrahedron, order::Int)
            if $dim == 2 && quad_type == :legendre
                data = _get_gauss_tridata(order)
            elseif $dim == 3 && quad_type == :legendre
                data = _get_gauss_tetdata(order)
            else
                throw(ArgumentError("unsupported quadrature rule"))
            end
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
