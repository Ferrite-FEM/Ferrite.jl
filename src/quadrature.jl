include("quadrature_tables/gaussquad_tri_table.jl")
include("quadrature_tables/gaussquad_tet_table.jl")

import Base.Cartesian: @nloops, @nref, @ntuple, @nexprs

"""
A `QuadratureRule` is used to approximate an integral on a domain by a weighted sum of
function values at specific points:

``\\int\\limits_\\Omega f(\\mathbf{x}) \\text{d} \\Omega \\approx \\sum\\limits_{q = 1}^{n_q} f(\\mathbf{x}_q) w_q``

The quadrature rule consists of ``n_q`` points in space ``\\mathbf{x}_q`` with corresponding weights ``w_q``.

There are different rules to determine the points and weights. In `JuAFEM` two different types are implemented:
`:legendre` and `:lobatto`, where `:lobatto` is only supported for `RefCube`. If the quadrature rule type is left out,
`:legendre` is used by default.

In `JuAFEM`, the `QuadratureRule` type is mostly used as one of the components to create a [`CellValues`](@ref)
or [`BoundaryValues`](@ref) object.

**Constructor:**

```julia
QuadratureRule{dim, shape}([quad_rule_type::Symbol], order::Int)
```

**Arguments:**

* `dim`: the space dimension of the reference shape
* `shape`: an [`AbstractRefShape`](@ref)
* `quad_rule_type`: `:legendre` or `:lobatto`, defaults to `:legendre`.
* `order`: the order of the quadrature rule

**Common methods:**

* [`getpoints`](@ref) : the points of the quadrature rule
* [`getweights`](@ref) : the weights of the quadrature rule

**Example:**

```jldoctest
julia> QuadratureRule{2, RefTetrahedron}(1)
JuAFEM.QuadratureRule{2,JuAFEM.RefTetrahedron,Float64}([0.5],Tensors.Tensor{1,2,Float64,2}[[0.333333,0.333333]])

julia> QuadratureRule{1, RefCube}(:lobatto, 2)
JuAFEM.QuadratureRule{1,JuAFEM.RefCube,Float64}([1.0,1.0],Tensors.Tensor{1,1,Float64,1}[[-1.0],[1.0]])
```
"""
immutable QuadratureRule{dim, shape, T}
    weights::Vector{T}
    points::Vector{Vec{dim, T}}
end

"""
The weights of the quadrature rule.

    getweights(qr::QuadratureRule) = qr.weights

**Arguments:**

* `qr`: the quadrature rule

**Example:**

```jldoctest
julia> qr = QuadratureRule{2, RefTetrahedron}(:legendre, 2);

julia> getweights(qr)
3-element Array{Float64,1}:
 0.166667
 0.166667
 0.166667
```

"""
getweights(qr::QuadratureRule) = qr.weights


"""
The points of the quadrature rule.

    getpoints(qr::QuadratureRule)

**Arguments:**

* `qr`: the quadrature rule

**Example:**

```jldoctest
julia> qr = QuadratureRule{2, RefTetrahedron}(:legendre, 2);

julia> getpoints(qr)
3-element Array{Tensors.Tensor{1,2,Float64,2},1}:
 [0.166667,0.166667]
 [0.166667,0.666667]
 [0.666667,0.166667]
```
"""
getpoints(qr::QuadratureRule) = qr.points

(::Type{QuadratureRule{dim, shape}}){dim, shape}(order::Int) = QuadratureRule{dim, shape}(:legendre, order)

# Special case for boundary integration of 1D problems
function (::Type{QuadratureRule{0, RefCube}})(quad_type::Symbol, order::Int)
    w = Float64[1.0]
    p = Vec{0,Float64}[]
    return QuadratureRule{0, RefCube, Float64}(w,p)
end

# Generate Gauss quadrature rules on cubes by doing an outer product
# over all dimensions
for dim in (1,2,3)
    @eval begin
        function (::Type{QuadratureRule{$dim, RefCube}})(quad_type::Symbol, order::Int)
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
                t = @ntuple $dim q-> p[$(Symbol("i"*"_q"))]
                points[count] = Vec{$dim, Float64}(t)
                weight = 1.0
                @nexprs $dim j->(weight *= w[i_{j}])
                weights[count] = weight
                count += 1
            end
            return QuadratureRule{$dim, RefCube, Float64}(weights, points)
        end
    end
end

for dim in (2, 3)
    @eval begin
        function (::Type{QuadratureRule{$dim, RefTetrahedron}})(quad_type::Symbol, order::Int)
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
            QuadratureRule{$dim, RefTetrahedron, Float64}(weights, points)
        end
    end
end

# Special version for boundary integration of triangles
function (::Type{QuadratureRule{1, RefTetrahedron}})(quad_type::Symbol, order::Int)
    if quad_type == :legendre
        p, weights = gausslegendre(order)
    elseif quad_type == :lobatto
        p, weights = gausslobatto(order)
    else
        throw(ArgumentError("unsupported quadrature rule"))
    end
    points = Vector{Vec{1, Float64}}(order)
    # Shift interval from (-1,1) to (0,1)
    weights *= 0.5
    p += 1.0; p /= 2.0

    for i in 1:length(weights)
        points[i] = Vec{1,Float64}((p[i],))
    end
    return QuadratureRule{1, RefTetrahedron, Float64}(weights, points)
end
