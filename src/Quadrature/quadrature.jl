include("gaussquad_tri_table.jl")
include("gaussquad_tet_table.jl")
include("generate_quadrature.jl")

import Base.Cartesian: @nloops, @nref, @ntuple, @nexprs

"""
    QuadratureRule{ξdim,shape}([quad_rule_type::Symbol], order::Int)

Create a `QuadratureRule` used for integration. `ξdim` is the (reference) space dimension,
`shape` an [`AbstractRefShape`](@ref) and `order` the order of the quadrature rule.
`quad_rule_type` is an optional argument determining the type of quadrature rule,
currently the `:legendre` and `:lobatto` rules are implemented.

A `QuadratureRule` is used to approximate an integral on a domain by a weighted sum of
function values at specific points:

``\\int\\limits_\\Omega f(\\mathbf{\\xi}) \\text{d} \\Omega \\approx \\sum\\limits_{q = 1}^{n_q} f(\\mathbf{\\xi}_q) w_q``

The quadrature rule consists of ``n_q`` points in space ``\\mathbf{\\xi}_q`` with corresponding weights ``w_q``.

In `JuAFEM`, the `QuadratureRule` type is mostly used as one of the components to create a [`CellValues`](@ref)
or [`FaceValues`](@ref) object.

**Common methods:**
* [`getpoints`](@ref) : the points of the quadrature rule
* [`getweights`](@ref) : the weights of the quadrature rule

**Example:**
```jldoctest
julia> QuadratureRule{2, RefTetrahedron}(1)
JuAFEM.QuadratureRule{2,JuAFEM.RefTetrahedron,Float64}([0.5], Tensors.Tensor{1,2,Float64,2}[[0.333333, 0.333333]])

julia> QuadratureRule{1, RefCube}(:lobatto, 2)
JuAFEM.QuadratureRule{1,JuAFEM.RefCube,Float64}([1.0, 1.0], Tensors.Tensor{1,1,Float64,1}[[-1.0], [1.0]])
```
"""
struct QuadratureRule{ξdim,shape,T}
    weights::Vector{T}
    points::Vector{Vec{ξdim,T}}
end

"""
    getweights(qr::QuadratureRule)

Return the weights of the quadrature rule.

# Examples
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
    getpoints(qr::QuadratureRule)

Return the points of the quadrature rule.

# Examples
```jldoctest
julia> qr = QuadratureRule{2, RefTetrahedron}(:legendre, 2);

julia> getpoints(qr)
3-element Array{Tensors.Tensor{1,2,Float64,2},1}:
 [0.166667, 0.166667]
 [0.166667, 0.666667]
 [0.666667, 0.166667]
```
"""
getpoints(qr::QuadratureRule) = qr.points

QuadratureRule{ξdim,shape}(order::Int) where {ξdim,shape} = QuadratureRule{ξdim,shape}(:legendre, order)

# Special case for face integration of 1D problems
function (::Type{QuadratureRule{0, RefCube}})(quad_type::Symbol, order::Int)
    w = Float64[1.0]
    p = Vec{0,Float64}[]
    return QuadratureRule{0,RefCube,Float64}(w,p)
end

# Generate Gauss quadrature rules on cubes by doing an outer product
# over all dimensions
for ξdim in (1,2,3)
    @eval begin
        function (::Type{QuadratureRule{$ξdim,RefCube}})(quad_type::Symbol, order::Int)
            if quad_type == :legendre
                p, w = GaussQuadrature.legendre(Float64, order)
            elseif quad_type == :lobatto
                p, w = GaussQuadrature.legendre(Float64, order, GaussQuadrature.both)
            else
                throw(ArgumentError("unsupported quadrature rule"))
            end
            weights = Vector{Float64}(undef, order^($ξdim))
            points = Vector{Vec{$ξdim,Float64}}(undef, order^($ξdim))
            count = 1
            @nloops $ξdim i j->(1:order) begin
                t = @ntuple $ξdim q-> p[$(Symbol("i"*"_q"))]
                points[count] = Vec{$ξdim,Float64}(t)
                weight = 1.0
                @nexprs $ξdim j->(weight *= w[i_{j}])
                weights[count] = weight
                count += 1
            end
            return QuadratureRule{$ξdim,RefCube,Float64}(weights, points)
        end
    end
end

for ξdim in (2, 3)
    @eval begin
        function (::Type{QuadratureRule{$ξdim, RefTetrahedron}})(quad_type::Symbol, order::Int)
            if $ξdim == 2 && quad_type == :legendre
                data = _get_gauss_tridata(order)
            elseif $ξdim == 3 && quad_type == :legendre
                data = _get_gauss_tetdata(order)
            else
                throw(ArgumentError("unsupported quadrature rule"))
            end
            n_points = size(data,1)
            points = Vector{Vec{$ξdim,Float64}}(undef, n_points)

            for p in 1:size(data, 1)
                points[p] = Vec{$ξdim,Float64}(@ntuple $ξdim i -> data[p, i])
            end
            weights = data[:, $ξdim + 1]
            QuadratureRule{$ξdim,RefTetrahedron,Float64}(weights, points)
        end
    end
end

# Special version for face integration of triangles
function (::Type{QuadratureRule{1,RefTetrahedron}})(quad_type::Symbol, order::Int)
    if quad_type == :legendre
        p, weights = GaussQuadrature.legendre(Float64,order)
    elseif quad_type == :lobatto
        p, weights = GaussQuadrature.legendre(Float64, order, GaussQuadrature.both)
    else
        throw(ArgumentError("unsupported quadrature rule"))
    end
    points = Vector{Vec{1,Float64}}(undef, order)
    # Shift interval from (-1,1) to (0,1)
    weights *= 0.5
    p .+= 1.0; p /= 2.0

    for i in 1:length(weights)
        points[i] = Vec{1,Float64}((p[i],))
    end
    return QuadratureRule{1,RefTetrahedron,Float64}(weights, points)
end
