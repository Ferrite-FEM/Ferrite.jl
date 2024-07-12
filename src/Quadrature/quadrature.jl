include("gaussquad_tri_table.jl")
include("gaussquad_tet_table.jl")
include("gaussquad_prism_table.jl")
include("gaussquad_pyramid_table.jl")
include("generate_quadrature.jl")

using Base.Cartesian: @nloops, @ntuple, @nexprs

##################
# QuadratureRule #
##################

"""
    QuadratureRule{shape}([::Type{T},] [quad_rule_type::Symbol,] order::Int)
    QuadratureRule{shape}(weights::AbstractVector{T}, points::AbstractVector{Vec{rdim, T}})

Create a `QuadratureRule` used for integration on the refshape `shape` (of type [`AbstractRefShape`](@ref)).
`order` is the order of the quadrature rule.
`quad_rule_type` is an optional argument determining the type of quadrature rule,
currently the `:legendre` and `:lobatto` rules are implemented for hypercubes.
For triangles up to order 8 the default rule is the one by `:dunavant` (see [Dun:1985:hde](@cite)) and for
tetrahedra the default rule is `keast_positive` (see [Keast:1986:mtq](@cite)). Wedges and pyramids default
to `:polyquad` (see [WitVin:2015:isq](@cite)).
Furthermore we have implemented
* `:gaussjacobi` for triangles (order 9-15)
* `:jinyun` (see [Jin:1984:sgq](@cite)) for tetrahedra (order 1-4)
* `:keast_minimal` (see [Keast:1986:mtq](@cite)) for tetrahedra (order 1-5)

A `QuadratureRule` is used to approximate an integral on a domain by a weighted sum of
function values at specific points:

``\\int\\limits_\\Omega f(\\mathbf{x}) \\text{d} \\Omega \\approx \\sum\\limits_{q = 1}^{n_q} f(\\mathbf{x}_q) w_q``

The quadrature rule consists of ``n_q`` points in space ``\\mathbf{x}_q`` with corresponding weights ``w_q``.

In `Ferrite`, the `QuadratureRule` type is mostly used as one of the components to create [`CellValues`](@ref).

**Common methods:**
* [`getpoints`](@ref) : the points of the quadrature rule
* [`getweights`](@ref) : the weights of the quadrature rule

**Example:**
```jldoctest
julia> qr = QuadratureRule{RefTriangle}(1)
QuadratureRule{RefTriangle, Float64, 2}([0.5], Vec{2, Float64}[[0.33333333333333, 0.33333333333333]])

julia> getpoints(qr)
1-element Vector{Vec{2, Float64}}:
 [0.33333333333333, 0.33333333333333]
```
"""
struct QuadratureRule{shape, WeightStorageType, PointStorageType}
    weights::WeightStorageType  # E.g. Vector{Float64}
    points::PointStorageType    # E.g. Vector{Vec{3, Float64}}
    function QuadratureRule{shape}(weights::AbstractVector{T}, points::AbstractVector{Vec{rdim, T}}) where {rdim, shape <: AbstractRefShape{rdim}, T}
        if length(weights) != length(points)
            throw(ArgumentError("number of weights and number of points do not match (#weights=$(length(weights)) != #points=$(length(points)))"))
        end
        new{shape, typeof(weights), typeof(points)}(weights, points)
    end
end

@inline _default_quadrature_rule(::Type{<:RefHypercube}) = :legendre
@inline _default_quadrature_rule(::Union{Type{RefPrism}, Type{RefPyramid}}) = :polyquad
@inline _default_quadrature_rule(::Type{RefTriangle}) = :dunavant
@inline _default_quadrature_rule(::Type{RefTetrahedron}) = :keast_minimal

# Fill in defaults with T=Float64
function QuadratureRule{shape}(order::Int) where {shape <: AbstractRefShape}
    return QuadratureRule{shape}(Float64, order)
end
function QuadratureRule{shape}(::Type{T}, order::Int) where {shape <: AbstractRefShape, T}
    quad_type = _default_quadrature_rule(shape)
    return QuadratureRule{shape}(T, quad_type, order)
end
function QuadratureRule{shape}(quad_type::Symbol, order::Int) where {shape <: AbstractRefShape}
    return QuadratureRule{shape}(Float64, quad_type, order)
end

# Generate Gauss quadrature rules on hypercubes by doing an outer product
# over all dimensions
for dim in 1:3
    @eval begin
        function QuadratureRule{RefHypercube{$dim}}(::Type{T}, quad_type::Symbol, order::Int) where T
            if quad_type === :legendre
                p, w = GaussQuadrature.legendre(T, order)
            elseif quad_type === :lobatto
                p, w = GaussQuadrature.legendre(T, order, GaussQuadrature.both)
            else
                throw(ArgumentError("unsupported quadrature rule"))
            end
            weights = Vector{T}(undef, order^($dim))
            points = Vector{Vec{$dim,T}}(undef, order^($dim))
            count = 1
            @nloops $dim i j->(1:order) begin
                t = @ntuple $dim q-> p[$(Symbol("i"*"_q"))]
                points[count] = Vec{$dim,T}(t)
                weight = 1.0
                @nexprs $dim j->(weight *= w[i_{j}])
                weights[count] = weight
                count += 1
            end
            return QuadratureRule{RefHypercube{$dim}}(weights, points)
        end
    end
end

for dim in 2:3
    @eval begin
        function QuadratureRule{RefSimplex{$dim}}(::Type{T}, quad_type::Symbol, order::Int) where T
            if $dim == 2 && quad_type === :dunavant
                data = _get_dunavant_gauss_tridata(order)
            elseif $dim == 2 && quad_type === :gaussjacobi
                data = _get_gaussjacobi_tridata(order)
            elseif $dim == 3 && quad_type === :jinyun
                data = _get_jinyun_tet_quadrature_data(order)
            elseif $dim == 3 && quad_type === :keast_minimal
                data = _get_keast_a_tet_quadrature_data(order)
            elseif $dim == 3 && quad_type === :keast_positive
                data = _get_keast_b_tet_quadrature_data(order)
            else
                throw(ArgumentError("unsupported quadrature rule"))
            end
            n_points = size(data,1)
            points = Vector{Vec{$dim,T}}(undef, n_points)

            for p in 1:size(data, 1)
                points[p] = Vec{$dim,T}(@ntuple $dim i -> data[p, i])
            end
            weights = T.(data[:, $dim + 1])
            QuadratureRule{RefSimplex{$dim}}(weights, points)
        end
    end
end

# Grab prism quadrature rule from table
function QuadratureRule{RefPrism}(::Type{T}, quad_type::Symbol, order::Int) where T
    if quad_type == :polyquad
        data = _get_gauss_prismdata_polyquad(order)
    else
        throw(ArgumentError("unsupported quadrature rule"))
    end
    n_points = size(data,1)
    points = Vector{Vec{3,T}}(undef, n_points)

    for p in 1:size(data, 1)
        points[p] = Vec{3,T}(@ntuple 3 i -> data[p, i])
    end
    weights = T.(data[:, 4])
    QuadratureRule{RefPrism}(weights, points)
end

# Grab pyramid quadrature rule from table
function QuadratureRule{RefPyramid}(::Type{T}, quad_type::Symbol, order::Int) where T
    if quad_type == :polyquad
        data = _get_gauss_pyramiddata_polyquad(order)
    else
        throw(ArgumentError("unsupported quadrature rule"))
    end
    n_points = size(data,1)
    points = Vector{Vec{3,T}}(undef, n_points)

    for p in 1:size(data, 1)
        points[p] = Vec{3,T}(@ntuple 3 i -> data[p, i])
    end
    weights = T.(data[:, 4])
    QuadratureRule{RefPyramid}(weights, points)
end

######################
# FacetQuadratureRule #
######################

"""
    FacetQuadratureRule{shape}([::Type{T},] [quad_rule_type::Symbol,] order::Int)
    FacetQuadratureRule{shape}(face_rules::NTuple{<:Any, <:QuadratureRule{shape}})
    FacetQuadratureRule{shape}(face_rules::AbstractVector{<:QuadratureRule{shape}})

Create a `FacetQuadratureRule` used for integration of the faces of the refshape `shape` (of
type [`AbstractRefShape`](@ref)). `order` is the order of the quadrature rule.
It defaults to the default rules for the individual face reference elements. Elements with mixed facet types
do not have a way of specifying the quadrature rule via symbol for now.

`FacetQuadratureRule` is used as one of the components to create [`FacetValues`](@ref).
"""
struct FacetQuadratureRule{shape, FacetRulesType}
    face_rules::FacetRulesType # E.g. Tuple{QuadratureRule{RefLine,...}, QuadratureRule{RefLine,...}}
    function FacetQuadratureRule{shape}(face_rules::Union{NTuple{<:Any, QRType}, AbstractVector{QRType}}) where {shape, QRType <: QuadratureRule{shape}}
        if length(face_rules) != nfacets(shape)
            throw(ArgumentError("number of quadrature rules does not not match number of facets (#rules=$(length(face_rules)) != #facets=$(nfacets(shape)))"))
        end
        return new{shape, typeof(face_rules)}(face_rules)
    end
end

function FacetQuadratureRule(face_rules::Union{NTuple{<:Any, QRType}, AbstractVector{QRType}}) where {shape, QRType <: QuadratureRule{shape}}
    return FacetQuadratureRule{shape}(face_rules)
end

# Fill in defaults T=Float64
function FacetQuadratureRule{shape}(order::Int) where {shape <: AbstractRefShape}
    return FacetQuadratureRule{shape}(Float64, order)
end
function FacetQuadratureRule{shape}(quad_type::Symbol, order::Int) where {shape <: AbstractRefShape}
    return FacetQuadratureRule{shape}(Float64, quad_type, order)
end

# For RefShapes with equal face-shapes: generate quad rule for the face shape
# and expand to each face
function FacetQuadratureRule{RefLine}(::Type{T}, ::Int) where T
    w, p = T[1], Vec{0, T}[Vec{0, T}(())]
    return create_facet_quad_rule(RefLine, w, p)
end
FacetQuadratureRule{RefQuadrilateral}(::Type{T}, order::Int) where T = FacetQuadratureRule{RefQuadrilateral}(T,_default_quadrature_rule(RefLine),order)
function FacetQuadratureRule{RefQuadrilateral}(::Type{T}, quad_type::Symbol, order::Int) where T
    qr = QuadratureRule{RefLine}(T, quad_type, order)
    return create_facet_quad_rule(RefQuadrilateral, qr.weights, qr.points)
end
FacetQuadratureRule{RefHexahedron}(::Type{T}, order::Int) where T = FacetQuadratureRule{RefHexahedron}(T,_default_quadrature_rule(RefQuadrilateral),order)
function FacetQuadratureRule{RefHexahedron}(::Type{T}, quad_type::Symbol, order::Int) where T
    qr = QuadratureRule{RefQuadrilateral}(T, quad_type, order)
    return create_facet_quad_rule(RefHexahedron, qr.weights, qr.points)
end
FacetQuadratureRule{RefTriangle}(::Type{T}, order::Int) where T = FacetQuadratureRule{RefTriangle}(T,_default_quadrature_rule(RefLine),order)
function FacetQuadratureRule{RefTriangle}(::Type{T}, quad_type::Symbol, order::Int) where T
    qr = QuadratureRule{RefLine}(T, quad_type, order)
    # Interval scaled and shifted in facet_to_element_transformation from (-1,1) to (0,1) -> half the length -> half quadrature weights
    return create_facet_quad_rule(RefTriangle, qr.weights/2, qr.points)
end
FacetQuadratureRule{RefTetrahedron}(::Type{T}, order::Int) where T = FacetQuadratureRule{RefTetrahedron}(T,_default_quadrature_rule(RefTriangle),order)
function FacetQuadratureRule{RefTetrahedron}(::Type{T}, quad_type::Symbol, order::Int) where T
    qr = QuadratureRule{RefTriangle}(T, quad_type, order)
    return create_facet_quad_rule(RefTetrahedron, qr.weights, qr.points)
end
FacetQuadratureRule{RefPrism}(::Type{T}, order::Int) where T = _FacetQuadratureRulePrism(T,(_default_quadrature_rule(RefTriangle), _default_quadrature_rule(RefQuadrilateral)),order)
function _FacetQuadratureRulePrism(::Type{T}, quad_types::Tuple{Symbol,Symbol}, order::Int) where T
    qr_quad = QuadratureRule{RefQuadrilateral}(T, quad_types[2], order)
    qr_tri = QuadratureRule{RefTriangle}(T, quad_types[1], order)
    # Interval scaled and shifted in facet_to_element_transformation for quadrilateral faces from (-1,1)² to (0,1)² -> quarter the area -> quarter the quadrature weights
    return create_facet_quad_rule(RefPrism, [2,3,4], qr_quad.weights/4, qr_quad.points,
        [1,5], qr_tri.weights, qr_tri.points)
end
FacetQuadratureRule{RefPyramid}(::Type{T}, order::Int) where T = _FacetQuadratureRulePyramid(T,(_default_quadrature_rule(RefTriangle), _default_quadrature_rule(RefQuadrilateral)),order)
function _FacetQuadratureRulePyramid(::Type{T}, quad_types::Tuple{Symbol,Symbol}, order::Int) where T
    qr_quad = QuadratureRule{RefQuadrilateral}(T, quad_types[2], order)
    qr_tri = QuadratureRule{RefTriangle}(T, quad_types[1], order)
    # Interval scaled and shifted in facet_to_element_transformation for quadrilateral faces from (-1,1)² to (0,1)² -> quarter the area -> quarter the quadrature weights
    return create_facet_quad_rule(RefPyramid, [1], qr_quad.weights/4, qr_quad.points,
        [2,3,4,5], qr_tri.weights, qr_tri.points)
end

##################
# Common methods #
##################

"""
    getnquadpoints(qr::QuadratureRule)

Return the number of quadrature points in `qr`.
"""
getnquadpoints(qr::QuadratureRule) = length(getweights(qr))

"""
    getnquadpoints(qr::FacetQuadratureRule, face::Int)

Return the number of quadrature points in `qr` for local face index `face`.
"""
getnquadpoints(qr::FacetQuadratureRule, face::Int) = getnquadpoints(qr.face_rules[face])

"""
    getweights(qr::QuadratureRule)
    getweights(qr::FacetQuadratureRule, face::Int)

Return the weights of the quadrature rule.

# Examples
```jldoctest
julia> qr = QuadratureRule{RefTriangle}(:legendre, 2);

julia> getweights(qr)
3-element Array{Float64,1}:
 0.166667
 0.166667
 0.166667
```
"""
getweights(qr::QuadratureRule) = qr.weights
getweights(qr::FacetQuadratureRule, face::Int) = getweights(qr.face_rules[face])


"""
    getpoints(qr::QuadratureRule)
    getpoints(qr::FacetQuadratureRule, face::Int)

Return the points of the quadrature rule.

# Examples
```jldoctest
julia> qr = QuadratureRule{RefTriangle}(:legendre, 2);

julia> getpoints(qr)
3-element Vector{Vec{2, Float64}}:
 [0.16666666666667, 0.16666666666667]
 [0.16666666666667, 0.66666666666667]
 [0.66666666666667, 0.16666666666667]
```
"""
getpoints(qr::QuadratureRule) = qr.points
getpoints(qr::FacetQuadratureRule, face::Int) = getpoints(qr.face_rules[face])

getrefshape(::QuadratureRule{RefShape}) where RefShape = RefShape

# TODO: This is used in copy(::(Cell|Face)Values), but it it useful to get an actual copy?
Base.copy(qr::Union{QuadratureRule, FacetQuadratureRule}) = qr
