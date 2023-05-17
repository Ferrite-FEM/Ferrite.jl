Base.@deprecate_binding DirichletBoundaryConditions ConstraintHandler
Base.@deprecate_binding DirichletBoundaryCondition Dirichlet

import Base: push!
@deprecate push!(dh::AbstractDofHandler, args...) add!(dh, args...)

@deprecate vertices(ip::Interpolation) vertexdof_indices(ip) false
@deprecate faces(ip::Interpolation) facedof_indices(ip) false
@deprecate edges(ip::Interpolation) edgedof_indices(ip) false
@deprecate nfields(dh::AbstractDofHandler) length(getfieldnames(dh)) false

@deprecate add!(ch::ConstraintHandler, fh::FieldHandler, dbc::Dirichlet) add!(ch, dbc)


struct Cell{refdim, nnodes, nfaces}
    function Cell{refdim, nnodes, nfaces}(nodes) where {refdim, nnodes, nfaces}
        params = (refdim, nnodes, nfaces)
        replacement = nothing
        if params == (1, 2, 2) || params == (2, 2, 1) || params == (3, 2, 0)
            replacement = Line
        elseif params == (1, 3, 2)
            replacement = QuadraticLine
        elseif params == (2, 3, 3)
            replacement = Triangle
        elseif params == (2, 6, 3)
            replacement = QuadraticTriangle
        elseif params == (2, 4, 4) || params == (3, 4, 1)
            replacement = Quadrilateral
        elseif params == (2, 9, 4)
            replacement = QuadraticQuadrilateral
        elseif params == (2, 8, 4)
            replacement = SerendipityQuadraticQuadrilateral
        elseif params == (3, 4, 4)
            replacement = Tetrahedron
        elseif params == (3, 10, 4)
            replacement = QuadraticTetrahedron
        elseif params == (3, 8, 6)
            replacement = Hexahedron
        elseif params == (3, 27, 6)
            replacement = QuadraticHexahedron
        elseif params == (3, 20, 6)
            replacement = SerendipityQuadraticHexahedron
        elseif params == (3, 6, 5)
            replacement = Wedge
        end
        if replacement === nothing
            error("The AbstractCell interface have been changed, see https://github.com/Ferrite-FEM/Ferrite.jl/pull/679")
        else
            Base.depwarn("Use `$(replacement)(nodes)` instead of `Cell{$refdim, $nnodes, $nfaces}(nodes)`.", :Cell)
            return replacement(nodes)
        end
    end
end
export Cell

Base.@deprecate_binding Line2D Line
Base.@deprecate_binding Line3D Line
Base.@deprecate_binding Quadrilateral3D Quadrilateral
export Line2D, Line3D, Quadrilateral3D

# Deprecation of auto-vectorized methods
function add!(dh::DofHandler, name::Symbol, dim::Int)
    celltype = getcelltype(dh.grid)
    if !isconcretetype(celltype)
        error("If you have more than one celltype in Grid, you must use add!(dh::DofHandler, fh::FieldHandler)")
    end
    Base.depwarn(
        "`add!(dh::DofHandler, name::Symbol, dim::Int)` is deprecated. Instead, pass the " *
        "interpolation explicitly, and vectorize it to `dim` for vector-valued " *
        "fields. See CHANGELOG for more details.",
        :add!,
    )
    ip = default_interpolation(celltype)
    add!(dh, name, dim == 1 ? ip : VectorizedInterpolation{dim}(ip))
end

function add!(dh::DofHandler, name::Symbol, dim::Int, ip::ScalarInterpolation)
    Base.depwarn(
        "`add!(dh::DofHandler, name::Symbol, dim::Int, ip::ScalarInterpolation)` is " *
        "deprecated. Instead, vectorize the interpolation to the appropriate dimension " *
        "and add it (`vip = ip^dim; add!(dh, name, vip)`). See CHANGELOG for more details.",
        :add!
    )
    add!(dh, name, dim == 1 ? ip : VectorizedInterpolation{dim}(ip))
end

# Deprecation of compute_vertex_values
@deprecate compute_vertex_values(nodes::Vector{<:Node}, f::Function) map(n -> f(n.x), nodes)
@deprecate compute_vertex_values(grid::AbstractGrid, f::Function) map(n -> f(n.x), getnodes(grid))
@deprecate compute_vertex_values(grid::AbstractGrid, v::Vector{Int}, f::Function) map(n -> f(n.x), getnodes(grid, v))
@deprecate compute_vertex_values(grid::AbstractGrid, set::String, f::Function) map(n -> f(n.x), getnodes(grid, set))

@deprecate reshape_to_nodes evaluate_at_grid_nodes

@deprecate start_assemble(f::Vector, K::Union{SparseMatrixCSC, Symmetric}; kwargs...) start_assemble(K, f; kwargs...)

@deprecate shape_derivative shape_gradient
@deprecate function_derivative function_gradient

# Deprecation of (Cell|Face|Point)(Scalar|Vector)Values.
# Define dummy types so that loading old code doesn't directly error, and let
# us print a descriptive error message in the constructor.
for VT in (
    :CellScalarValues, :CellVectorValues,
    :FaceScalarValues, :FaceVectorValues,
    :PointScalarValues, :PointVectorValues,
)
    str = string(VT)
    str_scalar = replace(str, "Vector" => "Scalar")
    str_vector = replace(str, "Scalar" => "Vector")
    str_new = replace(str_scalar, "Scalar" => "")
    io = IOBuffer()
    print(io, """
    The `$(str)` interface has been reworked for Ferrite.jl 0.4.0:

     - `$(str_scalar)` and `$(str_vector)` have been merged into a single type: `$(str_new)`
     - "Vectorization" of (scalar) interpolations should now be done on the interpolation
       instead of implicitly in the `CellValues` constructor.

    Upgrade as follows:
     - Scalar fields: Replace usage of
           $(str)(quad_rule, interpolation)
       with
           $(str_new)(quad_rule, interpolation)
     - Vector fields: Replace usage of
           $(str)(quad_rule, interpolation)
       with
           $(str_new)(quad_rule, interpolation^dim)
       where `dim` is the dimension to vectorize to.

    See CHANGELOG.md (https://github.com/Ferrite-FEM/Ferrite.jl/blob/master/CHANGELOG.md) for more details.
    """)
    message = String(take!(io))
    if occursin("Point", str)
        message = replace(message, "quad_rule, " => "")
        @eval begin
            struct $(VT){D, T, R, CV, IP}
                construct_me_if_you_can() = nothing
            end
        end
    else
        @eval begin
            struct $(VT){D, T, R}
                construct_me_if_you_can() = nothing
            end
        end
    end
    @eval begin
        function $(VT)(args...)
            error($message)
        end
        export $(VT)
    end
end

# (Cell|Face)Values with vector dofs
const _VectorValues = Union{CellValues{<:VectorInterpolation}, FaceValues{<:VectorInterpolation}}
@deprecate      function_value(fe_v::_VectorValues, q_point::Int, u::AbstractVector{Vec{dim,T}}) where {dim,T}      function_value(fe_v, q_point, reinterpret(T, u))
@deprecate   function_gradient(fe_v::_VectorValues, q_point::Int, u::AbstractVector{Vec{dim,T}}) where {dim,T}   function_gradient(fe_v, q_point, reinterpret(T, u))
@deprecate function_divergence(fe_v::_VectorValues, q_point::Int, u::AbstractVector{Vec{dim,T}}) where {dim,T} function_divergence(fe_v, q_point, reinterpret(T, u))
@deprecate       function_curl(fe_v::_VectorValues, q_point::Int, u::AbstractVector{Vec{dim,T}}) where {dim,T}       function_curl(fe_v, q_point, reinterpret(T, u))

# New reference shapes
struct RefCube end
export RefCube

function Lagrange{D, RefCube, O}() where {D, O}
    shape = D == 1 ? RefLine : D == 2 ? RefQuadrilateral : RefHexahedron
    Base.depwarn("`Lagrange{$D, RefCube, $O}()` is deprecated, use `Lagrange{$(shape), $O}()` instead.", :Lagrange)
    return Lagrange{D, shape, O}()
end
function Lagrange{2, RefTetrahedron, O}() where {O}
    Base.depwarn("`Lagrange{2, RefTetrahedron, $O}()` is deprecated, use `Lagrange{RefTriangle, $O}()` instead.", :Lagrange)
    return Lagrange{2, RefTriangle, O}()
end
function DiscontinuousLagrange{D, RefCube, O}() where {D, O}
    shape = D == 1 ? RefLine : D == 2 ? RefQuadrilateral : RefHexahedron
    Base.depwarn("`DiscontinuousLagrange{$D, RefCube, $O}()` is deprecated, use `DiscontinuousLagrange{$(shape), $O}()` instead.", :DiscontinuousLagrange)
    return DiscontinuousLagrange{D, shape, O}()
end
function BubbleEnrichedLagrange{2, RefTetrahedron, O}() where {O}
    Base.depwarn("`BubbleEnrichedLagrange{2, RefTetrahedron, $O}()` is deprecated, use `BubbleEnrichedLagrange{RefTriangle, $O}()` instead.", :BubbleEnrichedLagrange)
    return BubbleEnrichedLagrange{2, RefTriangle, O}()
end
function DiscontinuousLagrange{2, RefTetrahedron, O}() where {O}
    Base.depwarn("`DiscontinuousLagrange{2, RefTetrahedron, $O}()` is deprecated, use `DiscontinuousLagrange{RefTriangle, $O}()` instead.", :DiscontinuousLagrange)
    return DiscontinuousLagrange{2, RefTriangle, O}()
end
function Serendipity{D, RefCube, O}() where {D, O}
    shape = D == 1 ? RefLine : D == 2 ? RefQuadrilateral : RefHexahedron
    Base.depwarn("`Serendipity{$D, RefCube, $O}()` is deprecated, use `Serendipity{$(shape), $O}()` instead.", :Serendipity)
    return Serendipity{D, shape, O}()
end
function CrouzeixRaviart{2, 1}()
    Base.depwarn("`CrouzeixRaviart{2, 1}()` is deprecated, use `CrouzeixRaviart{RefTriangle, 1}()` instead.", :CrouzeixRaviart)
    return CrouzeixRaviart{2, RefTriangle, 1}()
end

# For the quadrature: This will be wrong for face integration, so then we warn
# in the FaceValue constructor...
function QuadratureRule{D, RefCube}(order::Int) where D
    shapes = (RefLine, RefQuadrilateral, RefHexahedron)
    msg = "`QuadratureRule{$D, RefCube}(order::Int)` is deprecated, use `QuadratureRule{$D, $(shapes[D])}(order)` instead"
    if D == 1 || D == 2
        msg *= " (or `QuadratureRule{$D, $(shapes[D+1])}(order)` if this is a face quadrature rule)"
    end
    msg *= "."
    Base.depwarn(msg, :QuadratureRule)
    return QuadratureRule{D, shapes[D]}(order)
end
function QuadratureRule{D, RefCube}(quad_type::Symbol, order::Int) where D
    shapes = (RefLine, RefQuadrilateral, RefHexahedron)
    msg = "`QuadratureRule{$D, RefCube}(quad_type::Symbol, order::Int)` is deprecated, use `QuadratureRule{$D, $(shapes[D])}(quad_type, order)` instead"
    if D == 1 || D == 2
        msg *= " (or `QuadratureRule{$D, $(shapes[D+1])}(order)` if this is a face quadrature rule)"
    end
    msg *= "."
    Base.depwarn(msg, :QuadratureRule)
    return QuadratureRule{D, shapes[D]}(quad_type, order)
end

function QuadratureRule{D, RefCube}(weights::AbstractVector{Tw}, points::AbstractVector{Vec{D,Tp}}) where {D, Tw, Tp}
    shapes = (RefLine, RefQuadrilateral, RefHexahedron)
    msg = "`QuadratureRule{$D, RefCube}(weights, points)` is deprecated, use `QuadratureRule{$D, $(shapes[D])}(weights, points)` instead"
    if D == 1 || D == 2
        msg *= " (or `QuadratureRule{$D, $(shapes[D+1])}(weights, points)` if this is a face quadrature rule)"
    end
    msg *= "."
    Base.depwarn(msg, :QuadratureRule)
    return QuadratureRule{D, shapes[D]}(weights, points)
end

function QuadratureRule{D, RefCube, T}(weights::AbstractVector, points::AbstractVector{<:Vec}) where {D, T}
    shapes = (RefLine, RefQuadrilateral, RefHexahedron)
    msg = "`QuadratureRule{$D, RefCube, T}(weights, points)` is deprecated, use `QuadratureRule{$D, $(shapes[D]), T}(weights, points)` instead"
    if D == 1 || D == 2
        msg *= " (or `QuadratureRule{$D, $(shapes[D+1]), T}(weights, points)` if this is a face quadrature rule)"
    end
    msg *= "."
    Base.depwarn(msg, :QuadratureRule)
    return QuadratureRule{D, shapes[D], T}(weights, points)
end

# These must be for a face, warn only here.
function QuadratureRule{0, RefCube}(order::Int)
    msg = "`QuadratureRule{0, RefCube}(order::Int)` is deprecated, use `QuadratureRule{0, RefLine}(order)` instead."
    Base.depwarn(msg, :QuadratureRule)
    return QuadratureRule{0, RefLine}(order)
end
function QuadratureRule{0, RefCube}(quad_type::Symbol, order::Int)
    msg = "`QuadratureRule{0, RefCube}(quad_type::Symbol, order::Int)` is deprecated, use `QuadratureRule{0, RefLine}(quad_type, order)` instead."
    Base.depwarn(msg, :QuadratureRule)
    return QuadratureRule{0, RefLine}(quad_type, order)
end
function QuadratureRule{1, RefTetrahedron}(order::Int)
    msg = "`QuadratureRule{1, RefTetrahedron}(order::Int)` is deprecated, use `QuadratureRule{1, RefTriangle}(order)` instead."
    Base.depwarn(msg, :QuadratureRule)
    return QuadratureRule{1, RefTriangle}(order)
end
function QuadratureRule{1, RefTetrahedron}(quad_type::Symbol, order::Int)
    msg = "`QuadratureRule{1, RefTetrahedron}(quad_type::Symbol, order::Int)` is deprecated, use `QuadratureRule{0, RefTriangle}(quad_type, order)` instead."
    Base.depwarn(msg, :QuadratureRule)
    return QuadratureRule{1, RefTriangle}(quad_type, order)
end

# Catch remaining cases in (Cell|Face)Value constructors
function CellValues(
    ::Type{T}, qr::QuadratureRule{2, RefTetrahedron, TQ}, ip::Interpolation{2, RefTriangle},
    gip::Interpolation{2, RefTriangle} = default_geometric_interpolation(ip),
) where {T, TQ}
    qr′ = QuadratureRule{2, RefTriangle, T}(qr.weights, qr.points)
    Base.depwarn("The input quadrature rule have the wrong reference shape, likely this comes from a constructor like `QuadratureRule{2, RefTetrahedron}(...)` which have been deprecated in favor of `QuadratureRule{2, RefTriangle}(...)`.", :CellValues)
    CellValues(T, qr′, ip, gip)
end
function FaceValues(
    ::Type{T}, qr::QuadratureRule{1, RefLine, TQ}, ip::Interpolation{2, RefQuadrilateral},
    gip::Interpolation{2, RefQuadrilateral} = default_geometric_interpolation(ip),
) where {T, TQ}
    Base.depwarn("The input quadrature rule have the wrong reference shape, likely this comes from a constructor like `QuadratureRule{1, RefCube}(...)` which have been deprecated in favor of `QuadratureRule{1, RefQuadrilateral}(...)`.", :FaceValues)
    qr′ = QuadratureRule{1, RefQuadrilateral, T}(qr.weights, qr.points)
    FaceValues(T, qr′, ip, gip)
end
function FaceValues(
    ::Type{T}, qr::QuadratureRule{2, RefQuadrilateral, TQ}, ip::Interpolation{3, RefHexahedron},
    gip::Interpolation{3, RefHexahedron} = default_geometric_interpolation(ip),
) where {T, TQ}
    Base.depwarn("The input quadrature rule have the wrong reference shape, likely this comes from a constructor like `QuadratureRule{2, RefCube}(...)` which have been deprecated in favor of `QuadratureRule{2, RefHexahedron}(...)`.", :FaceValues)
    qr′ = QuadratureRule{2, RefHexahedron, T}(qr.weights, qr.points)
    FaceValues(T, qr′, ip, gip)
end
