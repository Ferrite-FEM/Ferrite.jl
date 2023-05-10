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

function CellVectorValues(quad_rule::QuadratureRule, func_interpol::ScalarInterpolation,
        geom_interpol::Interpolation=default_geometric_interpolation(func_interpol))
    return CellVectorValues(Float64, quad_rule, func_interpol, geom_interpol)
end
function CellVectorValues(::Type{T}, quad_rule::QuadratureRule, func_interpol::ScalarInterpolation,
        geom_interpol::Interpolation=func_interpol) where {T}
    Base.depwarn(
        "passing scalar interpolations to CellVectorValues is deprectated. Instead, " *
        "vectorize the interpolation to the appropriate vector dimension first. " *
        "See CHANGELOG for more details.",
        :CellVectorValues
    )
    return CellVectorValues(T, quad_rule, VectorizedInterpolation(func_interpol), geom_interpol)
end
function FaceVectorValues(quad_rule::QuadratureRule, func_interpol::ScalarInterpolation,
        geom_interpol::Interpolation=default_geometric_interpolation(func_interpol))
    return FaceVectorValues(Float64, quad_rule, func_interpol, geom_interpol)
end
function FaceVectorValues(::Type{T}, quad_rule::QuadratureRule, func_interpol::ScalarInterpolation,
        geom_interpol::Interpolation=func_interpol) where {T}
    Base.depwarn(
        "passing scalar interpolations to FaceVectorValues is deprectated. Instead, " *
        "vectorize the interpolation to the appropriate vector dimension first. " *
        "See CHANGELOG for more details.",
        :FaceVectorValues
    )
    return FaceVectorValues(T, quad_rule, VectorizedInterpolation(func_interpol), geom_interpol)
end
function PointVectorValues(::Type{T}, ip::ScalarInterpolation, geom_interpol::Interpolation=default_geometric_interpolation(ip)) where {T}
    Base.depwarn(
        "passing scalar interpolations to PointVectorValues is deprectated. Instead, " *
        "vectorize the interpolation to the appropriate vector dimension first. " *
        "See CHANGELOG for more details.",
        :FaceVectorValues
    )
    return PointVectorValues(T, VectorizedInterpolation(ip), geom_interpol)
end

# (Cell|Face)VectorValues with vector dofs
@deprecate function_value(fe_v::Union{CellVectorValues,FaceVectorValues}, q_point::Int, u::AbstractVector{Vec{dim,T}}) where {dim,T} function_value(fe_v, q_point, reinterpret(T, u))
@deprecate function_value(vv::VectorValued, fe_v::Values, q_point::Int, u::AbstractVector{Vec{dim,T}}) where {dim,T} function_value(vv, fe_v, q_point, reinterpret(T, u))
@deprecate function_gradient(fe_v::Union{CellVectorValues,FaceVectorValues}, q_point::Int, u::AbstractVector{Vec{dim,T}}) where {dim,T} function_gradient(fe_v, q_point, reinterpret(T, u))
@deprecate function_gradient(vv::VectorValued, fe_v::Values{dim}, q_point::Int, u::AbstractVector{Vec{dim,T}}) where {dim,T} function_gradient(vv, fe_v, q_point, reinterpret(T, u))
@deprecate function_divergence(fe_v::Union{CellVectorValues,FaceVectorValues}, q_point::Int, u::AbstractVector{Vec{dim,T}}) where {dim,T} function_divergence(fe_v, q_point, reinterpret(T, u))
@deprecate function_divergence(::VectorValued, fe_v::Values{dim}, q_point::Int, u::AbstractVector{Vec{dim,T}}) where {dim,T} function_divergence(vv, fe_v, q_point, reinterpret(T, u))
@deprecate function_curl(fe_v::Union{CellVectorValues,FaceVectorValues}, q_point::Int, u::AbstractVector{Vec{3, T}}) where T function_curl(fe_v::Values, q_point::Int, reinterpret(T, u))

# Deprecation of compute_vertex_values
@deprecate compute_vertex_values(nodes::Vector{<:Node}, f::Function) map(n -> f(n.x), nodes)
@deprecate compute_vertex_values(grid::AbstractGrid, f::Function) map(n -> f(n.x), getnodes(grid))
@deprecate compute_vertex_values(grid::AbstractGrid, v::Vector{Int}, f::Function) map(n -> f(n.x), getnodes(grid, v))
@deprecate compute_vertex_values(grid::AbstractGrid, set::String, f::Function) map(n -> f(n.x), getnodes(grid, set))
