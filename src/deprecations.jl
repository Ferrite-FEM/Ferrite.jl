struct DeprecationError <: Exception
    msg::String
end
function DeprecationError(msg::Pair)
    io = _iobuffer()
    printstyled(io, "`$(msg.first)`", color=:red)
    print(io, " is deprecated, use ")
    printstyled(io, "`$(msg.second)`", color=:green)
    print(io, " instead.")
    DeprecationError(takestring(io))
end

function Base.showerror(io::IO, err::DeprecationError)
    print(io, "DeprecationError: ")
    print(io, err.msg)
end

function _iobuffer()
    io = IOBuffer()
    ioc = IOContext(io, IOContext(stderr))
    return ioc
end
function takestring(ioc)
    String(take!(ioc.io))
end

function Base.push!(::AbstractDofHandler, args...)
    throw(DeprecationError("push!(dh::AbstractDofHandler, args...)" => "add!(dh, args...)"))
end

for (a, b) in [(:vertices, :vertexdof_indices), (:faces, :facedof_indices), (:edges, :edgedof_indices)]
    @eval function $(a)(::Interpolation)
        throw(DeprecationError("$($(a))(ip::Interpolation)" => "`$($(b))(ip)"))
    end
end

function nfields(::AbstractDofHandler)
    throw(DeprecationError("nfields(dh::AbstractDofHandler)" => "length(getfieldnames(dh))"))
end

export getcoordinates
function getcoordinates(::Node)
    throw(DeprecationError("getcoordinates(node::Node)" => "get_node_coordinate(node)"))
end

function cellcoords!(x::Vector, dh::DofHandler, args...)
    throw(DeprecationError("cellcoords!(x::Vector, dh::DofHandler, args...)" => "getcoordinates!(x, dh.grid, args...)"))
end

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
            throw(DeprecationError("The AbstractCell interface have been changed, see https://github.com/Ferrite-FEM/Ferrite.jl/pull/679"))
        else
            throw(DeprecationError("Cell{$refdim, $nnodes, $nfaces}(nodes)" => "$replacement(nodes)"))
        end
    end
end
export Cell

const Line2D = Cell{2,2,1}
const Line3D = Cell{3,2,0}
const Quadrilateral3D = Cell{3,4,1}
export Line2D, Line3D, Quadrilateral3D

using WriteVTK: vtk_grid
export vtk_grid # To give better error

function WriteVTK.vtk_grid(::String, ::Union{AbstractGrid,AbstractDofHandler}; kwargs...)
    throw(DeprecationError(
        "The vtk interface has been updated in Ferrite v1.0. " *
        "See https://github.com/Ferrite-FEM/Ferrite.jl/pull/692. " *
        "Use VTKGridFile to open a vtk file, and the functions " *
        "write_solution, write_cell_data, and write_projection to save data."
    ))
end

# Deprecation of auto-vectorized methods
function add!(dh::DofHandler, name::Symbol, dim::Int)
    celltype = getcelltype(dh.grid)
    if !isconcretetype(celltype)
        error("If you have more than one celltype in Grid, you must use add!(dh::DofHandler, fh::FieldHandler)")
    end
    io = _iobuffer()
    printstyled(io, "`add!(dh::DofHandler, name::Symbol, dim::Int)`", color=:red)
    print(io, " is deprecated. Instead, pass the interpolation explicitly, and vectorize it to `dim` for vector-valued fields.")
    print(io, " See CHANGELOG for more details.")
    throw(DeprecationError(takestring(io)))
end

function add!(dh::DofHandler, name::Symbol, dim::Int, ip::ScalarInterpolation)
    io = _iobuffer()
    printstyled(io, "`add!(dh::DofHandler, name::Symbol, dim::Int, ip::ScalarInterpolation)`", color=:red)
    print(io, " is deprecated. Instead, vectorize the interpolation to the appropriate dimension and then `add!` it.")
    print(io, " See CHANGELOG for more details.")
    throw(DeprecationError(takestring(io)))
end

# Deprecation of compute_vertex_values
export compute_vertex_values
function compute_vertex_values(nodes::Vector{<:Node}, f::Function)
    throw(DeprecationError("compute_vertex_values(nodes::Vector{<:Node}, f::Function)" => "map(n -> f(n.x), nodes)"))
end
function compute_vertex_values(grid::AbstractGrid, f::Function)
    throw(DeprecationError("compute_vertex_values(grid::AbstractGrid, f::Function)" => "map(n -> f(n.x), getnodes(grid))"))
end
function compute_vertex_values(grid::AbstractGrid, v::Vector{Int}, f::Function)
    throw(DeprecationError("compute_vertex_values(grid::AbstractGrid, v::Vector{Int}, f::Function)" => "map(n -> f(n.x), getnodes(grid, v))"))
end
function compute_vertex_values(grid::AbstractGrid, set::String, f::Function)
    throw(DeprecationError("compute_vertex_values(grid::AbstractGrid, set::String, f::Function)" => "map(n -> f(n.x), getnodes(grid, set))"))
end

function reshape_to_nodes(args...)
    throw(DeprecationError("reshape_to_nodes(args...)" => "evaluate_at_grid_nodes(args...)"))
end

function start_assemble(f::Vector, K::Union{SparseMatrixCSC, Symmetric}; kwargs...)
    throw(DeprecationError("start_assemble(f::Vector, K::Union{SparseMatrixCSC, Symmetric}; kwargs...)" => "start_assemble(K, f; kwargs...)"))
end

function shape_derivative(args...)
    throw(DeprecationError("shape_derivative(args...)" => "shape_gradient(args...)"))
end
function function_derivative(args...)
    throw(DeprecationError("function_derivative(args...)" => "function_gradient(args...)"))
end

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
           $(str_scalar)(quad_rule, interpolation)
       with
           $(str_new)(quad_rule, interpolation)
     - Vector fields: Replace usage of
           $(str_vector)(quad_rule, interpolation)
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
            throw(DeprecationError($message))
        end
        export $(VT)
    end
end

# TODO: Are these needed to be deprecated - harder? with the new parameterization
# (Cell|Face)Values with vector dofs
const _VectorValues = Union{CellValues{<:FV}, FacetValues{<:FV}} where {FV <: FunctionValues{<:Any,<:VectorInterpolation}}
function function_value(::_VectorValues, ::Int, ::AbstractVector{Vec{dim,T}}) where {dim,T}
    throw(DeprecationError("function_value(fe_v::VectorValues, q_point::Int, u::AbstractVector{Vec{dim,T}})" => "function_value(fe_v, q_point, reinterpret(T, u))"))
end
function function_gradient(::_VectorValues, ::Int, ::AbstractVector{Vec{dim,T}}) where {dim,T}
    throw(DeprecationError("function_gradient(fe_v::VectorValues, q_point::Int, u::AbstractVector{Vec{dim,T}})" => "function_gradient(fe_v, q_point, reinterpret(T, u))"))
end
function function_divergence(::_VectorValues, ::Int, ::AbstractVector{Vec{dim,T}}) where {dim,T}
    throw(DeprecationError("function_divergence(fe_v::VectorValues, q_point::Int, u::AbstractVector{Vec{dim,T}})" => "function_divergence(fe_v, q_point, reinterpret(T, u))"))
end
function function_curl(::_VectorValues, ::Int, ::AbstractVector{Vec{dim,T}}) where {dim,T}
    throw(DeprecationError("function_curl(fe_v::VectorValues, q_point::Int, u::AbstractVector{Vec{dim,T}})" => "function_curl(fe_v, q_point, reinterpret(T, u))"))
end

# New reference shapes
struct RefCube end
export RefCube

function Lagrange{D, RefCube, O}() where {D, O}
    shape = D == 1 ? RefLine : D == 2 ? RefQuadrilateral : RefHexahedron
    throw(DeprecationError("Lagrange{$D, RefCube, $O}()" => "Lagrange{$(shape), $O}()"))
end
function Lagrange{2, RefTetrahedron, O}() where {O}
    throw(DeprecationError("Lagrange{2, RefTetrahedron, $O}()" => "Lagrange{RefTriangle, $O}()"))
end
function DiscontinuousLagrange{D, RefCube, O}() where {D, O}
    shape = D == 1 ? RefLine : D == 2 ? RefQuadrilateral : RefHexahedron
    throw(DeprecationError("DiscontinuousLagrange{$D, RefCube, $O}()" => "DiscontinuousLagrange{$(shape), $O}()"))
end
function BubbleEnrichedLagrange{2, RefTetrahedron, O}() where {O}
    throw(DeprecationError("BubbleEnrichedLagrange{2, RefTetrahedron, $O}()" => "BubbleEnrichedLagrange{RefTriangle, $O}()"))
end
function DiscontinuousLagrange{2, RefTetrahedron, O}() where {O}
    throw(DeprecationError("DiscontinuousLagrange{2, RefTetrahedron, $O}()" => "DiscontinuousLagrange{RefTriangle, $O}()"))
end
function Serendipity{D, RefCube, O}() where {D, O}
    shape = D == 1 ? RefLine : D == 2 ? RefQuadrilateral : RefHexahedron
    throw(DeprecationError("Serendipity{$D, RefCube, $O}()" => "Serendipity{$(shape), $O}()"))
end
function CrouzeixRaviart{2, 1}()
    throw(DeprecationError("CrouzeixRaviart{2, 1}()" => "CrouzeixRaviart{RefTriangle, 1}()"))
end

# For the quadrature: Some will be wrong for face integration, so then we warn
# in the FaceValue constructor...

# QuadratureRule{1, RefCube}(...) -> QuadratureRule{RefLine}(...)
# QuadratureRule{2, RefCube}(...) -> QuadratureRule{RefQuadrilateral}(...)
# QuadratureRule{3, RefCube}(...) -> QuadratureRule{RefHexahedron}(...)
# QuadratureRule{1, RefCube}(...) -> FacetQuadratureRule{RefQuadrilateral}(...)
# QuadratureRule{2, RefCube}(...) -> FacetQuadratureRule{RefHexahedron}(...)
function QuadratureRule{D, RefCube}(order::Int) where D
    shapes = (RefLine, RefQuadrilateral, RefHexahedron)
    msg = "`QuadratureRule{$D, RefCube}(order::Int)` is deprecated, use `QuadratureRule{$(shapes[D])}(order)` instead"
    if D == 1 || D == 2
        msg *= " (or `FacetQuadratureRule{$(shapes[D+1])}(order)` if this is a face quadrature rule)"
    end
    msg *= "."
    throw(DeprecationError(msg))
end
function QuadratureRule{D, RefCube}(quad_type::Symbol, order::Int) where D
    shapes = (RefLine, RefQuadrilateral, RefHexahedron)
    msg = "`QuadratureRule{$D, RefCube}(quad_type::Symbol, order::Int)` is deprecated, use `QuadratureRule{$(shapes[D])}(quad_type, order)` instead"
    if D == 1 || D == 2
        msg *= " (or `FacetQuadratureRule{$(shapes[D+1])}(quad_type, order)` if this is a face quadrature rule)"
    end
    msg *= "."
    throw(DeprecationError(msg))
end

# QuadratureRule{2, RefTetrahedron}(...) -> QuadratureRule{RefTriangle}(...)
# QuadratureRule{3, RefTetrahedron}(...) -> QuadratureRule{RefTetrahedron}(...)
# QuadratureRule{2, RefTetrahedron}(...) -> FacetQuadratureRule{RefTetrahedron}(...)
function QuadratureRule{D, RefTetrahedron}(order::Int) where D
    shapes = (nothing, RefTriangle, RefTetrahedron)
    msg = "`QuadratureRule{$D, RefTetrahedron}(order::Int)` is deprecated, use `QuadratureRule{$(shapes[D])}(order)` instead"
    if D == 2
        msg *= " (or `FacetQuadratureRule{RefTetrahedron)}(order)` if this is a face quadrature rule)"
    end
    msg *= "."
    throw(DeprecationError(msg))
end
function QuadratureRule{D, RefTetrahedron}(quad_type::Symbol, order::Int) where D
    shapes = (nothing, RefTriangle, RefTetrahedron)
    msg = "`QuadratureRule{$D, RefTetrahedron}(quad_type::Symbol, order::Int)` is deprecated, use `QuadratureRule{$(shapes[D])}(quad_type, order)` instead"
    if D == 2
        msg *= " (or `FacetQuadratureRule{RefTetrahedron)}(order)` if this is a face quadrature rule)"
    end
    msg *= "."
    throw(DeprecationError(msg))
end

# QuadratureRule{0, RefCube}(...) -> FacetQuadratureRule{RefLine}
function QuadratureRule{0, RefCube}(order::Int)
    msg = "`QuadratureRule{0, RefCube}(order::Int)` is deprecated, use `FacetQuadratureRule{RefLine}(order)` instead."
    throw(DeprecationError(msg))
end
function QuadratureRule{0, RefCube}(quad_type::Symbol, order::Int)
    msg = "`QuadratureRule{0, RefCube}(quad_type::Symbol, order::Int)` is deprecated, use `FacetQuadratureRule{RefLine}(quad_type, order)` instead."
    throw(DeprecationError(msg))
end

# QuadratureRule{1, RefTetrahedron}(...) -> FacetQuadratureRule{RefTriangle}
function QuadratureRule{1, RefTetrahedron}(order::Int)
    msg = "`QuadratureRule{1, RefTetrahedron}(order::Int)` is deprecated, use `FacetQuadratureRule{RefTriangle}(order)` instead."
    throw(DeprecationError(msg))
end
function QuadratureRule{1, RefTetrahedron}(quad_type::Symbol, order::Int)
    msg = "`QuadratureRule{1, RefTetrahedron}(quad_type::Symbol, order::Int)` is deprecated, use `FacetQuadratureRule{RefTriangle}(quad_type, order)` instead."
    throw(DeprecationError(msg))
end

# Catch remaining cases in (Cell|Face)Value constructors
function CellValues(
    ::Type{T}, qr::QuadratureRule{2, RefTetrahedron, TQ}, ip::Interpolation{RefTriangle},
    gip::Interpolation{RefTriangle} = default_geometric_interpolation(ip),
) where {T, TQ}
    msg = "The input quadrature rule have the wrong reference shape, likely this comes from a constructor like `QuadratureRule{2, RefTetrahedron}(...)` which have been deprecated in favor of `QuadratureRule{RefTriangle}(...)`."
    throw(DeprecationError(msg))
end
function FacetValues(qr::QuadratureRule, ip::Interpolation,
                    gip::Interpolation = default_geometric_interpolation(ip))
    return FacetValues(Float64, qr, ip, gip)
end
function FacetValues(
    ::Type{T}, qr::QuadratureRule{RefLine, TQ}, ip::Interpolation{RefQuadrilateral},
    gip::Interpolation{RefQuadrilateral} = default_geometric_interpolation(ip),
) where {T, TQ}
    msg = "The input quadrature rule have the wrong reference shape, likely this comes from a constructor like `QuadratureRule{1, RefCube}(...)` which have been deprecated in favor of `FacetQuadratureRule{RefQuadrilateral}(...)`."
    throw(DeprecationError(msg))
end
function FacetValues(
    ::Type{T}, qr::QuadratureRule{RefQuadrilateral, TQ}, ip::Interpolation{RefHexahedron},
    gip::Interpolation{RefHexahedron} = default_geometric_interpolation(ip),
) where {T, TQ}
    msg = "The input quadrature rule have the wrong reference shape, likely this comes from a constructor like `QuadratureRule{2, RefCube}(...)` which have been deprecated in favor of `FacetQuadratureRule{RefHexahedron}(...)`."
    throw(DeprecationError(msg))
end
function FacetValues(
    ::Type{T}, qr::QuadratureRule{RefTriangle, TQ}, ip::Interpolation{RefTetrahedron},
    gip::Interpolation{RefTetrahedron} = default_geometric_interpolation(ip),
) where {T, TQ}
    msg = "The input quadrature rule have the wrong reference shape, likely this comes from a constructor like `QuadratureRule{2, RefTetrahedron}(...)` which have been deprecated in favor of `FacetQuadratureRule{RefTetrahedron}(...)`."
    throw(DeprecationError(msg))
end

# Hide the last unused type param...
function Base.show(io::IO, ::DiscontinuousLagrange{shape, order}) where {shape, order}
    print(io, "DiscontinuousLagrange{$(shape), $(order)}()")
end
function Base.show(io::IO, ::Lagrange{shape, order}) where {shape, order}
    print(io, "Lagrange{$(shape), $(order)}()")
end
function Base.show(io::IO, ::Serendipity{shape, order}) where {shape, order}
    print(io, "Serendipity{$(shape), $(order)}()")
end
function Base.show(io::IO, ::CrouzeixRaviart{shape, order}) where {shape, order}
    print(io, "CrouzeixRaviart{$(shape), $(order)}()")
end

function value(ip::Interpolation, ξ::Vec)
    throw(DeprecationError("value(ip::Interpolation, ξ::Vec)" => "[reference_shape_value(ip, ξ, i) for i in 1:getnbasefunctions(ip)]"))
end
function derivative(ip::Interpolation, ξ::Vec)
    throw(DeprecationError("derivative(ip::Interpolation, ξ::Vec)" => "[reference_shape_gradient(ip, ξ, i) for i in 1:getnbasefunctions(ip)]"))
end
function value(ip::Interpolation, i::Int, ξ::Vec)
    throw(DeprecationError("value(ip::Interpolation, i::Int, ξ::Vec)" => "reference_shape_value(ip, ξ, i)"))
end

export MixedDofHandler
function MixedDofHandler(::AbstractGrid)
    throw(DeprecationError(
        "MixedDofHandler is the standard DofHandler in Ferrite now and has been renamed " *
        "to DofHandler. Use DofHandler even for mixed grids and fields on subdomains.",
    ))
end

export end_assemble
function end_assemble(args...)
    throw(DeprecationError("end_assemble(args...)" => "finish_assemble(args...)"))
end

export get_point_values
function get_point_values(args...)
    throw(DeprecationError("get_point_values(args...)" => "evaluate_at_points(args...)"))
end

export transform!
function transform!(args...)
    throw(DeprecationError("transform!(args...)" => "transform_coordinates!(args...)"))
end

export addfaceset! # deprecated, export for backwards compatibility.
# Use warn to show for standard users.
function addfaceset!(grid::AbstractGrid, name, set::Union{Set{FaceIndex}, Vector{FaceIndex}})
    msg = "addfaceset! is deprecated, use addfacetset! instead and convert the set to FacetIndex."
    throw(DeprecationError(msg))
end
function addfaceset!(grid, name, f::Function; kwargs...)
    throw(DeprecationError("addfaceset!(args...)" => "addfacetset!(args...)"))
end

export onboundary
function onboundary(::CellCache, ::Int)
    throw(DeprecationError("`onboundary` is deprecated, check just the facetset instead of first checking `onboundary`."))
end

getdim(args...) = throw(DeprecationError("`Ferrite.getdim` is deprecated, use `getrefdim` or `getspatialdim` instead"))
getfielddim(args...) = throw(DeprecationError("`Ferrite.getfielddim(::AbstractDofHandler, args...) is deprecated, use `n_components` instead"))

function default_interpolation(::Type{C}) where {C <: AbstractCell}
    msg = "Ferrite.default_interpolation is deprecated, use the exported `geometric_interpolation` instead"
    throw(DeprecationError(msg))
end

export create_sparsity_pattern
function create_sparsity_pattern(args...)
    throw(DeprecationError("create_sparsity_pattern(args...)" => "allocate_matrix(args...; kwargs...)"))
end

export VTKFile
function VTKFile(args...)
    throw(DeprecationError("VTKFile(args...)" => "VTKGridFile(args...)"))
end
