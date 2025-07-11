"""
    Interpolation{ref_shape, order}()

Abstract type for interpolations defined on `ref_shape`
(see [`AbstractRefShape`](@ref)).
`order` corresponds to the order of the interpolation.
The interpolation is used to define shape functions to interpolate
a function between nodes.

The following interpolations are implemented:

* `Lagrange{RefLine, 1}`
* `Lagrange{RefLine, 2}`
* `Lagrange{RefQuadrilateral, 1}`
* `Lagrange{RefQuadrilateral, 2}`
* `Lagrange{RefQuadrilateral, 3}`
* `Lagrange{RefTriangle, 1}`
* `Lagrange{RefTriangle, 2}`
* `Lagrange{RefTriangle, 3}`
* `Lagrange{RefTriangle, 4}`
* `Lagrange{RefTriangle, 5}`
* `BubbleEnrichedLagrange{RefTriangle, 1}`
* `CrouzeixRaviart{RefTriangle, 1}`
* `CrouzeixRaviart{RefTetrahedron, 1}`
* `RannacherTurek{RefQuadrilateral, 1}`
* `RannacherTurek{RefHexahedron, 1}`
* `Lagrange{RefHexahedron, 1}`
* `Lagrange{RefHexahedron, 2}`
* `Lagrange{RefTetrahedron, 1}`
* `Lagrange{RefTetrahedron, 2}`
* `Lagrange{RefPrism, 1}`
* `Lagrange{RefPrism, 2}`
* `Lagrange{RefPyramid, 1}`
* `Lagrange{RefPyramid, 2}`
* `Serendipity{RefQuadrilateral, 2}`
* `Serendipity{RefHexahedron, 2}`
* `Nedelec{RefTriangle, 1}`
* `Nedelec{RefTriangle, 2}`
* `Nedelec{RefQuadrilateral, 1}`
* `Nedelec{RefTetrahedron, 1}`
* `Nedelec{RefHexahedron, 1}`
* `RaviartThomas{RefTriangle, 1}`
* `RaviartThomas{RefTriangle, 2}`
* `RaviartThomas{RefQuadrilateral, 1}`
* `RaviartThomas{RefTetrahedron, 1}`
* `RaviartThomas{RefHexahedron, 1}`
* `BrezziDouglasMarini{RefTriangle, 1}`

Additionally, `DiscontinuousLagrange` is implemented for the same reference shapes
and orders as `Lagrange`, as well as for `order = 0` on any reference shape.

# Examples
```jldoctest
julia> ip = Lagrange{RefTriangle, 2}()
Lagrange{RefTriangle, 2}()

julia> getnbasefunctions(ip)
6
```
"""
abstract type Interpolation{shape #=<: AbstractRefShape=#, order} end

const InterpolationByDim{dim} = Interpolation{<:AbstractRefShape{dim}}

abstract type ScalarInterpolation{refshape, order} <: Interpolation{refshape, order} end
abstract type VectorInterpolation{vdim, refshape, order} <: Interpolation{refshape, order} end

struct L2Conformity end     # Discontinuous across cell boundaries
struct HdivConformity end   # Normal continuity across cell boundaries
struct HcurlConformity end  # Tangent continuity across cell boundaries
struct H1Conformity end     # Function continuity across cell boundaries

"""
    conformity(ip::Interpolation)

Return the conformity, i.e. `L2Conformity()`, `HdivConformity()`, `HcurlConformity()`,
or `H1Conformity()`, for the interpolation.
"""
function conformity end

# Number of components for the interpolation.
n_components(::ScalarInterpolation) = 1
n_components(::VectorInterpolation{vdim}) where {vdim} = vdim
# Number of components that are allowed to prescribe in e.g. Dirichlet BC
n_dbc_components(ip::Interpolation) = n_components(ip)

"""
    shape_value_type(ip::Interpolation, ::Type{T}) where T<:Number

Return the type of `shape_value(ip::Interpolation, ξ::Vec, ib::Int)`.
"""
shape_value_type(::Interpolation, ::Type{T}) where {T <: Number}

shape_value_type(::ScalarInterpolation, ::Type{T}) where {T <: Number} = T
shape_value_type(::VectorInterpolation{vdim}, ::Type{T}) where {vdim, T <: Number} = Vec{vdim, T}
#shape_value_type(::MatrixInterpolation, T::Type) = Tensor  #958

# TODO: Add a fallback that errors if there are multiple dofs per edge/face instead to force
#       interpolations to opt-out instead of silently do nothing.
"""
    adjust_dofs_during_distribution(::Interpolation)

This function must return `true` if the dofs should be adjusted (i.e. permuted) during dof
distribution. This is in contrast to i) adjusting the dofs during [`reinit!`](@ref) in the
assembly loop, or ii) not adjusting at all (which is not needed for low order
interpolations, generally).
"""
adjust_dofs_during_distribution(::Interpolation)

"""
    InterpolationInfo

Gathers all the information needed to distribute dofs for a given interpolation. Note that
this cache is of the same type no matter the interpolation: the purpose is to make
dof-distribution type-stable.
"""
struct InterpolationInfo
    nvertexdofs::Vector{Int}
    nedgedofs::Vector{Int}
    nfacedofs::Vector{Int}
    nvolumedofs::Int
    reference_dim::Int
    adjust_during_distribution::Bool
    n_copies::Int
end
function InterpolationInfo(interpolation::Interpolation{shape}, n_copies) where {rdim, shape <: AbstractRefShape{rdim}}
    info = InterpolationInfo(
        [length(i) for i in vertexdof_indices(interpolation)],
        [length(i) for i in edgedof_interior_indices(interpolation)],
        [length(i) for i in facedof_interior_indices(interpolation)],
        length(volumedof_interior_indices(interpolation)),
        rdim,
        adjust_dofs_during_distribution(interpolation),
        n_copies
    )
    return info
end
InterpolationInfo(interpolation::Interpolation) = InterpolationInfo(interpolation, 1)

nvertices(::Interpolation{RefShape}) where {RefShape} = nvertices(RefShape)
nedges(::Interpolation{RefShape}) where {RefShape} = nedges(RefShape)
nfaces(::Interpolation{RefShape}) where {RefShape} = nfaces(RefShape)

Base.copy(ip::Interpolation) = ip

"""
    Ferrite.getrefdim(::Interpolation)

Return the dimension of the reference element for a given interpolation.
"""
getrefdim(::Interpolation) # To make doc-filtering work
@inline getrefdim(::Interpolation{RefShape}) where {RefShape} = getrefdim(RefShape)

"""
    Ferrite.getrefshape(::Interpolation)::AbstractRefShape

Return the reference element shape of the interpolation.
"""
@inline getrefshape(::Interpolation{shape}) where {shape} = shape

"""
    Ferrite.getorder(::Interpolation)

Return order of the interpolation.
"""
@inline getorder(::Interpolation{shape, order}) where {shape, order} = order


#####################
# Utility functions #
#####################

"""
    Ferrite.getnbasefunctions(ip::Interpolation)

Return the number of base functions for the interpolation `ip`.
"""
getnbasefunctions(::Interpolation)

# The following functions are used to distribute the dofs. Definitions:
#   vertexdof: dof on a "corner" of the reference shape
#   facedof: dof in the dim-1 dimension (line in 2D, surface in 3D)
#   edgedof: dof on a line between 2 vertices (i.e. "corners") (3D only)
#   celldof: dof that is local to the element

"""
    reference_shape_values!(values::AbstractArray{T}, ip::Interpolation, ξ::Vec)

Evaluate all shape functions of `ip` at once at the reference point `ξ` and store them in
`values`.
"""
@propagate_inbounds function reference_shape_values!(values::AT, ip::IP, ξ::Vec) where {IP <: Interpolation, AT <: AbstractArray}
    @boundscheck checkbounds(values, 1:getnbasefunctions(ip))
    @inbounds for i in 1:getnbasefunctions(ip)
        values[i] = reference_shape_value(ip, ξ, i)
    end
    return
end

"""
    reference_shape_gradients!(gradients::AbstractArray, ip::Interpolation, ξ::Vec)

Evaluate all shape function gradients of `ip` at once at the reference point `ξ` and store
them in `gradients`.
"""
function reference_shape_gradients!(gradients::AT, ip::IP, ξ::Vec) where {IP <: Interpolation, AT <: AbstractArray}
    @boundscheck checkbounds(gradients, 1:getnbasefunctions(ip))
    @inbounds for i in 1:getnbasefunctions(ip)
        gradients[i] = reference_shape_gradient(ip, ξ, i)
    end
    return
end

"""
    reference_shape_gradients_and_values!(gradients::AbstractArray, values::AbstractArray, ip::Interpolation, ξ::Vec)

Evaluate all shape function gradients and values of `ip` at once at the reference point `ξ`
and store them in `values`.
"""
function reference_shape_gradients_and_values!(gradients::GAT, values::SAT, ip::IP, ξ::Vec) where {IP <: Interpolation, SAT <: AbstractArray, GAT <: AbstractArray}
    @boundscheck checkbounds(gradients, 1:getnbasefunctions(ip))
    @boundscheck checkbounds(values, 1:getnbasefunctions(ip))
    @inbounds for i in 1:getnbasefunctions(ip)
        gradients[i], values[i] = reference_shape_gradient_and_value(ip, ξ, i)
    end
    return
end

"""
    reference_shape_hessians_gradients_and_values!(hessians::AbstractVector, gradients::AbstractVector, values::AbstractVector, ip::Interpolation, ξ::Vec)

Evaluate all shape function hessians, gradients and values of `ip` at once at the reference point `ξ`
and store them in `hessians`, `gradients`, and `values`.
"""
@propagate_inbounds function reference_shape_hessians_gradients_and_values!(hessians::AbstractVector, gradients::AbstractVector, values::AbstractVector, ip::Interpolation, ξ::Vec)
    @boundscheck checkbounds(hessians, 1:getnbasefunctions(ip))
    @boundscheck checkbounds(gradients, 1:getnbasefunctions(ip))
    @boundscheck checkbounds(values, 1:getnbasefunctions(ip))
    @inbounds for i in 1:getnbasefunctions(ip)
        hessians[i], gradients[i], values[i] = reference_shape_hessian_gradient_and_value(ip, ξ, i)
    end
    return
end


"""
    reference_shape_value(ip::Interpolation, ξ::Vec, i::Int)

Evaluate the value of the `i`th shape function of the interpolation `ip`
at a point `ξ` on the reference element. The index `i` must
match the index in [`vertices(::Interpolation)`](@ref), [`faces(::Interpolation)`](@ref) and
[`edges(::Interpolation)`](@ref).

For nodal interpolations the indices also must match the
indices of [`reference_coordinates(::Interpolation)`](@ref).
"""
reference_shape_value(ip::Interpolation, ξ::Vec, i::Int)

"""
    reference_shape_gradient(ip::Interpolation, ξ::Vec, i::Int)

Evaluate the gradient of the `i`th shape function of the interpolation `ip` in
reference coordinate `ξ`.
"""
function reference_shape_gradient(ip::Interpolation, ξ::Vec, i::Int)
    return Tensors.gradient(x -> reference_shape_value(ip, x, i), ξ)
end

"""
    reference_shape_gradient_and_value(ip::Interpolation, ξ::Vec, i::Int)

Optimized version combining the evaluation [`Ferrite.reference_shape_value(::Interpolation)`](@ref)
and [`Ferrite.reference_shape_gradient(::Interpolation)`](@ref).
"""
function reference_shape_gradient_and_value(ip::Interpolation, ξ::Vec, i::Int)
    return gradient(x -> reference_shape_value(ip, x, i), ξ, :all)
end

"""
    reference_shape_hessian_gradient_and_value(ip::Interpolation, ξ::Vec, i::Int)

Optimized version combining the evaluation [`Ferrite.reference_shape_value(::Interpolation)`](@ref),
[`Ferrite.reference_shape_gradient(::Interpolation)`](@ref), and the gradient of the latter.
"""
function reference_shape_hessian_gradient_and_value(ip::Interpolation, ξ::Vec, i::Int)
    return hessian(x -> reference_shape_value(ip, x, i), ξ, :all)
end


"""
    reference_coordinates(ip::Interpolation)

Returns a vector of coordinates with length [`getnbasefunctions(::Interpolation)`](@ref)
and indices corresponding to the indices of a dof in [`vertices`](@ref), [`faces`](@ref) and
[`edges`](@ref).

    Only required for nodal interpolations.

    TODO: Separate nodal and non-nodal interpolations.
"""
reference_coordinates(::Interpolation)

"""
    vertexdof_indices(ip::Interpolation)

A tuple containing tuples of local dof indices for the respective vertex in local
enumeration on a cell defined by [`vertices(::Cell)`](@ref). The vertex enumeration must
match the vertex enumeration of the corresponding geometrical cell.

!!! note
    The dofs appearing in the tuple must be continuous and increasing! The first dof must be
    the 1, as vertex dofs are enumerated first.
"""
vertexdof_indices(ip::Interpolation) = ntuple(_ -> (), nvertices(ip))

"""
    dirichlet_vertexdof_indices(ip::Interpolation)

A tuple containing tuples of local dof indices for the respective vertex in local
enumeration on a cell defined by [`vertices(::Cell)`](@ref). The vertex enumeration must
match the vertex enumeration of the corresponding geometrical cell.
Used internally in [`ConstraintHandler`](@ref) and defaults to [`vertexdof_indices(ip::Interpolation)`](@ref) for continuous interpolation.

!!! note
    The dofs appearing in the tuple must be continuous and increasing! The first dof must be
    the 1, as vertex dofs are enumerated first.
"""
dirichlet_vertexdof_indices(ip::Interpolation) = vertexdof_indices(ip)

"""
    edgedof_indices(ip::Interpolation)

A tuple containing tuples of local dof indices for the respective edge in local enumeration
on a cell defined by [`edges(::Cell)`](@ref). The edge enumeration must match the edge
enumeration of the corresponding geometrical cell.

The dofs are guaranteed to be aligned with the local ordering of the entities on the oriented edge.
Here the first entries are the vertex dofs, followed by the edge interior dofs.
"""
edgedof_indices(::Interpolation)

"""
    dirichlet_edgedof_indices(ip::Interpolation)

A tuple containing tuples of local dof indices for the respective edge in local enumeration
on a cell defined by [`edges(::Cell)`](@ref). The edge enumeration must match the edge
enumeration of the corresponding geometrical cell.
Used internally in [`ConstraintHandler`](@ref) and defaults to [`edgedof_indices(ip::Interpolation)`](@ref) for continuous interpolation.

The dofs are guaranteed to be aligned with the local ordering of the entities on the oriented edge.
Here the first entries are the vertex dofs, followed by the edge interior dofs.
"""
dirichlet_edgedof_indices(ip::Interpolation) = edgedof_indices(ip)

"""
    edgedof_interior_indices(ip::Interpolation)

A tuple containing tuples of the local dof indices on the interior of the respective edge in
local enumeration on a cell defined by [`edges(::Cell)`](@ref). The edge enumeration must
match the edge enumeration of the corresponding geometrical cell. Note that the vertex dofs
are included here.

!!! note
    The dofs appearing in the tuple must be continuous and increasing! The first dof must be
    computed via "last vertex dof index + 1", if edge dofs exist.
"""
edgedof_interior_indices(::Interpolation)

"""
    facedof_indices(ip::Interpolation)

A tuple containing tuples of all local dof indices for the respective face in local
enumeration on a cell defined by [`faces(::Cell)`](@ref). The face enumeration must match
the face enumeration of the corresponding geometrical cell.
"""
facedof_indices(::Interpolation)

"""
    dirichlet_facedof_indices(ip::Interpolation)

A tuple containing tuples of all local dof indices for the respective face in local
enumeration on a cell defined by [`faces(::Cell)`](@ref). The face enumeration must match
the face enumeration of the corresponding geometrical cell.
Used internally in [`ConstraintHandler`](@ref) and defaults to [`facedof_indices(ip::Interpolation)`](@ref) for continuous interpolation.
"""
dirichlet_facedof_indices(ip::Interpolation) = facedof_indices(ip)

"""
    facedof_interior_indices(ip::Interpolation)

A tuple containing tuples of the local dof indices on the interior of the respective face in
local enumeration on a cell defined by [`faces(::Cell)`](@ref). The face enumeration must
match the face enumeration of the corresponding geometrical cell. Note that the vertex and
edge dofs are included here.

!!! note
    The dofs appearing in the tuple must be continuous and increasing! The first dof must be
    the computed via "last edge interior dof index + 1", if face dofs exist.
"""
facedof_interior_indices(::Interpolation)

"""
    volumedof_interior_indices(ip::Interpolation)

Tuple containing the dof indices associated with the interior of a volume.

!!! note
    The dofs appearing in the tuple must be continuous and increasing, volumedofs are
    enumerated last.
"""
volumedof_interior_indices(::Interpolation) = ()

# Some helpers to skip boilerplate
edgedof_indices(ip::Interpolation) = ntuple(_ -> (), nedges(ip))
edgedof_interior_indices(ip::Interpolation) = ntuple(_ -> (), nedges(ip))
facedof_indices(ip::Interpolation) = ntuple(_ -> (), nfaces(ip))
facedof_interior_indices(ip::Interpolation) = ntuple(_ -> (), nfaces(ip))

"""
    boundarydof_indices(::Type{<:BoundaryIndex})

Helper function to generically dispatch on the correct dof sets of a boundary entity.
"""
boundarydof_indices(::Type{<:BoundaryIndex})

boundarydof_indices(::Type{FaceIndex}) = facedof_indices
boundarydof_indices(::Type{EdgeIndex}) = edgedof_indices
boundarydof_indices(::Type{VertexIndex}) = vertexdof_indices

facetdof_indices(ip::InterpolationByDim{3}) = facedof_indices(ip)
facetdof_indices(ip::InterpolationByDim{2}) = edgedof_indices(ip)
facetdof_indices(ip::InterpolationByDim{1}) = vertexdof_indices(ip)
facetdof_interior_indices(ip::InterpolationByDim{3}) = facedof_interior_indices(ip)
facetdof_interior_indices(ip::InterpolationByDim{2}) = edgedof_interior_indices(ip)
facetdof_interior_indices(ip::InterpolationByDim{1}) = ntuple(_ -> (), nvertices(ip))
dirichlet_facetdof_indices(ip::InterpolationByDim{3}) = dirichlet_facedof_indices(ip)
dirichlet_facetdof_indices(ip::InterpolationByDim{2}) = dirichlet_edgedof_indices(ip)
dirichlet_facetdof_indices(ip::InterpolationByDim{1}) = dirichlet_vertexdof_indices(ip)

nfacets(ip::InterpolationByDim{3}) = nfaces(ip)
nfacets(ip::InterpolationByDim{2}) = nedges(ip)
nfacets(ip::InterpolationByDim{1}) = nvertices(ip)

"""
    dirichlet_boundarydof_indices(::Type{<:BoundaryIndex})

Helper function to generically dispatch on the correct dof sets of a boundary entity.
Used internally in [`ConstraintHandler`](@ref) and defaults to [`boundarydof_indices(ip::Interpolation)`](@ref) for continuous interpolation.
"""
dirichlet_boundarydof_indices(::Type{<:BoundaryIndex})

dirichlet_boundarydof_indices(::Type{FaceIndex}) = dirichlet_facedof_indices
dirichlet_boundarydof_indices(::Type{EdgeIndex}) = dirichlet_edgedof_indices
dirichlet_boundarydof_indices(::Type{VertexIndex}) = dirichlet_vertexdof_indices
dirichlet_boundarydof_indices(::Type{FacetIndex}) = dirichlet_facetdof_indices


get_edge_direction(cell, edgenr) = get_edge_direction(edges(cell)[edgenr])
get_face_direction(cell, facenr) = get_face_direction(faces(cell)[facenr])

#########################
# DiscontinuousLagrange #
#########################
# TODO generalize to arbitrary basis positionings.
"""
Piecewise discontinuous Lagrange basis via Gauss-Lobatto points.
"""
struct DiscontinuousLagrange{shape, order} <: ScalarInterpolation{shape, order}
    function DiscontinuousLagrange{shape, order}() where {shape <: AbstractRefShape, order}
        return new{shape, order}()
    end
end
conformity(::DiscontinuousLagrange) = L2Conformity()

adjust_dofs_during_distribution(::DiscontinuousLagrange) = false

getlowerorder(::DiscontinuousLagrange{shape, order}) where {shape, order} = DiscontinuousLagrange{shape, order - 1}()

getnbasefunctions(::DiscontinuousLagrange{shape, order}) where {shape, order} = getnbasefunctions(Lagrange{shape, order}())
getnbasefunctions(::DiscontinuousLagrange{shape, 0}) where {shape} = 1

# This just moves all dofs into the interior of the element.
volumedof_interior_indices(ip::DiscontinuousLagrange) = ntuple(i -> i, getnbasefunctions(ip))

# Mirror the Lagrange element for now to avoid repeating.
dirichlet_facedof_indices(ip::DiscontinuousLagrange{shape, order}) where {shape, order} = dirichlet_facedof_indices(Lagrange{shape, order}())
dirichlet_edgedof_indices(ip::DiscontinuousLagrange{shape, order}) where {shape, order} = dirichlet_edgedof_indices(Lagrange{shape, order}())
dirichlet_vertexdof_indices(ip::DiscontinuousLagrange{shape, order}) where {shape, order} = dirichlet_vertexdof_indices(Lagrange{shape, order}())

# Mirror the Lagrange element for now.
function reference_coordinates(ip::DiscontinuousLagrange{shape, order}) where {shape, order}
    return reference_coordinates(Lagrange{shape, order}())
end
function reference_shape_value(::DiscontinuousLagrange{shape, order}, ξ::Vec{dim}, i::Int) where {dim, shape <: AbstractRefShape{dim}, order}
    return reference_shape_value(Lagrange{shape, order}(), ξ, i)
end

# Excepting the L0 element.
function reference_coordinates(ip::DiscontinuousLagrange{RefHypercube{dim}, 0}) where {dim}
    return [Vec{dim, Float64}(ntuple(x -> 0.0, dim))]
end

function reference_coordinates(ip::DiscontinuousLagrange{RefTriangle, 0})
    return [Vec{2, Float64}((1 / 3, 1 / 3))]
end

function reference_coordinates(ip::DiscontinuousLagrange{RefTetrahedron, 0})
    return [Vec{3, Float64}((1 / 4, 1 / 4, 1 / 4))]
end

function reference_shape_value(ip::DiscontinuousLagrange{shape, 0}, ::Vec{dim, T}, i::Int) where {dim, shape <: AbstractRefShape{dim}, T}
    i == 1 && return one(T)
    throw(ArgumentError("no shape function $i for interpolation $ip"))
end

############
# Lagrange #
############
"""
    Lagrange{refshape, order} <: ScalarInterpolation

Standard continuous Lagrange polynomials with equidistant node placement.
"""
struct Lagrange{shape, order} <: ScalarInterpolation{shape, order}
    function Lagrange{shape, order}() where {shape <: AbstractRefShape, order}
        return new{shape, order}()
    end
end
conformity(::Lagrange) = H1Conformity()

adjust_dofs_during_distribution(::Lagrange) = true
adjust_dofs_during_distribution(::Lagrange{<:Any, 2}) = false
adjust_dofs_during_distribution(::Lagrange{<:Any, 1}) = false

# Vertices for all Lagrange interpolations are the same
vertexdof_indices(::Lagrange{RefLine}) = ((1,), (2,))
vertexdof_indices(::Lagrange{RefQuadrilateral}) = ((1,), (2,), (3,), (4,))
vertexdof_indices(::Lagrange{RefHexahedron}) = ((1,), (2,), (3,), (4,), (5,), (6,), (7,), (8,))
vertexdof_indices(::Lagrange{RefTriangle}) = ((1,), (2,), (3,))
vertexdof_indices(::Lagrange{RefTetrahedron}) = ((1,), (2,), (3,), (4,))
vertexdof_indices(::Lagrange{RefPrism}) = ((1,), (2,), (3,), (4,), (5,), (6,))
vertexdof_indices(::Lagrange{RefPyramid}) = ((1,), (2,), (3,), (4,), (5,))

getlowerorder(::Lagrange{shape, order}) where {shape, order} = Lagrange{shape, order - 1}()
getlowerorder(::Lagrange{shape, 1}) where {shape} = DiscontinuousLagrange{shape, 0}()

############################
# Lagrange RefLine order 1 #
############################
getnbasefunctions(::Lagrange{RefLine, 1}) = 2

edgedof_indices(::Lagrange{RefLine, 1}) = ((1, 2),)

function reference_coordinates(::Lagrange{RefLine, 1})
    return [
        Vec{1, Float64}((-1.0,)),
        Vec{1, Float64}((1.0,)),
    ]
end

function reference_shape_value(ip::Lagrange{RefLine, 1}, ξ::Vec{1}, i::Int)
    ξ_x = ξ[1]
    i == 1 && return (1 - ξ_x) / 2
    i == 2 && return (1 + ξ_x) / 2
    throw(ArgumentError("no shape function $i for interpolation $ip"))
end

############################
# Lagrange RefLine order 2 #
############################
getnbasefunctions(::Lagrange{RefLine, 2}) = 3

edgedof_indices(::Lagrange{RefLine, 2}) = ((1, 2, 3),)
edgedof_interior_indices(::Lagrange{RefLine, 2}) = ((3,),)

function reference_coordinates(::Lagrange{RefLine, 2})
    return [
        Vec{1, Float64}((-1.0,)),
        Vec{1, Float64}((1.0,)),
        Vec{1, Float64}((0.0,)),
    ]
end

function reference_shape_value(ip::Lagrange{RefLine, 2}, ξ::Vec{1}, i::Int)
    ξ_x = ξ[1]
    i == 1 && return ξ_x * (ξ_x - 1) / 2
    i == 2 && return ξ_x * (ξ_x + 1) / 2
    i == 3 && return 1 - ξ_x^2
    throw(ArgumentError("no shape function $i for interpolation $ip"))
end

#####################################
# Lagrange RefQuadrilateral order 1 #
#####################################
getnbasefunctions(::Lagrange{RefQuadrilateral, 1}) = 4

edgedof_indices(::Lagrange{RefQuadrilateral, 1}) = ((1, 2), (2, 3), (3, 4), (4, 1))
facedof_indices(ip::Lagrange{RefQuadrilateral, 1}) = (ntuple(i -> i, getnbasefunctions(ip)),)

function reference_coordinates(::Lagrange{RefQuadrilateral, 1})
    return [
        Vec{2, Float64}((-1.0, -1.0)),
        Vec{2, Float64}((1.0, -1.0)),
        Vec{2, Float64}((1.0, 1.0)),
        Vec{2, Float64}((-1.0, 1.0)),
    ]
end

function reference_shape_value(ip::Lagrange{RefQuadrilateral, 1}, ξ::Vec{2}, i::Int)
    ξ_x = ξ[1]
    ξ_y = ξ[2]
    i == 1 && return (1 - ξ_x) * (1 - ξ_y) / 4
    i == 2 && return (1 + ξ_x) * (1 - ξ_y) / 4
    i == 3 && return (1 + ξ_x) * (1 + ξ_y) / 4
    i == 4 && return (1 - ξ_x) * (1 + ξ_y) / 4
    throw(ArgumentError("no shape function $i for interpolation $ip"))
end

#####################################
# Lagrange RefQuadrilateral order 2 #
#####################################
getnbasefunctions(::Lagrange{RefQuadrilateral, 2}) = 9

edgedof_indices(::Lagrange{RefQuadrilateral, 2}) = ((1, 2, 5), (2, 3, 6), (3, 4, 7), (4, 1, 8))
edgedof_interior_indices(::Lagrange{RefQuadrilateral, 2}) = ((5,), (6,), (7,), (8,))
facedof_indices(ip::Lagrange{RefQuadrilateral, 2}) = (ntuple(i -> i, getnbasefunctions(ip)),)
facedof_interior_indices(::Lagrange{RefQuadrilateral, 2}) = ((9,),)

function reference_coordinates(::Lagrange{RefQuadrilateral, 2})
    return [
        Vec{2, Float64}((-1.0, -1.0)),
        Vec{2, Float64}((1.0, -1.0)),
        Vec{2, Float64}((1.0, 1.0)),
        Vec{2, Float64}((-1.0, 1.0)),
        Vec{2, Float64}((0.0, -1.0)),
        Vec{2, Float64}((1.0, 0.0)),
        Vec{2, Float64}((0.0, 1.0)),
        Vec{2, Float64}((-1.0, 0.0)),
        Vec{2, Float64}((0.0, 0.0)),
    ]
end

function reference_shape_value(ip::Lagrange{RefQuadrilateral, 2}, ξ::Vec{2}, i::Int)
    ξ_x = ξ[1]
    ξ_y = ξ[2]
    i == 1 && return (ξ_x^2 - ξ_x) * (ξ_y^2 - ξ_y) / 4
    i == 2 && return (ξ_x^2 + ξ_x) * (ξ_y^2 - ξ_y) / 4
    i == 3 && return (ξ_x^2 + ξ_x) * (ξ_y^2 + ξ_y) / 4
    i == 4 && return (ξ_x^2 - ξ_x) * (ξ_y^2 + ξ_y) / 4
    i == 5 && return (1 - ξ_x^2) * (ξ_y^2 - ξ_y) / 2
    i == 6 && return (ξ_x^2 + ξ_x) * (1 - ξ_y^2) / 2
    i == 7 && return (1 - ξ_x^2) * (ξ_y^2 + ξ_y) / 2
    i == 8 && return (ξ_x^2 - ξ_x) * (1 - ξ_y^2) / 2
    i == 9 && return (1 - ξ_x^2) * (1 - ξ_y^2)
    throw(ArgumentError("no shape function $i for interpolation $ip"))
end

#####################################
# Lagrange RefQuadrilateral order 3 #
#####################################
getnbasefunctions(::Lagrange{RefQuadrilateral, 3}) = 16

edgedof_indices(::Lagrange{RefQuadrilateral, 3}) = ((1, 2, 5, 6), (2, 3, 7, 8), (3, 4, 9, 10), (4, 1, 11, 12))
edgedof_interior_indices(::Lagrange{RefQuadrilateral, 3}) = ((5, 6), (7, 8), (9, 10), (11, 12))
facedof_indices(ip::Lagrange{RefQuadrilateral, 3}) = (ntuple(i -> i, getnbasefunctions(ip)),)
facedof_interior_indices(::Lagrange{RefQuadrilateral, 3}) = ((13, 14, 15, 16),)

function reference_coordinates(::Lagrange{RefQuadrilateral, 3})
    return [
        Vec{2, Float64}((-1.0, -1.0)),
        Vec{2, Float64}((1.0, -1.0)),
        Vec{2, Float64}((1.0, 1.0)),
        Vec{2, Float64}((-1.0, 1.0)),
        Vec{2, Float64}((-1 / 3, -1.0)),
        Vec{2, Float64}((1 / 3, -1.0)),
        Vec{2, Float64}((1.0, -1 / 3)),
        Vec{2, Float64}((1.0, 1 / 3)),
        Vec{2, Float64}((1 / 3, 1.0)),
        Vec{2, Float64}((-1 / 3, 1.0)),
        Vec{2, Float64}((-1.0, 1 / 3)),
        Vec{2, Float64}((-1.0, -1 / 3)),
        Vec{2, Float64}((-1 / 3, -1 / 3)),
        Vec{2, Float64}((1 / 3, -1 / 3)),
        Vec{2, Float64}((-1 / 3, 1 / 3)),
        Vec{2, Float64}((1 / 3, 1 / 3)),
    ]
end

function reference_shape_value(ip::Lagrange{RefQuadrilateral, 3}, ξ::Vec{2}, i::Int)
    # See https://defelement.org/elements/examples/quadrilateral-lagrange-equispaced-3.html
    # Transform domain from [-1, 1] × [-1, 1] to [0, 1] × [0, 1]
    ξ_x = (ξ[1] + 1) / 2
    ξ_y = (ξ[2] + 1) / 2
    i == 1 && return (81 * ξ_x^3 * ξ_y^3) / 4 - (81 * ξ_x^3 * ξ_y^2) / 2 + (99 * ξ_x^3 * ξ_y) / 4 - (9 * ξ_x^3) / 2 - (81 * ξ_x^2 * ξ_y^3) / 2 + (81 * ξ_x^2 * ξ_y^2) - (99 * ξ_x^2 * ξ_y) / 2 + (9 * ξ_x^2) + (99 * ξ_x * ξ_y^3) / 4 - (99 * ξ_x * ξ_y^2) / 2 + (121 * ξ_x * ξ_y) / 4 - (11 * ξ_x) / 2 - (9 * ξ_y^3) / 2 + 9 * ξ_y^2 - (11 * ξ_y) / 2 + 1
    i == 2 && return (ξ_x * (- 81 * ξ_x^2 * ξ_y^3 + 162 * ξ_x^2 * ξ_y^2 - 99 * ξ_x^2 * ξ_y + 18 * ξ_x^2 + 81 * ξ_x * ξ_y^3 - 162 * ξ_x * ξ_y^2 + 99 * ξ_x * ξ_y - 18 * ξ_x - 18 * ξ_y^3 + 36 * ξ_y^2 - 22 * ξ_y + 4)) / 4
    i == 4 && return (ξ_y * (- 81 * ξ_x^3 * ξ_y^2 + 81 * ξ_x^3 * ξ_y - 18 * ξ_x^3 + 162 * ξ_x^2 * ξ_y^2 - 162 * ξ_x^2 * ξ_y + 36 * ξ_x^2 - 99 * ξ_x * ξ_y^2 + 99 * ξ_x * ξ_y - 22 * ξ_x + 18 * ξ_y^2 - 18 * ξ_y + 4)) / 4
    i == 3 && return (ξ_x * ξ_y * (81 * ξ_x^2 * ξ_y^2 - 81 * ξ_x^2 * ξ_y + 18 * ξ_x^2 - 81 * ξ_x * ξ_y^2 + 81 * ξ_x * ξ_y - 18 * ξ_x + 18 * ξ_y^2 - 18 * ξ_y + 4)) / 4
    i == 5 && return (9 * ξ_x * (- 27 * ξ_x^2 * ξ_y^3 + 54 * ξ_x^2 * ξ_y^2 - 33 * ξ_x^2 * ξ_y + 6 * ξ_x^2 + 45 * ξ_x * ξ_y^3 - 90 * ξ_x * ξ_y^2 + 55 * ξ_x * ξ_y - 10 * ξ_x - 18 * ξ_y^3 + 36 * ξ_y^2 - 22 * ξ_y + 4)) / 4
    i == 6 && return (9 * ξ_x * (27 * ξ_x^2 * ξ_y^3 - 54 * ξ_x^2 * ξ_y^2 + 33 * ξ_x^2 * ξ_y - 6 * ξ_x^2 - 36 * ξ_x * ξ_y^3 + 72 * ξ_x * ξ_y^2 - 44 * ξ_x * ξ_y + 8 * ξ_x + 9 * ξ_y^3 - 18 * ξ_y^2 + 11 * ξ_y - 2)) / 4
    i == 12 && return (9 * ξ_y * (- 27 * ξ_x^3 * ξ_y^2 + 45 * ξ_x^3 * ξ_y - 18 * ξ_x^3 + 54 * ξ_x^2 * ξ_y^2 - 90 * ξ_x^2 * ξ_y + 36 * ξ_x^2 - 33 * ξ_x * ξ_y^2 + 55 * ξ_x * ξ_y - 22 * ξ_x + 6 * ξ_y^2 - 10 * ξ_y + 4)) / 4
    i == 11 && return (9 * ξ_y * (27 * ξ_x^3 * ξ_y^2 - 36 * ξ_x^3 * ξ_y + 9 * ξ_x^3 - 54 * ξ_x^2 * ξ_y^2 + 72 * ξ_x^2 * ξ_y - 18 * ξ_x^2 + 33 * ξ_x * ξ_y^2 - 44 * ξ_x * ξ_y + 11 * ξ_x - 6 * ξ_y^2 + 8 * ξ_y - 2)) / 4
    i == 7 && return (9 * ξ_x * ξ_y * (27 * ξ_x^2 * ξ_y^2 - 45 * ξ_x^2 * ξ_y + 18 * ξ_x^2 - 27 * ξ_x * ξ_y^2 + 45 * ξ_x * ξ_y - 18 * ξ_x + 6 * ξ_y^2 - 10 * ξ_y + 4)) / 4
    i == 8 && return (9 * ξ_x * ξ_y * (- 27 * ξ_x^2 * ξ_y^2 + 36 * ξ_x^2 * ξ_y - 9 * ξ_x^2 + 27 * ξ_x * ξ_y^2 - 36 * ξ_x * ξ_y + 9 * ξ_x - 6 * ξ_y^2 + 8 * ξ_y - 2)) / 4
    i == 10 && return (9 * ξ_x * ξ_y * (27 * ξ_x^2 * ξ_y^2 - 27 * ξ_x^2 * ξ_y + 6 * ξ_x^2 - 45 * ξ_x * ξ_y^2 + 45 * ξ_x * ξ_y - 10 * ξ_x + 18 * ξ_y^2 - 18 * ξ_y + 4)) / 4
    i == 9 && return (9 * ξ_x * ξ_y * (- 27 * ξ_x^2 * ξ_y^2 + 27 * ξ_x^2 * ξ_y - 6 * ξ_x^2 + 36 * ξ_x * ξ_y^2 - 36 * ξ_x * ξ_y + 8 * ξ_x - 9 * ξ_y^2 + 9 * ξ_y - 2)) / 4
    i == 13 && return (81 * ξ_x * ξ_y * (9 * ξ_x^2 * ξ_y^2 - 15 * ξ_x^2 * ξ_y + 6 * ξ_x^2 - 15 * ξ_x * ξ_y^2 + 25 * ξ_x * ξ_y - 10 * ξ_x + 6 * ξ_y^2 - 10 * ξ_y + 4)) / 4
    i == 14 && return (81 * ξ_x * ξ_y * (- 9 * ξ_x^2 * ξ_y^2 + 15 * ξ_x^2 * ξ_y - 6 * ξ_x^2 + 12 * ξ_x * ξ_y^2 - 20 * ξ_x * ξ_y + 8 * ξ_x - 3 * ξ_y^2 + 5 * ξ_y - 2)) / 4
    i == 15 && return (81 * ξ_x * ξ_y * (- 9 * ξ_x^2 * ξ_y^2 + 12 * ξ_x^2 * ξ_y - 3 * ξ_x^2 + 15 * ξ_x * ξ_y^2 - 20 * ξ_x * ξ_y + 5 * ξ_x - 6 * ξ_y^2 + 8 * ξ_y - 2)) / 4
    i == 16 && return (81 * ξ_x * ξ_y * (9 * ξ_x^2 * ξ_y^2 - 12 * ξ_x^2 * ξ_y + 3 * ξ_x^2 - 12 * ξ_x * ξ_y^2 + 16 * ξ_x * ξ_y - 4 * ξ_x + 3 * ξ_y^2 - 4 * ξ_y + 1)) / 4
    throw(ArgumentError("no shape function $i for interpolation $ip"))
end

################################
# Lagrange RefTriangle order 1 #
################################
getnbasefunctions(::Lagrange{RefTriangle, 1}) = 3

edgedof_indices(::Lagrange{RefTriangle, 1}) = ((1, 2), (2, 3), (3, 1))
facedof_indices(ip::Lagrange{RefTriangle, 1}) = (ntuple(i -> i, getnbasefunctions(ip)),)

function reference_coordinates(::Lagrange{RefTriangle, 1})
    return [
        Vec{2, Float64}((1.0, 0.0)),
        Vec{2, Float64}((0.0, 1.0)),
        Vec{2, Float64}((0.0, 0.0)),
    ]
end

function reference_shape_value(ip::Lagrange{RefTriangle, 1}, ξ::Vec{2}, i::Int)
    ξ_x = ξ[1]
    ξ_y = ξ[2]
    i == 1 && return ξ_x
    i == 2 && return ξ_y
    i == 3 && return 1 - ξ_x - ξ_y
    throw(ArgumentError("no shape function $i for interpolation $ip"))
end

################################
# Lagrange RefTriangle order 2 #
################################
getnbasefunctions(::Lagrange{RefTriangle, 2}) = 6

edgedof_indices(::Lagrange{RefTriangle, 2}) = ((1, 2, 4), (2, 3, 5), (3, 1, 6))
edgedof_interior_indices(::Lagrange{RefTriangle, 2}) = ((4,), (5,), (6,))
facedof_indices(ip::Lagrange{RefTriangle, 2}) = (ntuple(i -> i, getnbasefunctions(ip)),)

function reference_coordinates(::Lagrange{RefTriangle, 2})
    return [
        Vec{2, Float64}((1.0, 0.0)),
        Vec{2, Float64}((0.0, 1.0)),
        Vec{2, Float64}((0.0, 0.0)),
        Vec{2, Float64}((0.5, 0.5)),
        Vec{2, Float64}((0.0, 0.5)),
        Vec{2, Float64}((0.5, 0.0)),
    ]
end

function reference_shape_value(ip::Lagrange{RefTriangle, 2}, ξ::Vec{2}, i::Int)
    ξ_x = ξ[1]
    ξ_y = ξ[2]
    γ = 1 - ξ_x - ξ_y
    i == 1 && return ξ_x * (2ξ_x - 1)
    i == 2 && return ξ_y * (2ξ_y - 1)
    i == 3 && return γ * (2γ - 1)
    i == 4 && return 4ξ_x * ξ_y
    i == 5 && return 4ξ_y * γ
    i == 6 && return 4ξ_x * γ
    throw(ArgumentError("no shape function $i for interpolation $ip"))
end

######################################
# Lagrange RefTriangle order 3, 4, 5 #
######################################
# see https://getfem.readthedocs.io/en/latest/userdoc/appendixA.html

const Lagrange2Tri345 = Union{
    Lagrange{RefTriangle, 3},
    Lagrange{RefTriangle, 4},
    Lagrange{RefTriangle, 5},
}

function getnbasefunctions(ip::Lagrange2Tri345)
    order = getorder(ip)
    return (order + 1) * (order + 2) ÷ 2
end

# Permutation to switch numbering to Ferrite ordering
const permdof2DLagrange2Tri345 = Dict{Int, Vector{Int}}(
    1 => [1, 2, 3],
    2 => [3, 6, 1, 5, 4, 2],
    3 => [4, 10, 1, 7, 9, 8, 5, 2, 3, 6],
    4 => [5, 15, 1, 9, 12, 14, 13, 10, 6, 2, 3, 4, 7, 8, 11],
    5 => [6, 21, 1, 11, 15, 18, 20, 19, 16, 12, 7, 2, 3, 4, 5, 8, 9, 10, 13, 14, 17],
)

function edgedof_indices(ip::Lagrange2Tri345)
    order = getorder(ip)
    order == 1 && return ((1, 2), (2, 3), (3, 1))
    order == 2 && return ((1, 2, 4), (2, 3, 5), (3, 1, 6))
    order == 3 && return ((1, 2, 4, 5), (2, 3, 6, 7), (3, 1, 8, 9))
    order == 4 && return ((1, 2, 4, 5, 6), (2, 3, 7, 8, 9), (3, 1, 10, 11, 12))
    order == 5 && return ((1, 2, 4, 5, 6, 7), (2, 3, 8, 9, 10, 11), (3, 1, 12, 13, 14, 15))

    throw(ArgumentError("Unsupported order $order for Lagrange on triangles."))
end

function edgedof_interior_indices(ip::Lagrange2Tri345)
    order = getorder(ip)
    order == 1 && return ((), (), ())
    order == 2 && return ((4,), (5,), (6,))
    order == 3 && return ((4, 5), (6, 7), (8, 9))
    order == 4 && return ((4, 5, 6), (7, 8, 9), (10, 11, 12))
    order == 5 && return ((4, 5, 6, 7), (8, 9, 10, 11), (12, 13, 14, 15))
    throw(ArgumentError("Unsupported order $order for Lagrange on triangles."))
end

facedof_indices(ip::Lagrange2Tri345) = (ntuple(i -> i, getnbasefunctions(ip)),)

function facedof_interior_indices(ip::Lagrange2Tri345)
    order = getorder(ip)
    ncellintdofs = (order + 1) * (order + 2) ÷ 2 - 3 * order
    totaldofs = getnbasefunctions(ip)
    return (ntuple(i -> totaldofs - ncellintdofs + i, ncellintdofs),)
end

function reference_coordinates(ip::Lagrange2Tri345)
    order = getorder(ip)
    coordpts = Vector{Vec{2, Float64}}()
    for k in 0:order
        for l in 0:(order - k)
            push!(coordpts, Vec{2, Float64}((l / order, k / order)))
        end
    end
    return permute!(coordpts, permdof2DLagrange2Tri345[order])
end

function reference_shape_value(ip::Lagrange2Tri345, ξ::Vec{2}, i::Int)
    if !(0 < i <= getnbasefunctions(ip))
        throw(ArgumentError("no shape function $i for interpolation $ip"))
    end
    order = getorder(ip)
    i = permdof2DLagrange2Tri345[order][i]
    ξ_x = ξ[1]
    ξ_y = ξ[2]
    i1, i2, i3 = _numlin_basis2D(i, order)
    val = one(ξ_y)
    i1 ≥ 1 && (val *= prod((order - order * (ξ_x + ξ_y) - j) / (j + 1) for j in 0:(i1 - 1)))
    i2 ≥ 1 && (val *= prod((order * ξ_x - j) / (j + 1) for j in 0:(i2 - 1)))
    i3 ≥ 1 && (val *= prod((order * ξ_y - j) / (j + 1) for j in 0:(i3 - 1)))
    return val
end

function _numlin_basis2D(i, order)
    c, j1, j2, j3 = 0, 0, 0, 0
    for k in 0:order
        if i <= c + (order + 1 - k)
            j2 = i - c - 1
            break
        else
            j3 += 1
            c += order + 1 - k
        end
    end
    j1 = order - j2 - j3
    return j1, j2, j3
end

###################################
# Lagrange RefTetrahedron order 1 #
###################################
getnbasefunctions(::Lagrange{RefTetrahedron, 1}) = 4

facedof_indices(::Lagrange{RefTetrahedron, 1}) = ((1, 3, 2), (1, 2, 4), (2, 3, 4), (1, 4, 3))
edgedof_indices(::Lagrange{RefTetrahedron, 1}) = ((1, 2), (2, 3), (3, 1), (1, 4), (2, 4), (3, 4))

function reference_coordinates(::Lagrange{RefTetrahedron, 1})
    return [
        Vec{3, Float64}((0.0, 0.0, 0.0)),
        Vec{3, Float64}((1.0, 0.0, 0.0)),
        Vec{3, Float64}((0.0, 1.0, 0.0)),
        Vec{3, Float64}((0.0, 0.0, 1.0)),
    ]
end

function reference_shape_value(ip::Lagrange{RefTetrahedron, 1}, ξ::Vec{3}, i::Int)
    ξ_x = ξ[1]
    ξ_y = ξ[2]
    ξ_z = ξ[3]
    i == 1 && return 1 - ξ_x - ξ_y - ξ_z
    i == 2 && return ξ_x
    i == 3 && return ξ_y
    i == 4 && return ξ_z
    throw(ArgumentError("no shape function $i for interpolation $ip"))
end

###################################
# Lagrange RefTetrahedron order 2 #
###################################
getnbasefunctions(::Lagrange{RefTetrahedron, 2}) = 10

facedof_indices(::Lagrange{RefTetrahedron, 2}) = ((1, 3, 2, 7, 6, 5), (1, 2, 4, 5, 9, 8), (2, 3, 4, 6, 10, 9), (1, 4, 3, 8, 10, 7))
edgedof_indices(::Lagrange{RefTetrahedron, 2}) = ((1, 2, 5), (2, 3, 6), (3, 1, 7), (1, 4, 8), (2, 4, 9), (3, 4, 10))
edgedof_interior_indices(::Lagrange{RefTetrahedron, 2}) = ((5,), (6,), (7,), (8,), (9,), (10,))

function reference_coordinates(::Lagrange{RefTetrahedron, 2})
    return [
        Vec{3, Float64}((0.0, 0.0, 0.0)),
        Vec{3, Float64}((1.0, 0.0, 0.0)),
        Vec{3, Float64}((0.0, 1.0, 0.0)),
        Vec{3, Float64}((0.0, 0.0, 1.0)),
        Vec{3, Float64}((0.5, 0.0, 0.0)),
        Vec{3, Float64}((0.5, 0.5, 0.0)),
        Vec{3, Float64}((0.0, 0.5, 0.0)),
        Vec{3, Float64}((0.0, 0.0, 0.5)),
        Vec{3, Float64}((0.5, 0.0, 0.5)),
        Vec{3, Float64}((0.0, 0.5, 0.5)),
    ]
end

# http://www.colorado.edu/engineering/CAS/courses.d/AFEM.d/AFEM.Ch09.d/AFEM.Ch09.pdf
# http://www.colorado.edu/engineering/CAS/courses.d/AFEM.d/AFEM.Ch10.d/AFEM.Ch10.pdf
function reference_shape_value(ip::Lagrange{RefTetrahedron, 2}, ξ::Vec{3}, i::Int)
    ξ_x = ξ[1]
    ξ_y = ξ[2]
    ξ_z = ξ[3]
    i == 1  && return (-2 * ξ_x - 2 * ξ_y - 2 * ξ_z + 1) * (-ξ_x - ξ_y - ξ_z + 1)
    i == 2  && return ξ_x * (2 * ξ_x - 1)
    i == 3  && return ξ_y * (2 * ξ_y - 1)
    i == 4  && return ξ_z * (2 * ξ_z - 1)
    i == 5  && return ξ_x * (-4 * ξ_x - 4 * ξ_y - 4 * ξ_z + 4)
    i == 6  && return 4 * ξ_x * ξ_y
    i == 7  && return 4 * ξ_y * (-ξ_x - ξ_y - ξ_z + 1)
    i == 8  && return ξ_z * (-4 * ξ_x - 4 * ξ_y - 4 * ξ_z + 4)
    i == 9  && return 4 * ξ_x * ξ_z
    i == 10 && return 4 * ξ_y * ξ_z
    throw(ArgumentError("no shape function $i for interpolation $ip"))
end

##################################
# Lagrange RefHexahedron order 1 #
##################################
getnbasefunctions(::Lagrange{RefHexahedron, 1}) = 8

facedof_indices(::Lagrange{RefHexahedron, 1}) = ((1, 4, 3, 2), (1, 2, 6, 5), (2, 3, 7, 6), (3, 4, 8, 7), (1, 5, 8, 4), (5, 6, 7, 8))
edgedof_indices(::Lagrange{RefHexahedron, 1}) = ((1, 2), (2, 3), (3, 4), (4, 1), (5, 6), (6, 7), (7, 8), (8, 5), (1, 5), (2, 6), (3, 7), (4, 8))

function reference_coordinates(::Lagrange{RefHexahedron, 1})
    return [
        Vec{3, Float64}((-1.0, -1.0, -1.0)),
        Vec{3, Float64}((1.0, -1.0, -1.0)),
        Vec{3, Float64}((1.0, 1.0, -1.0)),
        Vec{3, Float64}((-1.0, 1.0, -1.0)),
        Vec{3, Float64}((-1.0, -1.0, 1.0)),
        Vec{3, Float64}((1.0, -1.0, 1.0)),
        Vec{3, Float64}((1.0, 1.0, 1.0)),
        Vec{3, Float64}((-1.0, 1.0, 1.0)),
    ]
end

function reference_shape_value(ip::Lagrange{RefHexahedron, 1}, ξ::Vec{3}, i::Int)
    ξ_x = ξ[1]
    ξ_y = ξ[2]
    ξ_z = ξ[3]
    i == 1 && return (1 - ξ_x) * (1 - ξ_y) * (1 - ξ_z) / 8
    i == 2 && return (1 + ξ_x) * (1 - ξ_y) * (1 - ξ_z) / 8
    i == 3 && return (1 + ξ_x) * (1 + ξ_y) * (1 - ξ_z) / 8
    i == 4 && return (1 - ξ_x) * (1 + ξ_y) * (1 - ξ_z) / 8
    i == 5 && return (1 - ξ_x) * (1 - ξ_y) * (1 + ξ_z) / 8
    i == 6 && return (1 + ξ_x) * (1 - ξ_y) * (1 + ξ_z) / 8
    i == 7 && return (1 + ξ_x) * (1 + ξ_y) * (1 + ξ_z) / 8
    i == 8 && return (1 - ξ_x) * (1 + ξ_y) * (1 + ξ_z) / 8
    throw(ArgumentError("no shape function $i for interpolation $ip"))
end


##################################
# Lagrange RefHexahedron order 2 #
##################################
# Based on vtkTriQuadraticHexahedron (see https://kitware.github.io/vtk-examples/site/Cxx/GeometricObjects/IsoparametricCellsDemo/)
getnbasefunctions(::Lagrange{RefHexahedron, 2}) = 27

facedof_indices(::Lagrange{RefHexahedron, 2}) = (
    (1, 4, 3, 2, 12, 11, 10, 9, 21),
    (1, 2, 6, 5, 9, 18, 13, 17, 22),
    (2, 3, 7, 6, 10, 19, 14, 18, 23),
    (3, 4, 8, 7, 11, 20, 15, 19, 24),
    (1, 5, 8, 4, 17, 16, 20, 12, 25),
    (5, 6, 7, 8, 13, 14, 15, 16, 26),
)
facedof_interior_indices(::Lagrange{RefHexahedron, 2}) = (
    (21,), (22,), (23,), (24,), (25,), (26,),
)

edgedof_indices(::Lagrange{RefHexahedron, 2}) = (
    (1, 2, 9),
    (2, 3, 10),
    (3, 4, 11),
    (4, 1, 12),
    (5, 6, 13),
    (6, 7, 14),
    (7, 8, 15),
    (8, 5, 16),
    (1, 5, 17),
    (2, 6, 18),
    (3, 7, 19),
    (4, 8, 20),
)
edgedof_interior_indices(::Lagrange{RefHexahedron, 2}) = (
    (9,), (10,), (11,), (12,), (13,), (14,), (15,), (16,), (17,), (18,), (19,), (20,),
)

volumedof_interior_indices(::Lagrange{RefHexahedron, 2}) = (27,)

function reference_coordinates(::Lagrange{RefHexahedron, 2})
    return [
        # vertex
        Vec{3, Float64}((-1.0, -1.0, -1.0)), # 1
        Vec{3, Float64}((1.0, -1.0, -1.0)), # 2
        Vec{3, Float64}((1.0, 1.0, -1.0)), # 3
        Vec{3, Float64}((-1.0, 1.0, -1.0)), # 4
        Vec{3, Float64}((-1.0, -1.0, 1.0)), # 5
        Vec{3, Float64}((1.0, -1.0, 1.0)), # 6
        Vec{3, Float64}((1.0, 1.0, 1.0)), # 7
        Vec{3, Float64}((-1.0, 1.0, 1.0)), # 8
        # edge
        Vec{3, Float64}((0.0, -1.0, -1.0)), # 9
        Vec{3, Float64}((1.0, 0.0, -1.0)),
        Vec{3, Float64}((0.0, 1.0, -1.0)),
        Vec{3, Float64}((-1.0, 0.0, -1.0)),
        Vec{3, Float64}((0.0, -1.0, 1.0)),
        Vec{3, Float64}((1.0, 0.0, 1.0)),
        Vec{3, Float64}((0.0, 1.0, 1.0)),
        Vec{3, Float64}((-1.0, 0.0, 1.0)),
        Vec{3, Float64}((-1.0, -1.0, 0.0)),
        Vec{3, Float64}((1.0, -1.0, 0.0)),
        Vec{3, Float64}((1.0, 1.0, 0.0)),
        Vec{3, Float64}((-1.0, 1.0, 0.0)), # 20
        Vec{3, Float64}((0.0, 0.0, -1.0)),
        Vec{3, Float64}((0.0, -1.0, 0.0)),
        Vec{3, Float64}((1.0, 0.0, 0.0)),
        Vec{3, Float64}((0.0, 1.0, 0.0)),
        Vec{3, Float64}((-1.0, 0.0, 0.0)),
        Vec{3, Float64}((0.0, 0.0, 1.0)), # 26
        # interior
        Vec{3, Float64}((0.0, 0.0, 0.0)), # 27
    ]
end

function reference_shape_value(ip::Lagrange{RefHexahedron, 2}, ξ::Vec{3, T}, i::Int) where {T}
    # Some local helpers.
    @inline φ₁(x::T) = -x * (1 - x) / 2
    @inline φ₂(x::T) = (1 + x) * (1 - x)
    @inline φ₃(x::T) = x * (1 + x) / 2
    (ξ_x, ξ_y, ξ_z) = ξ
    # vertices
    i == 1 && return φ₁(ξ_x) * φ₁(ξ_y) * φ₁(ξ_z)
    i == 2 && return φ₃(ξ_x) * φ₁(ξ_y) * φ₁(ξ_z)
    i == 3 && return φ₃(ξ_x) * φ₃(ξ_y) * φ₁(ξ_z)
    i == 4 && return φ₁(ξ_x) * φ₃(ξ_y) * φ₁(ξ_z)
    i == 5 && return φ₁(ξ_x) * φ₁(ξ_y) * φ₃(ξ_z)
    i == 6 && return φ₃(ξ_x) * φ₁(ξ_y) * φ₃(ξ_z)
    i == 7 && return φ₃(ξ_x) * φ₃(ξ_y) * φ₃(ξ_z)
    i == 8 && return φ₁(ξ_x) * φ₃(ξ_y) * φ₃(ξ_z)
    # edges
    i == 9 && return φ₂(ξ_x) * φ₁(ξ_y) * φ₁(ξ_z)
    i == 10 && return φ₃(ξ_x) * φ₂(ξ_y) * φ₁(ξ_z)
    i == 11 && return φ₂(ξ_x) * φ₃(ξ_y) * φ₁(ξ_z)
    i == 12 && return φ₁(ξ_x) * φ₂(ξ_y) * φ₁(ξ_z)
    i == 13 && return φ₂(ξ_x) * φ₁(ξ_y) * φ₃(ξ_z)
    i == 14 && return φ₃(ξ_x) * φ₂(ξ_y) * φ₃(ξ_z)
    i == 15 && return φ₂(ξ_x) * φ₃(ξ_y) * φ₃(ξ_z)
    i == 16 && return φ₁(ξ_x) * φ₂(ξ_y) * φ₃(ξ_z)
    i == 17 && return φ₁(ξ_x) * φ₁(ξ_y) * φ₂(ξ_z)
    i == 18 && return φ₃(ξ_x) * φ₁(ξ_y) * φ₂(ξ_z)
    i == 19 && return φ₃(ξ_x) * φ₃(ξ_y) * φ₂(ξ_z)
    i == 20 && return φ₁(ξ_x) * φ₃(ξ_y) * φ₂(ξ_z)
    # faces
    i == 21 && return φ₂(ξ_x) * φ₂(ξ_y) * φ₁(ξ_z)
    i == 22 && return φ₂(ξ_x) * φ₁(ξ_y) * φ₂(ξ_z)
    i == 23 && return φ₃(ξ_x) * φ₂(ξ_y) * φ₂(ξ_z)
    i == 24 && return φ₂(ξ_x) * φ₃(ξ_y) * φ₂(ξ_z)
    i == 25 && return φ₁(ξ_x) * φ₂(ξ_y) * φ₂(ξ_z)
    i == 26 && return φ₂(ξ_x) * φ₂(ξ_y) * φ₃(ξ_z)
    # interior
    i == 27 && return φ₂(ξ_x) * φ₂(ξ_y) * φ₂(ξ_z)
    throw(ArgumentError("no shape function $i for interpolation $ip"))
end


#############################
# Lagrange RefPrism order 1 #
#############################
# Build on https://defelement.org/elements/examples/prism-lagrange-equispaced-1.html
getnbasefunctions(::Lagrange{RefPrism, 1}) = 6

facedof_indices(::Lagrange{RefPrism, 1}) = ((1, 3, 2), (1, 2, 5, 4), (3, 1, 4, 6), (2, 3, 6, 5), (4, 5, 6))
edgedof_indices(::Lagrange{RefPrism, 1}) = ((2, 1), (1, 3), (1, 4), (3, 2), (2, 5), (3, 6), (4, 5), (4, 6), (6, 5))

function reference_coordinates(::Lagrange{RefPrism, 1})
    return [
        Vec{3, Float64}((0.0, 0.0, 0.0)),
        Vec{3, Float64}((1.0, 0.0, 0.0)),
        Vec{3, Float64}((0.0, 1.0, 0.0)),
        Vec{3, Float64}((0.0, 0.0, 1.0)),
        Vec{3, Float64}((1.0, 0.0, 1.0)),
        Vec{3, Float64}((0.0, 1.0, 1.0)),
    ]
end

function reference_shape_value(ip::Lagrange{RefPrism, 1}, ξ::Vec{3}, i::Int)
    (x, y, z) = ξ
    i == 1 && return 1 - x - y - z * (1 - x - y)
    i == 2 && return x * (1 - z)
    i == 3 && return y * (1 - z)
    i == 4 && return z * (1 - x - y)
    i == 5 && return x * z
    i == 6 && return y * z
    throw(ArgumentError("no shape function $i for interpolation $ip"))
end

#############################
# Lagrange RefPrism order 2 #
#############################
# Build on https://defelement.org/elements/examples/prism-lagrange-equispaced-2.html
# This is simply the tensor-product of a quadratic triangle with a quadratic line.
getnbasefunctions(::Lagrange{RefPrism, 2}) = 18

facedof_indices(::Lagrange{RefPrism, 2}) = (
    # Vertices, Edges, Face
    (1, 3, 2, 8, 10, 7),
    (1, 2, 5, 4, 7, 11, 13, 9, 16),
    (3, 1, 4, 6, 8, 9, 14, 12, 17),
    (2, 3, 6, 5, 10, 12, 15, 11, 18),
    (4, 5, 6, 13, 15, 14),
)
facedof_interior_indices(::Lagrange{RefPrism, 2}) = (
    # Face
    (),
    (16,),
    (17,),
    (18,),
    (),
)
edgedof_indices(::Lagrange{RefPrism, 2}) = (
    # Vertices, Edge
    (2, 1, 7),
    (1, 3, 8),
    (1, 4, 9),
    (3, 2, 10),
    (2, 5, 11),
    (3, 6, 12),
    (4, 5, 13),
    (4, 6, 14),
    (6, 5, 15),
)
edgedof_interior_indices(::Lagrange{RefPrism, 2}) = (
    # Edge
    (7,),
    (8,),
    (9,),
    (10,),
    (11,),
    (12,),
    (13,),
    (14,),
    (15,),
)

function reference_coordinates(::Lagrange{RefPrism, 2})
    return [
        Vec{3, Float64}((0.0, 0.0, 0.0)),
        Vec{3, Float64}((1.0, 0.0, 0.0)),
        Vec{3, Float64}((0.0, 1.0, 0.0)),
        Vec{3, Float64}((0.0, 0.0, 1.0)),
        Vec{3, Float64}((1.0, 0.0, 1.0)),
        Vec{3, Float64}((0.0, 1.0, 1.0)),
        Vec{3, Float64}((0.5, 0.0, 0.0)),
        Vec{3, Float64}((0.0, 0.5, 0.0)),
        Vec{3, Float64}((0.0, 0.0, 0.5)),
        Vec{3, Float64}((0.5, 0.5, 0.0)),
        Vec{3, Float64}((1.0, 0.0, 0.5)),
        Vec{3, Float64}((0.0, 1.0, 0.5)),
        Vec{3, Float64}((0.5, 0.0, 1.0)),
        Vec{3, Float64}((0.0, 0.5, 1.0)),
        Vec{3, Float64}((0.5, 0.5, 1.0)),
        Vec{3, Float64}((0.5, 0.0, 0.5)),
        Vec{3, Float64}((0.0, 0.5, 0.5)),
        Vec{3, Float64}((0.5, 0.5, 0.5)),
    ]
end

function reference_shape_value(ip::Lagrange{RefPrism, 2}, ξ::Vec{3}, i::Int)
    (x, y, z) = ξ
    x² = x * x
    y² = y * y
    z² = z * z
    i == 1  && return 4 * x² * z² - 6x² * z + 2x² + 8x * y * z² - 12x * y * z + 4x * y - 6x * z² + 9x * z - 3x + 4y² * z² - 6y² * z + 2y² - 6y * z² + 9y * z - 3 * y + 2z² - 3z + 1
    i == 2  && return x * (4x * z² - 6x * z + 2x - 2z² + 3z - 1)
    i == 3  && return y * (4y * z² - 6y * z + 2y - 2z² + 3z - 1)
    i == 4  && return z * (4x² * z - 2x² + 8x * y * z - 4x * y - 6x * z + 3x + 4y² * z - 2y² - 6y * z + 3y + 2z - 1)
    i == 5  && return x * z * (4x * z - 2x - 2z + 1)
    i == 6  && return y * z * (4y * z - 2y - 2z + 1)
    i == 7  && return 4x * (-2x * z² + 3x * z - x - 2 * y * z² + 3y * z - y + 2z² - 3z + 1)
    i == 8  && return 4y * (-2x * z² + 3x * z - x - 2 * y * z² + 3y * z - y + 2z² - 3z + 1)
    i == 9  && return 4z * (-2x² * z + 2x² - 4x * y * z + 4x * y + 3x * z - 3x - 2y² * z + 2y² + 3y * z - 3y - z + 1)
    i == 10 && return 4x * y * (2z² - 3z + 1)
    i == 11 && return 4x * z * (-2x * z + 2x + z - 1)
    i == 12 && return 4y * z * (-2y * z + 2y + z - 1)
    i == 13 && return 4x * z * (-2x * z + x - 2y * z + y + 2z - 1)
    i == 14 && return 4y * z * (-2x * z + x - 2y * z + y + 2z - 1)
    i == 15 && return 4x * y * z * (2z - 1)
    i == 16 && return 16x * z * (x * z - x + y * z - y - z + 1)
    i == 17 && return 16y * z * (x * z - x + y * z - y - z + 1)
    i == 18 && return 16x * y * z * (1 - z)
    throw(ArgumentError("no shape function $i for interpolation $ip"))
end


#####################################
# Lagrange dim 3 RefPyramid order 1 #
#####################################
getnbasefunctions(::Lagrange{RefPyramid, 1}) = 5
facedof_indices(::Lagrange{RefPyramid, 1}) = ((1, 3, 4, 2), (1, 2, 5), (1, 5, 3), (2, 4, 5), (3, 5, 4))
edgedof_indices(::Lagrange{RefPyramid, 1}) = ((1, 2), (1, 3), (1, 5), (2, 4), (2, 5), (4, 3), (3, 5), (4, 5))

function reference_coordinates(::Lagrange{RefPyramid, 1})
    return [
        Vec{3, Float64}((0.0, 0.0, 0.0)),
        Vec{3, Float64}((1.0, 0.0, 0.0)),
        Vec{3, Float64}((0.0, 1.0, 0.0)),
        Vec{3, Float64}((1.0, 1.0, 0.0)),
        Vec{3, Float64}((0.0, 0.0, 1.0)),
    ]
end

function reference_shape_value(ip::Lagrange{RefPyramid, 1}, ξ::Vec{3, T}, i::Int) where {T}
    (x, y, z) = ξ
    zzero = z ≈ one(T)
    i == 1 && return zzero ? zero(T) : (-x * y + (z - 1) * (-x - y - z + 1)) / (z - 1)
    i == 2 && return zzero ? zero(T) : x * (y + z - 1) / (z - 1)
    i == 3 && return zzero ? zero(T) : y * (x + z - 1) / (z - 1)
    i == 4 && return zzero ? zero(T) : -x * y / (z - 1)
    i == 5 && return z
    throw(ArgumentError("no shape function $i for interpolation $ip"))
end

#####################################
# Lagrange dim 3 RefPyramid order 2 #
#####################################
getnbasefunctions(::Lagrange{RefPyramid, 2}) = 14

facedof_indices(::Lagrange{RefPyramid, 2}) = (
    # Vertices, Edges, Face
    (1, 3, 4, 2, 7, 11, 9, 6, 14),
    (1, 2, 5, 6, 10, 8),
    (1, 5, 3, 7, 12, 8),
    (2, 4, 5, 9, 13, 10),
    (3, 5, 4, 12, 13, 11),
)
facedof_interior_indices(::Lagrange{RefPyramid, 2}) = (
    (14,),
    (),
    (),
    (),
    (),
)
edgedof_indices(::Lagrange{RefPyramid, 2}) = (
    (1, 2, 6),
    (1, 3, 7),
    (1, 5, 8),
    (2, 4, 9),
    (2, 5, 10),
    (4, 3, 11),
    (3, 5, 12),
    (4, 5, 13),
)
edgedof_interior_indices(::Lagrange{RefPyramid, 2}) = (
    (6,),
    (7,),
    (8,),
    (9,),
    (10,),
    (11,),
    (12,),
    (13,),
)
function reference_coordinates(::Lagrange{RefPyramid, 2})
    return [
        Vec{3, Float64}((0.0, 0.0, 0.0)),
        Vec{3, Float64}((1.0, 0.0, 0.0)),
        Vec{3, Float64}((0.0, 1.0, 0.0)),
        Vec{3, Float64}((1.0, 1.0, 0.0)),
        Vec{3, Float64}((0.0, 0.0, 1.0)),
        # edges
        Vec{3, Float64}((0.5, 0.0, 0.0)),
        Vec{3, Float64}((0.0, 0.5, 0.0)),
        Vec{3, Float64}((0.0, 0.0, 0.5)),
        Vec{3, Float64}((1.0, 0.5, 0.0)),
        Vec{3, Float64}((0.5, 0.0, 0.5)),
        Vec{3, Float64}((0.5, 1.0, 0.0)),
        Vec{3, Float64}((0.0, 0.5, 0.5)),
        Vec{3, Float64}((0.5, 0.5, 0.5)),
        # faces
        Vec{3, Float64}((0.5, 0.5, 0.0)),
    ]
end

function reference_shape_value(ip::Lagrange{RefPyramid, 2}, ξ::Vec{3, T}, i::Int) where {T}
    (x, y, z) = ξ
    x² = x * x
    y² = y * y
    z² = z * z
    zzero = z ≈ one(T)
    i == 1 && return zzero ? zero(T) : (4 * x² * y² * (z - 1) + x * y * (6x + 6y + z) * (z² - 2z + 1) + (z - 1) * (z² - 2z + 1) * (2x² + 9 * x * y + 4 * x * z - 3x + 2y² + 4 * y * z - 3y + 2z² - 3z + 1)) / ((z - 1) * (z² - 2z + 1))
    i == 2 && return zzero ? zero(T) : x * (4x * y² * (z - 1) + y * (6x + 2y - z) * (z² - 2z + 1) + (z - 1) * (2x + 3y - 1) * (z² - 2z + 1)) / ((z - 1) * (z² - 2z + 1))
    i == 3 && return zzero ? zero(T) : y * (4x² * y * (z - 1) + x * (2x + 6y - z) * (z² - 2z + 1) + (z - 1) * (3x + 2y - 1) * (z² - 2z + 1)) / ((z - 1) * (z² - 2z + 1))
    i == 4 && return zzero ? zero(T) : x * y * (4 * x * y + 2x * z - 2x + 2y * z - 2y + 2z² - 3z + 1) / (z² - 2z + 1)
    i == 5 && return z * (2z - 1)
    i == 6 && return zzero ? zero(T) : 4x * (2x * y² * (1 - z) - y * (3x + 2y) * (z² - 2z + 1) + (z - 1) * (z² - 2z + 1) * (-x - 3y - z + 1)) / ((z - 1) * (z² - 2z + 1))
    i == 7 && return zzero ? zero(T) : 4y * (2x² * y * (1 - z) - x * (2x + 3y) * (z² - 2z + 1) + (z - 1) * (z² - 2z + 1) * (-3x - y - z + 1)) / ((z - 1) * (z² - 2z + 1))
    i == 8 && return zzero ? zero(T) : 4z * (-x * y + (z - 1) * (-x - y - z + 1)) / (z - 1)
    i == 9 && return zzero ? zero(T) : 4 * x * y * (-2x * y - 2x * z + 2x - y * z + y - z² + 2 * z - 1) / (z² - 2z + 1)
    i == 10 && return zzero ? zero(T) : 4x * z * (y + z - 1) / (z - 1)
    i == 11 && return zzero ? zero(T) : 4 * x * y * (-2x * y - x * z + x - 2y * z + 2y - z² + 2z - 1) / (z² - 2z + 1)
    i == 12 && return zzero ? zero(T) : 4y * z * (x + z - 1) / (z - 1)
    i == 13 && return zzero ? zero(T) : -4x * y * z / (z - 1)
    i == 14 && return zzero ? zero(T) : 16x * y * (x * y + x * z - x + y * z - y + z² - 2z + 1) / (z² - 2z + 1)
    throw(ArgumentError("no shape function $i for interpolation $ip"))
end

###################
# Bubble elements #
###################
"""
Lagrange element with bubble stabilization.
"""
struct BubbleEnrichedLagrange{shape, order} <: ScalarInterpolation{shape, order}
    function BubbleEnrichedLagrange{shape, order}() where {shape <: AbstractRefShape, order}
        return new{shape, order}()
    end
end
conformity(::BubbleEnrichedLagrange) = H1Conformity()

#######################################
# Lagrange-Bubble RefTriangle order 1 #
#######################################
# Taken from https://defelement.org/elements/examples/triangle-bubble-enriched-lagrange-1.html
getnbasefunctions(::BubbleEnrichedLagrange{RefTriangle, 1}) = 4
adjust_dofs_during_distribution(::BubbleEnrichedLagrange{RefTriangle, 1}) = false

vertexdof_indices(::BubbleEnrichedLagrange{RefTriangle, 1}) = ((1,), (2,), (3,))
edgedof_indices(::BubbleEnrichedLagrange{RefTriangle, 1}) = ((1, 2), (2, 3), (3, 1))
facedof_indices(ip::BubbleEnrichedLagrange{RefTriangle, 1}) = (ntuple(i -> i, getnbasefunctions(ip)),)
facedof_interior_indices(::BubbleEnrichedLagrange{RefTriangle, 1}) = ((4,),)

function reference_coordinates(::BubbleEnrichedLagrange{RefTriangle, 1})
    return [
        Vec{2, Float64}((1.0, 0.0)),
        Vec{2, Float64}((0.0, 1.0)),
        Vec{2, Float64}((0.0, 0.0)),
        Vec{2, Float64}((1 / 3, 1 / 3)),
    ]
end

function reference_shape_value(ip::BubbleEnrichedLagrange{RefTriangle, 1}, ξ::Vec{2}, i::Int)
    ξ_x = ξ[1]
    ξ_y = ξ[2]
    i == 1 && return ξ_x * (9ξ_y^2 + 9ξ_x * ξ_y - 9ξ_y + 1)
    i == 2 && return ξ_y * (9ξ_x^2 + 9ξ_x * ξ_y - 9ξ_x + 1)
    i == 3 && return 9ξ_x^2 * ξ_y + 9ξ_x * ξ_y^2 - 9ξ_x * ξ_y - ξ_x - ξ_y + 1
    i == 4 && return 27ξ_x * ξ_y * (1 - ξ_x - ξ_y)
    throw(ArgumentError("no shape function $i for interpolation $ip"))
end

###############
# Serendipity #
###############
"""
    Serendipity{refshape, order} <: ScalarInterpolation

Serendipity element on hypercubes. Currently only second order variants are implemented.
"""
struct Serendipity{shape, order} <: ScalarInterpolation{shape, order}
    function Serendipity{shape, order}() where {shape <: AbstractRefShape, order}
        return new{shape, order}()
    end
end
conformity(::Serendipity) = H1Conformity()

# Note that the edgedofs for high order serendipity elements are defined in terms of integral moments,
# so no permutation exists in general. See e.g. Scroggs et al. [2022] for an example.
# adjust_dofs_during_distribution(::Serendipity{refshape, order}) where {refshape, order} = false
adjust_dofs_during_distribution(::Serendipity{<:Any, 2}) = false
adjust_dofs_during_distribution(::Serendipity{<:Any, 1}) = false

# Vertices for all Serendipity interpolations are the same
vertexdof_indices(::Serendipity{RefQuadrilateral}) = ((1,), (2,), (3,), (4,))
vertexdof_indices(::Serendipity{RefHexahedron}) = ((1,), (2,), (3,), (4,), (5,), (6,), (7,), (8,))

########################################
# Serendipity RefQuadrilateral order 2 #
########################################
getnbasefunctions(::Serendipity{RefQuadrilateral, 2}) = 8
getlowerorder(::Serendipity{RefQuadrilateral, 2}) = Lagrange{RefQuadrilateral, 1}()

edgedof_indices(::Serendipity{RefQuadrilateral, 2}) = ((1, 2, 5), (2, 3, 6), (3, 4, 7), (4, 1, 8))
edgedof_interior_indices(::Serendipity{RefQuadrilateral, 2}) = ((5,), (6,), (7,), (8,))
facedof_indices(ip::Serendipity{RefQuadrilateral, 2}) = (ntuple(i -> i, getnbasefunctions(ip)),)

function reference_coordinates(::Serendipity{RefQuadrilateral, 2})
    return [
        Vec{2, Float64}((-1.0, -1.0)),
        Vec{2, Float64}((1.0, -1.0)),
        Vec{2, Float64}((1.0, 1.0)),
        Vec{2, Float64}((-1.0, 1.0)),
        Vec{2, Float64}((0.0, -1.0)),
        Vec{2, Float64}((1.0, 0.0)),
        Vec{2, Float64}((0.0, 1.0)),
        Vec{2, Float64}((-1.0, 0.0)),
    ]
end

function reference_shape_value(ip::Serendipity{RefQuadrilateral, 2}, ξ::Vec{2}, i::Int)
    ξ_x = ξ[1]
    ξ_y = ξ[2]
    i == 1 && return (1 - ξ_x) * (1 - ξ_y) * (-ξ_x - ξ_y - 1) / 4
    i == 2 && return (1 + ξ_x) * (1 - ξ_y) * (ξ_x - ξ_y - 1) / 4
    i == 3 && return (1 + ξ_x) * (1 + ξ_y) * (ξ_x + ξ_y - 1) / 4
    i == 4 && return (1 - ξ_x) * (1 + ξ_y) * (-ξ_x + ξ_y - 1) / 4
    i == 5 && return (1 - ξ_x * ξ_x) * (1 - ξ_y) / 2
    i == 6 && return (1 + ξ_x) * (1 - ξ_y * ξ_y) / 2
    i == 7 && return (1 - ξ_x * ξ_x) * (1 + ξ_y) / 2
    i == 8 && return (1 - ξ_x) * (1 - ξ_y * ξ_y) / 2
    throw(ArgumentError("no shape function $i for interpolation $ip"))
end

#####################################
# Serendipity RefHexahedron order 2 #
#####################################
# Note that second order serendipity hex has no interior face indices.
getnbasefunctions(::Serendipity{RefHexahedron, 2}) = 20
getlowerorder(::Serendipity{RefHexahedron, 2}) = Lagrange{RefHexahedron, 1}()

facedof_indices(::Serendipity{RefHexahedron, 2}) = (
    (1, 4, 3, 2, 12, 11, 10, 9),
    (1, 2, 6, 5, 9, 18, 13, 17),
    (2, 3, 7, 6, 10, 19, 14, 18),
    (3, 4, 8, 7, 11, 20, 15, 19),
    (1, 5, 8, 4, 17, 16, 20, 12),
    (5, 6, 7, 8, 13, 14, 15, 16),
)
edgedof_indices(::Serendipity{RefHexahedron, 2}) = (
    (1, 2, 9),
    (2, 3, 10),
    (3, 4, 11),
    (4, 1, 12),
    (5, 6, 13),
    (6, 7, 14),
    (7, 8, 15),
    (8, 5, 16),
    (1, 5, 17),
    (2, 6, 18),
    (3, 7, 19),
    (4, 8, 20),
)

edgedof_interior_indices(::Serendipity{RefHexahedron, 2}) = (
    (9,), (10,), (11,), (12,), (13,), (14,), (15,), (16,), (17,), (18,), (19,), (20,),
)

function reference_coordinates(::Serendipity{RefHexahedron, 2})
    return [
        Vec{3, Float64}((-1.0, -1.0, -1.0)),
        Vec{3, Float64}((1.0, -1.0, -1.0)),
        Vec{3, Float64}((1.0, 1.0, -1.0)),
        Vec{3, Float64}((-1.0, 1.0, -1.0)),
        Vec{3, Float64}((-1.0, -1.0, 1.0)),
        Vec{3, Float64}((1.0, -1.0, 1.0)),
        Vec{3, Float64}((1.0, 1.0, 1.0)),
        Vec{3, Float64}((-1.0, 1.0, 1.0)),
        Vec{3, Float64}((0.0, -1.0, -1.0)),
        Vec{3, Float64}((1.0, 0.0, -1.0)),
        Vec{3, Float64}((0.0, 1.0, -1.0)),
        Vec{3, Float64}((-1.0, 0.0, -1.0)),
        Vec{3, Float64}((0.0, -1.0, 1.0)),
        Vec{3, Float64}((1.0, 0.0, 1.0)),
        Vec{3, Float64}((0.0, 1.0, 1.0)),
        Vec{3, Float64}((-1.0, 0.0, 1.0)),
        Vec{3, Float64}((-1.0, -1.0, 0.0)),
        Vec{3, Float64}((1.0, -1.0, 0.0)),
        Vec{3, Float64}((1.0, 1.0, 0.0)),
        Vec{3, Float64}((-1.0, 1.0, 0.0)),
    ]
end

# Inlined to resolve the recursion properly
@inline function reference_shape_value(ip::Serendipity{RefHexahedron, 2}, ξ::Vec{3}, i::Int)
    ξ_x = ξ[1]
    ξ_y = ξ[2]
    ξ_z = ξ[3]
    i == 1 && return (1 - ξ_x) * (1 - ξ_y) * (1 - ξ_z) / 8 - (reference_shape_value(ip, ξ, 12) + reference_shape_value(ip, ξ, 9) + reference_shape_value(ip, ξ, 17)) / 2
    i == 2 && return (1 + ξ_x) * (1 - ξ_y) * (1 - ξ_z) / 8 - (reference_shape_value(ip, ξ, 9) + reference_shape_value(ip, ξ, 10) + reference_shape_value(ip, ξ, 18)) / 2
    i == 3 && return (1 + ξ_x) * (1 + ξ_y) * (1 - ξ_z) / 8 - (reference_shape_value(ip, ξ, 10) + reference_shape_value(ip, ξ, 11) + reference_shape_value(ip, ξ, 19)) / 2
    i == 4 && return (1 - ξ_x) * (1 + ξ_y) * (1 - ξ_z) / 8 - (reference_shape_value(ip, ξ, 11) + reference_shape_value(ip, ξ, 12) + reference_shape_value(ip, ξ, 20)) / 2
    i == 5 && return (1 - ξ_x) * (1 - ξ_y) * (1 + ξ_z) / 8 - (reference_shape_value(ip, ξ, 16) + reference_shape_value(ip, ξ, 13) + reference_shape_value(ip, ξ, 17)) / 2
    i == 6 && return (1 + ξ_x) * (1 - ξ_y) * (1 + ξ_z) / 8 - (reference_shape_value(ip, ξ, 13) + reference_shape_value(ip, ξ, 14) + reference_shape_value(ip, ξ, 18)) / 2
    i == 7 && return (1 + ξ_x) * (1 + ξ_y) * (1 + ξ_z) / 8 - (reference_shape_value(ip, ξ, 14) + reference_shape_value(ip, ξ, 15) + reference_shape_value(ip, ξ, 19)) / 2
    i == 8 && return (1 - ξ_x) * (1 + ξ_y) * (1 + ξ_z) / 8 - (reference_shape_value(ip, ξ, 15) + reference_shape_value(ip, ξ, 16) + reference_shape_value(ip, ξ, 20)) / 2
    i == 9 && return (1 - ξ_x^2) * (1 - ξ_y) * (1 - ξ_z) / 4
    i == 10 && return (1 + ξ_x) * (1 - ξ_y^2) * (1 - ξ_z) / 4
    i == 11 && return (1 - ξ_x^2) * (1 + ξ_y) * (1 - ξ_z) / 4
    i == 12 && return (1 - ξ_x) * (1 - ξ_y^2) * (1 - ξ_z) / 4
    i == 13 && return (1 - ξ_x^2) * (1 - ξ_y) * (1 + ξ_z) / 4
    i == 14 && return (1 + ξ_x) * (1 - ξ_y^2) * (1 + ξ_z) / 4
    i == 15 && return (1 - ξ_x^2) * (1 + ξ_y) * (1 + ξ_z) / 4
    i == 16 && return (1 - ξ_x) * (1 - ξ_y^2) * (1 + ξ_z) / 4
    i == 17 && return (1 - ξ_x) * (1 - ξ_y) * (1 - ξ_z^2) / 4
    i == 18 && return (1 + ξ_x) * (1 - ξ_y) * (1 - ξ_z^2) / 4
    i == 19 && return (1 + ξ_x) * (1 + ξ_y) * (1 - ξ_z^2) / 4
    i == 20 && return (1 - ξ_x) * (1 + ξ_y) * (1 - ξ_z^2) / 4
    throw(ArgumentError("no shape function $i for interpolation $ip"))
end


#############################
# Crouzeix–Raviart Elements #
#############################
"""
    CrouzeixRaviart{refshape, order} <: ScalarInterpolation

Classical non-conforming Crouzeix–Raviart element.

For details we refer to the original paper [CroRav:1973:cnf](@cite).
"""
struct CrouzeixRaviart{shape, order} <: ScalarInterpolation{shape, order}
    CrouzeixRaviart{RefTriangle, 1}() = new{RefTriangle, 1}()
    CrouzeixRaviart{RefTetrahedron, 1}() = new{RefTetrahedron, 1}()
end
conformity(::CrouzeixRaviart) = L2Conformity()

# CR elements are characterized by not having vertex dofs
vertexdof_indices(ip::CrouzeixRaviart) = ntuple(i -> (), nvertices(ip))

#################################################
# Non-conforming Crouzeix-Raviart dim 2 order 1 #
#################################################
getnbasefunctions(::CrouzeixRaviart{RefTriangle, 1}) = 3

adjust_dofs_during_distribution(::CrouzeixRaviart) = true
adjust_dofs_during_distribution(::CrouzeixRaviart{<:Any, 1}) = false

edgedof_indices(::CrouzeixRaviart{RefTriangle, 1}) = ((1,), (2,), (3,))
edgedof_interior_indices(::CrouzeixRaviart{RefTriangle, 1}) = ((1,), (2,), (3,))
facedof_indices(ip::CrouzeixRaviart{RefTriangle, 1}) = (ntuple(i -> i, getnbasefunctions(ip)),)

function reference_coordinates(::CrouzeixRaviart{RefTriangle, 1})
    return [
        Vec{2, Float64}((0.5, 0.5)),
        Vec{2, Float64}((0.0, 0.5)),
        Vec{2, Float64}((0.5, 0.0)),
    ]
end

function reference_shape_value(ip::CrouzeixRaviart{RefTriangle, 1}, ξ::Vec{2}, i::Int)
    ξ_x = ξ[1]
    ξ_y = ξ[2]
    i == 1 && return 2 * ξ_x + 2 * ξ_y - 1
    i == 2 && return 1 - 2 * ξ_x
    i == 3 && return 1 - 2 * ξ_y
    throw(ArgumentError("no shape function $i for interpolation $ip"))
end

#################################################
# Non-conforming Crouzeix-Raviart dim 3 order 1 #
#################################################
getnbasefunctions(::CrouzeixRaviart{RefTetrahedron, 1}) = 4

facedof_indices(::CrouzeixRaviart{RefTetrahedron, 1}) = ((1,), (2,), (3,), (4,))
facedof_interior_indices(::CrouzeixRaviart{RefTetrahedron, 1}) = ((1,), (2,), (3,), (4,))

function reference_coordinates(::CrouzeixRaviart{RefTetrahedron, 1})
    return [
        Vec{3, Float64}((1 / 3, 1 / 3, 0.0)),
        Vec{3, Float64}((1 / 3, 0.0, 1 / 3)),
        Vec{3, Float64}((1 / 3, 1 / 3, 1 / 3)),
        Vec{3, Float64}((0.0, 1 / 3, 1 / 3)),
    ]
end

function reference_shape_value(ip::CrouzeixRaviart{RefTetrahedron, 1}, ξ::Vec{3}, i::Int)
    (x, y, z) = ξ
    i == 1 && return 1 - 3z
    i == 2 && return 1 - 3y
    i == 3 && return 3x + 3y + 3z - 2
    i == 4 && return 1 - 3x
    return throw(ArgumentError("no shape function $i for interpolation $ip"))
end

"""
    RannacherTurek{refshape, order} <: ScalarInterpolation

Classical non-conforming Rannacher-Turek element.

This element is basically the idea from Crouzeix and Raviart applied to
hypercubes. For details see the original paper [RanTur:1992:snq](@cite).
"""
struct RannacherTurek{shape, order} <: ScalarInterpolation{shape, order} end
conformity(::RannacherTurek) = L2Conformity()

# CR-type elements are characterized by not having vertex dofs
vertexdof_indices(ip::RannacherTurek) = ntuple(i -> (), nvertices(ip))

adjust_dofs_during_distribution(::RannacherTurek) = true
adjust_dofs_during_distribution(::RannacherTurek{<:Any, 1}) = false

#################################
# Rannacher-Turek dim 2 order 1 #
#################################
getnbasefunctions(::RannacherTurek{RefQuadrilateral, 1}) = 4

edgedof_indices(::RannacherTurek{RefQuadrilateral, 1}) = ((1,), (2,), (3,), (4,))
edgedof_interior_indices(::RannacherTurek{RefQuadrilateral, 1}) = ((1,), (2,), (3,), (4,))
facedof_indices(ip::RannacherTurek{RefQuadrilateral, 1}) = (ntuple(i -> i, getnbasefunctions(ip)),)

function reference_coordinates(::RannacherTurek{RefQuadrilateral, 1})
    return [
        Vec{2, Float64}((0.0, -1.0)),
        Vec{2, Float64}((1.0, 0.0)),
        Vec{2, Float64}((0.0, 1.0)),
        Vec{2, Float64}((-1.0, 0.0)),
    ]
end

function reference_shape_value(ip::RannacherTurek{RefQuadrilateral, 1}, ξ::Vec{2, T}, i::Int) where {T}
    (x, y) = ξ

    i == 1 && return -(x + 1)^2 / 4 + (y + 1)^2 / 4 + (x + 1) / 2 - (y + 1) + T(3) / 4
    i == 2 && return (x + 1)^2 / 4 - (y + 1)^2 / 4 + (y + 1) / 2 - T(1) / 4
    i == 3 && return -(x + 1)^2 / 4 + (y + 1)^2 / 4 + (x + 1) / 2 - T(1) / 4
    i == 4 && return (x + 1)^2 / 4 - (y + 1)^2 / 4 - (x + 1) + (y + 1) / 2 + T(3) / 4
    throw(ArgumentError("no shape function $i for interpolation $ip"))
end

#################################
# Rannacher-Turek dim 3 order 1 #
#################################
getnbasefunctions(::RannacherTurek{RefHexahedron, 1}) = 6

edgedof_indices(ip::RannacherTurek{RefHexahedron, 1}) = ntuple(i -> (), nedges(ip))
edgedof_interior_indices(ip::RannacherTurek{RefHexahedron, 1}) = ntuple(i -> (), nedges(ip))
facedof_indices(::RannacherTurek{RefHexahedron, 1}) = ((1,), (2,), (3,), (4,), (5,), (6,))
facedof_interior_indices(::RannacherTurek{RefHexahedron, 1}) = ((1,), (2,), (3,), (4,), (5,), (6,))

function reference_coordinates(::RannacherTurek{RefHexahedron, 1})
    return [
        Vec{3, Float64}((0.0, 0.0, -1.0)),
        Vec{3, Float64}((0.0, -1.0, 0.0)),
        Vec{3, Float64}((1.0, 0.0, 0.0)),
        Vec{3, Float64}((0.0, 1.0, 0.0)),
        Vec{3, Float64}((-1.0, 0.0, 0.0)),
        Vec{3, Float64}((0.0, 0.0, 1.0)),
    ]
end

function reference_shape_value(ip::RannacherTurek{RefHexahedron, 1}, ξ::Vec{3, T}, i::Int) where {T}
    (x, y, z) = ξ

    i == 1 && return -2((x + 1))^2 / 12 + 1(x + 1) / 3 - 2((y + 1))^2 / 12 + 1(y + 1) / 3 + 4((z + 1))^2 / 12 - 7(z + 1) / 6 + T(2) / 3
    i == 2 && return -2((x + 1))^2 / 12 + 1(x + 1) / 3 + 4((y + 1))^2 / 12 - 7(y + 1) / 6 - 2((z + 1))^2 / 12 + 1(z + 1) / 3 + T(2) / 3
    i == 3 && return 4((x + 1))^2 / 12 - 1(x + 1) / 6 - 2((y + 1))^2 / 12 + 1(y + 1) / 3 - 2((z + 1))^2 / 12 + 1(z + 1) / 3 - T(1) / 3
    i == 4 && return -2((x + 1))^2 / 12 + 1(x + 1) / 3 + 4((y + 1))^2 / 12 - 1(y + 1) / 6 - 2((z + 1))^2 / 12 + 1(z + 1) / 3 - T(1) / 3
    i == 5 && return 4((x + 1))^2 / 12 - 7(x + 1) / 6 - 2((y + 1))^2 / 12 + 1(y + 1) / 3 - 2((z + 1))^2 / 12 + 1(z + 1) / 3 + T(2) / 3
    i == 6 && return -2((x + 1))^2 / 12 + 1(x + 1) / 3 - 2((y + 1))^2 / 12 + 1(y + 1) / 3 + 4((z + 1))^2 / 12 - 1(z + 1) / 6 - T(1) / 3

    throw(ArgumentError("no shape function $i for interpolation $ip"))
end

##################################################
# VectorizedInterpolation{<:ScalarInterpolation} #
##################################################
struct VectorizedInterpolation{vdim, refshape, order, SI <: ScalarInterpolation{refshape, order}} <: VectorInterpolation{vdim, refshape, order}
    ip::SI
    function VectorizedInterpolation{vdim}(ip::SI) where {vdim, refshape, order, SI <: ScalarInterpolation{refshape, order}}
        return new{vdim, refshape, order, SI}(ip)
    end
end
conformity(ip::VectorizedInterpolation) = conformity(ip.ip)

adjust_dofs_during_distribution(ip::VectorizedInterpolation) = adjust_dofs_during_distribution(ip.ip)

# Vectorize to reference dimension by default
function VectorizedInterpolation(ip::ScalarInterpolation{shape}) where {refdim, shape <: AbstractRefShape{refdim}}
    return VectorizedInterpolation{refdim}(ip)
end

Base.:(^)(ip::ScalarInterpolation, vdim::Int) = VectorizedInterpolation{vdim}(ip)
function Base.literal_pow(::typeof(^), ip::ScalarInterpolation, ::Val{vdim}) where {vdim}
    return VectorizedInterpolation{vdim}(ip)
end

function Base.show(io::IO, mime::MIME"text/plain", ip::VectorizedInterpolation{vdim}) where {vdim}
    show(io, mime, ip.ip)
    print(io, "^", vdim)
    return
end

# Helper to get number of copies for DoF distribution
get_n_copies(::VectorizedInterpolation{vdim}) where {vdim} = vdim
InterpolationInfo(ip::VectorizedInterpolation) = InterpolationInfo(ip.ip, get_n_copies(ip))

# Error when trying to get dof indices from vectorized interpolations.
# Currently, this should only be done for the scalar interpolation.
function _entitydof_indices_vectorized_ip_error(f::Symbol)
    throw(ArgumentError(string(f, " is not implemented for VectorizedInterpolations and should be called on the scalar base interpolation")))
end
vertexdof_indices(::VectorizedInterpolation) = _entitydof_indices_vectorized_ip_error(:vertexdof_indices)
edgedof_indices(::VectorizedInterpolation) = _entitydof_indices_vectorized_ip_error(:edgedof_indices)
facedof_indices(::VectorizedInterpolation) = _entitydof_indices_vectorized_ip_error(:facedof_indices)
edgedof_interior_indices(::VectorizedInterpolation) = _entitydof_indices_vectorized_ip_error(:edgedof_interior_indices)
facedof_interior_indices(::VectorizedInterpolation) = _entitydof_indices_vectorized_ip_error(:facedof_interior_indices)
volumedof_interior_indices(::VectorizedInterpolation) = _entitydof_indices_vectorized_ip_error(:volumedof_interior_indices)

get_base_interpolation(ip::Interpolation) = ip
get_base_interpolation(ip::VectorizedInterpolation) = ip.ip

function getnbasefunctions(ipv::VectorizedInterpolation{vdim}) where {vdim}
    return vdim * getnbasefunctions(ipv.ip)
end
function reference_shape_value(ipv::VectorizedInterpolation{vdim, shape}, ξ::Vec{refdim, T}, I::Int) where {vdim, refdim, shape <: AbstractRefShape{refdim}, T}
    i0, c0 = divrem(I - 1, vdim)
    i = i0 + 1
    c = c0 + 1
    v = reference_shape_value(ipv.ip, ξ, i)
    return Vec{vdim, T}(j -> j == c ? v : zero(v))
end

# vdim == refdim
function reference_shape_gradient_and_value(ipv::VectorizedInterpolation{dim, shape}, ξ::Vec{dim}, I::Int) where {dim, shape <: AbstractRefShape{dim}}
    return invoke(reference_shape_gradient_and_value, Tuple{Interpolation, Vec, Int}, ipv, ξ, I)
end
# vdim != refdim
function reference_shape_gradient_and_value(ipv::VectorizedInterpolation{vdim, shape}, ξ::V, I::Int) where {vdim, refdim, shape <: AbstractRefShape{refdim}, T, V <: Vec{refdim, T}}
    tosvec(v::Vec) = SVector((v...,))
    tovec(sv::SVector) = Vec((sv...))
    val = reference_shape_value(ipv, ξ, I)
    grad = ForwardDiff.jacobian(sv -> tosvec(reference_shape_value(ipv, tovec(sv), I)), tosvec(ξ))
    return grad, val
end

# vdim == refdim
function reference_shape_hessian_gradient_and_value(ipv::VectorizedInterpolation{dim, shape}, ξ::Vec{dim}, I::Int) where {dim, shape <: AbstractRefShape{dim}}
    return invoke(reference_shape_hessian_gradient_and_value, Tuple{Interpolation, Vec, Int}, ipv, ξ, I)
end
# vdim != refdim
function reference_shape_hessian_gradient_and_value(ipv::VectorizedInterpolation{vdim, shape}, ξ::V, I::Int) where {vdim, refdim, shape <: AbstractRefShape{refdim}, T, V <: Vec{refdim, T}}
    return _reference_shape_hessian_gradient_and_value_static_array(ipv, ξ, I)
end
function _reference_shape_hessian_gradient_and_value_static_array(ipv::VectorizedInterpolation{vdim, shape}, ξ::V, I::Int) where {vdim, refdim, shape <: AbstractRefShape{refdim}, T, V <: Vec{refdim, T}}
    # Load with dual numbers and compute the value
    f = x -> reference_shape_value(ipv, x, I)
    ξd = Tensors._load(Tensors._load(ξ, ForwardDiff.Tag(f, V)), ForwardDiff.Tag(f, V))
    value_hess = f(ξd)
    # Extract the value and gradient
    val = Vec{vdim, T}(i -> ForwardDiff.value(ForwardDiff.value(value_hess[i])))
    grad = zero(MMatrix{vdim, refdim, T})
    hess = zero(MArray{Tuple{vdim, refdim, refdim}, T})
    for (i, vi) in pairs(value_hess)
        hess_values = ForwardDiff.value(vi)

        hess_values_partials = ForwardDiff.partials(hess_values)
        for (k, pk) in pairs(hess_values_partials)
            grad[i, k] = pk
        end

        hess_partials = ForwardDiff.partials(vi)
        for (j, partial_j) in pairs(hess_partials)
            hess_partials_partials = ForwardDiff.partials(partial_j)
            for (k, pk) in pairs(hess_partials_partials)
                hess[i, j, k] = pk
            end
        end
    end
    return SArray(hess), SMatrix(grad), val
end

reference_coordinates(ip::VectorizedInterpolation) = reference_coordinates(ip.ip)

"""
    mapping_type(ip::Interpolation)

Get the type of mapping from the reference cell to the real cell for an
interpolation `ip`. Subtypes of `ScalarInterpolation` and `VectorizedInterpolation`
return `IdentityMapping()`, but other non-scalar interpolations may request different
mapping types.
"""
function mapping_type end

mapping_type(::ScalarInterpolation) = IdentityMapping()
mapping_type(::VectorizedInterpolation) = IdentityMapping()

"""
    get_direction(interpolation::Interpolation, shape_nr::Int, cell::AbstractCell)

Return the direction, `±1`, of the cell entity (e.g. facet or edge) associated with
the `interpolation`'s shape function nr. `shape_nr`. This is only required for interpolations
with non-identity mappings, where the direction is required during the mapping of the shape values.

**TODO:** Move the following description to `get_edge_direction` and `get_face_direction`
following #1162

The direction of entities are defined as following the node numbers of the entity's
vertices, `vnodes`. For an edge, `vnodes[2] > vnodes[1]` implies positive direction.

For a face, we first find index, `i`, of the smallest value in `vnodes`. Considering
circular indexing, then a positive face has `vnodes[i-1] > vnodes[i+1]`.
"""
function get_direction end


#####################################
# RaviartThomas (1st kind), H(div)  #
#####################################
struct RaviartThomas{shape, order, vdim} <: VectorInterpolation{vdim, shape, order}
    function RaviartThomas{shape, order}() where {rdim, shape <: AbstractRefShape{rdim}, order}
        return new{shape, order, rdim}()
    end
end
mapping_type(::RaviartThomas) = ContravariantPiolaMapping()
conformity(::RaviartThomas) = HdivConformity()

# RefTriangle
edgedof_indices(ip::RaviartThomas{RefTriangle}) = edgedof_interior_indices(ip)
facedof_indices(ip::RaviartThomas{RefTriangle}) = (ntuple(i -> i, getnbasefunctions(ip)),)

# RefTriangle, 1st order Lagrange
# https://defelement.org/elements/examples/triangle-raviart-thomas-lagrange-0.html
function reference_shape_value(ip::RaviartThomas{RefTriangle, 1}, ξ::Vec{2}, i::Int)
    x, y = ξ
    i == 1 && return ξ                  # Flip sign
    i == 2 && return Vec(x - 1, y)      # Keep sign
    i == 3 && return Vec(x, y - 1)      # Flip sign
    throw(ArgumentError("no shape function $i for interpolation $ip"))
end

getnbasefunctions(::RaviartThomas{RefTriangle, 1}) = 3
edgedof_interior_indices(::RaviartThomas{RefTriangle, 1}) = ((1,), (2,), (3,))
facedof_interior_indices(::RaviartThomas{RefTriangle, 1}) = ((),)
adjust_dofs_during_distribution(::RaviartThomas) = false

function get_direction(::RaviartThomas{RefTriangle, 1}, shape_nr, cell)
    return get_edge_direction(cell, shape_nr) # shape_nr = edge_nr
end

# RefTriangle, 2st order Lagrange
# https://defelement.org/elements/examples/triangle-raviart-thomas-lagrange-1.html
function reference_shape_value(ip::RaviartThomas{RefTriangle, 2}, ξ::Vec{2}, i::Int)
    x, y = ξ
    # Face 1 (keep ordering, flip sign)
    i == 1 && return Vec(4x * (2x - 1), 2y * (4x - 1))
    i == 2 && return Vec(2x * (4y - 1), 4y * (2y - 1))
    # Face 2 (flip ordering, keep signs)
    i == 3 && return Vec(8x * y - 2x - 6y + 2, 4y * (2y - 1))
    i == 4 && return Vec(-8x^2 - 8x * y + 12x + 6y - 4, 2y * (-4x - 4y + 3))
    # Face 3 (keep ordering, flip sign)
    i == 5 && return Vec(2x * (3 - 4x - 4y), -8x * y + 6x - 8y^2 + 12y - 4)
    i == 6 && return Vec(4x * (2x - 1), 8x * y - 6x - 2y + 2)
    # Cell
    i == 7 && return Vec(8x * (-2x - y + 2), 8y * (-2x - y + 1))
    i == 8 && return Vec(8x * (-2y - x + 1), 8y * (-2y - x + 2))
    throw(ArgumentError("no shape function $i for interpolation $ip"))
end

getnbasefunctions(::RaviartThomas{RefTriangle, 2}) = 8
edgedof_interior_indices(::RaviartThomas{RefTriangle, 2}) = ((1, 2), (3, 4), (5, 6))
facedof_interior_indices(::RaviartThomas{RefTriangle, 2}) = ((7, 8),)
adjust_dofs_during_distribution(::RaviartThomas{RefTriangle, 2}) = true

function get_direction(::RaviartThomas{RefTriangle, 2}, shape_nr, cell)
    shape_nr > 6 && return 1
    edge_nr = (shape_nr + 1) ÷ 2
    return get_edge_direction(cell, edge_nr)
end

# RefQuadrilateral
edgedof_indices(ip::RaviartThomas{RefQuadrilateral}) = edgedof_interior_indices(ip)
facedof_indices(ip::RaviartThomas{RefQuadrilateral}) = (ntuple(i -> i, getnbasefunctions(ip)),)

# RefQuadrilateral, 1st order Lagrange
# https://defelement.org/elements/examples/quadrilateral-raviart-thomas-lagrange-1.html
function reference_shape_value(ip::RaviartThomas{RefQuadrilateral, 1}, ξ::Vec{2, T}, i::Int) where {T}
    x, y = ξ
    nil = zero(T)

    i == 1 && return Vec(nil, -(1 - y) / 4) # Changed sign, follow Ferrite's sign convention
    i == 2 && return Vec((1 + x) / 4, nil)  # Changed sign, follow Ferrite's sign convention
    i == 3 && return Vec(nil, (1 + y) / 4)
    i == 4 && return Vec((x - 1) / 4, nil)
    throw(ArgumentError("no shape function $i for interpolation $ip"))
end

getnbasefunctions(::RaviartThomas{RefQuadrilateral, 1}) = 4
edgedof_interior_indices(::RaviartThomas{RefQuadrilateral, 1}) = ((1,), (2,), (3,), (4,))
facedof_interior_indices(::RaviartThomas{RefQuadrilateral, 1}) = ((),)
adjust_dofs_during_distribution(::RaviartThomas{RefQuadrilateral, 1}) = false

function get_direction(::RaviartThomas{RefQuadrilateral, 1}, shape_nr, cell)
    return get_edge_direction(cell, shape_nr) # shape_nr = edge_nr
end

# RefTetrahedron, 1st order Lagrange
# https://defelement.com/elements/examples/tetrahedron-raviart-thomas-lagrange-1.html
function reference_shape_value(ip::RaviartThomas{RefTetrahedron, 1}, ξ::Vec{3}, i::Int)
    x, y, z = ξ
    i == 1 && return Vec(2 * x, 2 * y, 2 * (z - 1)) # Changed sign, follow Ferrite's sign convention
    i == 2 && return Vec(2 * x, 2 * (y - 1), 2 * z)
    i == 3 && return Vec(2 * x, 2 * y, 2 * z)
    i == 4 && return Vec(2 * (x - 1), 2 * y, 2 * z) # Changed sign, follow Ferrite's sign convention
    throw(ArgumentError("no shape function $i for interpolation $ip"))
end

getnbasefunctions(::RaviartThomas{RefTetrahedron, 1}) = 4
edgedof_interior_indices(::RaviartThomas{RefTetrahedron, 1}) = ntuple(_ -> (), 6)
edgedof_indices(ip::RaviartThomas{RefTetrahedron, 1}) = edgedof_interior_indices(ip)
facedof_interior_indices(::RaviartThomas{RefTetrahedron, 1}) = ((1,), (2,), (3,), (4,))
facedof_indices(ip::RaviartThomas{RefTetrahedron, 1}) = facedof_interior_indices(ip)
adjust_dofs_during_distribution(::RaviartThomas{RefTetrahedron, 1}) = false

function get_direction(::RaviartThomas{RefTetrahedron, 1}, shape_nr, cell)
    return get_face_direction(cell, shape_nr) # shape_nr = face_nr
end

# RefHexahedron, 1st order Lagrange
# https://defelement.com/elements/examples/hexahedron-raviart-thomas-lagrange-1.html
# Scale with factor 1/4 as Ferrite has 4 times defelement's face area.
function reference_shape_value(ip::RaviartThomas{RefHexahedron, 1}, ξ::Vec{3, T}, i::Int) where {T}
    x, y, z = ξ
    nil = zero(T)

    i == 1 && return Vec(nil, nil, (z - 1) / 8) # Changed sign, follow Ferrite's sign convention
    i == 2 && return Vec(nil, (y - 1) / 8, nil)
    i == 3 && return Vec((x + 1) / 8, nil, nil)
    i == 4 && return Vec(nil, (y + 1) / 8, nil) # Changed sign, follow Ferrite's sign convention
    i == 5 && return Vec((x - 1) / 8, nil, nil) # Changed sign, follow Ferrite's sign convention
    i == 6 && return Vec(nil, nil, (z + 1) / 8)
    throw(ArgumentError("no shape function $i for interpolation $ip"))
end

getnbasefunctions(::RaviartThomas{RefHexahedron, 1}) = 6
edgedof_interior_indices(::RaviartThomas{RefHexahedron, 1}) = ntuple(_ -> (), 12)
edgedof_indices(ip::RaviartThomas{RefHexahedron, 1}) = edgedof_interior_indices(ip)
facedof_interior_indices(::RaviartThomas{RefHexahedron, 1}) = ((1,), (2,), (3,), (4,), (5,), (6,))
facedof_indices(ip::RaviartThomas{RefHexahedron, 1}) = facedof_interior_indices(ip)
adjust_dofs_during_distribution(::RaviartThomas{RefHexahedron, 1}) = false

function get_direction(::RaviartThomas{RefHexahedron, 1}, shape_nr, cell)
    return get_face_direction(cell, shape_nr) # shape_nr = face_nr
end

#####################################
# Brezzi-Douglas–Marini, H(div)     #
#####################################
struct BrezziDouglasMarini{shape, order, vdim} <: VectorInterpolation{vdim, shape, order}
    function BrezziDouglasMarini{shape, order}() where {rdim, shape <: AbstractRefShape{rdim}, order}
        return new{shape, order, rdim}()
    end
end
mapping_type(::BrezziDouglasMarini) = ContravariantPiolaMapping()
conformity(::BrezziDouglasMarini) = HdivConformity()

# RefTriangle
edgedof_indices(ip::BrezziDouglasMarini{RefTriangle}) = edgedof_interior_indices(ip)
facedof_indices(ip::BrezziDouglasMarini{RefTriangle}) = (ntuple(i -> i, getnbasefunctions(ip)),)

# RefTriangle, 1st order Lagrange
# https://defelement.org/elements/examples/triangle-brezzi-douglas-marini-lagrange-1.html
function reference_shape_value(ip::BrezziDouglasMarini{RefTriangle, 1}, ξ::Vec{2}, i::Int)
    x, y = ξ
    # Edge 1: y=1-x, n = [1, 1]/√2 (Flip sign, pos. integration outwards)
    i == 1 && return Vec(4x, -2y) # N ⋅ n = (2√2 x - √2 (1-x)) = 3√2 x - √2
    i == 2 && return Vec(-2x, 4y) # N ⋅ n = (-√2x + 2√2 (1-x)) = 2√2 - 3√2x
    # Edge 2: x=0, n = [-1, 0] (reverse order to follow Ferrite convention)
    i == 3 && return Vec(-2x - 6y + 2, 4y) # N ⋅ n = (6y - 2)
    i == 4 && return Vec(4x + 6y - 4, -2y) # N ⋅ n = (4 - 6y)
    # Edge 3: y=0, n = [0, -1] (Flip sign, pos. integration outwards)
    i == 5 && return Vec(-2x, 6x + 4y - 4) # N ⋅ n = (4 - 6x)
    i == 6 && return Vec(4x, -6x - 2y + 2) # N ⋅ n = (6x - 2)
    throw(ArgumentError("no shape function $i for interpolation $ip"))
end

getnbasefunctions(::BrezziDouglasMarini{RefTriangle, 1}) = 6
edgedof_interior_indices(::BrezziDouglasMarini{RefTriangle, 1}) = ((1, 2), (3, 4), (5, 6))
adjust_dofs_during_distribution(::BrezziDouglasMarini{RefTriangle, 1}) = true

function get_direction(::BrezziDouglasMarini{RefTriangle, 1}, shape_nr, cell)
    edge_nr = (shape_nr + 1) ÷ 2
    return get_edge_direction(cell, edge_nr)
end

#####################################
# Nedelec (1st kind), H(curl)       #
#####################################
struct Nedelec{shape, order, vdim} <: VectorInterpolation{vdim, shape, order}
    function Nedelec{shape, order}() where {rdim, shape <: AbstractRefShape{rdim}, order}
        return new{shape, order, rdim}()
    end
end
mapping_type(::Nedelec) = CovariantPiolaMapping()
conformity(::Nedelec) = HcurlConformity()
edgedof_indices(ip::Nedelec) = edgedof_interior_indices(ip)

# 2D refshape (rdim == vdim for Nedelec)
facedof_indices(ip::Nedelec{<:AbstractRefShape{2}}) = (ntuple(i -> i, getnbasefunctions(ip)),)

# RefTriangle, 1st order Lagrange
# https://defelement.org/elements/examples/triangle-nedelec1-lagrange-0.html
function reference_shape_value(ip::Nedelec{RefTriangle, 1}, ξ::Vec{2}, i::Int)
    x, y = ξ
    i == 1 && return Vec(- y, x)
    i == 2 && return Vec(- y, x - 1) # Changed sign, follow Ferrite's sign convention
    i == 3 && return Vec(1 - y, x)
    throw(ArgumentError("no shape function $i for interpolation $ip"))
end

getnbasefunctions(::Nedelec{RefTriangle, 1}) = 3
edgedof_interior_indices(::Nedelec{RefTriangle, 1}) = ((1,), (2,), (3,))
adjust_dofs_during_distribution(::Nedelec{RefTriangle, 1}) = false

function get_direction(::Nedelec{RefTriangle, 1}, shape_nr, cell)
    return get_edge_direction(cell, shape_nr) # shape_nr = edge_nr
end

# RefTriangle, 2nd order Lagrange
# https://defelement.org/elements/examples/triangle-nedelec1-lagrange-1.html
function reference_shape_value(ip::Nedelec{RefTriangle, 2}, ξ::Vec{2}, i::Int)
    x, y = ξ
    # Edge 1
    i == 1 && return Vec(2 * y * (1 - 4 * x), 4 * x * (2 * x - 1))
    i == 2 && return Vec(4 * y * (1 - 2 * y), 2 * x * (4 * y - 1))
    # Edge 2 (flip order and sign compared to defelement)
    i == 3 && return Vec(4 * y * (1 - 2 * y), 8 * x * y - 2 * x - 6 * y + 2)
    i == 4 && return Vec(2 * y * (4 * x + 4 * y - 3), -8 * x^2 - 8 * x * y + 12 * x + 6 * y - 4)
    # Edge 3
    i == 5 && return Vec(8 * x * y - 6 * x + 8 * y^2 - 12 * y + 4, 2 * x * (-4 * x - 4 * y + 3))
    i == 6 && return Vec(
        -8 * x * y + 6 * x + 2 * y - 2, 4 * x * (2 * x - 1)
    )
    # Face
    i == 7 && return Vec(8 * y * (-x - 2 * y + 2), 8 * x * (x + 2 * y - 1))
    i == 8 && return Vec(8 * y * (2 * x + y - 1), 8 * x * (-2 * x - y + 2))
    throw(ArgumentError("no shape function $i for interpolation $ip"))
end

getnbasefunctions(::Nedelec{RefTriangle, 2}) = 8
edgedof_interior_indices(::Nedelec{RefTriangle, 2}) = ((1, 2), (3, 4), (5, 6))
facedof_interior_indices(::Nedelec{RefTriangle, 2}) = ((7, 8),)
adjust_dofs_during_distribution(::Nedelec{RefTriangle, 2}) = true

function get_direction(::Nedelec{RefTriangle, 2}, shape_nr, cell)
    shape_nr > 6 && return 1
    edge_nr = (shape_nr + 1) ÷ 2
    return get_edge_direction(cell, edge_nr)
end

# RefQuadrilateral, 1st order Lagrange
# https://defelement.org/elements/examples/quadrilateral-nedelec1-lagrange-1.html
# Scaled by 1/2 as the reference edge length in Ferrite is length 2, but 1 in DefElement.
function reference_shape_value(ip::Nedelec{RefQuadrilateral, 1}, ξ::Vec{2, T}, i::Int) where {T}
    x, y = ξ
    nil = zero(T)

    i == 1 && return Vec((1 - y) / 4, nil)
    i == 2 && return Vec(nil, (1 + x) / 4)
    i == 3 && return Vec(-(1 + y) / 4, nil) # Changed sign, follow Ferrite's sign convention
    i == 4 && return Vec(nil, -(1 - x) / 4) # Changed sign, follow Ferrite's sign convention
    throw(ArgumentError("no shape function $i for interpolation $ip"))
end

getnbasefunctions(::Nedelec{RefQuadrilateral, 1}) = 4
edgedof_interior_indices(::Nedelec{RefQuadrilateral, 1}) = ((1,), (2,), (3,), (4,))
adjust_dofs_during_distribution(::Nedelec{RefQuadrilateral, 1}) = false

function get_direction(::Nedelec{RefQuadrilateral, 1}, shape_nr, cell)
    return get_edge_direction(cell, shape_nr) # shape_nr = edge_nr
end

# RefTetrahedron, 1st order Lagrange
# https://defelement.org/elements/examples/tetrahedron-nedelec1-lagrange-1.html
function reference_shape_value(ip::Nedelec{RefTetrahedron, 1}, ξ::Vec{3, T}, i::Int) where {T}
    x, y, z = ξ
    nil = zero(T)

    i == 1 && return Vec(1 - y - z, x, x)
    i == 2 && return Vec(-y, x, nil)
    i == 3 && return Vec(-y, x + z - 1, -y) # Changed sign, follow Ferrite's sign convention
    i == 4 && return Vec(z, z, 1 - x - y)
    i == 5 && return Vec(-z, nil, x)
    i == 6 && return Vec(nil, -z, y)
    throw(ArgumentError("no shape function $i for interpolation $ip"))
end

getnbasefunctions(::Nedelec{RefTetrahedron, 1}) = 6
edgedof_interior_indices(::Nedelec{RefTetrahedron, 1}) = ((1,), (2,), (3,), (4,), (5,), (6,))
facedof_indices(::Nedelec{RefTetrahedron, 1}) = ((1, 2, 3), (1, 4, 5), (2, 5, 6), (3, 4, 6))
adjust_dofs_during_distribution(::Nedelec{RefTetrahedron, 1}) = false

function get_direction(::Nedelec{RefTetrahedron, 1}, shape_nr, cell)
    return get_edge_direction(cell, shape_nr) # shape_nr = edge_nr
end

# RefHexahedron, 1st order Lagrange
# https://defelement.org/elements/examples/hexahedron-nedelec1-lagrange-1.html
function reference_shape_value(ip::Nedelec{RefHexahedron, 1}, ξ::Vec{3, T}, i::Int) where {T}
    x, y, z = ξ
    nil = zero(T)

    i == 1 && return Vec((1 - y) * (1 - z) / 8, nil, nil)
    i == 2 && return Vec(nil, (1 + x) * (1 - z) / 8, nil)
    i == 3 && return Vec(-(1 + y) * (1 - z) / 8, nil, nil) # Changed sign, follow Ferrite's sign convention
    i == 4 && return Vec(nil, -(1 - x) * (1 - z) / 8, nil)  # Changed sign, follow Ferrite's sign convention
    i == 5 && return Vec((1 - y) * (1 + z) / 8, nil, nil)
    i == 6 && return Vec(nil, (1 + x) * (1 + z) / 8, nil)
    i == 7 && return Vec(-(1 + y) * (1 + z) / 8, nil, nil)  # Changed sign, follow Ferrite's sign convention
    i == 8 && return Vec(nil, -(1 - x) * (1 + z) / 8, nil) # Changed sign, follow Ferrite's sign convention
    i == 9 && return Vec(nil, nil, (1 - x) * (1 - y) / 8)
    i == 10 && return Vec(nil, nil, (1 + x) * (1 - y) / 8)
    i == 11 && return Vec(nil, nil, (1 + x) * (1 + y) / 8)
    i == 12 && return Vec(nil, nil, (1 - x) * (1 + y) / 8)
    throw(ArgumentError("no shape function $i for interpolation $ip"))
end

getnbasefunctions(::Nedelec{RefHexahedron, 1}) = 12
edgedof_interior_indices(::Nedelec{RefHexahedron, 1}) = ((1,), (2,), (3,), (4,), (5,), (6,), (7,), (8,), (9,), (10,), (11,), (12,))
facedof_indices(::Nedelec{RefHexahedron, 1}) = ((1, 2, 3, 4), (1, 5, 9, 10), (2, 6, 10, 11), (3, 7, 11, 12), (4, 8, 9, 12), (5, 6, 7, 8))
adjust_dofs_during_distribution(::Nedelec{RefHexahedron, 1}) = false

function get_direction(::Nedelec{RefHexahedron, 1}, shape_nr, cell)
    return get_edge_direction(cell, shape_nr) # shape_nr = edge_nr
end
