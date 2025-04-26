"""
    Interpolation{ref_shape, order}()

Abstract type for interpolations defined on `ref_shape`
(see [`AbstractRefShape`](@ref)).
`order` corresponds to the order of the interpolation.
The interpolation is used to define shape functions to interpolate
a function between nodes.

The following interpolations are implemented:

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
    is_discontinuous::Bool
end
function InterpolationInfo(interpolation::Interpolation{shape}, n_copies) where {rdim, shape <: AbstractRefShape{rdim}}
    info = InterpolationInfo(
        [length(i) for i in vertexdof_indices(interpolation)],
        [length(i) for i in edgedof_interior_indices(interpolation)],
        [length(i) for i in facedof_interior_indices(interpolation)],
        length(volumedof_interior_indices(interpolation)),
        rdim,
        adjust_dofs_during_distribution(interpolation),
        n_copies,
        is_discontinuous(interpolation)
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
    is_discontinuous(::Interpolation)
    is_discontinuous(::Type{<:Interpolation})

Checks whether the interpolation is discontinuous (i.e. `DiscontinuousLagrange`)
"""
is_discontinuous(ip::Interpolation) = is_discontinuous(typeof(ip))
is_discontinuous(::Type{<:Interpolation}) = false

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

function get_edge_direction(edgenodes::NTuple{2, Int})
    positive = edgenodes[2] > edgenodes[1]
    return ifelse(positive, 1, -1)
end

function get_face_direction(facenodes::NTuple{N, Int}) where {N}
    N > 2 || throw(ArgumentError("A face must have at least 3 nodes"))
    min_idx = argmin(facenodes)
    if min_idx == 1
        positive = facenodes[2] < facenodes[end]
    elseif min_idx == length(facenodes)
        positive = facenodes[1] < facenodes[end - 1]
    else
        positive = facenodes[min_idx + 1] < facenodes[min_idx - 1]
    end
    return ifelse(positive, 1, -1)
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

is_discontinuous(::Type{<:VectorizedInterpolation{<:Any, <:Any, <:Any, ip}}) where {ip} = is_discontinuous(ip)

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


# Scalar interpolations
include("Interpolations/Lagrange.jl")
include("Interpolations/BubbleLagrange.jl")
include("Interpolations/DiscontinuousLagrange.jl")

include("Interpolations/Serendipity.jl")

include("Interpolations/CrouzeixRaviart.jl")
include("Interpolations/RannacherTurek.jl")

# Vector interpolations
## H(div) conforming
include("Interpolations/RaviartThomas.jl")
include("Interpolations/BrezziDouglasMarini.jl")

## H(curl) conforming
include("Interpolations/Nedelec1.jl")
