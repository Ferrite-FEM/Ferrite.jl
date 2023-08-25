"""
    Interpolation{ref_shape, order}()

Abstract type for interpolations defined on `ref_shape`
(see [`AbstractRefShape`](@ref)).
`order` corresponds to the order of the interpolation.
The interpolation is used to define shape functions to interpolate
a function between nodes.

The following interpolations are implemented with fixed order:

* `Lagrange{RefLine,1}`
* `Lagrange{RefLine,2}`
* `Lagrange{RefQuadrilateral,1}`
* `Lagrange{RefQuadrilateral,2}`
* `Lagrange{RefQuadrilateral,3}`
* `Lagrange{RefTriangle,1}`
* `Lagrange{RefTriangle,2}`
* `BubbleEnrichedLagrange{RefTriangle,1}`
* `CrouzeixRaviart{RefTriangle, 1}`
* `Lagrange{RefHexahedron,1}`
* `Lagrange{RefHexahedron,2}`
* `Lagrange{RefTetrahedron,1}`
* `Lagrange{RefTetrahedron,2}`
* `Lagrange{RefPrism,1}`
* `Lagrange{RefPrism,2}`
* `Lagrange{RefPyramid,1}`
* `Lagrange{RefPyramid,2}`
* `Serendipity{RefQuadrilateral,2}`
* `Serendipity{RefHexahedron,2}`

The following interpolations are implemented with arbitrary order:

* `Lagrange{RefTriangle,order}`
* `ArbitraryOrderLagrange{RefHypercube,order}`
* `ArbitraryOrderDiscontinuousLagrange{RefHypercube,order}`

The following interpolations are implemented with arbitrary basis coordinates:

* `ArbitraryOrderLagrange{RefHypercube,order}`
* `ArbitraryOrderDiscontinuousLagrange{RefHypercube,order}`

# Examples
```jldoctest
julia> ip = Lagrange{RefTriangle, 2}()
Lagrange{RefTriangle, 2}()

julia> getnbasefunctions(ip)
6
```
"""
abstract type Interpolation{shape #=<: AbstractRefShape=#, order, unused} end

const InterpolationByDim{dim} = Interpolation{<:AbstractRefShape{dim}}

abstract type ScalarInterpolation{      refshape, order} <: Interpolation{refshape, order, Nothing} end
abstract type VectorInterpolation{vdim, refshape, order} <: Interpolation{refshape, order, Nothing} end

# Number of components for the interpolation.
n_components(::ScalarInterpolation)                    = 1
n_components(::VectorInterpolation{vdim}) where {vdim} = vdim
# Number of components that are allowed to prescribe in e.g. Dirichlet BC
n_dbc_components(ip::Interpolation) = n_components(ip)
# n_dbc_components(::Union{RaviartThomas,Nedelec}) = 1

# TODO: Remove: this is a hotfix to apply constraints to embedded elements.
edges(ip::InterpolationByDim{2}) = faces(ip)
edgedof_indices(ip::InterpolationByDim{2}) = facedof_indices(ip)
edgedof_interior_indices(ip::InterpolationByDim{2}) = facedof_interior_indices(ip)
facedof_indices(ip::InterpolationByDim{1}) = vertexdof_indices(ip)

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
    ncelldofs::Int
    reference_dim::Int
    adjust_during_distribution::Bool
    n_copies::Int
    is_discontinuous::Bool
    function InterpolationInfo(interpolation::InterpolationByDim{3})
        n_copies = 1
        if interpolation isa VectorizedInterpolation
            n_copies = get_n_copies(interpolation)
            interpolation = interpolation.ip
        end
        new(
            [length(i) for i ∈ vertexdof_indices(interpolation)],
            [length(i) for i ∈ edgedof_interior_indices(interpolation)],
            [length(i) for i ∈ facedof_interior_indices(interpolation)],
            length(celldof_interior_indices(interpolation)),
            3,
            adjust_dofs_during_distribution(interpolation),
            n_copies,
            is_discontinuous(interpolation)
        )
    end
    function InterpolationInfo(interpolation::InterpolationByDim{2})
        n_copies = 1
        if interpolation isa VectorizedInterpolation
            n_copies = get_n_copies(interpolation)
            interpolation = interpolation.ip
        end
        new(
            [length(i) for i ∈ vertexdof_indices(interpolation)],
            Int[],
            [length(i) for i ∈ facedof_interior_indices(interpolation)],
            length(celldof_interior_indices(interpolation)),
            2,
            adjust_dofs_during_distribution(interpolation),
            n_copies,
            is_discontinuous(interpolation)
        )
    end
    function InterpolationInfo(interpolation::InterpolationByDim{1})
        n_copies = 1
        if interpolation isa VectorizedInterpolation
            n_copies = get_n_copies(interpolation)
            interpolation = interpolation.ip
        end
        new(
            [length(i) for i ∈ vertexdof_indices(interpolation)],
            Int[],
            Int[],
            length(celldof_interior_indices(interpolation)),
            1,
            adjust_dofs_during_distribution(interpolation),
            n_copies,
            is_discontinuous(interpolation)
        )
    end
end

# Some redundant information about the geometry of the reference cells.
nfaces(::Interpolation{RefHypercube{dim}}) where {dim} = 2*dim
nfaces(::Interpolation{RefTriangle}) = 3
nfaces(::Interpolation{RefTetrahedron}) = 4
nfaces(::Interpolation{RefPrism}) = 5
nfaces(::Interpolation{RefPyramid}) = 5

nedges(::Interpolation{RefLine}) = 0
nedges(::Interpolation{RefQuadrilateral}) = 0
nedges(::Interpolation{RefHexahedron}) = 12
nedges(::Interpolation{RefTriangle}) = 0
nedges(::Interpolation{RefTetrahedron}) = 6
nedges(::Interpolation{RefPrism}) = 9
nedges(::Interpolation{RefPyramid}) =  8

nvertices(::Interpolation{RefHypercube{dim}}) where {dim} = 2^dim
nvertices(::Interpolation{RefTriangle}) = 3
nvertices(::Interpolation{RefTetrahedron}) = 4
nvertices(::Interpolation{RefPrism}) = 6
nvertices(::Interpolation{RefPyramid}) = 5

Base.copy(ip::Interpolation) = ip

"""
    Ferrite.getdim(::Interpolation)

Return the dimension of the reference element for a given interpolation.
"""
@inline getdim(::Interpolation{shape}) where {dim, shape <: AbstractRefShape{dim}} = dim

"""
    Ferrite.getrefshape(::Interpolation)::AbstractRefShape

Return the reference element shape of the interpolation.
"""
@inline getrefshape(::Interpolation{shape}) where {shape} = shape

"""
    Ferrite.getorder(::Interpolation)

Return order of the interpolation.
"""
@inline getorder(::Interpolation{shape,order}) where {shape,order} = order


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
    shape_value(ip::Interpolation, ξ::Vec, i::Int)

Evaluate the value of the `i`th shape function of the interpolation `ip`
at a point `ξ` on the reference element. The index `i` must
match the index in [`vertices(::Interpolation)`](@ref), [`faces(::Interpolation)`](@ref) and
[`edges(::Interpolation)`](@ref).

For nodal interpolations the indices also must match the
indices of [`reference_coordinates(::Interpolation)`](@ref).
"""
shape_value(ip::Interpolation, ξ::Vec, i::Int)

"""
    shape_gradient(ip::Interpolation, ξ::Vec, i::Int)

Evaluate the gradient of the `i`th shape function of the interpolation `ip` in
reference coordinate `ξ`.
"""
function shape_gradient(ip::Interpolation, ξ::Vec, i::Int)
    return Tensors.gradient(x -> shape_value(ip, x, i), ξ)
end

"""
    shape_gradient_and_value(ip::Interpolation, ξ::Vec, i::Int)

Optimized version combining the evaluation [`Ferrite.shape_value(::Interpolation)`](@ref)
and [`Ferrite.shape_gradient(::Interpolation)`](@ref).
"""
function shape_gradient_and_value(ip::Interpolation, ξ::Vec, i::Int)
    return gradient(x -> shape_value(ip, x, i), ξ, :all)
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
    celldof_interior_indices(ip::Interpolation)

Tuple containing the dof indices associated with the interior of the cell.

!!! note
    The dofs appearing in the tuple must be continuous and increasing! Celldofs are
    enumerated last.
"""
celldof_interior_indices(::Interpolation) = ()

# Some helpers to skip boilerplate
edgedof_indices(ip::InterpolationByDim{3}) = ntuple(_ -> (), nedges(ip))
edgedof_interior_indices(ip::InterpolationByDim{3}) = ntuple(_ -> (), nedges(ip))
facedof_indices(ip::Union{InterpolationByDim{2}, InterpolationByDim{3}}) =  ntuple(_ -> (), nfaces(ip))
facedof_interior_indices(ip::Union{InterpolationByDim{2}, InterpolationByDim{3}}) =  ntuple(_ -> (), nfaces(ip))

"""
    boundarydof_indices(::Type{<:BoundaryIndex})

Helper function to generically dispatch on the correct dof sets of a boundary entity.
"""
boundarydof_indices(::Type{<:BoundaryIndex})

boundarydof_indices(::Type{FaceIndex}) = Ferrite.facedof_indices
boundarydof_indices(::Type{EdgeIndex}) = Ferrite.edgedof_indices
boundarydof_indices(::Type{VertexIndex}) = Ferrite.vertexdof_indices

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

dirichlet_boundarydof_indices(::Type{FaceIndex}) = Ferrite.dirichlet_facedof_indices
dirichlet_boundarydof_indices(::Type{EdgeIndex}) = Ferrite.dirichlet_edgedof_indices
dirichlet_boundarydof_indices(::Type{VertexIndex}) = Ferrite.dirichlet_vertexdof_indices

#########################
# DiscontinuousLagrange #
#########################
# TODO generalize to arbitrary basis positionings.
"""
Piecewise discontinuous Lagrange interpolation using equidistant basis.

To use arbitrary order and basis positionings consider using:
* [`ArbitraryOrderDiscontinuousLagrange{refshape,order}(basis)`](@ref) for RefLine and RefHypercube
* [`DiscontinuousLagrange{RefTriangle,order}`](@ref) for RefTriangle with equidistant basis positionings
"""
struct DiscontinuousLagrange{shape, order, unused} <: ScalarInterpolation{shape, order}
    function DiscontinuousLagrange{shape, order}() where {shape <: AbstractRefShape, order}
        new{shape, order, Nothing}()
    end
end

adjust_dofs_during_distribution(::DiscontinuousLagrange) = false

getlowerorder(::DiscontinuousLagrange{shape,order}) where {shape,order} = DiscontinuousLagrange{shape,order-1}()

getnbasefunctions(::DiscontinuousLagrange{shape,order}) where {shape,order} = getnbasefunctions(Lagrange{shape,order}())
getnbasefunctions(::DiscontinuousLagrange{shape,0}) where {shape} = 1

# This just moves all dofs into the interior of the element.
celldof_interior_indices(ip::DiscontinuousLagrange) = ntuple(i->i, getnbasefunctions(ip))

# Mirror the Lagrange element for now to avoid repeating.
dirichlet_facedof_indices(ip::DiscontinuousLagrange{shape, order}) where {shape, order} = dirichlet_facedof_indices(Lagrange{shape, order}())
dirichlet_edgedof_indices(ip::DiscontinuousLagrange{shape, order}) where {shape, order} = dirichlet_edgedof_indices(Lagrange{shape, order}())
dirichlet_vertexdof_indices(ip::DiscontinuousLagrange{shape, order}) where {shape, order} = dirichlet_vertexdof_indices(Lagrange{shape, order}())

# Mirror the Lagrange element for now.
function reference_coordinates(ip::DiscontinuousLagrange{shape, order}) where {shape, order}
    return reference_coordinates(Lagrange{shape,order}())
end
function shape_value(::DiscontinuousLagrange{shape, order}, ξ::Vec{dim}, i::Int) where {dim, shape <: AbstractRefShape{dim}, order}
    return shape_value(Lagrange{shape, order}(), ξ, i)
end

# Excepting the L0 element.
function reference_coordinates(ip::DiscontinuousLagrange{RefHypercube{dim},0}) where dim
    return [Vec{dim, Float64}(ntuple(x->0.0, dim))]
end

function reference_coordinates(ip::DiscontinuousLagrange{RefTriangle,0})
    return [Vec{2,Float64}((1/3,1/3))]
end

function reference_coordinates(ip::DiscontinuousLagrange{RefTetrahedron,0})
   return [Vec{3,Float64}((1/4,1/4,1/4))]
end

function shape_value(ip::DiscontinuousLagrange{shape, 0}, ::Vec{dim, T}, i::Int) where {dim, shape <: AbstractRefShape{dim}, T}
    i > 1 && throw(ArgumentError("no shape function $i for interpolation $ip"))
    return one(T)
end

is_discontinuous(::Type{<:DiscontinuousLagrange}) = true

############
# Lagrange #
############
"""
Continuous Lagrange interpolation using equidistant basis.

To use arbitrary order and basis positionings consider using:
* [`ArbitraryOrderLagrange{refshape,order}(basis)`](@ref) for RefLine and RefHypercube
* [`Lagrange{RefTriangle,order}`](@ref) for RefTriangle with equidistant basis positionings
"""
struct Lagrange{shape, order, unused} <: ScalarInterpolation{shape, order}
    function Lagrange{shape, order}() where {shape <: AbstractRefShape, order}
        new{shape, order, Nothing}()
    end
end

adjust_dofs_during_distribution(::Lagrange) = true
adjust_dofs_during_distribution(::Lagrange{<:Any, 2}) = false
adjust_dofs_during_distribution(::Lagrange{<:Any, 1}) = false

# Vertices for all Lagrange interpolations are the same
vertexdof_indices(::Lagrange{RefLine}) = ((1,),(2,))
vertexdof_indices(::Lagrange{RefQuadrilateral}) = ((1,),(2,),(3,),(4,))
vertexdof_indices(::Lagrange{RefHexahedron}) = ((1,),(2,),(3,),(4,),(5,),(6,),(7,),(8,))
vertexdof_indices(::Lagrange{RefTriangle}) = ((1,),(2,),(3,))
vertexdof_indices(::Lagrange{RefTetrahedron}) = ((1,),(2,),(3,),(4,))
vertexdof_indices(::Lagrange{RefPrism}) = ((1,), (2,), (3,), (4,), (5,), (6,))
vertexdof_indices(::Lagrange{RefPyramid}) = ((1,), (2,), (3,), (4,), (5,),)

getlowerorder(::Lagrange{shape,order}) where {shape,order} = Lagrange{shape,order-1}()
getlowerorder(::Lagrange{shape,1}) where {shape} = DiscontinuousLagrange{shape,0}()

############################
# Lagrange RefLine order 1 #
############################
getnbasefunctions(::Lagrange{RefLine,1}) = 2

function reference_coordinates(::Lagrange{RefLine,1})
    return [Vec{1, Float64}((-1.0,)),
            Vec{1, Float64}(( 1.0,))]
end

function shape_value(ip::Lagrange{RefLine, 1}, ξ::Vec{1}, i::Int)
    ξ_x = ξ[1]
    i == 1 && return (1 - ξ_x) * 0.5
    i == 2 && return (1 + ξ_x) * 0.5
    throw(ArgumentError("no shape function $i for interpolation $ip"))
end

############################
# Lagrange RefLine order 2 #
############################
getnbasefunctions(::Lagrange{RefLine,2}) = 3

facedof_indices(::Lagrange{RefLine,2}) = ((1,), (2,))
celldof_interior_indices(::Lagrange{RefLine,2}) = (3,)

function reference_coordinates(::Lagrange{RefLine,2})
    return [Vec{1, Float64}((-1.0,)),
            Vec{1, Float64}(( 1.0,)),
            Vec{1, Float64}(( 0.0,))]
end

function shape_value(ip::Lagrange{RefLine, 2}, ξ::Vec{1}, i::Int)
    ξ_x = ξ[1]
    i == 1 && return ξ_x * (ξ_x - 1) * 0.5
    i == 2 && return ξ_x * (ξ_x + 1) * 0.5
    i == 3 && return 1 - ξ_x^2
    throw(ArgumentError("no shape function $i for interpolation $ip"))
end

#####################################
# Lagrange RefQuadrilateral order 1 #
#####################################
getnbasefunctions(::Lagrange{RefQuadrilateral,1}) = 4

facedof_indices(::Lagrange{RefQuadrilateral,1}) = ((1,2), (2,3), (3,4), (4,1))

function reference_coordinates(::Lagrange{RefQuadrilateral,1})
    return [Vec{2, Float64}((-1.0, -1.0)),
            Vec{2, Float64}(( 1.0, -1.0)),
            Vec{2, Float64}(( 1.0,  1.0,)),
            Vec{2, Float64}((-1.0,  1.0,))]
end

function shape_value(ip::Lagrange{RefQuadrilateral, 1}, ξ::Vec{2}, i::Int)
    ξ_x = ξ[1]
    ξ_y = ξ[2]
    i == 1 && return (1 - ξ_x) * (1 - ξ_y) * 0.25
    i == 2 && return (1 + ξ_x) * (1 - ξ_y) * 0.25
    i == 3 && return (1 + ξ_x) * (1 + ξ_y) * 0.25
    i == 4 && return (1 - ξ_x) * (1 + ξ_y) * 0.25
    throw(ArgumentError("no shape function $i for interpolation $ip"))
end

#####################################
# Lagrange RefQuadrilateral order 2 #
#####################################
getnbasefunctions(::Lagrange{RefQuadrilateral,2}) = 9

facedof_indices(::Lagrange{RefQuadrilateral,2}) = ((1,2, 5), (2,3, 6), (3,4, 7), (4,1, 8))
facedof_interior_indices(::Lagrange{RefQuadrilateral,2}) = ((5,), (6,), (7,), (8,))
celldof_interior_indices(::Lagrange{RefQuadrilateral,2}) = (9,)

function reference_coordinates(::Lagrange{RefQuadrilateral,2})
    return [Vec{2, Float64}((-1.0, -1.0)),
            Vec{2, Float64}(( 1.0, -1.0)),
            Vec{2, Float64}(( 1.0,  1.0)),
            Vec{2, Float64}((-1.0,  1.0)),
            Vec{2, Float64}(( 0.0, -1.0)),
            Vec{2, Float64}(( 1.0,  0.0)),
            Vec{2, Float64}(( 0.0,  1.0)),
            Vec{2, Float64}((-1.0,  0.0)),
            Vec{2, Float64}(( 0.0,  0.0))]
end

function shape_value(ip::Lagrange{RefQuadrilateral, 2}, ξ::Vec{2}, i::Int)
    ξ_x = ξ[1]
    ξ_y = ξ[2]
    i == 1 && return (ξ_x^2 - ξ_x) * (ξ_y^2 - ξ_y) * 0.25
    i == 2 && return (ξ_x^2 + ξ_x) * (ξ_y^2 - ξ_y) * 0.25
    i == 3 && return (ξ_x^2 + ξ_x) * (ξ_y^2 + ξ_y) * 0.25
    i == 4 && return (ξ_x^2 - ξ_x) * (ξ_y^2 + ξ_y) * 0.25
    i == 5 && return (1 - ξ_x^2) * (ξ_y^2 - ξ_y) * 0.5
    i == 6 && return (ξ_x^2 + ξ_x) * (1 - ξ_y^2) * 0.5
    i == 7 && return (1 - ξ_x^2) * (ξ_y^2 + ξ_y) * 0.5
    i == 8 && return (ξ_x^2 - ξ_x) * (1 - ξ_y^2) * 0.5
    i == 9 && return (1 - ξ_x^2) * (1 - ξ_y^2)
    throw(ArgumentError("no shape function $i for interpolation $ip"))
end

#####################################
# Lagrange RefQuadrilateral order 3 #
#####################################
getnbasefunctions(::Lagrange{RefQuadrilateral, 3}) = 16

facedof_indices(::Lagrange{RefQuadrilateral, 3}) = ((1,2, 5,6), (2,3, 7,8), (3,4, 9,10), (4,1, 11,12))
facedof_interior_indices(::Lagrange{RefQuadrilateral, 3}) = ((5,6), (7,8), (9,10), (11,12))
celldof_interior_indices(::Lagrange{RefQuadrilateral, 3}) = (13,14,15,16)

function reference_coordinates(::Lagrange{RefQuadrilateral, 3})
    return [Vec{2, Float64}((-1.0, -1.0)),
            Vec{2, Float64}(( 1.0, -1.0)),
            Vec{2, Float64}(( 1.0,  1.0)),
            Vec{2, Float64}((-1.0,  1.0)),
            Vec{2, Float64}((-1/3, -1.0)),
            Vec{2, Float64}(( 1/3, -1.0)),
            Vec{2, Float64}(( 1.0, -1/3)),
            Vec{2, Float64}(( 1.0,  1/3)),
            Vec{2, Float64}(( 1/3,  1.0)),
            Vec{2, Float64}((-1/3,  1.0)),
            Vec{2, Float64}((-1.0,  1/3)),
            Vec{2, Float64}((-1.0, -1/3)),
            Vec{2, Float64}((-1/3, -1/3)),
            Vec{2, Float64}(( 1/3, -1/3)),
            Vec{2, Float64}((-1/3,  1/3)),
            Vec{2, Float64}(( 1/3,  1/3))]
end

function shape_value(ip::Lagrange{RefQuadrilateral, 3}, ξ::Vec{2}, i::Int)
    # See https://defelement.com/elements/examples/quadrilateral-Q-3.html
    # Transform domain from [-1, 1] × [-1, 1] to [0, 1] × [0, 1]
    ξ_x = ξ[1]*0.5 + 0.5
    ξ_y = ξ[2]*0.5 + 0.5
    i ==  1 && return (81*ξ_x^3*ξ_y^3)/4 - (81*ξ_x^3*ξ_y^2)/2 + (99*ξ_x^3*ξ_y)/4 - (9*ξ_x^3)/2 - (81*ξ_x^2*ξ_y^3)/2 + (81*ξ_x^2*ξ_y^2) - (99*ξ_x^2*ξ_y)/2 + (9*ξ_x^2) + (99*ξ_x*ξ_y^3)/4 - (99*ξ_x*ξ_y^2)/2 + (121*ξ_x*ξ_y)/4 - (11*ξ_x)/2 - (9*ξ_y^3)/2 + 9*ξ_y^2 - (11*ξ_y)/2 + 1
    i ==  2 && return (ξ_x*( - 81*ξ_x^2*ξ_y^3 + 162*ξ_x^2*ξ_y^2 - 99*ξ_x^2*ξ_y + 18*ξ_x^2 + 81*ξ_x*ξ_y^3 - 162*ξ_x*ξ_y^2 + 99*ξ_x*ξ_y - 18*ξ_x - 18*ξ_y^3 + 36*ξ_y^2 - 22*ξ_y + 4))/4
    i ==  4 && return (ξ_y*( - 81*ξ_x^3*ξ_y^2 + 81*ξ_x^3*ξ_y - 18*ξ_x^3 + 162*ξ_x^2*ξ_y^2 - 162*ξ_x^2*ξ_y + 36*ξ_x^2 - 99*ξ_x*ξ_y^2 + 99*ξ_x*ξ_y - 22*ξ_x + 18*ξ_y^2 - 18*ξ_y + 4))/4
    i ==  3 && return (ξ_x*ξ_y*(81*ξ_x^2*ξ_y^2 - 81*ξ_x^2*ξ_y + 18*ξ_x^2 - 81*ξ_x*ξ_y^2 + 81*ξ_x*ξ_y - 18*ξ_x + 18*ξ_y^2 - 18*ξ_y + 4))/4
    i ==  5 && return (9*ξ_x*( - 27*ξ_x^2*ξ_y^3 + 54*ξ_x^2*ξ_y^2 - 33*ξ_x^2*ξ_y + 6*ξ_x^2 + 45*ξ_x*ξ_y^3 - 90*ξ_x*ξ_y^2 + 55*ξ_x*ξ_y - 10*ξ_x - 18*ξ_y^3 + 36*ξ_y^2 - 22*ξ_y + 4))/4
    i ==  6 && return (9*ξ_x*(27*ξ_x^2*ξ_y^3 - 54*ξ_x^2*ξ_y^2 + 33*ξ_x^2*ξ_y - 6*ξ_x^2 - 36*ξ_x*ξ_y^3 + 72*ξ_x*ξ_y^2 - 44*ξ_x*ξ_y + 8*ξ_x + 9*ξ_y^3 - 18*ξ_y^2 + 11*ξ_y - 2))/4
    i ==  12 && return (9*ξ_y*( - 27*ξ_x^3*ξ_y^2 + 45*ξ_x^3*ξ_y - 18*ξ_x^3 + 54*ξ_x^2*ξ_y^2 - 90*ξ_x^2*ξ_y + 36*ξ_x^2 - 33*ξ_x*ξ_y^2 + 55*ξ_x*ξ_y - 22*ξ_x + 6*ξ_y^2 - 10*ξ_y + 4))/4
    i ==  11 && return (9*ξ_y*(27*ξ_x^3*ξ_y^2 - 36*ξ_x^3*ξ_y + 9*ξ_x^3 - 54*ξ_x^2*ξ_y^2 + 72*ξ_x^2*ξ_y - 18*ξ_x^2 + 33*ξ_x*ξ_y^2 - 44*ξ_x*ξ_y + 11*ξ_x - 6*ξ_y^2 + 8*ξ_y - 2))/4
    i ==  7 && return (9*ξ_x*ξ_y*(27*ξ_x^2*ξ_y^2 - 45*ξ_x^2*ξ_y + 18*ξ_x^2 - 27*ξ_x*ξ_y^2 + 45*ξ_x*ξ_y - 18*ξ_x + 6*ξ_y^2 - 10*ξ_y + 4))/4
    i == 8 && return (9*ξ_x*ξ_y*( - 27*ξ_x^2*ξ_y^2 + 36*ξ_x^2*ξ_y - 9*ξ_x^2 + 27*ξ_x*ξ_y^2 - 36*ξ_x*ξ_y + 9*ξ_x - 6*ξ_y^2 + 8*ξ_y - 2))/4
    i == 10 && return (9*ξ_x*ξ_y*(27*ξ_x^2*ξ_y^2 - 27*ξ_x^2*ξ_y + 6*ξ_x^2 - 45*ξ_x*ξ_y^2 + 45*ξ_x*ξ_y - 10*ξ_x + 18*ξ_y^2 - 18*ξ_y + 4))/4
    i == 9 && return (9*ξ_x*ξ_y*( - 27*ξ_x^2*ξ_y^2 + 27*ξ_x^2*ξ_y - 6*ξ_x^2 + 36*ξ_x*ξ_y^2 - 36*ξ_x*ξ_y + 8*ξ_x - 9*ξ_y^2 + 9*ξ_y - 2))/4
    i == 13 && return (81*ξ_x*ξ_y*(9*ξ_x^2*ξ_y^2 - 15*ξ_x^2*ξ_y + 6*ξ_x^2 - 15*ξ_x*ξ_y^2 + 25*ξ_x*ξ_y - 10*ξ_x + 6*ξ_y^2 - 10*ξ_y + 4))/4
    i == 14 && return (81*ξ_x*ξ_y*( - 9*ξ_x^2*ξ_y^2 + 15*ξ_x^2*ξ_y - 6*ξ_x^2 + 12*ξ_x*ξ_y^2 - 20*ξ_x*ξ_y + 8*ξ_x - 3*ξ_y^2 + 5*ξ_y - 2))/4
    i == 15 && return (81*ξ_x*ξ_y*( - 9*ξ_x^2*ξ_y^2 + 12*ξ_x^2*ξ_y - 3*ξ_x^2 + 15*ξ_x*ξ_y^2 - 20*ξ_x*ξ_y + 5*ξ_x - 6*ξ_y^2 + 8*ξ_y - 2))/4
    i == 16 && return (81*ξ_x*ξ_y*(9*ξ_x^2*ξ_y^2 - 12*ξ_x^2*ξ_y + 3*ξ_x^2 - 12*ξ_x*ξ_y^2 + 16*ξ_x*ξ_y - 4*ξ_x + 3*ξ_y^2 - 4*ξ_y + 1))/4
    throw(ArgumentError("no shape function $i for interpolation $ip"))
end

################################
# Lagrange RefTriangle order 1 #
################################
getnbasefunctions(::Lagrange{RefTriangle,1}) = 3

facedof_indices(::Lagrange{RefTriangle,1}) = ((1,2), (2,3), (3,1))

function reference_coordinates(::Lagrange{RefTriangle,1})
    return [Vec{2, Float64}((1.0, 0.0)),
            Vec{2, Float64}((0.0, 1.0)),
            Vec{2, Float64}((0.0, 0.0))]
end

function shape_value(ip::Lagrange{RefTriangle, 1}, ξ::Vec{2}, i::Int)
    ξ_x = ξ[1]
    ξ_y = ξ[2]
    i == 1 && return ξ_x
    i == 2 && return ξ_y
    i == 3 && return 1. - ξ_x - ξ_y
    throw(ArgumentError("no shape function $i for interpolation $ip"))
end

################################
# Lagrange RefTriangle order 2 #
################################
getnbasefunctions(::Lagrange{RefTriangle,2}) = 6

facedof_indices(::Lagrange{RefTriangle,2}) = ((1,2,4), (2,3,5), (3,1,6))
facedof_interior_indices(::Lagrange{RefTriangle,2}) = ((4,), (5,), (6,))

function reference_coordinates(::Lagrange{RefTriangle,2})
    return [Vec{2, Float64}((1.0, 0.0)),
            Vec{2, Float64}((0.0, 1.0)),
            Vec{2, Float64}((0.0, 0.0)),
            Vec{2, Float64}((0.5, 0.5)),
            Vec{2, Float64}((0.0, 0.5)),
            Vec{2, Float64}((0.5, 0.0))]
end

function shape_value(ip::Lagrange{RefTriangle, 2}, ξ::Vec{2}, i::Int)
    ξ_x = ξ[1]
    ξ_y = ξ[2]
    γ = 1. - ξ_x - ξ_y
    i == 1 && return ξ_x * (2ξ_x - 1)
    i == 2 && return ξ_y * (2ξ_y - 1)
    i == 3 && return γ * (2γ - 1)
    i == 4 && return 4ξ_x * ξ_y
    i == 5 && return 4ξ_y * γ
    i == 6 && return 4ξ_x * γ
    throw(ArgumentError("no shape function $i for interpolation $ip"))
end

########################################
# Lagrange RefTriangle abritrary order #
########################################
# see https://getfem.readthedocs.io/en/latest/userdoc/appendixA.html

function getnbasefunctions(::Lagrange{RefTriangle,N}) where N
    return (N + 1) * (N + 2) ÷ 2
end

# Permutation to switch numbering to Ferrite ordering
function getPermLagrangeTri(order::Int)
    result = Array{Int, 1}(undef, (order + 1) * (order + 2) ÷ 2)
    result[1:3] .= (order + 1, (order + 1) * (order + 2) ÷ 2, 1)
    idx = 4
    for i in order:-1:2 # Face 1
        result[idx] = sum(i:order+1)
        idx += 1
    end
    for i in 3:(order+1) # Face 2
        result[idx] = sum(i:order+1)+1
        idx += 1
    end
    for i in 2:order # Face 3
        result[idx] = i
        idx += 1
    end
    for j in (order+1):-1:4 # Interior
        for i in sum(j:order+1)+2 : sum(j:order+1)+j-2
            result[idx] = i
            idx += 1
        end
    end
    return result
end

function facedof_interior_indices(::Lagrange{RefTriangle,order}) where order
    order == 1 && return ((),(),()) # Workaround for nightly test, will see if it works
    return (SVector{order-1}((i+3 for i in 1:order-1)),
        SVector{order-1}((i+order+2 for i in 1:order-1)),
        SVector{order-1}((i+2*order+1 for i in 1:order-1)))
end

function facedof_indices(ip::Lagrange{RefTriangle,order}) where order
    interior = facedof_interior_indices(ip)
    face1 = SVector{order + 1}((i == 1 ? 1 : i == 2 ? 2 : interior[1][i-2] for i in 1:order+1))    
    face2 = SVector{order + 1}((i == 1 ? 2 : i == 2 ? 3 : interior[2][i-2] for i in 1:order+1))    
    face3 = SVector{order + 1}((i == 1 ? 3 : i == 2 ? 1 : interior[3][i-2] for i in 1:order+1))    
    return (face1, face2, face3)
end

function celldof_interior_indices(ip::Lagrange{RefTriangle,order}) where order
    ncellintdofs = (order - 2) * (order - 1) ÷ 2
    totaldofs = getnbasefunctions(ip)
    return SVector{ncellintdofs}((totaldofs-ncellintdofs+i for i in 1:ncellintdofs))
end

function reference_coordinates(::Lagrange{RefTriangle,order}) where order
    coordpts = Vector{Vec{2, Float64}}()
    for k = 0:order
        for l = 0:(order - k)
            push!(coordpts, Vec{2, Float64}((l / order, k / order)))
        end
    end
    return permute!(coordpts, getPermLagrangeTri(order))
end

function shape_value(ip::Lagrange{RefTriangle,order}, ξ::Vec{2}, i::Int) where order
    if !(0 < i <= getnbasefunctions(ip))
        throw(ArgumentError("no shape function $i for interpolation $ip"))
    end
    i = getPermLagrangeTri(order)[i]
    ξ_x = ξ[1]
    ξ_y = ξ[2]
    i1, i2, i3 = _numlin_basis2D(i, order)
    val = one(ξ_y)
    i1 ≥ 1 && (val *= prod((order - order * (ξ_x + ξ_y ) - j) / (j + 1) for j = 0:(i1 - 1)))
    i2 ≥ 1 && (val *= prod((order * ξ_x - j) / (j + 1) for j = 0:(i2 - 1)))
    i3 ≥ 1 && (val *= prod((order * ξ_y - j) / (j + 1) for j = 0:(i3 - 1)))
    return val
end

function _numlin_basis2D(i, order)
    c, j1, j2, j3 = 0, 0, 0, 0
    for k = 0:order
        if i <= c + (order + 1 - k)
            j2 = i - c - 1
            break
        else
            j3 += 1
            c += order + 1 - k
        end
    end
    j1 = order - j2 -j3
    return j1, j2, j3
end

###################################
# Lagrange RefTetrahedron order 1 #
###################################
getnbasefunctions(::Lagrange{RefTetrahedron,1}) = 4

facedof_indices(::Lagrange{RefTetrahedron,1}) = ((1,3,2), (1,2,4), (2,3,4), (1,4,3))
edgedof_indices(::Lagrange{RefTetrahedron,1}) = ((1,2), (2,3), (3,1), (1,4), (2,4), (3,4))

function reference_coordinates(::Lagrange{RefTetrahedron,1})
    return [Vec{3, Float64}((0.0, 0.0, 0.0)),
            Vec{3, Float64}((1.0, 0.0, 0.0)),
            Vec{3, Float64}((0.0, 1.0, 0.0)),
            Vec{3, Float64}((0.0, 0.0, 1.0))]
end

function shape_value(ip::Lagrange{RefTetrahedron, 1}, ξ::Vec{3}, i::Int)
    ξ_x = ξ[1]
    ξ_y = ξ[2]
    ξ_z = ξ[3]
    i == 1 && return 1.0 - ξ_x - ξ_y - ξ_z
    i == 2 && return ξ_x
    i == 3 && return ξ_y
    i == 4 && return ξ_z
    throw(ArgumentError("no shape function $i for interpolation $ip"))
end

###################################
# Lagrange RefTetrahedron order 2 #
###################################
getnbasefunctions(::Lagrange{RefTetrahedron,2}) = 10

facedof_indices(::Lagrange{RefTetrahedron,2}) = ((1,3,2,7,6,5), (1,2,4,5,9,8), (2,3,4,6,10,9), (1,4,3,8,10,7))
edgedof_indices(::Lagrange{RefTetrahedron,2}) = ((1,2,5), (2,3,6), (3,1,7), (1,4,8), (2,4,9), (3,4,10))
edgedof_interior_indices(::Lagrange{RefTetrahedron,2}) = ((5,), (6,), (7,), (8,), (9,), (10,))

function reference_coordinates(::Lagrange{RefTetrahedron,2})
    return [Vec{3, Float64}((0.0, 0.0, 0.0)),
            Vec{3, Float64}((1.0, 0.0, 0.0)),
            Vec{3, Float64}((0.0, 1.0, 0.0)),
            Vec{3, Float64}((0.0, 0.0, 1.0)),
            Vec{3, Float64}((0.5, 0.0, 0.0)),
            Vec{3, Float64}((0.5, 0.5, 0.0)),
            Vec{3, Float64}((0.0, 0.5, 0.0)),
            Vec{3, Float64}((0.0, 0.0, 0.5)),
            Vec{3, Float64}((0.5, 0.0, 0.5)),
            Vec{3, Float64}((0.0, 0.5, 0.5))]
end

# http://www.colorado.edu/engineering/CAS/courses.d/AFEM.d/AFEM.Ch09.d/AFEM.Ch09.pdf
# http://www.colorado.edu/engineering/CAS/courses.d/AFEM.d/AFEM.Ch10.d/AFEM.Ch10.pdf
function shape_value(ip::Lagrange{RefTetrahedron, 2}, ξ::Vec{3}, i::Int)
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
getnbasefunctions(::Lagrange{RefHexahedron,1}) = 8

facedof_indices(::Lagrange{RefHexahedron,1}) = ((1,4,3,2), (1,2,6,5), (2,3,7,6), (3,4,8,7), (1,5,8,4), (5,6,7,8))
edgedof_indices(::Lagrange{RefHexahedron,1}) = ((1,2), (2,3), (3,4), (4,1), (5,6), (6,7), (7,8), (8,5), (1,5), (2,6), (3,7), (4,8))

function reference_coordinates(::Lagrange{RefHexahedron,1})
    return [Vec{3, Float64}((-1.0, -1.0, -1.0)),
            Vec{3, Float64}(( 1.0, -1.0, -1.0)),
            Vec{3, Float64}(( 1.0,  1.0, -1.0)),
            Vec{3, Float64}((-1.0,  1.0, -1.0)),
            Vec{3, Float64}((-1.0, -1.0,  1.0)),
            Vec{3, Float64}(( 1.0, -1.0,  1.0)),
            Vec{3, Float64}(( 1.0,  1.0,  1.0)),
            Vec{3, Float64}((-1.0,  1.0,  1.0))]
end

function shape_value(ip::Lagrange{RefHexahedron, 1}, ξ::Vec{3}, i::Int)
    ξ_x = ξ[1]
    ξ_y = ξ[2]
    ξ_z = ξ[3]
    i == 1 && return 0.125(1 - ξ_x) * (1 - ξ_y) * (1 - ξ_z)
    i == 2 && return 0.125(1 + ξ_x) * (1 - ξ_y) * (1 - ξ_z)
    i == 3 && return 0.125(1 + ξ_x) * (1 + ξ_y) * (1 - ξ_z)
    i == 4 && return 0.125(1 - ξ_x) * (1 + ξ_y) * (1 - ξ_z)
    i == 5 && return 0.125(1 - ξ_x) * (1 - ξ_y) * (1 + ξ_z)
    i == 6 && return 0.125(1 + ξ_x) * (1 - ξ_y) * (1 + ξ_z)
    i == 7 && return 0.125(1 + ξ_x) * (1 + ξ_y) * (1 + ξ_z)
    i == 8 && return 0.125(1 - ξ_x) * (1 + ξ_y) * (1 + ξ_z)
    throw(ArgumentError("no shape function $i for interpolation $ip"))
end


##################################
# Lagrange RefHexahedron order 2 #
##################################
# Based on vtkTriQuadraticHexahedron (see https://kitware.github.io/vtk-examples/site/Cxx/GeometricObjects/IsoparametricCellsDemo/)
getnbasefunctions(::Lagrange{RefHexahedron,2}) = 27

facedof_indices(::Lagrange{RefHexahedron,2}) = (
    (1,4,3,2, 12,11,10,9, 21),
    (1,2,6,5, 9,18,13,17, 22),
    (2,3,7,6, 10,19,14,18, 23),
    (3,4,8,7, 11,20,15,19, 24),
    (1,5,8,4, 17,16,20,12, 25),
    (5,6,7,8, 13,14,15,16, 26),
)
facedof_interior_indices(::Lagrange{RefHexahedron,2}) = (
    (21,), (22,), (23,), (24,), (25,), (26,),
)

edgedof_indices(::Lagrange{RefHexahedron,2}) = (
    (1,2, 9),
    (2,3, 10),
    (3,4, 11),
    (4,1, 12),
    (5,6, 13),
    (6,7, 14),
    (7,8, 15),
    (8,5, 16),
    (1,5, 17),
    (2,6, 18),
    (3,7, 19),
    (4,8, 20),
)
edgedof_interior_indices(::Lagrange{RefHexahedron,2}) = (
    (9,), (10,), (11,), (12,), (13,), (14,), (15,), (16,), (17), (18,), (19,), (20,)
)

celldof_interior_indices(::Lagrange{RefHexahedron,2}) = (27,)

function reference_coordinates(::Lagrange{RefHexahedron,2})
           # vertex
    return [Vec{3, Float64}((-1.0, -1.0, -1.0)), #  1
            Vec{3, Float64}(( 1.0, -1.0, -1.0)), #  2
            Vec{3, Float64}(( 1.0,  1.0, -1.0)), #  3
            Vec{3, Float64}((-1.0,  1.0, -1.0)), #  4
            Vec{3, Float64}((-1.0, -1.0,  1.0)), #  5
            Vec{3, Float64}(( 1.0, -1.0,  1.0)), #  6
            Vec{3, Float64}(( 1.0,  1.0,  1.0)), #  7
            Vec{3, Float64}((-1.0,  1.0,  1.0)), #  8
            # edge
            Vec{3, Float64}(( 0.0, -1.0, -1.0)), #  9
            Vec{3, Float64}(( 1.0,  0.0, -1.0)),
            Vec{3, Float64}(( 0.0,  1.0, -1.0)),
            Vec{3, Float64}((-1.0,  0.0, -1.0)),
            Vec{3, Float64}(( 0.0, -1.0,  1.0)),
            Vec{3, Float64}(( 1.0,  0.0,  1.0)),
            Vec{3, Float64}(( 0.0,  1.0,  1.0)),
            Vec{3, Float64}((-1.0,  0.0,  1.0)),
            Vec{3, Float64}((-1.0, -1.0,  0.0)),
            Vec{3, Float64}(( 1.0, -1.0,  0.0)),
            Vec{3, Float64}(( 1.0,  1.0,  0.0)),
            Vec{3, Float64}((-1.0,  1.0,  0.0)), # 20
            Vec{3, Float64}(( 0.0,  0.0, -1.0)),
            Vec{3, Float64}(( 0.0, -1.0,  0.0)),
            Vec{3, Float64}(( 1.0,  0.0,  0.0)),
            Vec{3, Float64}(( 0.0,  1.0,  0.0)),
            Vec{3, Float64}((-1.0,  0.0,  0.0)),
            Vec{3, Float64}(( 0.0,  0.0,  1.0)), # 26
            # interior
            Vec{3, Float64}((0.0, 0.0, 0.0)),    # 27
            ]
end

function shape_value(ip::Lagrange{RefHexahedron, 2}, ξ::Vec{3, T}, i::Int) where {T}
    # Some local helpers.
    @inline φ₁(x::T) = -0.5*x*(1-x)
    @inline φ₂(x::T) = (1+x)*(1-x)
    @inline φ₃(x::T) = 0.5*x*(1+x)
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
    i ==  9 && return φ₂(ξ_x) * φ₁(ξ_y) * φ₁(ξ_z)
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
# Build on https://defelement.com/elements/examples/prism-Lagrange-1.html
getnbasefunctions(::Lagrange{RefPrism,1}) = 6

facedof_indices(::Lagrange{RefPrism,1}) = ((1,3,2), (1,2,5,4), (3,1,4,6), (2,3,6,5), (4,5,6))
edgedof_indices(::Lagrange{RefPrism,1}) = ((2,1), (1,3), (1,4), (3,2), (2,5), (3,6), (4,5), (4,6), (6,5))

function reference_coordinates(::Lagrange{RefPrism,1})
    return [Vec{3, Float64}((0.0, 0.0, 0.0)),
            Vec{3, Float64}((1.0, 0.0, 0.0)),
            Vec{3, Float64}((0.0, 1.0, 0.0)),
            Vec{3, Float64}((0.0, 0.0, 1.0)),
            Vec{3, Float64}((1.0, 0.0, 1.0)),
            Vec{3, Float64}((0.0, 1.0, 1.0))]
end

function shape_value(ip::Lagrange{RefPrism,1}, ξ::Vec{3}, i::Int)
    (x,y,z) = ξ
    i == 1 && return 1-x-y -z*(1-x-y)
    i == 2 && return x*(1-z)
    i == 3 && return y*(1-z)
    i == 4 && return z*(1-x-y)
    i == 5 && return x*z
    i == 6 && return y*z
    throw(ArgumentError("no shape function $i for interpolation $ip"))
end

#############################
# Lagrange RefPrism order 2 #
#############################
# Build on https://defelement.com/elements/examples/prism-Lagrange-2.html .
# This is simply the tensor-product of a quadratic triangle with a quadratic line.
getnbasefunctions(::Lagrange{RefPrism,2}) = 18

facedof_indices(::Lagrange{RefPrism,2}) = (
    #Vertices| Edges  | Face 
    (1,3,2  , 8,10,7         ),
    (1,2,5,4, 7,11,13,9,   16), 
    (3,1,4,6, 8,9,14,12,   17),
    (2,3,6,5, 10,12,15,11, 18),
    (4,5,6  , 13,15,14       ),
)
facedof_interior_indices(::Lagrange{RefPrism,2}) = (
    #Vertices| Edges  | Face 
    (), 
    (16,), 
    (17,), 
    (18,), 
    (),
)
edgedof_indices(::Lagrange{RefPrism,2}) = (
    #Vert|Edge
    (2,1, 7),
    (1,3, 8),
    (1,4, 9),
    (3,2, 10),
    (2,5, 11),
    (3,6, 12),
    (4,5, 13),
    (4,6, 14),
    (6,5, 15),
)
edgedof_interior_indices(::Lagrange{RefPrism,2}) = (
    #Vert|Edge
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

function reference_coordinates(::Lagrange{RefPrism,2})
    return [Vec{3, Float64}((0.0, 0.0, 0.0)),
            Vec{3, Float64}((1.0, 0.0, 0.0)),
            Vec{3, Float64}((0.0, 1.0, 0.0)),
            Vec{3, Float64}((0.0, 0.0, 1.0)),
            Vec{3, Float64}((1.0, 0.0, 1.0)),
            Vec{3, Float64}((0.0, 1.0, 1.0)),
            Vec{3, Float64}((1/2, 0.0, 0.0)),
            Vec{3, Float64}((0.0, 1/2, 0.0)),
            Vec{3, Float64}((0.0, 0.0, 1/2)),
            Vec{3, Float64}((1/2, 1/2, 0.0)),
            Vec{3, Float64}((1.0, 0.0, 1/2)),
            Vec{3, Float64}((0.0, 1.0, 1/2)),
            Vec{3, Float64}((1/2, 0.0, 1.0)),
            Vec{3, Float64}((0.0, 1/2, 1.0)),
            Vec{3, Float64}((1/2, 1/2, 1.0)),
            Vec{3, Float64}((1/2, 0.0, 1/2)),
            Vec{3, Float64}((0.0, 1/2, 1/2)),
            Vec{3, Float64}((1/2, 1/2, 1/2)),]
end

function shape_value(ip::Lagrange{RefPrism, 2}, ξ::Vec{3}, i::Int)
    (x,y,z) = ξ
    x² = x*x
    y² = y*y
    z² = z*z
    i == 1  && return 4*x²*z² - 6x²*z +2x² +8x*y*z² -12x*y*z +4x*y -6x*z² +9x*z -3x +4y²*z² -6y²*z + 2y² -6y*z² +9y*z -3*y +2z² -3z +1
    i == 2  && return x*(4x*z² -6x*z +2x -2z² +3z -1)
    i == 3  && return y*(4y*z² -6y*z +2y -2z² +3z -1)
    i == 4  && return z*(4x²*z -2x² + 8x*y*z -4x*y -6x*z +3x +4y²*z -2y² -6y*z +3y +2z -1)
    i == 5  && return x*z*(4x*z -2x -2z +1)
    i == 6  && return y*z*(4y*z -2y -2z +1)
    i == 7  && return 4x*(-2x*z² +3x*z -x -2*y*z² +3y*z -y +2z² -3z +1)
    i == 8  && return 4y*(-2x*z² +3x*z -x -2*y*z² +3y*z -y +2z² -3z +1)
    i == 9  && return 4z*(-2x²*z +2x² -4x*y*z +4x*y +3x*z -3x -2y²*z +2y² +3y*z -3y -z +1)
    i == 10 && return 4x*y*(2z² -3z +1)
    i == 11 && return 4x*z*(-2x*z +2x +z -1)
    i == 12 && return 4y*z*(-2y*z +2y +z -1)
    i == 13 && return 4x*z*(-2x*z +x -2y*z +y +2z -1)
    i == 14 && return 4y*z*(-2x*z +x -2y*z +y +2z -1)
    i == 15 && return 4x*y*z*(2z -1)
    i == 16 && return 16x*z*(x*z -x +y*z -y -z +1)
    i == 17 && return 16y*z*(x*z -x +y*z -y -z +1)
    i == 18 && return 16x*y*z*(1 -z)
    throw(ArgumentError("no shape function $i for interpolation $ip"))
end


#####################################
# Lagrange dim 3 RefPyramid order 1 #
#####################################
getnbasefunctions(::Lagrange{RefPyramid,1}) = 5
facedof_indices(::Lagrange{RefPyramid,1}) = ((1,3,4,2), (1,2,5), (1,5,3), (2,4,5), (3,5,4), )
edgedof_indices(::Lagrange{RefPyramid,1}) = ((1,2), (1,3), (1,5), (2,4), (2,5), (4,3), (3,5), (4,5))
 
function reference_coordinates(::Lagrange{RefPyramid,1})
    return [Vec{3, Float64}((0.0, 0.0, 0.0)),
            Vec{3, Float64}((1.0, 0.0, 0.0)),
            Vec{3, Float64}((0.0, 1.0, 0.0)),
            Vec{3, Float64}((1.0, 1.0, 0.0)),
            Vec{3, Float64}((0.0, 0.0, 1.0))]
end

function shape_value(ip::Lagrange{RefPyramid,1}, ξ::Vec{3,T}, i::Int) where T
    (x,y,z) = ξ
    zzero = z ≈ one(T)
    i == 1 && return zzero ? zero(T) : (-x*y+(z-1)*(-x-y-z+1))/(z-1)
    i == 2 && return zzero ? zero(T) : x*(y+z-1)/(z-1)
    i == 3 && return zzero ? zero(T) : y*(x+z-1)/(z-1)
    i == 4 && return zzero ? zero(T) : -x*y/(z-1)
    i == 5 && return z
    throw(ArgumentError("no shape function $i for interpolation $ip"))
end

#####################################
# Lagrange dim 3 RefPyramid order 2 #
#####################################
getnbasefunctions(::Lagrange{RefPyramid,2}) = 14

facedof_indices(::Lagrange{RefPyramid,2}) = (
    #Vertices | Edges  | Face 
    (1,3,4,2, 7,11,9,6, 14), 
    (1,2,5  , 6,10,8      ), 
    (1,5,3  , 7,12,8      ), 
    (2,4,5  , 9,13,10     ), 
    (3,5,4  , 12,13,11    ), 
)
facedof_interior_indices(::Lagrange{RefPyramid,2}) = (
    (14,), 
    (), 
    (), 
    (), 
    (),
)
edgedof_indices(::Lagrange{RefPyramid,2}) = (
    (1,2,6), 
    (1,3,7), 
    (1,5,8), 
    (2,4,9), 
    (2,5,10), 
    (4,3,11), 
    (3,5,12), 
    (4,5,13)
)
edgedof_interior_indices(::Lagrange{RefPyramid,2}) = (
    (6,),
    (7,),
    (8,),
    (9,),
    (10,),
    (11,),
    (12,),
    (13,),
)
function reference_coordinates(::Lagrange{RefPyramid,2})
    return [Vec{3, Float64}((0.0, 0.0, 0.0)),
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
            Vec{3, Float64}((0.5, 0.5, 0.0))]
end

function shape_value(ip::Lagrange{RefPyramid,2}, ξ::Vec{3,T}, i::Int) where T
    (x,y,z) = ξ
    x² = x*x
    y² = y*y
    z² = z*z
    zzero = z ≈ one(T)
    i == 1 && return zzero ? zero(T) : (4*x²*y²*(z-1) + x*y*(6x+6y+z)*(z²-2z+1) + (z-1)*(z² - 2z + 1)*(2x² + 9*x*y + 4*x*z - 3x + 2y² + 4*y*z - 3y + 2z² - 3z + 1)) / ((z-1)*(z²-2z+1))
    i == 2 && return zzero ? zero(T) : x*(4x*y²*(z-1) + y*(6x+2y-z)*(z²-2z+1) + (z-1)*(2x+3y-1)*(z²-2z+1))/((z-1)*(z²-2z+1))
    i == 3 && return zzero ? zero(T) : y*(4x²*y*(z-1) + x*(2x+6y-z)*(z²-2z+1) + (z-1)*(3x+2y-1)*(z²-2z+1))/((z-1)*(z²-2z+1))
    i == 4 && return zzero ? zero(T) : x*y*(4*x*y + 2x*z - 2x + 2y*z - 2y + 2z² - 3z + 1)/(z²-2z+1)
    i == 5 && return                   z*(2z-1)
    i == 6 && return zzero ? zero(T) : 4x*(2x*y²*(1-z) - y*(3x+2y)*(z²-2z+1) + (z-1)*(z²-2z+1)*(-x-3y-z+1))/((z-1)*(z²-2z+1))
    i == 7 && return zzero ? zero(T) : 4y*(2x²*y*(1-z) - x*(2x+3y)*(z²-2z+1) + (z-1)*(z²-2z+1)*(-3x-y-z+1))/((z-1)*(z²-2z+1))
    i == 8 && return zzero ? zero(T) : 4z*(-x*y + (z-1)*(-x-y-z+1))/(z-1)
    i == 9 && return zzero ? zero(T) : 4*x*y*(-2x*y - 2x*z + 2x - y*z + y - z² + 2*z - 1)/(z²-2z+1)
    i == 10 && return zzero ? zero(T) : 4x*z*(y + z - 1)/(z-1)
    i == 11 && return zzero ? zero(T) : 4*x*y*(-2x*y - x*z + x - 2y*z + 2y - z² + 2z -1)/(z²-2z+1)
    i == 12 && return zzero ? zero(T) : 4y*z*(x + z - 1)/(z-1)
    i == 13 && return zzero ? zero(T) : -4x*y*z/(z-1)
    i == 14 && return zzero ? zero(T) : 16x*y*(x*y + x*z - x + y*z - y + z² - 2z + 1)/(z²-2z+1)
    throw(ArgumentError("no shape function $i for interpolation $ip"))
end

###################
# Bubble elements #
###################
"""
Lagrange element with bubble stabilization.
"""
struct BubbleEnrichedLagrange{shape, order, unused} <: ScalarInterpolation{shape, order}
    function BubbleEnrichedLagrange{shape, order}() where {shape <: AbstractRefShape, order}
        new{shape, order, Nothing}()
    end
end

#######################################
# Lagrange-Bubble RefTriangle order 1 #
#######################################
# Taken from https://defelement.com/elements/bubble-enriched-lagrange.html
getnbasefunctions(::BubbleEnrichedLagrange{RefTriangle,1}) = 4

vertexdof_indices(::BubbleEnrichedLagrange{RefTriangle,1}) = ((1,), (2,), (3,))
facedof_indices(::BubbleEnrichedLagrange{RefTriangle,1}) = ((1,2), (2,3), (3,1))
celldof_interior_indices(::BubbleEnrichedLagrange{RefTriangle,1}) = (4,)

function reference_coordinates(::BubbleEnrichedLagrange{RefTriangle,1})
    return [Vec{2, Float64}((1.0, 0.0)),
            Vec{2, Float64}((0.0, 1.0)),
            Vec{2, Float64}((0.0, 0.0)),
            Vec{2, Float64}((1/3, 1/3)),]
end

function shape_value(ip::BubbleEnrichedLagrange{RefTriangle, 1}, ξ::Vec{2}, i::Int)
    ξ_x = ξ[1]
    ξ_y = ξ[2]
    i == 1 && return ξ_x*(9ξ_y^2 + 9ξ_x*ξ_y - 9ξ_y + 1)
    i == 2 && return ξ_y*(9ξ_x^2 + 9ξ_x*ξ_y - 9ξ_x + 1)
    i == 3 && return 9ξ_x^2*ξ_y + 9ξ_x*ξ_y^2 - 9ξ_x*ξ_y - ξ_x - ξ_y + 1
    i == 4 && return 27ξ_x*ξ_y*(1 - ξ_x - ξ_y)
    throw(ArgumentError("no shape function $i for interpolation $ip"))
end

############################
# Arbitrary Order Lagrange #
############################
"""
Arbitrary order continuous Lagrange interpolation with arbitrary basis positionings for hypercubes.

Basis positionings default to Gauss-Lobatto points

To use arbitrary order with triangles consider using [`Lagrange{RefTriangle,order}`](@ref) which in implemented for equidistant basis only
"""
ArbitraryOrderLagrange

"""
Arbitrary order discontinuous Lagrange interpolation with arbitrary basis positionings for hypercubes.

Basis positionings default to Gauss-Legendre points

To use arbitrary order with triangles consider using [`DiscontinuousLagrange{RefTriangle,order}`](@ref) which in implemented for equidistant basis only
"""
ArbitraryOrderDiscontinuousLagrange

for name in (
    :ArbitraryOrderLagrange,
    :ArbitraryOrderDiscontinuousLagrange
    )
    @eval begin
        struct $(name){shape, order, prod_T, ref_T, perm_T} <: ScalarInterpolation{shape, order}
            product_of::prod_T
            reference_coordinates::ref_T
            perm::perm_T
            inv_perm::perm_T
        end
        function reference_coordinates(ip::$(name){_, order}) where {_, order}
            return ip.reference_coordinates[ip.perm]
        end
    end
end

vertexdof_indices(::ArbitraryOrderLagrange{shape,order}) where {shape, order} = vertexdof_indices(Lagrange{shape,order}())
function dirichlet_vertexdof_indices(ip::ArbitraryOrderDiscontinuousLagrange{shape,order}) where {shape, order}
    (ip.reference_coordinates[1] ≉ Vec(-1.0) || ip.reference_coordinates[end] ≉ Vec(1.0)) &&
        error("dirichlet_vertexdof_indices is not implemented for L2 elements with no basis on the boundaries")
    return vertexdof_indices(Lagrange{shape,order}())
end
celldof_interior_indices(ip::ArbitraryOrderDiscontinuousLagrange{shape,order}) where {shape, order} = SVector{(order+1)^getdim(ip)}(1:(order+1)^getdim(ip))        

facedof_interior_indices(ip::ArbitraryOrderLagrange) = _facedof_interior_indices(ip)
edgedof_interior_indices(ip::ArbitraryOrderLagrange) = _edgedof_interior_indices(ip)

adjust_dofs_during_distribution(::ArbitraryOrderLagrange) = true
adjust_dofs_during_distribution(::ArbitraryOrderDiscontinuousLagrange) = false

equidistant(order::Int) = [(i*2-order)/order for i in 0:order]

####################################
# Arbitrary Order Lagrange RefLine #
####################################
getPermLagrangeLine(order::Int) = SVector{order+1}((1, order+1, SVector{order-1}(i for i in 2:order)...))

for (name,  default_coords) in (
    (:ArbitraryOrderLagrange,               :both),
    (:ArbitraryOrderDiscontinuousLagrange,  :neither)
    )
    @eval begin
        function $(name){RefLine, order}(points::Vector{Float64} = GaussQuadrature.legendre(order+1, GaussQuadrature.$(default_coords))[1]) where order
            $(name == :ArbitraryOrderLagrange) && (points[1] ≉ -1.0 || points[end] ≉ 1.0) &&
                throw(ArgumentError("Continuous nodal interpolations must have basis on the boundaries"))
            product_of = nothing
            ref_coord = Array{Vec{1,Float64},1}(undef,order+1)
            for i in 1:order+1
                ref_coord[i] = Vec(points[i])
            end
            perm = getPermLagrangeLine(order)
            inv_perm = sortperm(perm)
            $(name){RefLine, order, typeof(product_of), typeof(ref_coord), typeof(perm)}(product_of, ref_coord, perm, inv_perm)
        end
        getnbasefunctions(::$(name){RefLine,order}) where order = order + 1
    end
end

for (name,  facefunc) in (
    (:ArbitraryOrderLagrange,               :facedof_indices          ),
    (:ArbitraryOrderDiscontinuousLagrange,  :dirichlet_facedof_indices)
    )
    @eval begin
        function $(facefunc)(ip::$(name){RefLine,order}) where order
            $(name == :ArbitraryOrderDiscontinuousLagrange) &&
                (ip.reference_coordinates[1] ≉ Vec(-1.0) || ip.reference_coordinates[end] ≉ Vec(1.0)) &&
                error("$($facefunc) is not implemented for L2 elements with no basis on the boundaries")
            return ((1,), (2,))
        end
        function shape_value(ip::$(name){RefLine, order}, ξ::Vec{1}, j::Int) where order
            j > getnbasefunctions(ip) && throw(ArgumentError("no shape function $j for interpolation $ip"))
            ξ_x = ξ[1]
            result = 1.0
            j = ip.perm[j]
            coeff = ip.reference_coordinates
            for k in 1:order+1
                k == j && continue
                result *= (ξ_x - coeff[k][1])/(coeff[j][1]-coeff[k][1])
            end
            return result
        end
    end
end

celldof_interior_indices(::ArbitraryOrderLagrange{RefLine,order}) where order = SVector{order-1}(i+2 for i in 1:order - 1)        

##############################
# Lagrange RefQuadrilateral  #
##############################

# Permutation to switch numbering to Ferrite ordering
function getPermLagrangeQ(order)
    result = Array{Int,1}(undef, (order+1)^2)
    result[1:4] .= (1, order + 1, (order + 1)^2, order * (order + 1) + 1)
    idx = 5
    for i in 2:order
        result[idx] = i
        idx += 1
    end
    for i in 2:order
        result[idx] = (order+1)*i
        idx += 1
    end
    for i in 1:order-1
        result[idx] = (order+1)^2-i
        idx += 1
    end
    for i in order-1:-1:1
        result[idx] = (order+1)*i+1
        idx += 1
    end
    for j in 1:(order-1), i in 1:(order-1)
        result[idx] = i+1+(order+1)*j
        idx += 1
    end
    return result
end


for (name,  default_coords) in (
    (:ArbitraryOrderLagrange,               :both),
    (:ArbitraryOrderDiscontinuousLagrange,  :neither)
    )
    @eval begin
        function $(name){RefQuadrilateral, order}(points::Vector{Float64} = GaussQuadrature.legendre(order+1, GaussQuadrature.$(default_coords))[1]) where order
            $(name == :ArbitraryOrderLagrange) && (points[1] ≉ -1.0 || points[end] ≉ 1.0) &&
                throw(ArgumentError("Continuous nodal interpolations must have basis on the boundaries"))
            product_of = $(name){RefLine,order}(points)
            ref_coord = Array{Vec{2,Float64},1}(undef,(order+1)^2)
            for i in 1:order+1, j in 1:order+1
                ref_coord[i+(j-1)*(order+1)] = Vec(points[i],points[j])
            end
            perm = getPermLagrangeQ(order)
            inv_perm = sortperm(perm)
            $(name){RefQuadrilateral, order, typeof(product_of), typeof(ref_coord), typeof(perm)}(product_of, ref_coord, perm, inv_perm)
        end
        getnbasefunctions(::$(name){RefQuadrilateral,order}) where order = (order + 1)^2
    end
end

for (name,  facefunc) in (
    (:ArbitraryOrderLagrange,               :facedof_indices),
    (:ArbitraryOrderDiscontinuousLagrange,  :dirichlet_facedof_indices)
    )
    @eval begin
        function _facedof_interior_indices(::$(name){RefQuadrilateral,order}) where order
            return (SVector{order-1}((i+4 for i in 1:order-1)),
                SVector{order-1}((i+order+3 for i in 1:order-1)),
                SVector{order-1}((i+2*order+2 for i in 1:order-1)),
                SVector{order-1}((i+3*order+1 for i in 1:order-1)))
        end
        function $(facefunc)(ip::$(name){RefQuadrilateral,order}) where order
            $(name == :ArbitraryOrderDiscontinuousLagrange)  &&
                (ip.reference_coordinates[1] ≉ Vec(-1.0,-1.0) || ip.reference_coordinates[end] ≉ Vec(1.0,1.0)) &&
                error("$($facefunc) is not implemented for L2 elements with no basis on the boundaries")
            interior = _facedof_interior_indices(ip)
            face1 = SVector{order + 1}((i == 1 ? 1 : i == 2 ? 2 : interior[1][i-2] for i in 1:order+1))    
            face2 = SVector{order + 1}((i == 1 ? 2 : i == 2 ? 3 : interior[2][i-2] for i in 1:order+1))    
            face3 = SVector{order + 1}((i == 1 ? 3 : i == 2 ? 4 : interior[3][i-2] for i in 1:order+1))    
            face4 = SVector{order + 1}((i == 1 ? 4 : i == 2 ? 1 : interior[4][i-2] for i in 1:order+1))    
            return (face1, face2, face3, face4)
        end
        function shape_value(ip::$(name){RefQuadrilateral, order}, ξ::Vec{2}, i::Int) where order
            i > getnbasefunctions(ip) && throw(ArgumentError("no shape function $i for interpolation $ip"))
            ξ_x = ξ[1]
            ξ_y = ξ[2]
            i = ip.perm[i]
            i_x = (i-1)%(order+1) + 1
            i_y = (i-1)÷(order+1) + 1
            ip2 = ip.product_of
            i_x = ip2.inv_perm[i_x]
            i_y = ip2.inv_perm[i_y]
            return shape_value(ip2,Vec(ξ_x),i_x) * shape_value(ip2,Vec(ξ_y),i_y)
        end
    end
end

function celldof_interior_indices(ip::ArbitraryOrderLagrange{RefQuadrilateral,order}) where order
    ncellintdofs = (order - 1)^2
    totaldofs = getnbasefunctions(ip)
    return SVector{ncellintdofs}((totaldofs-ncellintdofs+i for i in 1:ncellintdofs))
end

##########################
# Lagrange RefHexahedron #
##########################
# Based on vtkTriQuadraticHexahedron (see https://kitware.github.io/vtk-examples/site/Cxx/GeometricObjects/IsoparametricCellsDemo/)
# Permutation to switch numbering to Ferrite ordering
function getPermLagrangeHex(order)
    result = Array{Int,1}(undef, (order+1)^3)
    # Vertices
    result[1:4] .= (1, order + 1, (order + 1)^2, order * (order + 1) + 1)
    result[5:8] .= order * (order + 1)^2 .+ (1, order + 1, (order + 1)^2, order * (order + 1) + 1)
    idx = 9
    # Edge 1, 2, 3, 4
    for i in 2:order
        result[idx] = i
        idx += 1
    end
    for i in 2:order
        result[idx] = (order+1)*i
        idx += 1
    end
    for i in 1:order-1
        result[idx] = (order+1)^2-i
        idx += 1
    end
    for i in order-1:-1:1
        result[idx] = (order+1)*i+1
        idx += 1
    end
    # Edge 5, 6, 7, 8
    for i in 2:order
        result[idx] = (order+1)^2 * order + i
        idx += 1
    end
    for i in 2:order
        result[idx] = (order+1)^2 * order + (order+1)*i
        idx += 1
    end
    for i in 1:order-1
        result[idx] = (order+1)^2 * order + (order+1)^2-i
        idx += 1
    end
    for i in order-1:-1:1
        result[idx] = (order+1)^2 * order + (order+1)*i+1
        idx += 1
    end
    # Edges 9, 10, 11, 12
    for i in 1:order-1
        result[idx] = (order+1)^2 * i + 1
        idx += 1
    end
    for i in 1:order-1
        result[idx] = (order+1)^2 * i + (order+1)
        idx += 1
    end
    for i in :1:order-1
        result[idx] = (order+1)^2 * (i+1)
        idx += 1
    end
    for i in 1:order-1
        result[idx] = (order+1)^2 * (i+1) - order
        idx += 1
    end
    # Face 1
    for j in 1:(order-1), i in 1:(order-1)
        result[idx] = i+1+(order+1)*j
        idx += 1
    end
    # Face 2
    for k in 1:(order-1), i in 1:(order-1)
        result[idx] = i+1+(order+1)^2*k
        idx += 1
    end
    # Face 3
    for k in 1:(order-1), j in 1:(order-1)
        result[idx] = j*(order+1)+order+1+(order+1)^2*k
        idx += 1
    end
    # Face 4
    for k in 1:(order-1), i in (order-1):-1:1
        result[idx] = i+(order)*(order+1)+1+(order+1)^2*k
        idx += 1
    end
    # Face 5
    for k in 1:(order-1), j in (order-1):-1:1
        result[idx] = j*(order+1)+1+(order+1)^2*k
        idx += 1
    end
    # Face 6
    for j in 1:(order-1), i in 1:(order-1)
        result[idx] = i+1+(order+1)*j + (order+1)^2 * order
        idx += 1
    end
    # Interior
    for k in 1:(order-1), j in 1:(order-1), i in 1:(order-1)
        result[idx] = k+1+(order+1)*j + (order+1)^2*i
        idx += 1
    end
    return result
end

for (name,  default_coords) in (
    (:ArbitraryOrderLagrange,               :both),
    (:ArbitraryOrderDiscontinuousLagrange,  :neither)
    )
    @eval begin
        function $(name){RefHexahedron, order}(points::Vector{Float64} = GaussQuadrature.legendre(order+1, GaussQuadrature.$(default_coords))[1]) where order
            $(name == :ArbitraryOrderLagrange) && (points[1] ≉ -1.0 || points[end] ≉ 1.0) &&
                throw(ArgumentError("Continuous nodal interpolations must have basis on the boundaries"))
            product_of = $(name){RefLine,order}(points)
            ref_coord = Array{Vec{3,Float64},1}(undef,(order+1)^3)
            for i in 1:order+1, j in 1:order+1, k in 1:order+1
                ref_coord[i+(j-1)*(order+1)+(k-1)*(order+1)^2] = Vec(points[i],points[j],points[k])
            end
            perm = getPermLagrangeHex(order)
            inv_perm = sortperm(perm)
            $(name){RefHexahedron, order, typeof(product_of), typeof(ref_coord), typeof(perm)}(product_of, ref_coord, perm, inv_perm)
        end
        getnbasefunctions(::$(name){RefHexahedron, order}) where order = (order+1)^3
    end
end

for (name,  facefunc,  edgefunc) in (
    (:ArbitraryOrderLagrange,               :facedof_indices,           :edgedof_indices),
    (:ArbitraryOrderDiscontinuousLagrange,  :dirichlet_facedof_indices, :dirichlet_edgedof_indices)
    )
    @eval begin
        _edgedof_interior_indices(::$(name){RefHexahedron, order}) where order = (
            SVector{order-1}(9:8+1*(order-1)),
            SVector{order-1}(9+(order-1):8+2*(order-1)),
            SVector{order-1}(9+2*(order-1):8+3*(order-1)),
            SVector{order-1}(9+3*(order-1):8+4*(order-1)),
            SVector{order-1}(9+4*(order-1):8+5*(order-1)),
            SVector{order-1}(9+5*(order-1):8+6*(order-1)),
            SVector{order-1}(9+6*(order-1):8+7*(order-1)),
            SVector{order-1}(9+7*(order-1):8+8*(order-1)),
            SVector{order-1}(9+8*(order-1):8+9*(order-1)),
            SVector{order-1}(9+9*(order-1):8+10*(order-1)),
            SVector{order-1}(9+10*(order-1):8+11*(order-1)),
            SVector{order-1}(9+11*(order-1):8+12*(order-1)),
            )

        _facedof_interior_indices(::$(name){RefHexahedron, order}) where order = 
            (
                SVector{(order-1)^2}((order-1)*12+9:(order-1)*12+8 + (order-1)^2),
                SVector{(order-1)^2}((order-1)*12+9 + (order-1)^2:(order-1)*12+8 + 2*(order-1)^2),
                SVector{(order-1)^2}((order-1)*12+9 + 2*(order-1)^2:(order-1)*12+8 + 3*(order-1)^2),
                SVector{(order-1)^2}((order-1)*12+9 + 3*(order-1)^2:(order-1)*12+8 + 4*(order-1)^2),
                SVector{(order-1)^2}((order-1)*12+9 + 4*(order-1)^2:(order-1)*12+8 + 5*(order-1)^2),
                SVector{(order-1)^2}((order-1)*12+9 + 5*(order-1)^2:(order-1)*12+8 + 6*(order-1)^2),     
            )

        function $(facefunc)(ip::$(name){RefHexahedron, order}) where order
            $(name == :ArbitraryOrderDiscontinuousLagrange)  &&
                (ip.reference_coordinates[1] ≉ Vec(-1.0,-1.0,-1.0) || ip.reference_coordinates[end] ≉ Vec(1.0,1.0,1.0)) &&
                error("$($facefunc) is not implemented for L2 elements with no basis on the boundaries")
            fdofi = _facedof_interior_indices(ip)
            edofi = _edgedof_interior_indices(ip)
            face1 = Array{Int,1}(undef,(order+1)^2)
            face2 = Array{Int,1}(undef,(order+1)^2)
            face3 = Array{Int,1}(undef,(order+1)^2)
            face4 = Array{Int,1}(undef,(order+1)^2)
            face5 = Array{Int,1}(undef,(order+1)^2)
            face6 = Array{Int,1}(undef,(order+1)^2)
        
            # face 1
            face1[1:4] .= (1,4,3,2)
            face1[5:4+(order-1)*1] .= @view edofi[4][end:-1:1]
            face1[5+(order-1)*1:4+(order-1)*2] .= @view edofi[3][end:-1:1]
            face1[5+(order-1)*2:4+(order-1)*3] .= @view edofi[2][end:-1:1]
            face1[5+(order-1)*3:4+(order-1)*4] .= @view edofi[1][end:-1:1]
            face1[5+(order-1)*4:end] .= fdofi[1]
        
            # face 2
            face2[1:4] .= (1,2,6,5)
            face2[5:4+(order-1)*1] .= edofi[1]
            face2[5+(order-1)*1:4+(order-1)*2] .= edofi[10]
            face2[5+(order-1)*2:4+(order-1)*3] .= @view edofi[5][end:-1:1]
            face2[5+(order-1)*3:4+(order-1)*4] .= @view edofi[9][end:-1:1]
            face2[5+(order-1)*4:end] .= fdofi[2]
        
            # face 3
            face3[1:4] .= (2,3,7,6)
            face3[5:4+(order-1)*1] .= edofi[2]
            face3[5+(order-1)*1:4+(order-1)*2] .= edofi[11]
            face3[5+(order-1)*2:4+(order-1)*3] .= @view edofi[6][end:-1:1]
            face3[5+(order-1)*3:4+(order-1)*4] .= @view edofi[10][end:-1:1]
            face3[5+(order-1)*4:end] .= fdofi[3]
        
            # face 4
            face4[1:4] .= (3,4,8,7)
            face4[5:4+(order-1)*1] .= edofi[3]
            face4[5+(order-1)*1:4+(order-1)*2] .= edofi[12]
            face4[5+(order-1)*2:4+(order-1)*3] .= @view edofi[7][end:-1:1]
            face4[5+(order-1)*3:4+(order-1)*4] .= @view edofi[11][end:-1:1]
            face4[5+(order-1)*4:end] .= fdofi[4]
        
            # face 5
            face5[1:4] .= (1,5,8,4)
            face5[5:4+(order-1)*1] .= edofi[9]
            face5[5+(order-1)*1:4+(order-1)*2] .= @view edofi[8][end:-1:1]
            face5[5+(order-1)*2:4+(order-1)*3] .= @view edofi[12][end:-1:1]
            face5[5+(order-1)*3:4+(order-1)*4] .= edofi[4]
            face5[5+(order-1)*4:end] .= fdofi[5]
        
            # face 6
            face6[1:4] .= (5,6,7,8)
            face6[5:4+(order-1)*1] .= edofi[5]
            face6[5+(order-1)*1:4+(order-1)*2] .= edofi[6]
            face6[5+(order-1)*2:4+(order-1)*3] .= edofi[7]
            face6[5+(order-1)*3:4+(order-1)*4] .= edofi[8]
            face6[5+(order-1)*4:end] .= fdofi[6]
        
            return (face1, face2, face3, face4, face5, face6)
        end
        
        function $(edgefunc)(ip::$(name){RefHexahedron, order}) where order 
            $(name == :ArbitraryOrderDiscontinuousLagrange) && (ip.reference_coordinates[1] ≉ Vec(-1.0,-1.0,-1.0) || ip.reference_coordinates[end] ≉ Vec(1.0,1.0,1.0)) &&
                error("$($edgefunc) is not implemented for L2 elements with no basis on the boundaries")
            edofi = _edgedof_interior_indices(ip)
            return (
                (1,2, edofi[1]...),
                (2,3, edofi[2]...),
                (3,4, edofi[3]...),
                (4,1, edofi[4]...),
                (5,6, edofi[5]...),
                (6,7, edofi[6]...),
                (7,8, edofi[7]...),
                (8,5, edofi[8]...),
                (1,5, edofi[9]...),
                (2,6, edofi[10]...),
                (3,7, edofi[11]...),
                (4,8, edofi[12]...),
            )
        end
        function shape_value(ip::$(name){RefHexahedron, order}, ξ::Vec{3, T}, i::Int) where {T, order}
            i > getnbasefunctions(ip) && throw(ArgumentError("no shape function $i for interpolation $ip"))
            ξ_x = ξ[1]
            ξ_y = ξ[2]
            ξ_z = ξ[3]
            i = ip.perm[i]
            i_x = (i-1)%(order+1) + 1
            i_y = ((i-1)%(order+1)^2)÷(order+1) + 1
            i_z = (i-1)÷(order+1)^2 + 1
            ip2 = ip.product_of
            i_x = ip2.inv_perm[i_x]
            i_y = ip2.inv_perm[i_y]
            i_z = ip2.inv_perm[i_z]
            return shape_value(ip2,Vec(ξ_x),i_x) * shape_value(ip2,Vec(ξ_y),i_y) * shape_value(ip2,Vec(ξ_z),i_z)
        end
    end
end

function celldof_interior_indices(ip::ArbitraryOrderLagrange{RefHexahedron,order}) where order
    ncellintdofs = (order - 1)^3
    totaldofs = getnbasefunctions(ip)
    return SVector{ncellintdofs}((totaldofs-ncellintdofs+i for i in 1:ncellintdofs))
end

###############
# Serendipity #
###############
struct Serendipity{shape, order, unused} <: ScalarInterpolation{shape,order}
    function Serendipity{shape, order}() where {shape <: AbstractRefShape, order}
        new{shape, order, Nothing}()
    end
end

# Note that the edgedofs for high order serendipity elements are defined in terms of integral moments, 
# so no permutation exists in general. See e.g. Scroggs et al. [2022] for an example.
# adjust_dofs_during_distribution(::Serendipity{refshape, order}) where {refshape, order} = false
adjust_dofs_during_distribution(::Serendipity{<:Any, 2}) = false
adjust_dofs_during_distribution(::Serendipity{<:Any, 1}) = false

# Vertices for all Serendipity interpolations are the same
vertexdof_indices(::Serendipity{RefQuadrilateral}) = ((1,),(2,),(3,),(4,))
vertexdof_indices(::Serendipity{RefHexahedron}) = ((1,),(2,),(3,),(4,),(5,),(6,),(7,),(8,))

########################################
# Serendipity RefQuadrilateral order 2 #
########################################
getnbasefunctions(::Serendipity{RefQuadrilateral,2}) = 8
getlowerorder(::Serendipity{RefQuadrilateral,2}) = Lagrange{RefQuadrilateral,1}()

facedof_indices(::Serendipity{RefQuadrilateral,2}) = ((1,2,5), (2,3,6), (3,4,7), (4,1,8))
facedof_interior_indices(::Serendipity{RefQuadrilateral,2}) = ((5,), (6,), (7,), (8,))

function reference_coordinates(::Serendipity{RefQuadrilateral,2})
    return [Vec{2, Float64}((-1.0, -1.0)),
            Vec{2, Float64}(( 1.0, -1.0)),
            Vec{2, Float64}(( 1.0,  1.0)),
            Vec{2, Float64}((-1.0,  1.0)),
            Vec{2, Float64}(( 0.0, -1.0)),
            Vec{2, Float64}(( 1.0,  0.0)),
            Vec{2, Float64}(( 0.0,  1.0)),
            Vec{2, Float64}((-1.0,  0.0))]
end

function shape_value(ip::Serendipity{RefQuadrilateral,2}, ξ::Vec{2}, i::Int)
    ξ_x = ξ[1]
    ξ_y = ξ[2]
    i == 1 && return (1 - ξ_x) * (1 - ξ_y) * 0.25(-ξ_x - ξ_y - 1)
    i == 2 && return (1 + ξ_x) * (1 - ξ_y) * 0.25( ξ_x - ξ_y - 1)
    i == 3 && return (1 + ξ_x) * (1 + ξ_y) * 0.25( ξ_x + ξ_y - 1)
    i == 4 && return (1 - ξ_x) * (1 + ξ_y) * 0.25(-ξ_x + ξ_y - 1)
    i == 5 && return 0.5(1 - ξ_x * ξ_x) * (1 - ξ_y)
    i == 6 && return 0.5(1 + ξ_x) * (1 - ξ_y * ξ_y)
    i == 7 && return 0.5(1 - ξ_x * ξ_x) * (1 + ξ_y)
    i == 8 && return 0.5(1 - ξ_x) * (1 - ξ_y * ξ_y)
    throw(ArgumentError("no shape function $i for interpolation $ip"))
end

#####################################
# Serendipity RefHexahedron order 2 #
#####################################
# Note that second order serendipity hex has no interior face indices.
getnbasefunctions(::Serendipity{RefHexahedron,2}) = 20
getlowerorder(::Serendipity{RefHexahedron,2}) = Lagrange{RefHexahedron,1}()

facedof_indices(::Serendipity{RefHexahedron,2}) = (
    (1,4,3,2, 12,11,10,9),
    (1,2,6,5, 9,18,13,17),
    (2,3,7,6, 10,19,14,18),
    (3,4,8,7, 11,20,15,19),
    (1,5,8,4, 17,16,20,12),
    (5,6,7,8, 13,14,15,16)
)
edgedof_indices(::Serendipity{RefHexahedron,2}) = (
    (1,2, 9),
    (2,3, 10),
    (3,4, 11),
    (4,1, 12),
    (5,6, 13),
    (6,7, 14),
    (7,8, 15),
    (8,5, 16),
    (1,5, 17),
    (2,6, 18),
    (3,7, 19),
    (4,8, 20),
)

edgedof_interior_indices(::Serendipity{RefHexahedron,2}) = (
    (9,), (10,), (11,), (12,), (13,), (14,), (15,), (16,), (17), (18,), (19,), (20,)
)

function reference_coordinates(::Serendipity{RefHexahedron,2})
    return [Vec{3, Float64}((-1.0, -1.0, -1.0)),
            Vec{3, Float64}(( 1.0, -1.0, -1.0)),
            Vec{3, Float64}(( 1.0,  1.0, -1.0)),
            Vec{3, Float64}((-1.0,  1.0, -1.0)),
            Vec{3, Float64}((-1.0, -1.0,  1.0)),
            Vec{3, Float64}(( 1.0, -1.0,  1.0)),
            Vec{3, Float64}(( 1.0,  1.0,  1.0)),
            Vec{3, Float64}((-1.0,  1.0,  1.0)),
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
            Vec{3, Float64}((-1.0, 1.0, 0.0)),]
end

function shape_value(ip::Serendipity{RefHexahedron, 2}, ξ::Vec{3}, i::Int)
    ξ_x = ξ[1]
    ξ_y = ξ[2]
    ξ_z = ξ[3]
    i == 1 && return 0.125(1 - ξ_x) * (1 - ξ_y) * (1 - ξ_z) - 0.5(shape_value(ip, ξ, 12) + shape_value(ip, ξ, 9) + shape_value(ip, ξ, 17))
    i == 2 && return 0.125(1 + ξ_x) * (1 - ξ_y) * (1 - ξ_z) - 0.5(shape_value(ip, ξ, 9) + shape_value(ip, ξ, 10) + shape_value(ip, ξ, 18))
    i == 3 && return 0.125(1 + ξ_x) * (1 + ξ_y) * (1 - ξ_z) - 0.5(shape_value(ip, ξ, 10) + shape_value(ip, ξ, 11) + shape_value(ip, ξ, 19))
    i == 4 && return 0.125(1 - ξ_x) * (1 + ξ_y) * (1 - ξ_z) - 0.5(shape_value(ip, ξ, 11) + shape_value(ip, ξ, 12) + shape_value(ip, ξ, 20))
    i == 5 && return 0.125(1 - ξ_x) * (1 - ξ_y) * (1 + ξ_z) - 0.5(shape_value(ip, ξ, 16) + shape_value(ip, ξ, 13) + shape_value(ip, ξ, 17))
    i == 6 && return 0.125(1 + ξ_x) * (1 - ξ_y) * (1 + ξ_z) - 0.5(shape_value(ip, ξ, 13) + shape_value(ip, ξ, 14) + shape_value(ip, ξ, 18))
    i == 7 && return 0.125(1 + ξ_x) * (1 + ξ_y) * (1 + ξ_z) - 0.5(shape_value(ip, ξ, 14) + shape_value(ip, ξ, 15) + shape_value(ip, ξ, 19))
    i == 8 && return 0.125(1 - ξ_x) * (1 + ξ_y) * (1 + ξ_z) - 0.5(shape_value(ip, ξ, 15) + shape_value(ip, ξ, 16) + shape_value(ip, ξ, 20))
    i == 9 && return 0.25(1 - ξ_x^2) * (1 - ξ_y) * (1 - ξ_z)
    i == 10 && return 0.25(1 + ξ_x) * (1 - ξ_y^2) * (1 - ξ_z)
    i == 11 && return 0.25(1 - ξ_x^2) * (1 + ξ_y) * (1 - ξ_z)
    i == 12 && return 0.25(1 - ξ_x) * (1 - ξ_y^2) * (1 - ξ_z)
    i == 13 && return 0.25(1 - ξ_x^2) * (1 - ξ_y) * (1 + ξ_z)
    i == 14 && return 0.25(1 + ξ_x) * (1 - ξ_y^2) * (1 + ξ_z)
    i == 15 && return 0.25(1 - ξ_x^2) * (1 + ξ_y) * (1 + ξ_z)
    i == 16 && return 0.25(1 - ξ_x) * (1 - ξ_y^2) * (1 + ξ_z)
    i == 17 && return 0.25(1 - ξ_x) * (1 - ξ_y) * (1 - ξ_z^2)
    i == 18 && return 0.25(1 + ξ_x) * (1 - ξ_y) * (1 - ξ_z^2)
    i == 19 && return 0.25(1 + ξ_x) * (1 + ξ_y) * (1 - ξ_z^2)
    i == 20 && return 0.25(1 - ξ_x) * (1 + ξ_y) * (1 - ξ_z^2)
    throw(ArgumentError("no shape function $i for interpolation $ip"))
end


#############################
# Crouzeix–Raviart Elements #
#############################
"""
Classical non-conforming Crouzeix–Raviart element.

For details we refer to the original paper:
M. Crouzeix and P. Raviart. "Conforming and nonconforming finite element 
methods for solving the stationary Stokes equations I." ESAIM: Mathematical Modelling 
and Numerical Analysis-Modélisation Mathématique et Analyse Numérique 7.R3 (1973): 33-75.
"""
struct CrouzeixRaviart{shape, order, unused} <: ScalarInterpolation{shape, order}
    CrouzeixRaviart{RefTriangle, 1}() = new{RefTriangle, 1, Nothing}()
end

adjust_dofs_during_distribution(::CrouzeixRaviart) = true
adjust_dofs_during_distribution(::CrouzeixRaviart{<:Any, 1}) = false

getnbasefunctions(::CrouzeixRaviart) = 3

facedof_indices(::CrouzeixRaviart) = ((1,), (2,), (3,))
facedof_interior_indices(::CrouzeixRaviart) = ((1,), (2,), (3,))

function reference_coordinates(::CrouzeixRaviart)
    return [Vec{2, Float64}((0.5, 0.5)),
            Vec{2, Float64}((0.0, 0.5)),
            Vec{2, Float64}((0.5, 0.0))]
end

function shape_value(ip::CrouzeixRaviart, ξ::Vec{2}, i::Int)
    ξ_x = ξ[1]
    ξ_y = ξ[2]
    i == 1 && return 2*ξ_x + 2*ξ_y - 1.0
    i == 2 && return 1.0 - 2*ξ_x
    i == 3 && return 1.0 - 2*ξ_y
    throw(ArgumentError("no shape function $i for interpolation $ip"))
end

##################################################
# VectorizedInterpolation{<:ScalarInterpolation} #
##################################################
struct VectorizedInterpolation{vdim, refshape, order, SI <: ScalarInterpolation{refshape, order}} <: VectorInterpolation{vdim, refshape,order}
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
function Base.literal_pow(::typeof(^), ip::ScalarInterpolation, ::Val{vdim}) where vdim
    return VectorizedInterpolation{vdim}(ip)
end

function Base.show(io::IO, mime::MIME"text/plain", ip::VectorizedInterpolation{vdim}) where vdim
    show(io, mime, ip.ip)
    print(io, "^", vdim)
end

# Helper to get number of copies for DoF distribution
get_n_copies(::VectorizedInterpolation{vdim}) where vdim = vdim

function getnbasefunctions(ipv::VectorizedInterpolation{vdim}) where vdim
    return vdim * getnbasefunctions(ipv.ip)
end
function shape_value(ipv::VectorizedInterpolation{vdim, shape}, ξ::Vec{refdim, T}, I::Int) where {vdim, refdim, shape <: AbstractRefShape{refdim}, T}
    i0, c0 = divrem(I - 1, vdim)
    i = i0 + 1
    c = c0 + 1
    v = shape_value(ipv.ip, ξ, i)
    return Vec{vdim, T}(j -> j == c ? v : zero(v))
end

# vdim == refdim
function shape_gradient_and_value(ipv::VectorizedInterpolation{dim, shape}, ξ::Vec{dim}, I::Int) where {dim, shape <: AbstractRefShape{dim}}
    return invoke(shape_gradient_and_value, Tuple{Interpolation, Vec, Int}, ipv, ξ, I)
end
# vdim != refdim
function shape_gradient_and_value(ipv::VectorizedInterpolation{vdim, shape}, ξ::V, I::Int) where {vdim, refdim, shape <: AbstractRefShape{refdim}, T, V <: Vec{refdim, T}}
    # Load with dual numbers and compute the value
    f = x -> shape_value(ipv, x, I)
    ξd = Tensors._load(ξ, Tensors.Tag(f, V))
    value_grad = f(ξd)
    # Extract the value and gradient
    val = Vec{vdim, T}(i -> Tensors.value(value_grad[i]))
    grad = zero(MMatrix{vdim, refdim, T})
    for (i, vi) in pairs(value_grad)
        p = Tensors.partials(vi)
        for (j, pj) in pairs(p)
            grad[i, j] = pj
        end
    end
    return SMatrix(grad), val
end

reference_coordinates(ip::VectorizedInterpolation) = reference_coordinates(ip.ip)

is_discontinuous(::Type{<:VectorizedInterpolation{<:Any, <:Any, <:Any, ip}}) where {ip} = is_discontinuous(ip)
