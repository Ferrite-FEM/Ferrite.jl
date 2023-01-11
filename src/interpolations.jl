"""
    Interpolation{ref_dim, ref_shape, order}()

Return an `Interpolation` on a `ref_dim`-dimensional reference shape
(see [`AbstractRefShape`](@ref)) `ref_shape` and order `order`.
`order` corresponds to the order of the interpolation.
The interpolation is used to define shape functions to interpolate
a function between nodes.

The following interpolations are implemented:

* `Lagrange{1,RefCube,1}`
* `Lagrange{1,RefCube,2}`
* `Lagrange{2,RefCube,1}`
* `Lagrange{2,RefCube,2}`
* `Lagrange{2,RefTetrahedron,1}`
* `Lagrange{2,RefTetrahedron,2}`
* `Lagrange{2,RefTetrahedron,3}`
* `Lagrange{2,RefTetrahedron,4}`
* `Lagrange{2,RefTetrahedron,5}`
* `BubbleEnrichedLagrange{2,RefTetrahedron,1}`
* `CrouzeixRaviart{2,1}`
* `Lagrange{3,RefCube,1}`
* `Lagrange{3,RefCube,2}`
* `Lagrange{3,RefTetrahedron,1}`
* `Lagrange{3,RefTetrahedron,2}`
* `Serendipity{2,RefCube,2}`
* `Serendipity{3,RefCube,2}`

# Examples
```jldoctest
julia> ip = Lagrange{2,RefTetrahedron,2}()
Ferrite.Lagrange{2,Ferrite.RefTetrahedron,2}()

julia> getnbasefunctions(ip)
6
```
"""
abstract type Interpolation{dim,shape,order} end

# struct that gathers all the information needed to distribute
# dofs for a given interpolation.
struct InterpolationInfo
    nvertexdofs::Vector{Int}
    nedgedofs::Vector{Int}
    nfacedofs::Vector{Int}
    ncelldofs::Int
    dim::Int
    function InterpolationInfo(interpolation::Interpolation{dim}) where {dim}
        new(
            [length(i) for i ∈ vertexdof_indices(interpolation)],
            [length(i) for i ∈ edgedof_interior_indices(interpolation)],
            [length(i) for i ∈ facedof_interior_indices(interpolation)],
            length(celldof_interior_indices(interpolation)),
            dim,
        )
    end
end

# Some redundant information about the geometry of the reference cells.
nfaces(::Interpolation{dim, RefCube}) where {dim}= 2*dim
nfaces(::Interpolation{2, RefTetrahedron})       = 3
nfaces(::Interpolation{3, RefTetrahedron})       = 4
nfaces(::Interpolation{3, RefPrism})             = 5

nedges(::Interpolation{1, RefCube})          =  0
nedges(::Interpolation{2, RefCube})          =  0
nedges(::Interpolation{3, RefCube})          = 12
nedges(::Interpolation{2, RefTetrahedron})   =  0
nedges(::Interpolation{3, RefTetrahedron})   =  6
nedges(::Interpolation{3, RefPrism})         =  9

nvertices(::Interpolation{dim, RefCube}) where {dim} = 2^dim
nvertices(::Interpolation{2, RefTetrahedron})        = 3
nvertices(::Interpolation{3, RefTetrahedron})        = 4
nvertices(::Interpolation{3, RefPrism})              = 6

Base.copy(ip::Interpolation) = ip

"""
    Ferrite.getdim(::Interpolation)

Return the dimension of the reference element for a given interpolation.
"""
@inline getdim(::Interpolation{dim}) where {dim} = dim

"""
    Ferrite.getrefshape(::Interpolation)::AbstractRefShape

Return the reference element shape of the interpolation.
"""
@inline getrefshape(::Interpolation{dim,shape}) where {dim,shape} = shape

"""
    Ferrite.getorder(::Interpolation)

Return order of the interpolation.
"""
@inline getorder(::Interpolation{dim,shape,order}) where {dim,shape,order} = order

"""
    Ferrite.value(ip::Interpolation, ξ::Vec)

Return a vector, of length [`getnbasefunctions(ip::Interpolation)`](@ref), with the value of each shape functions
of `ip`, evaluated in the reference coordinate `ξ`. This calls [`Ferrite.value(ip::Interpolation, i::Int, ξ::Vec)`](@ref), where `i`
is the shape function number, which each concrete interpolation should implement.
"""
function value(ip::Interpolation{dim}, ξ::Vec{dim,T}) where {dim,T}
    [value(ip, i, ξ) for i in 1:getnbasefunctions(ip)]
end

"""
    Ferrite.derivative(ip::Interpolation, ξ::Vec)

Return a vector, of length [`getnbasefunctions(ip::Interpolation)`](@ref), with the derivative (w.r.t. the
reference coordinate) of each shape functions of `ip`, evaluated in the reference coordinate
`ξ`. This uses automatic differentiation and uses `ip`s implementation of [`Ferrite.value(ip::Interpolation, i::Int, ξ::Vec)`](@ref).
"""
function derivative(ip::Interpolation{dim}, ξ::Vec{dim,T}) where {dim,T}
    [gradient(ξ -> value(ip, i, ξ), ξ) for i in 1:getnbasefunctions(ip)]
end

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

# Necessary for correct distribution of dofs.
"""
    value(ip::Interpolation, i::Int, ξ::Vec)

Evaluates the `i`'th basis function of the interpolation `ip` 
at a point `ξ` on the reference element. The index `i` must 
match the index in [`vertices(::Interpolation)`](@ref), [`faces(::Interpolation)`](@ref) and
[`edges(::Interpolation)`](@ref).

For nodal interpolations the indices also must match the 
indices of [`reference_coordinates(::Interpolation)`](@ref).
"""
value(ip::Interpolation, i::Int, ξ::Vec)

"""
    reference_coordinates(::Interpolation)

Returns a vector of coordinates with length [`getnbasefunctions(::Interpolation)`](@ref) 
and indices corresponding to the indices of a dof in [`vertices`](@ref), [`faces`](@ref) and
[`edges`](@ref).

    Only required for nodal interpolations.
    
    TODO: Separate nodal and non-nodal interpolations.
"""
reference_coordinates(ip::Interpolation)

"""
    vertexdof_indices(ip::Interpolation)
A tuple containing tuples of local dof indices for the respective 
vertex in local enumeration on a cell defined by [`vertices(::Cell)`](@ref). The vertex enumeration must 
match the vertex enumeration of the corresponding geometrical cell.
"""
vertexdof_indices(ip::Interpolation) = ntuple(_ -> (), nvertices(ip))

"""
    edgedof_indices(::Interpolation)
A tuple containing tuples of local dof indices for the respective 
edge in local enumeration on a cell defined by [`edges(::Cell)`](@ref). The edge enumeration must 
match the edge enumeration of the corresponding geometrical cell.
Note that the vertex dofs are included here.
"""
edgedof_indices(ip::Interpolation{3}) = ntuple(_ -> (), nedges(ip))

"""
    edgedof_interior_indices(ip::Interpolation)
A tuple containing tuples of the local dof indices on the interior of the respective
edge in local enumeration on a cell defined by [`edges(::Cell)`](@ref). The edge enumeration must 
match the edge enumeration of the corresponding geometrical cell.
"""
edgedof_interior_indices(ip::Interpolation{3}) = ntuple(_ -> (), nedges(ip))

"""
    facedof_indices(::Interpolation)
A tuple containing tuples of all local dof indices for the respective 
face in local enumeration on a cell defined by [`faces(::Cell)`](@ref). The face enumeration must 
match the face enumeration of the corresponding geometrical cell.
Note that the vertex and edge dofs are included here.
"""
facedof_indices(ip::Union{Interpolation{2}, Interpolation{3}}) =  ntuple(_ -> (), nfaces(ip))

"""
    facedof_interior_indices(ip::Interpolation)
A tuple containing tuples of the local dof indices on the interior of the respective 
face in local enumeration on a cell defined by [`faces(::Cell)`](@ref). The face enumeration must 
match the face enumeration of the corresponding geometrical cell.
Note that the vertex and edge dofs are included here.
"""
facedof_interior_indices(ip::Union{Interpolation{2}, Interpolation{3}}) = ntuple(_ -> (), nfaces(ip))

"""
    celldof_interior_indices(::Interpolation)
Tuple containing the dof indices associated with the interior of the cell.
"""
celldof_interior_indices(::Interpolation) = ()

# Needed for distributing dofs on shells correctly (face in 2d is edge in 3d)
# Ferrite.edgedof_indices(ip::Interpolation{2}) = Ferrite.facedof_indices(ip)
# Ferrite.edgedof_interior_indices(ip::Interpolation{2}) = Ferrite.facedof_interior_indices(ip)

#########################
# DiscontinuousLagrange #
#########################
# TODO generalize to arbitrary basis positionings.
"""
Piecewise discontinous Lagrange basis via Gauss-Lobatto points.
"""
struct DiscontinuousLagrange{dim,shape,order} <: Interpolation{dim,shape,order} end

getlowerdim(::DiscontinuousLagrange{dim,shape,order}) where {dim,shape,order} = DiscontinuousLagrange{dim-1,shape,order}()
getlowerorder(::DiscontinuousLagrange{dim,shape,order}) where {dim,shape,order} = DiscontinuousLagrange{dim,shape,order-1}()

getnbasefunctions(::DiscontinuousLagrange{dim,shape,order}) where {dim,shape,order} = getnbasefunctions(Lagrange{dim,shape,order}())
getnbasefunctions(::DiscontinuousLagrange{dim,shape,0}) where {dim,shape} = 1

# This just moves all dofs into the interior of the element.
celldof_interior_indices(ip::DiscontinuousLagrange{dim,shape,order}) where {dim,shape,order} = (collect(1:getnbasefunctions(ip))...,)

# Mirror the Lagrange element for now.
function reference_coordinates(ip::DiscontinuousLagrange{dim,shape,order}) where {dim,shape,order}
    return reference_coordinates(Lagrange{dim,shape,order}())
end
function value(ip::DiscontinuousLagrange{dim,shape,order}, i::Int, ξ::Vec{dim}) where {dim,shape,order}
    return value(Lagrange{dim, shape, order}(), i, ξ)
end

# Excepting the L0 element.
function reference_coordinates(ip::DiscontinuousLagrange{dim,RefCube,0}) where dim
    return [Vec{dim, Float64}(ntuple(x->0.0, dim))]
end

function reference_coordinates(ip::DiscontinuousLagrange{2,RefTetrahedron,0})
    return [Vec{2,Float64}((1/3,1/3))]
end

function reference_coordinates(ip::DiscontinuousLagrange{3,RefTetrahedron,0})
   return [Vec{3,Float64}((1/4,1/4,1/4))]
end

function value(ip::DiscontinuousLagrange{dim,shape,0}, i::Int, ξ::Vec{dim}) where {dim,shape}
    i > 1 && throw(BoundsError("no shape function $i for interpolation $ip"))
    return 1.0
end

############
# Lagrange #
############
struct Lagrange{dim,shape,order} <: Interpolation{dim,shape,order} end

# Vertices for all Lagrange interpolations are the same
vertices(::Lagrange{2,RefCube,order}) where {order} = ((1,),(2,),(3,),(4,))
vertices(::Lagrange{3,RefCube,order}) where {order} = ((1,),(2,),(3,),(4,),(5,),(6,),(7,),(8,))
vertices(::Lagrange{2,RefTetrahedron,order}) where {order} = ((1,),(2,),(3,))
vertices(::Lagrange{3,RefTetrahedron,order}) where {order} = ((1,),(2,),(3,),(4,))
nvertexdofs(::Lagrange{dim,shape,order}) where {dim,shape,order} = 1

getlowerdim(::Lagrange{dim,shape,order}) where {dim,shape,order} = Lagrange{dim-1,shape,order}()
getlowerorder(::Lagrange{dim,shape,order}) where {dim,shape,order} = Lagrange{dim,shape,order-1}()
getlowerorder(::Lagrange{dim,shape,1}) where {dim,shape} = DiscontinuousLagrange{dim,shape,0}()

##################################
# Lagrange dim 1 RefCube order 1 #
##################################
getnbasefunctions(::Lagrange{1,RefCube,1}) = 2

vertexdof_indices(::Lagrange{1,RefCube,1}) = ((1,), (2,))
facedof_indices(::Lagrange{1,RefCube,1}) = ((1,), (2,))

function reference_coordinates(::Lagrange{1,RefCube,1})
    return [Vec{1, Float64}((-1.0,)),
            Vec{1, Float64}(( 1.0,))]
end

function value(ip::Lagrange{1,RefCube,1}, i::Int, ξ::Vec{1})
    ξ_x = ξ[1]
    i == 1 && return (1 - ξ_x) * 0.5
    i == 2 && return (1 + ξ_x) * 0.5
    throw(BoundsError("no shape function $i for interpolation $ip"))
end

##################################
# Lagrange dim 1 RefCube order 2 #
##################################
getnbasefunctions(::Lagrange{1,RefCube,2}) = 3
<<<<<<< HEAD
ncelldofs(::Lagrange{1,RefCube,2}) = 1
=======
>>>>>>> 8b971b38f (Introduce new dof api and extend tests to capture the most important invariants.)

vertexdof_indices(::Lagrange{1,RefCube,2}) = ((1,), (2,))
facedof_indices(::Lagrange{1,RefCube,2}) = ((1,), (2,))
celldof_interior_indices(::Lagrange{1,RefCube,2}) = (3,)

function reference_coordinates(::Lagrange{1,RefCube,2})
    return [Vec{1, Float64}((-1.0,)),
            Vec{1, Float64}(( 1.0,)),
            Vec{1, Float64}(( 0.0,))]
end

function value(ip::Lagrange{1,RefCube,2}, i::Int, ξ::Vec{1})
    ξ_x = ξ[1]
    i == 1 && return ξ_x * (ξ_x - 1) * 0.5
    i == 2 && return ξ_x * (ξ_x + 1) * 0.5
    i == 3 && return 1 - ξ_x^2
    throw(BoundsError("no shape function $i for interpolation $ip"))
end

##################################
# Lagrange dim 2 RefCube order 1 #
##################################
getnbasefunctions(::Lagrange{2,RefCube,1}) = 4

vertexdof_indices(::Lagrange{2,RefCube,1}) = ((1,), (2,), (3,), (4,))
facedof_indices(::Lagrange{2,RefCube,1}) = ((1,2), (2,3), (3,4), (4,1))

function reference_coordinates(::Lagrange{2,RefCube,1})
    return [Vec{2, Float64}((-1.0, -1.0)),
            Vec{2, Float64}(( 1.0, -1.0)),
            Vec{2, Float64}(( 1.0,  1.0,)),
            Vec{2, Float64}((-1.0,  1.0,))]
end

function value(ip::Lagrange{2,RefCube,1}, i::Int, ξ::Vec{2})
    ξ_x = ξ[1]
    ξ_y = ξ[2]
    i == 1 && return (1 - ξ_x) * (1 - ξ_y) * 0.25
    i == 2 && return (1 + ξ_x) * (1 - ξ_y) * 0.25
    i == 3 && return (1 + ξ_x) * (1 + ξ_y) * 0.25
    i == 4 && return (1 - ξ_x) * (1 + ξ_y) * 0.25
    throw(BoundsError("no shape function $i for interpolation $ip"))
end

##################################
# Lagrange dim 2 RefCube order 2 #
##################################
getnbasefunctions(::Lagrange{2,RefCube,2}) = 9
<<<<<<< HEAD
nfacedofs(::Lagrange{2,RefCube,2}) = 1
ncelldofs(::Lagrange{2,RefCube,2}) = 1
=======
>>>>>>> 8b971b38f (Introduce new dof api and extend tests to capture the most important invariants.)

vertexdof_indices(::Lagrange{2,RefCube,2}) = ((1,), (2,), (3,), (4,))
facedof_indices(::Lagrange{2,RefCube,2}) = ((1,2, 5), (2,3, 6), (3,4, 7), (4,1, 8))
facedof_interior_indices(::Lagrange{2,RefCube,2}) = ((5,), (6,), (7,), (8,))
celldof_interior_indices(::Lagrange{2,RefCube,2}) = (9,)

function reference_coordinates(::Lagrange{2,RefCube,2})
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

function value(ip::Lagrange{2,RefCube,2}, i::Int, ξ::Vec{2})
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
    throw(BoundsError("no shape function $i for interpolation $ip"))
end

#########################################
# Lagrange dim 2 RefTetrahedron order 1 #
#########################################
getnbasefunctions(::Lagrange{2,RefTetrahedron,1}) = 3
getlowerdim(::Lagrange{2, RefTetrahedron, order}) where {order} = Lagrange{1, RefCube, order}()

vertexdof_indices(::Lagrange{2,RefTetrahedron,1}) = (1,2,3)
facedof_indices(::Lagrange{2,RefTetrahedron,1}) = ((1,2), (2,3), (3,1))

function reference_coordinates(::Lagrange{2,RefTetrahedron,1})
    return [Vec{2, Float64}((1.0, 0.0)),
            Vec{2, Float64}((0.0, 1.0)),
            Vec{2, Float64}((0.0, 0.0))]
end

function value(ip::Lagrange{2,RefTetrahedron,1}, i::Int, ξ::Vec{2})
    ξ_x = ξ[1]
    ξ_y = ξ[2]
    i == 1 && return ξ_x
    i == 2 && return ξ_y
    i == 3 && return 1. - ξ_x - ξ_y
    throw(BoundsError("no shape function $i for interpolation $ip"))
end

#########################################
# Lagrange dim 2 RefTetrahedron order 2 #
#########################################
getnbasefunctions(::Lagrange{2,RefTetrahedron,2}) = 6
<<<<<<< HEAD
nfacedofs(::Lagrange{2,RefTetrahedron,2}) = 1
=======
>>>>>>> 8b971b38f (Introduce new dof api and extend tests to capture the most important invariants.)

vertexdof_indices(::Lagrange{2,RefTetrahedron,2}) = (1,2,3)
facedof_indices(::Lagrange{2,RefTetrahedron,2}) = ((1,2,4), (2,3,5), (3,1,6))
facedof_interior_indices(::Lagrange{2,RefTetrahedron,2}) = ((4,), (5,), (6,))

function reference_coordinates(::Lagrange{2,RefTetrahedron,2})
    return [Vec{2, Float64}((1.0, 0.0)),
            Vec{2, Float64}((0.0, 1.0)),
            Vec{2, Float64}((0.0, 0.0)),
            Vec{2, Float64}((0.5, 0.5)),
            Vec{2, Float64}((0.0, 0.5)),
            Vec{2, Float64}((0.5, 0.0))]
end

function value(ip::Lagrange{2,RefTetrahedron,2}, i::Int, ξ::Vec{2})
    ξ_x = ξ[1]
    ξ_y = ξ[2]
    γ = 1. - ξ_x - ξ_y
    i == 1 && return ξ_x * (2ξ_x - 1)
    i == 2 && return ξ_y * (2ξ_y - 1)
    i == 3 && return γ * (2γ - 1)
    i == 4 && return 4ξ_x * ξ_y
    i == 5 && return 4ξ_y * γ
    i == 6 && return 4ξ_x * γ
    throw(BoundsError("no shape function $i for interpolation $ip"))
end
###############################################
# Lagrange dim 2 RefTetrahedron order 3, 4, 5 #
###############################################
# see https://getfem.readthedocs.io/en/latest/userdoc/appendixA.html

const Lagrange2Tri345 = Union{
    Lagrange{2,RefTetrahedron,3},
    Lagrange{2,RefTetrahedron,4},
    Lagrange{2,RefTetrahedron,5},
}

function getnbasefunctions(ip::Lagrange2Tri345)
    order = getorder(ip)
    return (order + 1) * (order + 2) ÷ 2
end

# Permutation to switch numbering to Ferrite ordering
const permdof2D = Dict{Int,Vector{Int}}(
    1 => [1, 2, 3],
    2 => [3, 6, 1, 5, 4, 2],
    3 => [4, 10, 1, 7, 9, 8, 5, 2, 3, 6],
    4 => [5, 15, 1, 9, 12, 14, 13, 10, 6, 2, 3, 4, 7, 8, 11],
    5 => [6, 21, 1, 11, 15, 18, 20, 19, 16, 12, 7, 2, 3, 4, 5, 8, 9, 10, 13, 14, 17],
)

vertexdof_indices(::Lagrange2Tri345) = (1, 2, 3)

function facedof_indices(ip::Lagrange2Tri345)
    order = getorder(ip)
    order == 1 && return ((1,2), (2,3), (3,1))
    order == 2 && return ((1,2,4), (2,3,5), (3,1,6))
    order == 3 && return ((1,2,4,5), (2,3,6,7), (3,1,8,9))
    order == 4 && return ((1,2,4,5,6), (2,3,7,8,9), (3,1,10,11,12))
    order == 5 && return ((1,2,4,5,6,7), (2,3,8,9,10,11), (3,1,12,13,14,15))

    throw(ArgumentError("Unsupported order $order for Lagrange on triangles."))
end

function facedof_interior_indices(ip::Lagrange2Tri345)
    order = getorder(ip)
    order == 1 && return ((), (), ())
    order == 2 && return ((4,), (5,), (6,))
    order == 3 && return ((4,5), (6,7), (8,9))
    order == 4 && return ((4,5,6), (7,8,9), (10,11,12))
    order == 5 && return ((4,5,6,7), (8,9,10,11), (12,13,14,15))
    
    throw(ArgumentError("Unsupported order $order for Lagrange on triangles."))
end

function celldof_interior_indices(ip::Lagrange2Tri345)
    order = getorder(ip)
    ncellintdofs = (order + 1) * (order + 2) ÷ 2 - 3 * order
    totaldofs = getnbasefunctions(ip)
    return (collect((totaldofs-ncellintdofs+1):totaldofs)...,)
end

function reference_coordinates(ip::Lagrange2Tri345)
    order = getorder(ip)
    coordpts = Vector{Vec{2, Float64}}()
    for k = 0:order
        for l = 0:(order - k)
            push!(coordpts, Vec{2, Float64}((l / order, k / order)))
        end
    end
    return permute!(coordpts, permdof2D[order])
end

function value(ip::Lagrange2Tri345, i::Int, ξ::Vec{2})
    order = getorder(ip)
    i = permdof2D[order][i]
    ξ_x = ξ[1]
    ξ_y = ξ[2]
    γ = 1. - ξ_x - ξ_y
    i1, i2, i3 = _numlin_basis2D(i, order)
    val = one(γ)
    i1 ≥ 1 && (val *= prod((order * γ - j) / (j + 1) for j = 0:(i1 - 1)))
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

#########################################
# Lagrange dim 3 RefTetrahedron order 1 #
#########################################
getnbasefunctions(::Lagrange{3,RefTetrahedron,1}) = 4

vertexdof_indices(::Lagrange{3,RefTetrahedron,1}) = ((1,), (2,), (3,), (4,))
facedof_indices(::Lagrange{3,RefTetrahedron,1}) = ((1,3,2), (1,2,4), (2,3,4), (1,4,3))
edgedof_indices(::Lagrange{3,RefTetrahedron,1}) = ((1,2), (2,3), (3,1), (1,4), (2,4), (3,4))

function reference_coordinates(::Lagrange{3,RefTetrahedron,1})
    return [Vec{3, Float64}((0.0, 0.0, 0.0)),
            Vec{3, Float64}((1.0, 0.0, 0.0)),
            Vec{3, Float64}((0.0, 1.0, 0.0)),
            Vec{3, Float64}((0.0, 0.0, 1.0))]
end

function value(ip::Lagrange{3,RefTetrahedron,1}, i::Int, ξ::Vec{3})
    ξ_x = ξ[1]
    ξ_y = ξ[2]
    ξ_z = ξ[3]
    i == 1 && return 1.0 - ξ_x - ξ_y - ξ_z
    i == 2 && return ξ_x
    i == 3 && return ξ_y
    i == 4 && return ξ_z
    throw(BoundsError("no shape function $i for interpolation $ip"))
end

#########################################
# Lagrange dim 3 RefTetrahedron order 2 #
#########################################
getnbasefunctions(::Lagrange{3,RefTetrahedron,2}) = 10
<<<<<<< HEAD
nedgedofs(::Lagrange{3,RefTetrahedron,2}) = 1
=======
>>>>>>> 8b971b38f (Introduce new dof api and extend tests to capture the most important invariants.)

vertexdof_indices(::Lagrange{3,RefTetrahedron,2}) = ((1,), (2,), (3,), (4,))
facedof_indices(::Lagrange{3,RefTetrahedron,2}) = ((1,3,2,7,6,5), (1,2,4,5,9,8), (2,3,4,6,10,9), (1,4,3,8,10,7))
edgedof_indices(::Lagrange{3,RefTetrahedron,2}) = ((1,2,5), (2,3,6), (3,1,7), (1,4,8), (2,4,9), (3,4,10))
edgedof_interior_indices(::Lagrange{3,RefTetrahedron,2}) = ((5,), (6,), (7,), (8,), (9,), (10,))

function reference_coordinates(::Lagrange{3,RefTetrahedron,2})
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
function value(ip::Lagrange{3,RefTetrahedron,2}, i::Int, ξ::Vec{3})
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
    throw(BoundsError("no shape function $i for interpolation $ip"))
end

##################################
# Lagrange dim 3 RefCube order 1 #
##################################
getnbasefunctions(::Lagrange{3,RefCube,1}) = 8

vertexdof_indices(::Lagrange{3,RefCube,1}) = ((1,), (2,), (3,), (4,), (5,), (6,), (7,), (8,))
facedof_indices(::Lagrange{3,RefCube,1}) = ((1,4,3,2), (1,2,6,5), (2,3,7,6), (3,4,8,7), (1,5,8,4), (5,6,7,8))
edgedof_indices(::Lagrange{3,RefCube,1}) = ((1,2), (2,3), (3,4), (4,1), (1,5), (2,6), (3,7), (4,8), (5,6), (6,7), (7,8), (8,5))

function reference_coordinates(::Lagrange{3,RefCube,1})
    return [Vec{3, Float64}((-1.0, -1.0, -1.0)),
            Vec{3, Float64}(( 1.0, -1.0, -1.0)),
            Vec{3, Float64}(( 1.0,  1.0, -1.0)),
            Vec{3, Float64}((-1.0,  1.0, -1.0)),
            Vec{3, Float64}((-1.0, -1.0,  1.0)),
            Vec{3, Float64}(( 1.0, -1.0,  1.0)),
            Vec{3, Float64}(( 1.0,  1.0,  1.0)),
            Vec{3, Float64}((-1.0,  1.0,  1.0))]
end

function value(ip::Lagrange{3,RefCube,1}, i::Int, ξ::Vec{3})
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
    throw(BoundsError("no shape function $i for interpolation $ip"))
end


##################################
# Lagrange dim 3 RefCube order 2 #
##################################
# Based on vtkTriQuadraticHexahedron (see https://kitware.github.io/vtk-examples/site/Cxx/GeometricObjects/IsoparametricCellsDemo/)
getnbasefunctions(::Lagrange{3,RefCube,2}) = 27
<<<<<<< HEAD
nedgedofs(::Lagrange{3,RefCube,2}) = 1
nfacedofs(::Lagrange{3,RefCube,2}) = 1
ncelldofs(::Lagrange{3,RefCube,2}) = 1
=======
>>>>>>> 8b971b38f (Introduce new dof api and extend tests to capture the most important invariants.)

vertexdof_indices(::Lagrange{3,RefCube,2}) = ((1,), (2,), (3,), (4,), (5,), (6,), (7,), (8,))
facedof_indices(::Lagrange{3,RefCube,2}) = (
    (1,4,3,2, 12,11,10,9, 21),
    (1,2,6,5, 9,18,13,17, 22),
    (2,3,7,6, 10,19,14,18, 23),
    (3,4,8,7, 11,20,15,19, 24),
    (1,5,8,4, 17,16,20,12, 25),
    (5,6,7,8, 13,14,15,16, 26),
)
facedof_interior_indices(::Lagrange{3,RefCube,2}) = (
    (21,), (22,), (23,), (24,), (25,), (26,),
)

edgedof_indices(::Lagrange{3,RefCube,2}) = (
    (1,2, 9),
    (2,3, 10),
    (3,4, 11),
    (4,1, 12),
    (1,5, 17),
    (2,6, 18),
    (3,7, 19),
    (4,8, 20),
    (5,6, 13),
    (6,7, 14),
    (7,8, 15),
    (8,5, 16)
)
edgedof_interior_indices(::Lagrange{3,RefCube,2}) = (
    (9,), (10,), (11,), (12,), (17), (18,), (19,), (20,), (13,), (14,), (15,), (16,)
)

celldof_interior_indices(::Lagrange{3,RefCube,2}) = (27,)

function reference_coordinates(::Lagrange{3,RefCube,2})
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

function value(ip::Lagrange{3,RefCube,2}, i::Int, ξ::Vec{3, T}) where {T}
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
    throw(BoundsError("no shape function $i for interpolation $ip"))
end


###################################
# Lagrange dim 3 RefPrism order 1 #
###################################
# Build on https://defelement.com/elements/examples/prism-Lagrange-1.html
getnbasefunctions(::Lagrange{3,RefPrism,1}) = 6

vertexdof_indices(::Lagrange{3,RefPrism,1}) = ((1,), (2,), (3,), (4,), (5,), (6,))
facedof_indices(::Lagrange{3,RefPrism,1}) = ((1,3,2), (1,2,5,4), (3,1,4,6), (2,3,6,5), (4,5,6))
edgedof_indices(::Lagrange{3,RefPrism,1}) = ((2,1), (1,3), (1,4), (3,2), (2,5), (3,6), (4,5), (4,6), (6,5))

function reference_coordinates(::Lagrange{3,RefPrism,1})
    return [Vec{3, Float64}((0.0, 0.0, 0.0)),
            Vec{3, Float64}((1.0, 0.0, 0.0)),
            Vec{3, Float64}((0.0, 1.0, 0.0)),
            Vec{3, Float64}((0.0, 0.0, 1.0)),
            Vec{3, Float64}((1.0, 0.0, 1.0)),
            Vec{3, Float64}((0.0, 1.0, 1.0))]
end

function value(ip::Lagrange{3,RefPrism,1}, i::Int, ξ::Vec{3})
    (x,y,z) = ξ
    i == 1 && return 1-x-y -z*(1-x-y)
    i == 2 && return x*(1-z)
    i == 3 && return y*(1-z)
    i == 4 && return z*(1-x-y)
    i == 5 && return x*z
    i == 6 && return y*z
    throw(BoundsError("no shape function $i for interpolation $ip"))
end

###################################
# Lagrange dim 3 RefPrism order 2 #
###################################
# Build on https://defelement.com/elements/examples/prism-Lagrange-2.html .
# This is simply the tensor-product of a quadratic triangle with a quadratic line.
getnbasefunctions(::Lagrange{3,RefPrism,2}) = 18

vertexdof_indices(::Lagrange{3,RefPrism,2}) = ((1,), (2,), (3,), (4,), (5,), (6,),)
facedof_indices(::Lagrange{3,RefPrism,2}) = (
    #Vertices| Edges  | Face 
    (1,3,2  , 8,9,6          ), 
    (1,2,5,4, 7,11,13,9,   16), 
    (3,1,4,6, 8,9,14,15,   17), 
    (2,3,6,5, 10,12,15,11, 18), 
    (4,5,6  , 13,15,14       ),
)
facedof_interior_indices(::Lagrange{3,RefPrism,2}) = (
    #Vertices| Edges  | Face 
    (), 
    (16,), 
    (17,), 
    (18,), 
    (),
)
edgedof_indices(::Lagrange{3,RefPrism,2}) = (
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
edgedof_interior_indices(::Lagrange{3,RefPrism,2}) = (
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

function reference_coordinates(::Lagrange{3,RefPrism,2})
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

function value(ip::Lagrange{3,RefPrism,2}, i::Int, ξ::Vec{3})
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
    throw(BoundsError("no shape function $i for interpolation $ip"))
end

###################
# Bubble elements #
###################
"""
Lagrange element with bubble stabilization.
"""
struct BubbleEnrichedLagrange{dim,ref_shape,order} <: Interpolation{dim,ref_shape,order} end
getlowerdim(ip::BubbleEnrichedLagrange{dim,ref_shape,order}) where {dim,ref_shape,order} = Lagrange{dim-1,ref_shape,order}()

# Vertices for all Bubble interpolations are the same
vertices(::BubbleEnrichedLagrange{2,RefTetrahedron,order}) where {order} = ((1,),(2,),(3,))
nvertexdofs(::BubbleEnrichedLagrange{ref_dim,ref_shape,order}) where {ref_dim,ref_shape, order} = 1

################################################
# Lagrange-Bubble dim 2 RefTetrahedron order 1 #
################################################
# Taken from https://defelement.com/elements/bubble-enriched-lagrange.html
getnbasefunctions(::BubbleEnrichedLagrange{2,RefTetrahedron,1}) = 4
<<<<<<< HEAD
ncelldofs(::BubbleEnrichedLagrange{2,RefTetrahedron,1}) = 1
=======
>>>>>>> 8b971b38f (Introduce new dof api and extend tests to capture the most important invariants.)

vertexdof_indices(::BubbleEnrichedLagrange{2,RefTetrahedron,1}) = ((1,), (2,), (3,))
facedof_indices(::BubbleEnrichedLagrange{2,RefTetrahedron,1}) = ((1,2), (2,3), (3,1))
celldof_interior_indices(::BubbleEnrichedLagrange{2,RefTetrahedron,1}) = (4,)

function reference_coordinates(::BubbleEnrichedLagrange{2,RefTetrahedron,1})
    return [Vec{2, Float64}((1.0, 0.0)),
            Vec{2, Float64}((0.0, 1.0)),
            Vec{2, Float64}((0.0, 0.0)),
            Vec{2, Float64}((1/3, 1/3)),]
end

function value(ip::BubbleEnrichedLagrange{2,RefTetrahedron,1}, i::Int, ξ::Vec{2})
    ξ_x = ξ[1]
    ξ_y = ξ[2]
    i == 1 && return ξ_x*(9ξ_y^2 + 9ξ_x*ξ_y - 9ξ_y + 1)
    i == 2 && return ξ_y*(9ξ_x^2 + 9ξ_x*ξ_y - 9ξ_x + 1)
    i == 3 && return 9ξ_x^2*ξ_y + 9ξ_x*ξ_y^2 - 9ξ_x*ξ_y - ξ_x - ξ_y + 1
    i == 4 && return 27ξ_x*ξ_y*(1 - ξ_x - ξ_y)
    throw(BoundsError("no shape function $i for interpolation $ip"))
end

###############
# Serendipity #
###############
struct Serendipity{dim,shape,order} <: Interpolation{dim,shape,order} end

# Vertices for all Serendipity interpolations are the same
vertices(::Serendipity{2,RefCube,order}) where {order} = ((1,),(2,),(3,),(4,))
vertices(::Serendipity{3,RefCube,order}) where {order} = ((1,),(2,),(3,),(4,),(5,),(6,),(7,),(8,))
nvertexdofs(::Serendipity{ref_dim,ref_shape,order}) where {ref_dim,ref_shape, order} = 1

#####################################
# Serendipity dim 2 RefCube order 2 #
#####################################
getnbasefunctions(::Serendipity{2,RefCube,2}) = 8
getlowerdim(::Serendipity{2,RefCube,2}) = Lagrange{1,RefCube,2}()
getlowerorder(::Serendipity{2,RefCube,2}) = Lagrange{2,RefCube,1}()
<<<<<<< HEAD
nfacedofs(::Serendipity{2,RefCube,2}) = 1
=======
>>>>>>> 8b971b38f (Introduce new dof api and extend tests to capture the most important invariants.)

vertexdof_indices(::Serendipity{2,RefCube,2}) = ((1,), (2,), (3,), (4,))
facedof_indices(::Serendipity{2,RefCube,2}) = ((1,2,5), (2,3,6), (3,4,7), (4,1,8))
facedof_interior_indices(::Serendipity{2,RefCube,2}) = ((5,), (6,), (7,), (8,))

function reference_coordinates(::Serendipity{2,RefCube,2})
    return [Vec{2, Float64}((-1.0, -1.0)),
            Vec{2, Float64}(( 1.0, -1.0)),
            Vec{2, Float64}(( 1.0,  1.0)),
            Vec{2, Float64}((-1.0,  1.0)),
            Vec{2, Float64}(( 0.0, -1.0)),
            Vec{2, Float64}(( 1.0,  0.0)),
            Vec{2, Float64}(( 0.0,  1.0)),
            Vec{2, Float64}((-1.0,  0.0))]
end

function value(ip::Serendipity{2,RefCube,2}, i::Int, ξ::Vec{2})
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
    throw(BoundsError("no shape function $i for interpolation $ip"))
end

#####################################
# Serendipity dim 3 RefCube order 2 #
#####################################
# Note that second order serendipity hex has no interior face indices.
getnbasefunctions(::Serendipity{3,RefCube,2}) = 20
getlowerdim(::Serendipity{3,RefCube,2}) = Serendipity{2,RefCube,2}()
getlowerorder(::Serendipity{3,RefCube,2}) = Lagrange{3,RefCube,1}()
<<<<<<< HEAD
nedgedofs(::Serendipity{3,RefCube,2}) = 1
=======
>>>>>>> 8b971b38f (Introduce new dof api and extend tests to capture the most important invariants.)

vertexdof_indices(::Serendipity{3,RefCube,2}) = ((1,), (2,), (3,), (4,), (5,), (6,), (7,), (8,))
facedof_indices(::Serendipity{3,RefCube,2}) = (
    (1,4,3,2, 12,11,10,9),
    (1,2,6,5, 9,18,13,17),
    (2,3,7,6, 10,19,14,18),
    (3,4,8,7, 11,20,15,19),
    (1,5,8,4, 17,16,20,12),
    (5,6,7,8, 13,14,15,16)
)
edgedof_indices(::Serendipity{3,RefCube,2}) = (
    (1,2, 9),
    (2,3, 10),
    (3,4, 11),
    (4,1, 12),
    (1,5, 17),
    (2,6, 18),
    (3,7, 19),
    (4,8, 20),
    (5,6, 13),
    (6,7, 14),
    (7,8, 15),
    (8,5, 16)
)

edgedof_interior_indices(::Serendipity{3,RefCube,2}) = (
    (9,), (10,), (11,), (12,), (17), (18,), (19,), (20,), (13,), (14,), (15,), (16,)
)

function reference_coordinates(::Serendipity{3,RefCube,2})
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

function value(ip::Serendipity{3,RefCube,2}, i::Int, ξ::Vec{3})
    ξ_x = ξ[1]
    ξ_y = ξ[2]
    ξ_z = ξ[3]
    i == 1 && return 0.125(1 - ξ_x) * (1 - ξ_y) * (1 - ξ_z) - 0.5(value(ip,12,ξ) + value(ip,9,ξ) + value(ip,17,ξ))
    i == 2 && return 0.125(1 + ξ_x) * (1 - ξ_y) * (1 - ξ_z) - 0.5(value(ip,9,ξ) + value(ip,10,ξ) + value(ip,18,ξ))
    i == 3 && return 0.125(1 + ξ_x) * (1 + ξ_y) * (1 - ξ_z) - 0.5(value(ip,10,ξ) + value(ip,11,ξ) + value(ip,19,ξ))
    i == 4 && return 0.125(1 - ξ_x) * (1 + ξ_y) * (1 - ξ_z) - 0.5(value(ip,11,ξ) + value(ip,12,ξ) + value(ip,20,ξ))
    i == 5 && return 0.125(1 - ξ_x) * (1 - ξ_y) * (1 + ξ_z) - 0.5(value(ip,16,ξ) + value(ip,13,ξ) + value(ip,17,ξ))
    i == 6 && return 0.125(1 + ξ_x) * (1 - ξ_y) * (1 + ξ_z) - 0.5(value(ip,13,ξ) + value(ip,14,ξ) + value(ip,18,ξ))
    i == 7 && return 0.125(1 + ξ_x) * (1 + ξ_y) * (1 + ξ_z) - 0.5(value(ip,14,ξ) + value(ip,15,ξ) + value(ip,19,ξ))
    i == 8 && return 0.125(1 - ξ_x) * (1 + ξ_y) * (1 + ξ_z) - 0.5(value(ip,15,ξ) + value(ip,16,ξ) + value(ip,20,ξ))
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
    throw(BoundsError("no shape function $i for interpolation $ip"))
end


#############################
# Crouzeix–Raviart Elements #
#############################
"""
Classical non-conforming Crouzeix–Raviart element.

For details we refer ot the original paper:
M. Crouzeix and P. Raviart. "Conforming and nonconforming finite element 
methods for solving the stationary Stokes equations I." ESAIM: Mathematical Modelling 
and Numerical Analysis-Modélisation Mathématique et Analyse Numérique 7.R3 (1973): 33-75.
"""
struct CrouzeixRaviart{dim,order} <: Interpolation{dim,RefTetrahedron,order} end

getnbasefunctions(::CrouzeixRaviart{2,1}) = 3
<<<<<<< HEAD
nfacedofs(::CrouzeixRaviart{2,1}) = 1
faces(::CrouzeixRaviart{2,1}) = ((1,), (2,), (3,))
=======

facedof_indices(::CrouzeixRaviart{2,1}) = ((1,), (2,), (3,))
facedof_interior_indices(::CrouzeixRaviart{2,1}) = ((1,), (2,), (3,))
>>>>>>> 8b971b38f (Introduce new dof api and extend tests to capture the most important invariants.)

function reference_coordinates(::CrouzeixRaviart{2,1})
    return [Vec{2, Float64}((0.5, 0.5)),
            Vec{2, Float64}((0.0, 0.5)),
            Vec{2, Float64}((0.5, 0.0))]
end

function value(ip::CrouzeixRaviart{2,1}, i::Int, ξ::Vec{2})
    ξ_x = ξ[1]
    ξ_y = ξ[2]
    i == 1 && return 2*ξ_x + 2*ξ_y - 1.0
    i == 2 && return 1.0 - 2*ξ_x
    i == 3 && return 1.0 - 2*ξ_y
    throw(BoundsError("no shape function $i for interpolation $ip"))
end
