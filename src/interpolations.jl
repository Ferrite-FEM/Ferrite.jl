"""
    Interpolation{dim, ref_shape, order}()

Return an `Interpolation` of given dimension `dim`, reference shape
(see see [`AbstractRefShape`](@ref)) `ref_shape` and order `order`.
`order` corresponds to the highest order term in the polynomial.
The interpolation is used to define shape functions to interpolate
a function between nodes.

The following interpolations are implemented:

* `Lagrange{1,RefCube,1}`
* `Lagrange{1,RefCube,2}`
* `Lagrange{2,RefCube,1}`
* `Lagrange{2,RefCube,2}`
* `Lagrange{2,RefTetrahedron,1}`
* `Lagrange{2,RefTetrahedron,2}`
* `Lagrange{3,RefCube,1}`
* `Serendipity{2,RefCube,2}`
* `Serendipity{3,RefCube,2}`
* `Lagrange{3,RefTetrahedron,1}`
* `Lagrange{3,RefTetrahedron,2}`

# Examples
```jldoctest
julia> ip = Lagrange{2,RefTetrahedron,2}()
Ferrite.Lagrange{2,Ferrite.RefTetrahedron,2}()

julia> getnbasefunctions(ip)
6
```
"""
abstract type Interpolation{dim,shape,order} end

Base.copy(ip::Interpolation) = ip

"""
Return the dimension of an `Interpolation`
"""
@inline getdim(ip::Interpolation{dim}) where {dim} = dim

"""
Return the reference shape of an `Interpolation`
"""
@inline getrefshape(ip::Interpolation{dim,shape}) where {dim,shape} = shape

"""
Return the polynomial order of the `Interpolation`
"""
@inline getorder(ip::Interpolation{dim,shape,order}) where {dim,shape,order} = order

"""
Compute the value of the shape functions at a point ξ for a given interpolation
"""
function value(ip::Interpolation{dim}, ξ::Vec{dim,T}) where {dim,T}
    [value(ip, i, ξ) for i in 1:getnbasefunctions(ip)]
end

"""
Compute the gradients of the shape functions at a point ξ for a given interpolation
"""
function derivative(ip::Interpolation{dim}, ξ::Vec{dim,T}) where {dim,T}
    [gradient(ξ -> value(ip, i, ξ), ξ) for i in 1:getnbasefunctions(ip)]
end

#####################
# Utility functions #
#####################

"""
Return the number of base functions for an [`Interpolation`](@ref) or `Values` object.
"""
getnbasefunctions

# struct that gathers all the information needed to distribute
# dofs for a given interpolation.
struct InterpolationInfo
    # TODO: Can be smaller than `Int` if that matters...
    nvertexdofs::Int
    nedgedofs::Int
    nfacedofs::Int
    ncelldofs::Int
    dim::Int
    InterpolationInfo(interpolation::Interpolation{dim}) where {dim} =
        new(nvertexdofs(interpolation), nedgedofs(interpolation),
            nfacedofs(interpolation),   ncelldofs(interpolation), dim)
end

# The following functions are used to distribute the dofs. Definitions:
#   vertexdof: dof on a "corner" of the reference shape
#   facedof: dof in the dim-1 dimension (line in 2D, surface in 3D)
#   edgedof: dof on a line between 2 vertices (i.e. "corners") (3D only)
#   celldof: dof that is local to the element

# Fallbacks for the interpolations which are used to distribute the dofs correctly
"""
Number of dofs per vertex
"""
nvertexdofs(::Interpolation) = 0
"""
Number of dofs per edge
"""
nedgedofs(::Interpolation)   = 0
"""
Number of dofs per face
"""
nfacedofs(::Interpolation)   = 0
"""
Total number of dofs in the interior
"""
ncelldofs(::Interpolation)   = 0

# Needed for distributing dofs on shells correctly (face in 2d is edge in 3d)
edges(ip::Interpolation{2}) = faces(ip)
nedgedofs(ip::Interpolation{2}) = nfacedofs(ip)

# Fallbacks for vertices
vertices(::Interpolation{2,RefCube}) = (1,2,3,4)
vertices(::Interpolation{3,RefCube}) = (1,2,3,4,5,6,7,8)
vertices(::Interpolation{2,RefTetrahedron}) = (1,2,3)
vertices(::Interpolation{3,RefTetrahedron}) = (1,2,3,4)

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

# TODO generalize properly
# ncelldofs(::DiscontinuousLagrange{dim,RefCube,order}) where {dim,order} = (order+1)^dim
# ncelldofs(::DiscontinuousLagrange{2,RefTetrahedron,order}) where {order} = (order+1)*(order+2)/2
# ncelldofs(::DiscontinuousLagrange{3,RefTetrahedron,order}) where {order} = (order+1)*(order+2)/2
getnbasefunctions(::DiscontinuousLagrange{dim,shape,order}) where {dim,shape,order} = getnbasefunctions(Lagrange{dim,shape,order}())
ncelldofs(::DiscontinuousLagrange{dim,shape,order}) where {dim,shape,order} = getnbasefunctions(DiscontinuousLagrange{dim,shape,order}())

getnbasefunctions(::DiscontinuousLagrange{dim,shape,0}) where {dim,shape} = 1
ncelldofs(::DiscontinuousLagrange{dim,shape,0}) where {dim,shape} = 1

faces(::DiscontinuousLagrange{dim,shape,order}) where {dim,shape,order} = ()

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
    return 1.0
end

############
# Lagrange #
############
struct Lagrange{dim,shape,order} <: Interpolation{dim,shape,order} end

getlowerdim(::Lagrange{dim,shape,order}) where {dim,shape,order} = Lagrange{dim-1,shape,order}()
getlowerorder(::Lagrange{dim,shape,order}) where {dim,shape,order} = Lagrange{dim,shape,order-1}()
getlowerorder(::Lagrange{dim,shape,1}) where {dim,shape} = DiscontinuousLagrange{dim,shape,0}()

##################################
# Lagrange dim 1 RefCube order 1 #
##################################
getnbasefunctions(::Lagrange{1,RefCube,1}) = 2
nvertexdofs(::Lagrange{1,RefCube,1}) = 1

faces(::Lagrange{1,RefCube,1}) = ((1,), (2,))

function reference_coordinates(::Lagrange{1,RefCube,1})
    return [Vec{1, Float64}((-1.0,)),
            Vec{1, Float64}(( 1.0,))]
end

function value(ip::Lagrange{1,RefCube,1}, i::Int, ξ::Vec{1})
    ξ_x = ξ[1]
    i == 1 && return (1 - ξ_x) * 0.5
    i == 2 && return (1 + ξ_x) * 0.5
    throw(ArgumentError("no shape function $i for interpolation $ip"))
end

##################################
# Lagrange dim 1 RefCube order 2 #
##################################
getnbasefunctions(::Lagrange{1,RefCube,2}) = 3
nvertexdofs(::Lagrange{1,RefCube,2}) = 1
ncelldofs(::Lagrange{1,RefCube,2}) = 1

faces(::Lagrange{1,RefCube,2}) = ((1,), (2,))

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
    throw(ArgumentError("no shape function $i for interpolation $ip"))
end

##################################
# Lagrange dim 2 RefCube order 1 #
##################################
getnbasefunctions(::Lagrange{2,RefCube,1}) = 4
nvertexdofs(::Lagrange{2,RefCube,1}) = 1

faces(::Lagrange{2,RefCube,1}) = ((1,2), (2,3), (3,4), (4,1))

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
    throw(ArgumentError("no shape function $i for interpolation $ip"))
end

##################################
# Lagrange dim 2 RefCube order 2 #
##################################
getnbasefunctions(::Lagrange{2,RefCube,2}) = 9
nvertexdofs(::Lagrange{2,RefCube,2}) = 1
nfacedofs(::Lagrange{2,RefCube,2}) = 1
ncelldofs(::Lagrange{2,RefCube,2}) = 1

faces(::Lagrange{2,RefCube,2}) = ((1,2,5), (2,3,6), (3,4,7), (4,1,8))

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
    throw(ArgumentError("no shape function $i for interpolation $ip"))
end

#########################################
# Lagrange dim 2 RefTetrahedron order 1 #
#########################################
getnbasefunctions(::Lagrange{2,RefTetrahedron,1}) = 3
getlowerdim(::Lagrange{2, RefTetrahedron, order}) where {order} = Lagrange{1, RefCube, order}()
nvertexdofs(::Lagrange{2,RefTetrahedron,1}) = 1

vertices(::Lagrange{2,RefTetrahedron,1}) = (1,2,3)
faces(::Lagrange{2,RefTetrahedron,1}) = ((1,2), (2,3), (3,1))

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
    throw(ArgumentError("no shape function $i for interpolation $ip"))
end

#########################################
# Lagrange dim 2 RefTetrahedron order 2 #
#########################################
getnbasefunctions(::Lagrange{2,RefTetrahedron,2}) = 6
nvertexdofs(::Lagrange{2,RefTetrahedron,2}) = 1
nfacedofs(::Lagrange{2,RefTetrahedron,2}) = 1

vertices(::Lagrange{2,RefTetrahedron,2}) = (1,2,3)
faces(::Lagrange{2,RefTetrahedron,2}) = ((1,2,4), (2,3,5), (3,1,6))

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
    throw(ArgumentError("no shape function $i for interpolation $ip"))
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

nvertexdofs(::Lagrange2Tri345) = 1
nedgedofs(ip::Lagrange2Tri345) = getorder(ip) - 1
nfacedofs(ip::Lagrange2Tri345) = getorder(ip) - 1
function ncelldofs(ip::Lagrange2Tri345)
    order = getorder(ip)
    return (order + 1) * (order + 2) ÷ 2 - 3 * order
end

# Permutation to switch numbering to Ferrite ordering
const permdof2D = Dict{Int,Vector{Int}}(
    1 => [1, 2, 3],
    2 => [3, 6, 1, 5, 4, 2],
    3 => [4, 10, 1, 7, 9, 8, 5, 2, 3, 6],
    4 => [5, 15, 1, 9, 12, 14, 13, 10, 6, 2, 3, 4, 7, 8, 11],
    5 => [6, 21, 1, 11, 15, 18, 20, 19, 16, 12, 7, 2, 3, 4, 5, 8, 9, 10, 13, 14, 17],
)

vertices(::Lagrange2Tri345) = (1, 2, 3)

function faces(ip::Lagrange2Tri345)
    order = getorder(ip)
    if order == 1
        return ((1,2), (2,3), (3,1))
    elseif order == 2
        return ((1,2,4), (2,3,5), (3,1,6))
    elseif order == 3
        return ((1,2,4,5), (2,3,6,7), (3,1,8,9))
    elseif order == 4
        return ((1,2,4,5,6), (2,3,7,8,9), (3,1,10,11,12))
    elseif order == 5
        return ((1,2,4,5,6,7), (2,3,8,9,10,11), (3,1,12,13,14,15))
    else
        error("unreachable")
    end
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
nvertexdofs(::Lagrange{3,RefTetrahedron,1}) = 1

faces(::Lagrange{3,RefTetrahedron,1}) = (
    (1,3,2),
    (1,2,4),
    (2,3,4),
    (1,4,3))
edges(::Lagrange{3,RefTetrahedron,1}) = (
    (1,2),
    (2,3),
    (3,1),
    (1,4),
    (2,4),
    (3,4)
)

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
    throw(ArgumentError("no shape function $i for interpolation $ip"))
end

#########################################
# Lagrange dim 3 RefTetrahedron order 2 #
#########################################
getnbasefunctions(::Lagrange{3,RefTetrahedron,2}) = 10
nvertexdofs(::Lagrange{3,RefTetrahedron,2}) = 1
nedgedofs(::Lagrange{3,RefTetrahedron,2}) = 1

faces(::Lagrange{3,RefTetrahedron,2}) = (
    (1,3,2, 7, 6,5),
    (1,2,4, 5, 9,8),
    (2,3,4, 6,10,9),
    (1,4,3, 8,10,7)
)
edges(::Lagrange{3,RefTetrahedron,2}) = (
    (1,2,  5),
    (2,3,  6),
    (3,1,  7),
    (1,4,  8),
    (2,4,  9),
    (3,4, 10)
)

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
    throw(ArgumentError("no shape function $i for interpolation $ip"))
end

##################################
# Lagrange dim 3 RefCube order 1 #
##################################
getnbasefunctions(::Lagrange{3,RefCube,1}) = 8
nvertexdofs(::Lagrange{3,RefCube,1}) = 1

faces(::Lagrange{3,RefCube,1}) = (
    (1,4,3,2), # Local face index 1
    (1,2,6,5),
    (2,3,7,6),
    (3,4,8,7),
    (1,5,8,4),
    (5,6,7,8)  # Local face index 6
)

edges(::Lagrange{3,RefCube,1}) = (
    (1,2),  # Local edge index 1
    (2,3), 
    (3,4),
    (4,1),
    (5,6),
    (6,7),
    (7,8),
    (8,5),
    (1,5),
    (2,6),
    (3,7),
    (4,8)   # Local edge index 12
)

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
    throw(ArgumentError("no shape function $i for interpolation $ip"))
end


##################################
# Lagrange dim 3 RefCube order 2 #
##################################
# Based on vtkTriQuadraticHexahedron (see https://kitware.github.io/vtk-examples/site/Cxx/GeometricObjects/IsoparametricCellsDemo/)
getnbasefunctions(::Lagrange{3,RefCube,2}) = 27
nvertexdofs(::Lagrange{3,RefCube,2}) = 1
nedgedofs(::Lagrange{3,RefCube,2}) = 1
nfacedofs(::Lagrange{3,RefCube,2}) = 1
ncelldofs(::Lagrange{3,RefCube,2}) = 1

faces(::Lagrange{3,RefCube,2}) = (
    (1,4,3,2, 12,11,10,9, 21),  # Local face index 1
    (1,2,6,5, 9,18,13,17, 22),
    (2,3,7,6, 10,19,14,18, 23),
    (3,4,8,7, 11,20,15,19, 24),
    (1,5,8,4, 17,16,20,12, 25),
    (5,6,7,8, 13,14,15,16, 26), # Local face index 6
)
edges(::Lagrange{3,RefCube,2}) = (
    (1,2,  9), # Local edge index 1
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
    (4,8, 20)  # Local edge index 12
)

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
    throw(ArgumentError("no shape function $i for interpolation $ip"))
end

###################
# Bubble elements #
###################
"""
Lagrange element with bubble stabilization.
"""
struct BubbleEnrichedLagrange{dim,ref_shape,order} <: Interpolation{dim,ref_shape,order} end
getlowerdim(ip::BubbleEnrichedLagrange{dim,ref_shape,order}) where {dim,ref_shape,order} = Lagrange{dim-1,ref_shape,order}()

################################################
# Lagrange-Bubble dim 2 RefTetrahedron order 1 #
################################################
# Taken from https://defelement.com/elements/bubble-enriched-lagrange.html
getnbasefunctions(::BubbleEnrichedLagrange{2,RefTetrahedron,1}) = 4
nvertexdofs(::BubbleEnrichedLagrange{2,RefTetrahedron,1}) = 1
ncelldofs(::BubbleEnrichedLagrange{2,RefTetrahedron,1}) = 1

faces(::BubbleEnrichedLagrange{2,RefTetrahedron,1}) = ((1,2), (2,3), (3,1))

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
    throw(ArgumentError("no shape function $i for interpolation $ip"))
end

###############
# Serendipity #
###############
struct Serendipity{dim,shape,order} <: Interpolation{dim,shape,order} end

#####################################
# Serendipity dim 2 RefCube order 2 #
#####################################
getnbasefunctions(::Serendipity{2,RefCube,2}) = 8
getlowerdim(::Serendipity{2,RefCube,2}) = Lagrange{1,RefCube,2}()
getlowerorder(::Serendipity{2,RefCube,2}) = Lagrange{2,RefCube,1}()
nvertexdofs(::Serendipity{2,RefCube,2}) = 1
nfacedofs(::Serendipity{2,RefCube,2}) = 1

faces(::Serendipity{2,RefCube,2}) = ((1,2,5), (2,3,6), (3,4,7), (4,1,8))

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
    throw(ArgumentError("no shape function $i for interpolation $ip"))
end

#####################################
# Serendipity dim 3 RefCube order 2 #
#####################################
getnbasefunctions(::Serendipity{3,RefCube,2}) = 20
getlowerdim(::Serendipity{3,RefCube,2}) = Serendipity{2,RefCube,2}()
getlowerorder(::Serendipity{3,RefCube,2}) = Lagrange{3,RefCube,1}()
nvertexdofs(::Serendipity{3,RefCube,2}) = 1
nedgedofs(::Serendipity{3,RefCube,2}) = 1

faces(::Serendipity{3,RefCube,2}) = (
    (1,4,3,2, 12,11,10, 9),
    (1,2,6,5,  9,18,13,17),
    (2,3,7,6, 10,19,14,18),
    (3,4,8,7, 11,20,15,19),
    (1,5,8,4, 17,16,20,12),
    (5,6,7,8, 13,14,15,16)
)
edges(::Serendipity{3,RefCube,2}) = (
    (1,2,  9), # Local edge index 1
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
    (4,8, 20)  # Local edge index 12
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
    throw(ArgumentError("no shape function $i for interpolation $ip"))
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
nfacedofs(::CrouzeixRaviart{2,1}) = 1

vertices(::CrouzeixRaviart{2,1}) = ()
faces(::CrouzeixRaviart{2,1}) = ((1,), (2,), (3,))

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
    throw(ArgumentError("no shape function $i for interpolation $ip"))
end
