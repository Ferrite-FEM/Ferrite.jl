"""
An `Interpolation` is used to define shape functions to interpolate
a function between nodes.

**Constructor:**

```julia
Interpolation{dim, reference_shape, order}()
```

**Arguments:**

* `dim`: the dimension the interpolation lives in
* `shape`: a reference shape, see [`AbstractRefShape`](@ref)
* `order`: the highest order term in the polynomial

The following interpolations are implemented:

* `Lagrange{1, RefCube, 1}`
* `Lagrange{1, RefCube, 2}`
* `Lagrange{2, RefCube, 1}`
* `Lagrange{2, RefCube, 2}`
* `Lagrange{2, RefTetrahedron, 1}`
* `Lagrange{2, RefTetrahedron, 2}`
* `Lagrange{3, RefCube, 1}`
* `Serendipity{2, RefCube, 2}`
* `Lagrange{3, RefTetrahedron, 1}`
* `Lagrange{3, RefTetrahedron, 2}`

**Common methods:**

* [`getnbasefunctions`](@ref)
* [`getdim`](@ref)
* [`getrefshape`](@ref)
* [`getorder`](@ref)


**Example:**

```jldoctest
julia> ip = Lagrange{2, RefTetrahedron, 2}()
JuAFEM.Lagrange{2,JuAFEM.RefTetrahedron,2}()

julia> getnbasefunctions(ip)
6
```
"""
abstract type Interpolation{dim, shape, order} end

"""
Returns the dimension of an `Interpolation`
"""
@inline getdim(ip::Interpolation{dim}) where {dim} = dim

"""
Returns the reference shape of an `Interpolation`
"""
@inline getrefshape(ip::Interpolation{dim, shape}) where {dim, shape} = shape

"""
Returns the polynomial order of the `Interpolation`
"""
@inline getorder(ip::Interpolation{dim, shape, order}) where {dim, shape, order} = order

"""
Computes the value of the shape functions at a point ξ for a given interpolation
"""
function value(ip::Interpolation{dim}, ξ::Vec{dim, T}) where {dim, T}
    [value(ip, i, ξ) for i in 1:getnbasefunctions(ip)]
end

"""
Computes the gradients of the shape functions at a point ξ for a given interpolation
"""
function derivative(ip::Interpolation{dim}, ξ::Vec{dim, T}) where {dim, T}
    [gradient(ξ -> value(ip, i, ξ), ξ) for i in 1:getnbasefunctions(ip)]
end

#####################
# Utility functions #
#####################
getnfaces(::Interpolation{dim, RefCube}) where {dim} = 2*dim
getnfaces(::Interpolation{2, RefTetrahedron}) = 3
getnfaces(::Interpolation{3, RefTetrahedron}) = 4

getfacelist(i::Interpolation) = getfacelist(typeof(i))

"""
Returns the number of base functions for an [`Interpolation`](@ref) or `Values` object.
"""
getnbasefunctions

############
# Lagrange #
############
struct Lagrange{dim, shape, order} <: Interpolation{dim, shape, order} end

getlowerdim(::Lagrange{dim,shape,order}) where {dim,shape,order} = Lagrange{dim-1,shape,order}()
getlowerorder(::Lagrange{dim,shape,order}) where {dim,shape,order} = Lagrange{dim,shape,order-1}()

##################################
# Lagrange dim 1 RefCube order 1 #
##################################
getnbasefunctions(::Lagrange{1, RefCube, 1}) = 2
getnfacenodes(::Lagrange{1, RefCube, 1}) = 1
getfacelist(::Type{Lagrange{1, RefCube, 1}}) = ((1,),(2,))

function value(ip::Lagrange{1, RefCube, 1}, i::Int, ξ::Vec{1})
    @inbounds begin
        ξ_x = ξ[1]
        i == 1 && return (1 - ξ_x) * 0.5
        i == 2 && return (1 + ξ_x) * 0.5
    end
    throw(ArgumentError("no shape function $i for interpolation $ip"))
end

##################################
# Lagrange dim 1 RefCube order 2 #
##################################
getnbasefunctions(::Lagrange{1, RefCube, 2}) = 3
getnfacenodes(::Lagrange{1, RefCube, 2}) = 1
getfacelist(::Type{Lagrange{1, RefCube, 2}}) = ((1,),(2,))

function value(ip::Lagrange{1, RefCube, 2}, i::Int, ξ::Vec{1})
    @inbounds begin
        ξ_x = ξ[1]
        i == 1 && return ξ_x * (ξ_x - 1) * 0.5
        i == 2 && return ξ_x * (ξ_x + 1) * 0.5
        i == 3 && return 1 - ξ_x^2
    end
    throw(ArgumentError("no shape function $i for interpolation $ip"))
end

##################################
# Lagrange dim 2 RefCube order 1 #
##################################
getnbasefunctions(::Lagrange{2, RefCube, 1}) = 4
getnfacenodes(::Lagrange{2, RefCube, 1}) = 2
getfacelist(::Type{Lagrange{2, RefCube, 1}}) = ((1,2),(2,3),(3,4),(1,4))

function value(ip::Lagrange{2, RefCube, 1}, i::Int, ξ::Vec{2})
    @inbounds begin
        ξ_x = ξ[1]
        ξ_y = ξ[2]
        i == 1 && return (1 - ξ_x) * (1 - ξ_y) * 0.25
        i == 2 && return (1 + ξ_x) * (1 - ξ_y) * 0.25
        i == 3 && return (1 + ξ_x) * (1 + ξ_y) * 0.25
        i == 4 && return (1 - ξ_x) * (1 + ξ_y) * 0.25
    end
    throw(ArgumentError("no shape function $i for interpolation $ip"))
end

##################################
# Lagrange dim 2 RefCube order 2 #
##################################
getnbasefunctions(::Lagrange{2, RefCube, 2}) = 9
getnfacenodes(::Lagrange{2, RefCube, 2}) = 3
getfacelist(::Type{Lagrange{2, RefCube, 2}}) = ((1,2,5),(2,3,6),(3,4,7),(1,4,8))

function value(ip::Lagrange{2, RefCube, 2}, i::Int, ξ::Vec{2})
    @inbounds begin
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
    end
    throw(ArgumentError("no shape function $i for interpolation $ip"))
end

#########################################
# Lagrange dim 2 RefTetrahedron order 1 #
#########################################
getnbasefunctions(::Lagrange{2, RefTetrahedron, 1}) = 3
getlowerdim(::Lagrange{2, RefTetrahedron, order}) where {order} = Lagrange{1, RefCube, order}()
getnfacenodes(::Lagrange{2, RefTetrahedron, 1}) = 2
getfacelist(::Type{Lagrange{2, RefTetrahedron, 1}}) = ((1,2),(2,3),(1,3))

function value(ip::Lagrange{2, RefTetrahedron, 1}, i::Int, ξ::Vec{2})
    @inbounds begin
        ξ_x = ξ[1]
        ξ_y = ξ[2]
        i == 1 && return ξ_x
        i == 2 && return ξ_y
        i == 3 && return 1. - ξ_x - ξ_y
    end
    throw(ArgumentError("no shape function $i for interpolation $ip"))
end

#########################################
# Lagrange dim 2 RefTetrahedron order 2 #
#########################################
getnbasefunctions(::Lagrange{2, RefTetrahedron, 2}) = 6
getnfacenodes(::Lagrange{2, RefTetrahedron, 2}) = 3
getfacelist(::Type{Lagrange{2, RefTetrahedron, 2}}) = ((1,2,4),(2,3,5),(1,3,6))

function value(ip::Lagrange{2, RefTetrahedron, 2}, i::Int, ξ::Vec{2})
    @inbounds begin
        ξ_x = ξ[1]
        ξ_y = ξ[2]
        γ = 1. - ξ_x - ξ_y
        i == 1 && return ξ_x * (2ξ_x - 1)
        i == 2 && return ξ_y * (2ξ_y - 1)
        i == 3 && return γ * (2γ - 1)
        i == 4 && return 4ξ_x * ξ_y
        i == 5 && return 4ξ_y * γ
        i == 6 && return 4ξ_x * γ
    end
    throw(ArgumentError("no shape function $i for interpolation $ip"))
end

#########################################
# Lagrange dim 3 RefTetrahedron order 1 #
#########################################
getnbasefunctions(::Lagrange{3, RefTetrahedron, 1}) = 4
getnfacenodes(::Lagrange{3, RefTetrahedron, 1}) = 3
getfacelist(::Type{Lagrange{3, RefTetrahedron, 1}}) = ((1,2,3),(1,2,4),(2,3,4),(1,3,4))

function value(ip::Lagrange{3, RefTetrahedron, 1}, i::Int, ξ::Vec{3})
    @inbounds begin
        ξ_x = ξ[1]
        ξ_y = ξ[2]
        ξ_z = ξ[3]
        i == 1 && return 1.0 - ξ_x - ξ_y - ξ_z
        i == 2 && return ξ_x
        i == 3 && return ξ_y
        i == 4 && return ξ_z
    end
    throw(ArgumentError("no shape function $i for interpolation $ip"))
end

#########################################
# Lagrange dim 3 RefTetrahedron order 2 #
#########################################
getnbasefunctions(::Lagrange{3, RefTetrahedron, 2}) = 10
getnfacenodes(::Lagrange{3, RefTetrahedron, 2}) = 6
getfacelist(::Lagrange{3, RefTetrahedron, 2}) = ((1,2,3,5,6,7),(1,2,4,5,8,9),(2,3,4,6,9,10),(1,3,4,7,8,10))

# http://www.colorado.edu/engineering/CAS/courses.d/AFEM.d/AFEM.Ch09.d/AFEM.Ch09.pdf
# http://www.colorado.edu/engineering/CAS/courses.d/AFEM.d/AFEM.Ch10.d/AFEM.Ch10.pdf
function value(ip::Lagrange{3, RefTetrahedron, 2}, i::Int, ξ::Vec{3})
    @inbounds begin
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
    end
    throw(ArgumentError("no shape function $i for interpolation $ip"))
end

##################################
# Lagrange dim 3 RefCube order 1 #
##################################
getnbasefunctions(::Lagrange{3, RefCube, 1}) = 8
getnfacenodes(::Lagrange{3, RefCube, 1}) = 4
getfacelist(::Type{Lagrange{3, RefCube, 1}}) = ((1,2,3,4),(1,2,5,6),(2,3,6,7),(3,4,7,8),(1,4,5,8),(5,6,7,8))

function value(ip::Lagrange{3, RefCube, 1}, i::Int, ξ::Vec{3})
    @inbounds begin
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
    end
    throw(ArgumentError("no shape function $i for interpolation $ip"))
end

###############
# Serendipity #
###############
struct Serendipity{dim, shape, order} <: Interpolation{dim, shape, order} end

#####################################
# Serendipity dim 2 RefCube order 2 #
#####################################
getnbasefunctions(::Serendipity{2, RefCube, 2}) = 8
getlowerdim(::Serendipity{2, RefCube, 2}) = Lagrange{1, RefCube, 2}()
getlowerorder(::Serendipity{2, RefCube, 2}) = Lagrange{2, RefCube, 1}()
getnfacenodes(::Serendipity{2, RefCube, 2}) = 3
getfacelist(::Type{Serendipity{2, RefCube, 2}}) = ((1,2,5),(2,3,6),(3,4,7),(1,4,8))

function value(ip::Serendipity{2, RefCube, 2}, i::Int, ξ::Vec{2})
    @inbounds begin
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
    end
    throw(ArgumentError("no shape function $i for interpolation $ip"))
end
