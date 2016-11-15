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
abstract Interpolation{dim, shape, order}

"""
Returns the dimension of an `Interpolation`
"""
@inline getdim{dim}(ip::Interpolation{dim}) = dim

"""
Returns the reference shape of an `Interpolation`
"""
@inline getrefshape{dim, shape}(ip::Interpolation{dim, shape}) = shape

"""
Returns the polynomial order of the `Interpolation`
"""
@inline getorder{dim, shape, order}(ip::Interpolation{dim, shape, order}) = order

"""
Computes the value of the shape functions at a point ξ for a given interpolation
"""
function value{dim, T}(ip::Interpolation{dim}, ξ::Vec{dim, T})
    value!(ip, zeros(T, getnbasefunctions(ip)), ξ)
end

"""
Computes the gradients of the shape functions at a point ξ for a given interpolation
"""
function derivative{dim, T}(ip::Interpolation{dim}, ξ::Vec{dim, T})
    derivative!(ip, [zero(Tensor{1, dim, T}) for i in 1:getnbasefunctions(ip)], ξ)
end

@inline function checkdim_value{dim}(ip::Interpolation{dim}, N::AbstractVector, ξ::Vec{dim})
    n_base = getnbasefunctions(ip)
    length(N) == n_base || throw(ArgumentError("N must have length $(n_base)"))
end

@inline function checkdim_derivative{dim, T}(ip::Interpolation{dim}, dN::AbstractVector{Vec{dim, T}}, ξ::Vec{dim, T})
    n_base = getnbasefunctions(ip)
    length(dN) == n_base || throw(ArgumentError("dN must have length $(n_base)"))
end

#####################
# Utility functions #
#####################
"""
Returns the number of base functions for an [`Interpolation`](@ref) or `Values` object.
"""
getnboundaries{dim}(::Interpolation{dim, RefCube}) = 2*dim
getnboundaries(::Interpolation{2, RefTetrahedron}) = 3
getnboundaries(::Interpolation{3, RefTetrahedron}) = 4
getboundarylist(interp::Interpolation) = getboundarylist(typeof(interp))
getnbasefunctions(interp::Interpolation) = getnbasefunctions(typeof(interp))
getnboundarynodes(interp::Interpolation) = getnboundarynodes(typeof(interp))


############
# Lagrange #
############
immutable Lagrange{dim, shape, order} <: Interpolation{dim, shape, order} end

getlowerdim{dim,shape,order}(::Lagrange{dim,shape,order}) = Lagrange{dim-1,shape,order}()
getlowerorder{dim,shape,order}(::Lagrange{dim,shape,order}) = Lagrange{dim,shape,order-1}()

##################################
# Lagrange dim 1 RefCube order 1 #
##################################
getnbasefunctions(::Type{Lagrange{1, RefCube, 1}}) = 2
getnboundarynodes(::Type{Lagrange{1, RefCube, 1}}) = 1
getboundarylist(  ::Type{Lagrange{1, RefCube, 1}}) = ((1,),(2,))

function value!(ip::Lagrange{1, RefCube, 1}, N::AbstractVector, ξ::Vec{1})
    checkdim_value(ip, N, ξ)

    @inbounds begin
        ξ_x = ξ[1]

        N[1] = (1 - ξ_x) * 0.5
        N[2] = (1 + ξ_x) * 0.5
    end

    return N
end

function derivative!{T}(ip::Lagrange{1, RefCube, 1}, dN::AbstractVector{Vec{1, T}}, ξ::Vec{1, T})
    checkdim_derivative(ip, dN, ξ)

    @inbounds begin
        dN[1] = Vec{1,T}((-0.5,))
        dN[2] = Vec{1,T}((0.5,))
    end

    return dN
end

##################################
# Lagrange dim 1 RefCube order 2 #
##################################
getnbasefunctions(::Type{Lagrange{1, RefCube, 2}}) = 3
getnboundarynodes(::Type{Lagrange{1, RefCube, 2}}) = 1
getboundarylist(  ::Type{Lagrange{1, RefCube, 2}}) = ((1,),(2,))

function value!(ip::Lagrange{1, RefCube, 2}, N::AbstractVector, ξ::Vec{1})
    checkdim_value(ip, N, ξ)

    @inbounds begin
        ξ_x = ξ[1]

        N[1] = ξ_x * (ξ_x - 1) * 0.5
        N[2] = ξ_x * (ξ_x + 1) * 0.5
        N[3] = 1 - ξ_x^2
    end

    return N
end

function derivative!{T}(ip::Lagrange{1, RefCube, 2}, dN::AbstractVector{Vec{1, T}}, ξ::Vec{1, T})
    checkdim_derivative(ip, dN, ξ)

    @inbounds begin
        ξ_x = ξ[1]

        dN[1] = Vec{1,T}((ξ_x - 0.5,))
        dN[2] = Vec{1,T}((ξ_x + 0.5,))
        dN[3] = Vec{1,T}((-2 * ξ_x,))
    end

    return dN
end

##################################
# Lagrange dim 2 RefCube order 1 #
##################################
getnbasefunctions(::Type{Lagrange{2, RefCube, 1}}) = 4
getnboundarynodes(::Type{Lagrange{2, RefCube, 1}}) = 2
getboundarylist(  ::Type{Lagrange{2, RefCube, 1}}) = ((1,2),(2,3),(3,4),(1,4))

function value!(ip::Lagrange{2, RefCube, 1}, N::AbstractVector, ξ::Vec{2})
    checkdim_value(ip, N, ξ)

    @inbounds begin
        ξ_x = ξ[1]
        ξ_y = ξ[2]

        N[1] = (1 - ξ_x) * (1 - ξ_y) * 0.25
        N[2] = (1 + ξ_x) * (1 - ξ_y) * 0.25
        N[3] = (1 + ξ_x) * (1 + ξ_y) * 0.25
        N[4] = (1 - ξ_x) * (1 + ξ_y) * 0.25
    end

    return N
end

function derivative!{T}(ip::Lagrange{2, RefCube, 1}, dN::AbstractVector{Vec{2, T}}, ξ::Vec{2, T})
    checkdim_derivative(ip, dN, ξ)

    @inbounds begin
        ξ_x = ξ[1]
        ξ_y = ξ[2]

        dN[1] = Vec{2, T}((-(1 - ξ_y) * 0.25,
                           -(1 - ξ_x) * 0.25))

        dN[2] = Vec{2, T}(( (1 - ξ_y) * 0.25,
                           -(1 + ξ_x) * 0.25))

        dN[3] =  Vec{2, T}(((1 + ξ_y) * 0.25,
                            (1 + ξ_x) * 0.25))

        dN[4] = Vec{2, T}((-(1 + ξ_y) * 0.25,
                            (1 - ξ_x) * 0.25))
    end

    return dN
end

##################################
# Lagrange dim 2 RefCube order 2 #
##################################
getnbasefunctions(::Type{Lagrange{2, RefCube, 2}}) = 9
getnboundarynodes(::Type{Lagrange{2, RefCube, 2}}) = 3
getboundarylist(  ::Type{Lagrange{2, RefCube, 2}}) = ((1,2,5),(2,3,6),(3,4,7),(1,4,8))

function value!(ip::Lagrange{2, RefCube, 2}, N::AbstractVector, ξ::Vec{2})
    checkdim_value(ip, N, ξ)

    @inbounds begin
        ξ_x = ξ[1]
        ξ_y = ξ[2]

        N[1] = (ξ_x^2 - ξ_x) * (ξ_y^2 - ξ_y) * 0.25
        N[2] = (ξ_x^2 + ξ_x) * (ξ_y^2 - ξ_y) * 0.25
        N[3] = (ξ_x^2 + ξ_x) * (ξ_y^2 + ξ_y) * 0.25
        N[4] = (ξ_x^2 - ξ_x) * (ξ_y^2 + ξ_y) * 0.25
        N[5] = (1 - ξ_x^2) * (ξ_y^2 - ξ_y) * 0.5
        N[6] = (ξ_x^2 + ξ_x) * (1 - ξ_y^2) * 0.5
        N[7] = (1 - ξ_x^2) * (ξ_y^2 + ξ_y) * 0.5
        N[8] = (ξ_x^2 - ξ_x) * (1 - ξ_y^2) * 0.5
        N[9] = (1 - ξ_x^2) * (1 - ξ_y^2)
    end

    return N
end

function derivative!{T}(ip::Lagrange{2, RefCube, 2}, dN::AbstractVector{Vec{2, T}}, ξ::Vec{2, T})
    checkdim_derivative(ip, dN, ξ)

    @inbounds begin
        ξ_x = ξ[1]
        ξ_y = ξ[2]

        dN[1] = Vec{2, T}((ξ_y * (2 * ξ_x - 1) * (ξ_y - 1) * 0.25,
                           ξ_x * (2 * ξ_y - 1) * (ξ_x - 1) * 0.25))

        dN[2] = Vec{2, T}((ξ_y * (2 * ξ_x + 1) * (ξ_y - 1) * 0.25,
                           ξ_x * (2 * ξ_y - 1) * (ξ_x + 1) * 0.25))

        dN[3] = Vec{2, T}((ξ_y * (2 * ξ_x + 1) * (ξ_y + 1) * 0.25,
                           ξ_x * (2 * ξ_y + 1) * (ξ_x + 1) * 0.25))

        dN[4] = Vec{2, T}((ξ_y * (2 * ξ_x - 1) * (ξ_y + 1) * 0.25,
                           ξ_x * (2 * ξ_y + 1) * (ξ_x - 1) * 0.25))

        dN[5] = Vec{2, T}((ξ_y * ξ_x * (1 - ξ_y),
                           (1 - 2 * ξ_y) * (ξ_x^2 - 1) * 0.5))

        dN[6] = Vec{2, T}(((1 - ξ_y^2) * (2 * ξ_x + 1) * 0.5,
                           - ξ_y * ξ_x * (ξ_x + 1)))

        dN[7] = Vec{2, T}((- ξ_y * ξ_x * (ξ_y + 1),
                           (2 * ξ_y + 1) * (1 - ξ_x^2) * 0.5))

        dN[8] = Vec{2, T}(((1 - ξ_y^2) * (2 * ξ_x - 1) * 0.5,
                           ξ_y * ξ_x * (1 - ξ_x)))

        dN[9] = Vec{2, T}((2 * ξ_x * (ξ_y^2 - 1),
                           2 * ξ_y * (ξ_x^2 - 1)))
    end

    return dN
end

#########################################
# Lagrange dim 2 RefTetrahedron order 1 #
#########################################

getlowerdim{order}(::Lagrange{2, RefTetrahedron, order}) = Lagrange{1, RefCube, order}()
getnbasefunctions(::Type{Lagrange{2, RefTetrahedron, 1}}) = 3
getnboundarynodes(::Type{Lagrange{2, RefTetrahedron, 1}}) = 2
getboundarylist(  ::Type{Lagrange{2, RefTetrahedron, 1}}) = ((1,2),(2,3),(1,3))

function value!(ip::Lagrange{2, RefTetrahedron, 1}, N::AbstractVector, ξ::Vec{2})
    checkdim_value(ip, N, ξ)

    @inbounds begin
        ξ_x = ξ[1]
        ξ_y = ξ[2]

        N[1] = ξ_x
        N[2] = ξ_y
        N[3] = 1. - ξ_x - ξ_y
    end

    return N
end

function derivative!{T}(ip::Lagrange{2, RefTetrahedron, 1}, dN::AbstractVector{Vec{2, T}}, ξ::Vec{2, T})
    checkdim_derivative(ip, dN, ξ)

    @inbounds begin
        dN[1] = Vec{2, T}(( 1.0,  0.0))
        dN[2] = Vec{2, T}(( 0.0,  1.0))
        dN[3] = Vec{2, T}((-1.0, -1.0))
    end

    return dN
end

#########################################
# Lagrange dim 2 RefTetrahedron order 2 #
#########################################
getnbasefunctions(::Type{Lagrange{2, RefTetrahedron, 2}}) = 6
getnboundarynodes(::Type{Lagrange{2, RefTetrahedron, 2}}) = 3
getboundarylist(  ::Type{Lagrange{2, RefTetrahedron, 2}}) = ((1,2,4),(2,3,5),(1,3,6))


function value!(ip::Lagrange{2, RefTetrahedron, 2}, N::AbstractVector, ξ::Vec{2})
    checkdim_value(ip, N, ξ)

    @inbounds begin
        ξ_x = ξ[1]
        ξ_y = ξ[2]

        γ = 1. - ξ_x - ξ_y

        N[1] = ξ_x * (2ξ_x - 1)
        N[2] = ξ_y * (2ξ_y - 1)
        N[3] = γ * (2γ - 1)
        N[4] = 4ξ_x * ξ_y
        N[5] = 4ξ_y * γ
        N[6] = 4ξ_x * γ
    end

    return N
end

function derivative!{T}(ip::Lagrange{2, RefTetrahedron, 2}, dN::AbstractVector{Vec{2, T}}, ξ::Vec{2, T})
    checkdim_derivative(ip, dN, ξ)

    @inbounds begin

        ξ_x = ξ[1]
        ξ_y = ξ[2]

        γ = 1 - ξ_x - ξ_y

        dN[1] = Vec{2,T}((4ξ_x - 1, zero(T)))
        dN[2] = Vec{2,T}((zero(T),  4ξ_y - 1))
        dN[3] = Vec{2,T}((-4γ + 1,  -4γ + 1))
        dN[4] = Vec{2,T}((4ξ_y,  4ξ_x))
        dN[5] = Vec{2,T}((-4ξ_y,  4(γ - ξ_y)))
        dN[6] = Vec{2,T}((4(γ - ξ_x),  -4ξ_x))
    end

    return dN
end

#########################################
# Lagrange dim 3 RefTetrahedron order 1 #
#########################################
getnbasefunctions(::Type{Lagrange{3, RefTetrahedron, 1}}) = 4
getnboundarynodes(::Type{Lagrange{3, RefTetrahedron, 1}}) = 3
getboundarylist(  ::Type{Lagrange{3, RefTetrahedron, 1}}) = ((1,2,3),(1,2,4),(2,3,4),(1,3,4))

function value!(ip::Lagrange{3, RefTetrahedron, 1}, N::AbstractVector, ξ::Vec{3})
    checkdim_value(ip, N, ξ)

    @inbounds begin
        ξ_x = ξ[1]
        ξ_y = ξ[2]
        ξ_z = ξ[3]

        N[1] = 1.0 - ξ_x - ξ_y - ξ_z
        N[2] = ξ_x
        N[3] = ξ_y
        N[4] = ξ_z
    end

    return N
end

function derivative!{T}(ip::Lagrange{3, RefTetrahedron, 1}, dN::AbstractVector{Vec{3, T}}, ξ::Vec{3, T})
    checkdim_derivative(ip, dN, ξ)

    @inbounds begin
        dN[1] = Vec{3, T}((-1.0, -1.0, -1.0))
        dN[2] = Vec{3, T}(( 1.0,  0.0, 0.0))
        dN[3] = Vec{3, T}(( 0.0,  1.0, 0.0))
        dN[4] = Vec{3, T}(( 0.0,  0.0, 1.0))
    end

    return dN
end

##################################
# Lagrange dim 3 RefCube order 1 #
##################################
getnbasefunctions(::Type{Lagrange{3, RefCube, 1}}) = 8
getnboundarynodes(::Type{Lagrange{3, RefCube, 1}}) = 4
getboundarylist(  ::Type{Lagrange{3, RefCube, 1}}) = ((1,2,3,4),(1,2,5,6),(2,3,6,7),(3,4,7,8),(1,4,5,8),(5,6,7,8))

function value!(ip::Lagrange{3, RefCube, 1}, N::AbstractVector, ξ::Vec{3})
    checkdim_value(ip, N, ξ)

    @inbounds begin
        ξ_x = ξ[1]
        ξ_y = ξ[2]
        ξ_z = ξ[3]

        N[1]  = 0.125(1 - ξ_x) * (1 - ξ_y) * (1 - ξ_z)
        N[2]  = 0.125(1 + ξ_x) * (1 - ξ_y) * (1 - ξ_z)
        N[3]  = 0.125(1 + ξ_x) * (1 + ξ_y) * (1 - ξ_z)
        N[4]  = 0.125(1 - ξ_x) * (1 + ξ_y) * (1 - ξ_z)
        N[5]  = 0.125(1 - ξ_x) * (1 - ξ_y) * (1 + ξ_z)
        N[7]  = 0.125(1 + ξ_x) * (1 + ξ_y) * (1 + ξ_z)
        N[6]  = 0.125(1 + ξ_x) * (1 - ξ_y) * (1 + ξ_z)
        N[8]  = 0.125(1 - ξ_x) * (1 + ξ_y) * (1 + ξ_z)
    end

    return N
end

function derivative!{T}(ip::Lagrange{3, RefCube, 1}, dN::AbstractVector{Vec{3, T}}, ξ::Vec{3, T})
    checkdim_derivative(ip, dN, ξ)

    @inbounds begin
        ξ_x = ξ[1]
        ξ_y = ξ[2]
        ξ_z = ξ[3]

        dN[1]  = Vec{3,T}(( -0.125(1 - ξ_y) * (1 - ξ_z),   -0.125(1 - ξ_x) * (1 - ξ_z),   -0.125(1 - ξ_x) * (1 - ξ_y)))
        dN[2]  = Vec{3,T}((  0.125(1 - ξ_y) * (1 - ξ_z),   -0.125(1 + ξ_x) * (1 - ξ_z),   -0.125(1 + ξ_x) * (1 - ξ_y)))
        dN[3]  = Vec{3,T}((  0.125(1 + ξ_y) * (1 - ξ_z),    0.125(1 + ξ_x) * (1 - ξ_z),   -0.125(1 + ξ_x) * (1 + ξ_y)))
        dN[4]  = Vec{3,T}(( -0.125(1 + ξ_y) * (1 - ξ_z),    0.125(1 - ξ_x) * (1 - ξ_z),   -0.125(1 - ξ_x) * (1 + ξ_y)))
        dN[5]  = Vec{3,T}(( -0.125(1 - ξ_y) * (1 + ξ_z),   -0.125(1 - ξ_x) * (1 + ξ_z),    0.125(1 - ξ_x) * (1 - ξ_y)))
        dN[6]  = Vec{3,T}((  0.125(1 - ξ_y) * (1 + ξ_z),   -0.125(1 + ξ_x) * (1 + ξ_z),    0.125(1 + ξ_x) * (1 - ξ_y)))
        dN[7]  = Vec{3,T}((  0.125(1 + ξ_y) * (1 + ξ_z),    0.125(1 + ξ_x) * (1 + ξ_z),    0.125(1 + ξ_x) * (1 + ξ_y)))
        dN[8]  = Vec{3,T}(( -0.125(1 + ξ_y) * (1 + ξ_z),    0.125(1 - ξ_x) * (1 + ξ_z),    0.125(1 - ξ_x) * (1 + ξ_y)))
    end

    return dN
end

###############
# Serendipity #
###############
immutable Serendipity{dim, shape, order} <: Interpolation{dim, shape, order} end

#####################################
# Serendipity dim 2 RefCube order 2 #
#####################################
getlowerdim(::Serendipity{2, RefCube, 2}) = Lagrange{1, RefCube, 2}()
getlowerorder(::Serendipity{2, RefCube, 2}) = Lagrange{2, RefCube, 1}()

getnbasefunctions(::Type{Serendipity{2, RefCube, 2}}) = 8
getnboundarynodes(::Type{Serendipity{2, RefCube, 2}}) = 3
getboundarylist(  ::Type{Serendipity{2, RefCube, 2}}) = ((1,2,5),(2,3,6),(3,4,7),(1,4,8))

function value!(ip::Serendipity{2, RefCube, 2}, N::AbstractVector, ξ::Vec{2})
    checkdim_value(ip, N, ξ)

    ξ_x = ξ[1]
    ξ_y = ξ[2]

    @inbounds begin
        N[1] = (1 - ξ_x) * (1 - ξ_y) * 0.25(-ξ_x - ξ_y - 1)
        N[2] = (1 + ξ_x) * (1 - ξ_y) * 0.25( ξ_x - ξ_y - 1)
        N[3] = (1 + ξ_x) * (1 + ξ_y) * 0.25( ξ_x + ξ_y - 1)
        N[4] = (1 - ξ_x) * (1 + ξ_y) * 0.25(-ξ_x + ξ_y - 1)
        N[6] = 0.5(1 + ξ_x) * (1 - ξ_y * ξ_y)
        N[5] = 0.5(1 - ξ_x * ξ_x) * (1 - ξ_y)
        N[7] = 0.5(1 - ξ_x * ξ_x) * (1 + ξ_y)
        N[8] = 0.5(1 - ξ_x) * (1 - ξ_y * ξ_y)
    end
    return N
end

function derivative!{T}(ip::Serendipity{2, RefCube, 2}, dN::AbstractVector{Vec{2, T}}, ξ::Vec{2, T})
    checkdim_derivative(ip, dN, ξ)

    ξ_x = ξ[1]
    ξ_y = ξ[2]

    @inbounds begin
        dN[1] = Vec{2, T}((-0.25(1 - ξ_y) * (-2ξ_x - ξ_y), -0.25(1 - ξ_x) * (-2ξ_y - ξ_x)))
        dN[2] = Vec{2, T}(( 0.25(1 - ξ_y) * ( 2ξ_x - ξ_y), -0.25(1 + ξ_x) * (-2ξ_y + ξ_x)))
        dN[3] = Vec{2, T}(( 0.25(1 + ξ_y) * ( 2ξ_x + ξ_y),  0.25(1 + ξ_x) * ( 2ξ_y + ξ_x)))
        dN[4] = Vec{2, T}((-0.25(1 + ξ_y) * (-2ξ_x + ξ_y),  0.25(1 - ξ_x) * ( 2ξ_y - ξ_x)))
        dN[5] = Vec{2, T}(( -ξ_x*(1 - ξ_y),  -0.5(1 - ξ_x * ξ_x)))
        dN[6] = Vec{2, T}((  0.5(1 - ξ_y * ξ_y),  -ξ_y*(1 + ξ_x)))
        dN[7] = Vec{2, T}(( -ξ_x*(1 + ξ_y),   0.5(1 - ξ_x * ξ_x)))
        dN[8] = Vec{2, T}(( -0.5(1 - ξ_y * ξ_y),  -ξ_y*(1 - ξ_x)))
    end
    return dN
end
