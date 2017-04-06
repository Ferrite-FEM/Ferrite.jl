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
@compat abstract type Interpolation{dim, shape, order} end

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


_get_n_celldofs(::Interpolation) = 0
_get_n_facedofs(::Interpolation) = 0
_get_n_edgedofs(::Interpolation) = 0
_get_n_vertexdofs(::Interpolation) = 0

"""
Computes the gradients of the shape functions at a point ξ for a given interpolation
"""
function derivative{dim, T}(ip::Interpolation{dim}, ξ::Vec{dim, T})
    derivative!(ip, [zero(Tensor{1, dim, T}) for i in 1:getnbasefunctions(ip)], ξ)
end

@inline function checkdim_value{dim}(ip::Interpolation{dim}, N::AbstractVector, ξ::AbstractVector)
    @assert length(ξ) == dim
    n_base = getnbasefunctions(ip)
    length(N) == n_base || throw(ArgumentError("N must have length $(n_base)"))
end
    n_base = getnbasefunctions(ip)
    length(dN) == n_base || throw(ArgumentError("dN must have length $(n_base)"))
end

function derivative!{T, order, shape, dim}(ip::Interpolation{dim, shape, order}, dN::AbstractVector{Vec{dim, T}}, ξ::Vec{dim, T})
    checkdim_derivative(ip, dN, ξ)
    f!(N, x) = value!(ip, N, x)
    NArray = zeros(T, getnbasefunctions(ip))
    dNArray = ForwardDiff.jacobian(f!, NArray, ξ)
    # Want to use reinterpret but that crashes Julia 0.6, #20847
    for i in 1:length(dN)
        dN[i] = Vec{dim, T}((dNArray[i, :]...))
    end
    return dN
end

"""
Returns the number of base functions for an [`Interpolation`](@ref) or `Values` object.
"""
getnbasefunctions

function derivative!{T}(ip::Interpolation{2, RefCube, 2}, dN::AbstractVector{Vec{2, T}}, ξ::Vec{2, T})
    value!
    ForwardDiff.jacobian( ξ)
end

############
# Lagrange #
############
immutable Lagrange{dim, shape, order} <: Interpolation{dim, shape, order} end


getnbasefunctions{dim, order}(::Lagrange{dim, RefCube, order}) = (order + 1)^dim
getnfacenodes{order}(::Lagrange{1, RefCube, order}) = (order + 1)^(dim - 1)
getnfacenodes{dim, order}(::Lagrange{dim, RefCube, order}) = (order + 1)^(dim - 1)

function value!(ip::Lagrange{1, RefCube, 1}, N::AbstractVector, ξ::AbstractVector)
    checkdim_value(ip, N, ξ)

# Evaluates the lagrange polynomial number j at point x with xs as basis points
function lagrange_polynomial(x::Number, xs::AbstractVector, j::Int)
    @assert j <= length(xs)
    num, den = one(x), one(x)
    @inbounds for i in 1:length(xs)
        i == j && continue
        num *= (x - xs[i])
        den *= (xs[j] - xs[i])
    end
    return num / den
end

function evaluate_Nmatrix{T, order, dim}(ip::Lagrange{dim, 2, order}, ξ::AbstractVector{T})
    @assert length(ξ) == dim
    x = gausslobatto(order + 1)[1]
    N =  ones(T, ntuple(i -> 1+order, dim)...)
    for k in 1:dim
        for i in 1:order+1
            # taken from the code in slicedim
            N[( n==k ? i : indices(N, n) for n in 1:ndims(N) )...] .*= lagrange_polynomial(ξ[k], x, i)
        end
    end
    return N
end

##################################
# Lagrange dim 1 RefCube order 2 #
##################################
getnbasefunctions(::Lagrange{1, RefCube, 2}) = 3

function value!(ip::Lagrange{1, RefCube, 2}, N::AbstractVector, ξ::AbstractVector)
    checkdim_value(ip, N, ξ)

    @inbounds begin
        ξ_x = ξ[1]

        N[1] = ξ_x * (ξ_x - 1) * 0.5
        N[2] = ξ_x * (ξ_x + 1) * 0.5
        N[3] = 1 - ξ_x^2
    end
    return N
end

##################################
# Lagrange dim 2 RefCube order 1 #
##################################
getnbasefunctions(::Lagrange{2, RefCube, 1}) = 4

function value!(ip::Lagrange{2, RefCube, 1}, N::AbstractVector, ξ::AbstractVector)
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

##################################
# Lagrange dim 2 RefCube order 2 #
##################################
getnbasefunctions(::Lagrange{2, RefCube, 2}) = 9

function value!(ip::Lagrange{2, RefCube, 2}, N::AbstractVector, ξ::AbstractVector)
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

#########################################
# Lagrange dim 2 RefTetrahedron order 1 #
#########################################
getnbasefunctions(::Lagrange{2, RefTetrahedron, 1}) = 3

function value!(ip::Lagrange{2, RefTetrahedron, 1}, N::AbstractVector, ξ::AbstractVector)
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

#########################################
# Lagrange dim 2 RefTetrahedron order 2 #
#########################################
getnbasefunctions(::Lagrange{2, RefTetrahedron, 2}) = 6

function value!(ip::Lagrange{2, RefTetrahedron, 2}, N::AbstractVector, ξ::AbstractVector)
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

#########################################
# Lagrange dim 3 RefTetrahedron order 1 #
#########################################
getnbasefunctions(::Lagrange{3, RefTetrahedron, 1}) = 4

function value!(ip::Lagrange{3, RefTetrahedron, 1}, N::AbstractVector, ξ::AbstractVector)
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

#########################################
# Lagrange dim 3 RefTetrahedron order 2 #
#########################################
getnbasefunctions(::Lagrange{3, RefTetrahedron, 2}) = 10

# http://www.colorado.edu/engineering/CAS/courses.d/AFEM.d/AFEM.Ch09.d/AFEM.Ch09.pdf
# http://www.colorado.edu/engineering/CAS/courses.d/AFEM.d/AFEM.Ch10.d/AFEM.Ch10.pdf
function value!(ip::Lagrange{3, RefTetrahedron, 2}, N::AbstractVector, ξ::AbstractVector)
    checkdim_value(ip, N, ξ)

    @inbounds begin
        ξ_x = ξ[1]
        ξ_y = ξ[2]
        ξ_z = ξ[3]

        N[1]  = (-2 * ξ_x - 2 * ξ_y - 2 * ξ_z + 1) * (-ξ_x - ξ_y - ξ_z + 1)
        N[2]  = ξ_x * (2 * ξ_x - 1)
        N[3]  = ξ_y * (2 * ξ_y - 1)
        N[4]  = ξ_z * (2 * ξ_z - 1)
        N[5]  = ξ_x * (-4 * ξ_x - 4 * ξ_y - 4 * ξ_z + 4)
        N[6]  = 4 * ξ_x * ξ_y
        N[7]  = 4 * ξ_y * (-ξ_x - ξ_y - ξ_z + 1)
        N[8]  = ξ_z * (-4 * ξ_x - 4 * ξ_y - 4 * ξ_z + 4)
        N[9]  = 4 * ξ_x * ξ_z
        N[10] = 4 * ξ_y * ξ_z
    end

    return N
end

##################################
# Lagrange dim 3 RefCube order 1 #
##################################
getnbasefunctions(::Lagrange{3, RefCube, 1}) = 8

function value!(ip::Lagrange{3, RefCube, 1}, N::AbstractVector, ξ::AbstractVector)
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

###############
# Serendipity #
###############
immutable Serendipity{dim, shape, order} <: Interpolation{dim, shape, order} end

#####################################
# Serendipity dim 2 RefCube order 2 #
#####################################
getnbasefunctions(::Serendipity{2, RefCube, 2}) = 8

function value!(ip::Serendipity{2, RefCube, 2}, N::AbstractVector, ξ::AbstractVector)
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
