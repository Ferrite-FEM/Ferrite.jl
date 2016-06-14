"""

**Constructor:**

    FunctionSpace{dim, reference_shape, order}()

**Parameters:**

* `dim` The dimension the function space lives in
* `shape` A reference shape, see [`AbstractRefShape`](@ref)
* `order` The highest order term in the polynomial

The following function spaces are implemented:

* `Lagrange{1, RefCube, 1}`
* `Lagrange{1, RefCube, 2}`
* `Lagrange{2, RefCube, 1}`
* `Lagrange{2, RefTetrahedron, 1}`
* `Lagrange{2, RefTetrahedron, 2}`
* `Lagrange{3, RefCube, 1}`
* `Serendipity{2, RefCube, 2}`
* `Lagrange{3, RefTetrahedron, 1}`
"""
abstract FunctionSpace{dim, shape, order}

@inline n_dim{dim}(fs::FunctionSpace{dim}) = dim
@inline ref_shape{dim, shape}(fs::FunctionSpace{dim, shape}) = shape()
@inline order{dim, shape, order}(fs::FunctionSpace{dim, shape, order}) = order

"""
Computes the value of the shape functions at a point ξ for a given function space
"""
function value{dim, T}(fs::FunctionSpace{dim}, ξ::Vec{dim, T})
    value!(fs, zeros(T, n_basefunctions(fs)), ξ)
end

"""
Computes the gradients of the shape functions at a point ξ for a given function space
"""
function derivative{dim, T}(fs::FunctionSpace{dim}, ξ::Vec{dim, T})
    derivative!(fs, [zero(Tensor{1, dim, T}) for i in 1:n_basefunctions(fs)], ξ)
end

@inline function checkdim_value{dim}(fs::FunctionSpace{dim}, N::Vector, ξ::Vec{dim})
    n_base = n_basefunctions(fs)
    length(N) == n_base || throw(ArgumentError("N must have length $(n_base)"))
end

@inline function checkdim_derivative{dim, T}(fs::FunctionSpace{dim}, dN::Vector{Vec{dim, T}}, ξ::Vec{dim, T})
    n_base = n_basefunctions(fs)
    length(dN) == n_base || throw(ArgumentError("dN must have length $(n_base)"))
end

############
# Lagrange
############

type Lagrange{dim, shape, order} <: FunctionSpace{dim, shape, order} end

#################################
# Lagrange dim 1 RefCube order 1 #
#################################

n_basefunctions(::Lagrange{1, RefCube, 1}) = 2

function value!(fs::Lagrange{1, RefCube, 1}, N::Vector, ξ::Vec{1})
    checkdim_value(fs, N, ξ)

    @inbounds begin
        ξ_x = ξ[1]

        N[1] = (1 - ξ_x) * 0.5
        N[2] = (1 + ξ_x) * 0.5
    end

    return N
end

function derivative!{T}(fs::Lagrange{1, RefCube, 1}, dN::Vector{Vec{1, T}}, ξ::Vec{1, T})
    checkdim_derivative(fs, dN, ξ)

    @inbounds begin
        dN[1] = Vec{1,T}((-0.5,))
        dN[2] = Vec{1,T}((0.5,))
    end

    return dN
end

function reference_coordinates(fs::Lagrange{1, RefCube, 1})
    return (Vec{1, Float64}((-1.0,)),
            Vec{1, Float64}(( 1.0,)))
end

#################################
# Lagrange dim 1 RefCube order 2 #
#################################

n_basefunctions(::Lagrange{1, RefCube, 2}) = 3

function value!(fs::Lagrange{1, RefCube, 2}, N::Vector, ξ::Vec{1})
    checkdim_value(fs, N, ξ)

    @inbounds begin
        ξ_x = ξ[1]

        N[1] = ξ_x * (ξ_x - 1) * 0.5
        N[2] = 1 - ξ_x^2
        N[3] = ξ_x * (ξ_x + 1) * 0.5
    end

    return N
end



function derivative!{T}(fs::Lagrange{1, RefCube, 2}, dN::Vector{Vec{1, T}}, ξ::Vec{1, T})
    checkdim_derivative(fs, dN, ξ)

    @inbounds begin
        ξ_x = ξ[1]

        dN[1] = Vec{1,T}((ξ_x - 0.5,))
        dN[2] = Vec{1,T}((-2 * ξ_x,))
        dN[3] = Vec{1,T}((ξ_x + 0.5,))
    end

    return dN
end

function reference_coordinates(fs::Lagrange{1, RefCube, 2})
    return (Vec{1, Float64}((-1.0,)),
            Vec{1, Float64}(( 0.0,)),
            Vec{1, Float64}(( 1.0,)))
end

#################################
# Lagrange dim 2 RefCube order 1 #
#################################

n_basefunctions(::Lagrange{2, RefCube, 1}) = 4

function value!(fs::Lagrange{2, RefCube, 1}, N::Vector, ξ::Vec{2})
    checkdim_value(fs, N, ξ)

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

function derivative!{T}(fs::Lagrange{2, RefCube, 1}, dN::Vector{Vec{2, T}}, ξ::Vec{2, T})
    checkdim_derivative(fs, dN, ξ)

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

function reference_coordinates(fs::Lagrange{2, RefCube, 1})
    return (Vec{2, Float64}((-1.0, -1.0)),
            Vec{2, Float64}(( 1.0, -1.0)),
            Vec{2, Float64}(( 1.0,  1.0,)),
            Vec{2, Float64}((-1.0,  1.0,)))
end


###################################
# Lagrange dim 2 RefTetrahedron order 1 #
###################################

n_basefunctions(::Lagrange{2, RefTetrahedron, 1}) = 3

function value!(fs::Lagrange{2, RefTetrahedron, 1}, N::Vector, ξ::Vec{2})
    checkdim_value(fs, N, ξ)

    @inbounds begin
        ξ_x = ξ[1]
        ξ_y = ξ[2]

        γ = 1 - ξ_x - ξ_y

        N[1] = ξ_x
        N[2] = ξ_y
        N[3] = γ
    end

    return N
end

function derivative!{T}(fs::Lagrange{2, RefTetrahedron, 1}, dN::Vector{Vec{2, T}}, ξ::Vec{2, T})
    checkdim_derivative(fs, dN, ξ)

    @inbounds begin
        dN[1] = Vec{2, T}(( 1.0,  0.0))
        dN[2] = Vec{2, T}(( 0.0,  1.0))
        dN[3] = Vec{2, T}((-1.0, -1.0))
    end

    return dN
end

function reference_coordinates(fs::Lagrange{2, RefTetrahedron, 1})
    return (Vec{2, Float64}((1.0, 0.0)),
            Vec{2, Float64}((0.0, 1.0)),
            Vec{2, Float64}((0.0, 0.0)))
end


###################################
# Lagrange dim 2 RefTetrahedron order 2 #
###################################

n_basefunctions(::Lagrange{2, RefTetrahedron, 2}) = 6

function value!(fs::Lagrange{2, RefTetrahedron, 2}, N::Vector, ξ::Vec{2})
    checkdim_value(fs, N, ξ)

    @inbounds begin
        ξ_x = ξ[1]
        ξ_y = ξ[2]

        γ = 1 - ξ_x - ξ_y

        N[1] = ξ_x * (2ξ_x - 1)
        N[2] = ξ_y * (2ξ_y - 1)
        N[3] = γ * (2γ - 1)
        N[4] = 4ξ_x * ξ_y
        N[5] = 4ξ_y * γ
        N[6] = 4ξ_x * γ
    end

    return N
end

function derivative!{T}(fs::Lagrange{2, RefTetrahedron, 2}, dN::Vector{Vec{2, T}}, ξ::Vec{2, T})
    checkdim_derivative(fs, dN, ξ)

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

function reference_coordinates(fs::Lagrange{2, RefTetrahedron, 2})
    return (Vec{2, Float64}((1.0, 0.0)),
            Vec{2, Float64}((0.0, 1.0)),
            Vec{2, Float64}((0.0, 0.0)),
            Vec{2, Float64}((0.5, 0.5)),
            Vec{2, Float64}((0.0, 0.5)),
            Vec{2, Float64}((0.5, 0.0)))
end


###################################
# Lagrange dim 3 RefTetrahedron order 1 #
###################################

n_basefunctions(::Lagrange{3, RefTetrahedron, 1}) = 4

function value!(fs::Lagrange{3, RefTetrahedron, 1}, N::Vector, ξ::Vec{3})
    checkdim_value(fs, N, ξ)

    @inbounds begin
        ξ_x = ξ[1]
        ξ_y = ξ[2]
        ξ_z = ξ[3]

        γ = 1. - ξ_x - ξ_y - ξ_z

        N[1] = ξ_x
        N[2] = ξ_y
        N[3] = ξ_z
        N[4] = γ
    end

    return N
end

function derivative!{T}(fs::Lagrange{3, RefTetrahedron, 1}, dN::Vector{Vec{3, T}}, ξ::Vec{3, T})
    checkdim_derivative(fs, dN, ξ)

    @inbounds begin
        dN[1] = Vec{3, T}(( 1.0,  0.0, 0.0))
        dN[2] = Vec{3, T}(( 0.0,  1.0, 0.0))
        dN[3] = Vec{3, T}(( 0.0,  0.0, 1.0))
        dN[4] = Vec{3, T}((-1.0, -1.0, -1.0))
    end

    return dN
end

function reference_coordinates(fs::Lagrange{3, RefTetrahedron, 1})
    return (Vec{3, Float64}((1.0, 0.0, 0.0)),
            Vec{3, Float64}((0.0, 1.0, 0.0)),
            Vec{3, Float64}((0.0, 0.0, 1.0)),
            Vec{3, Float64}((0.0, 0.0, 0.0)))
end

VTK_type(fs::Lagrange{3, RefTetrahedron, 1}) = VTKCellType.VTK_TETRA

###################################
# Lagrange dim 3 RefCube order 1 #
###################################

n_basefunctions(::Lagrange{3, RefCube, 1}) = 8

function value!(fs::Lagrange{3, RefCube, 1}, N::Vector, ξ::Vec{3})
    checkdim_value(fs, N, ξ)

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

function derivative!{T}(fs::Lagrange{3, RefCube, 1}, dN::Vector{Vec{3, T}}, ξ::Vec{3, T})
    checkdim_derivative(fs, dN, ξ)

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

function reference_coordinates(fs::Lagrange{3, RefCube, 1})
    return (Vec{3, Float64}((-1.0, -1.0, -1.0)),
            Vec{3, Float64}(( 1.0, -1.0, -1.0)),
            Vec{3, Float64}(( 1.0,  1.0, -1.0)),
            Vec{3, Float64}((-1.0,  1.0, -1.0)),
            Vec{3, Float64}((-1.0, -1.0,  1.0)),
            Vec{3, Float64}(( 1.0, -1.0,  1.0)),
            Vec{3, Float64}(( 1.0,  1.0,  1.0)),
            Vec{3, Float64}((-1.0,  1.0,  1.0)))
end


####################################
# Serendipity dim 2 RefCube order 2 #
####################################

type Serendipity{dim, shape, order} <: FunctionSpace{dim, shape, order} end

n_basefunctions(::Serendipity{2, RefCube, 2}) = 8

function value!(fs::Serendipity{2, RefCube, 2}, N::Vector, ξ::Vec{2})
    checkdim_value(fs, N, ξ)

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

function derivative!{T}(fs::Serendipity{2, RefCube, 2}, dN::Vector{Vec{2, T}}, ξ::Vec{2, T})
    checkdim_derivative(fs, dN, ξ)

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

function reference_coordinates(fs::Serendipity{2, RefCube, 2})
    return (Vec{2, Float64}((-1.0, -1.0)),
            Vec{2, Float64}(( 1.0, -1.0)),
            Vec{2, Float64}(( 1.0,  1.0)),
            Vec{2, Float64}((-1.0,  1.0)),
            Vec{2, Float64}(( 0.0, -1.0)),
            Vec{2, Float64}(( 1.0,  0.0)),
            Vec{2, Float64}(( 0.0,  1.0)),
            Vec{2, Float64}((-1.0,  0.0)))
end
