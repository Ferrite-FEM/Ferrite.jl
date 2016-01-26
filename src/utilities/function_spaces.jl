abstract FunctionSpace

@inline n_dim(fs::FunctionSpace) = n_dim(ref_shape(fs))

############
# Lagrange
############

type Lagrange{Order, Shape} <: FunctionSpace end

@inline ref_shape{Order, Shape}(fs::Lagrange{Order, Shape}) = Shape()
@inline order{Order, Shape}(fs::Lagrange{Order, Shape}) = Order

###################
# Lagrange 1 Square
###################

n_basefunctions(::Lagrange{1, Square}) = 4

"""
Computes the shape functions at a point for
a bilinear quadriterial element
"""
value(fs::Lagrange{1, Square}, ξ::Vector) = value!(fs, zeros(eltype(ξ), 4), ξ)

function value!(::Lagrange{1, Square}, N::Vector, ξ::Vector)
    length(N) == 4 || throw(ArgumentError("N must have length 4"))
    length(ξ) == 2 || throw(ArgumentError("ξ must have length 2"))

    @inbounds begin
        ξ_x = ξ[1]
        ξ_y = ξ[2]

        N[1] = (1 + ξ_x) * (1 + ξ_y) * 0.25
        N[2] = (1 - ξ_x) * (1 + ξ_y) * 0.25
        N[3] = (1 - ξ_x) * (1 - ξ_y) * 0.25
        N[4] = (1 + ξ_x) * (1 - ξ_y) * 0.25
    end

    return N
end

"""
Computes the derivatives of the shape functions at a point for
a bilinear quadriterial element
"""
derivative(fs::Lagrange{1, Square}, ξ::Vector) = derivative!(fs, zeros(eltype(ξ), 2, 4), ξ)

function derivative!(::Lagrange{1, Square}, dN::Matrix, ξ::Vector)
    size(dN) == (2, 4) || throw(ArgumentError("dN must have size (2, 4)"))
    length(ξ) == 2 || throw(ArgumentError("ξ must have length 2"))

    @inbounds begin
        ξ_x = ξ[1]
        ξ_y = ξ[2]

        dN[1,1] =  (1 + ξ_y) * 0.25
        dN[2,1] =  (1 + ξ_x) * 0.25

        dN[1,2] = -(1 + ξ_y) * 0.25
        dN[2,2] =  (1 - ξ_x) * 0.25

        dN[1,3] = -(1 - ξ_y) * 0.25
        dN[2,3] = -(1 - ξ_x) * 0.25

        dN[1,4] =  (1 - ξ_y) * 0.25
        dN[2,4] = -(1 + ξ_x) * 0.25
    end

    return dN
end

#####################
# Lagrange 1 Triangle
#####################

n_basefunctions(::Lagrange{1, Triangle}) = 3

"""
Computes the shape functions at a point for
a linear triangle element
"""
value(fs::Lagrange{1, Triangle}, ξ::Vector) = value!(fs, zeros(eltype(ξ), 3), ξ)

function value!(::Lagrange{1, Triangle}, N::Vector, ξ::Vector)
    length(N) == 3 || throw(ArgumentError("N must have length 3"))
    length(ξ) == 2 || throw(ArgumentError("ξ must have length 2"))

    @inbounds begin
        ξ_x = ξ[1]
        ξ_y = ξ[2]

        N[1] = ξ_x
        N[2] = ξ_y
        N[3] = 1.0 - ξ_x - ξ_y
    end

    return N
end

"""
Computes the derivatives of the shape functions at a point for
a linear triangle element
"""
derivative(fs::Lagrange{1, Triangle}, ξ::Vector) = derivative!(fs, zeros(eltype(ξ), 2, 3), ξ)

function derivative!(::Lagrange{1, Triangle}, dN::Matrix, ξ::Vector)
    size(dN) == (2, 3) || throw(ArgumentError("dN must have size (2, 3)"))
    length(ξ) == 2 || throw(ArgumentError("ξ must have length 2"))

    @inbounds begin
        dN[1,1] =  1.0
        dN[2,1] =  0.0

        dN[1,2] = 0.0
        dN[2,2] = 1.0

        dN[1,3] = -1.0
        dN[2,3] = -1.0
    end

    return dN
end

#################
# Lagrange 1 Cube
#################

n_basefunctions(::Lagrange{1, Cube}) = 8

"""
Computes the shape functions at a point for
a linear cubic element
"""
value(fs::Lagrange{1, Cube}, ξ::Vector) = value!(fs, zeros(eltype(ξ), 8), ξ)

function value!(::Lagrange{1, Cube}, N::Vector, ξ::Vector)
    length(N) == 8 || throw(ArgumentError("N must have length 8"))
    length(ξ) == 3 || throw(ArgumentError("ξ must have length 3"))

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

"""
Computes the derivatives of the shape functions at a point for
a quadratic hexaedric element
"""
derivative(fs::Lagrange{1, Cube}, ξ::Vector) = derivative!(fs, zeros(eltype(ξ), 3, 8), ξ)

function derivative!(fs::Lagrange{1, Cube}, dN::Matrix, ξ::Vector)

    size(dN) == (3, 8) || throw(ArgumentError("dN must have size (3, 8)"))
    length(ξ) == 3 || throw(ArgumentError("ξ must have length 3"))

    @inbounds begin
        ξ_x = ξ[1]
        ξ_y = ξ[2]
        ξ_z = ξ[3]

        dN[1, 1] = -0.125(1 - ξ_y) * (1 - ξ_z)
        dN[1, 2] =  0.125(1 - ξ_y) * (1 - ξ_z)
        dN[1, 3] =  0.125(1 + ξ_y) * (1 - ξ_z)
        dN[1, 4] = -0.125(1 + ξ_y) * (1 - ξ_z)
        dN[1, 5] = -0.125(1 - ξ_y) * (1 + ξ_z)
        dN[1, 6] =  0.125(1 - ξ_y) * (1 + ξ_z)
        dN[1, 7] =  0.125(1 + ξ_y) * (1 + ξ_z)
        dN[1, 8] = -0.125(1 + ξ_y) * (1 + ξ_z)

        dN[2, 1] = -0.125(1 - ξ_x) * (1 - ξ_z)
        dN[2, 2] = -0.125(1 + ξ_x) * (1 - ξ_z)
        dN[2, 3] =  0.125(1 + ξ_x) * (1 - ξ_z)
        dN[2, 4] =  0.125(1 - ξ_x) * (1 - ξ_z)
        dN[2, 5] = -0.125(1 - ξ_x) * (1 + ξ_z)
        dN[2, 6] = -0.125(1 + ξ_x) * (1 + ξ_z)
        dN[2, 7] =  0.125(1 + ξ_x) * (1 + ξ_z)
        dN[2, 8] =  0.125(1 - ξ_x) * (1 + ξ_z)

        dN[3, 1] = -0.125(1 - ξ_x) * (1 - ξ_y)
        dN[3, 2] = -0.125(1 + ξ_x) * (1 - ξ_y)
        dN[3, 3] = -0.125(1 + ξ_x) * (1 + ξ_y)
        dN[3, 4] = -0.125(1 - ξ_x) * (1 + ξ_y)
        dN[3, 5] =  0.125(1 - ξ_x) * (1 - ξ_y)
        dN[3, 6] =  0.125(1 + ξ_x) * (1 - ξ_y)
        dN[3, 7] =  0.125(1 + ξ_x) * (1 + ξ_y)
        dN[3, 8] =  0.125(1 - ξ_x) * (1 + ξ_y)
    end

    return dN
end


#################
# Serendipity Q 2
#################


type Serendipity{Order, Shape} <: FunctionSpace end

@inline ref_shape{Order, Shape}(fs::Serendipity{Order, Shape}) = Shape()
@inline order{Order, Shape}(fs::Serendipity{Order, Shape}) = Order

n_basefunctions(::Serendipity{2, Square}) = 8

"""
Computes the shape functions at a point for
a quadratic quadrilateral element
"""
value(fs::Serendipity{2, Square}, ξ::Vector) = value!(fs, zeros(eltype(ξ), 8), ξ)

function value!(::Serendipity{2, Square}, N::Vector, ξ::Vector)
    length(N) == 8 || throw(ArgumentError("N must have length 3"))
    length(ξ) == 2 || throw(ArgumentError("ξ must have length 2"))

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


"""
Computes the derivatives of the shape functions at a point for
a quadratic quadrilateral element
"""
derivative(fs::Serendipity{2, Square}, ξ::Vector) = derivative!(fs, zeros(eltype(ξ), 2, 8), ξ)

function derivative!(::Serendipity{2, Square}, dN::Matrix, ξ::Vector)
    size(dN) == (2, 8) || throw(ArgumentError("dN must have size (2, 8)"))
    length(ξ) == 2 || throw(ArgumentError("ξ must have length 2"))

    ξ_x = ξ[1]
    ξ_y = ξ[2]

    @inbounds begin
        dN[1, 1] = -0.25(1 - ξ_y) * (-2ξ_x - ξ_y)
        dN[1, 2] =  0.25(1 - ξ_y) * ( 2ξ_x - ξ_y)
        dN[1, 3] =  0.25(1 + ξ_y) * ( 2ξ_x + ξ_y)
        dN[1, 4] = -0.25(1 + ξ_y) * (-2ξ_x + ξ_y)
        dN[1, 5] = -ξ_x*(1 - ξ_y)
        dN[1, 6] =  0.5(1 - ξ_y * ξ_y)
        dN[1, 7] = -ξ_x*(1 + ξ_y)
        dN[1, 8] = -0.5(1 - ξ_y * ξ_y)

        dN[2, 1] = -0.25(1 - ξ_x) * (-2ξ_y - ξ_x)
        dN[2, 2] = -0.25(1 + ξ_x) * (-2ξ_y + ξ_x)
        dN[2, 3] =  0.25(1 + ξ_x) * ( 2ξ_y + ξ_x)
        dN[2, 4] =  0.25(1 - ξ_x) * ( 2ξ_y - ξ_x)
        dN[2, 5] = -0.5(1 - ξ_x * ξ_x)
        dN[2, 6] = -ξ_y*(1 + ξ_x)
        dN[2, 7] =  0.5(1 - ξ_x * ξ_x)
        dN[2, 8] = -ξ_y*(1 - ξ_x)
    end
    return dN
end


