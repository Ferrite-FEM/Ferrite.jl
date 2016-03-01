abstract FunctionSpace{order, shape, dim}

@inline n_dim{order, shape, dim}(fs::FunctionSpace{order, shape, dim}) = dim
@inline ref_shape{order, shape, dim}(fs::FunctionSpace{order, shape, dim}) = shape()
@inline order{order, shape, dim}(fs::FunctionSpace{order, shape, dim}) = order

"""
Computes the value of the shape functions at a point ξ for a given function space
"""
function value{order, shape, dim}(fs::FunctionSpace{order, shape, dim}, ξ::Vector)
    value!(fs, zeros(eltype(ξ), n_basefunctions(fs)), ξ)
end

"""
Computes the gradients of the shape functions at a point ξ for a given function space
"""
function derivative{order, shape, dim}(fs::FunctionSpace{order, shape, dim}, ξ::Vector)
    derivative!(fs, zeros(eltype(ξ), n_dim(fs), n_basefunctions(fs)), ξ)
end

############
# Lagrange
############

type Lagrange{order, shape, dim} <: FunctionSpace{order, shape, dim} end

#####################
# Lagrange 1 Square #
#####################

n_basefunctions(::Lagrange{1, Square, 1}) = 2

function value!(::Lagrange{1, Square, 1}, N::Vector, ξ::Vector)
    length(N) == 2 || throw(ArgumentError("N must have length 2"))
    length(ξ) == 1 || throw(ArgumentError("ξ must have length 1"))

    @inbounds begin
        ξ_x = ξ[1]

        N[1] = (1 - ξ_x) * 0.5
        N[2] = (1 + ξ_x) * 0.5
    end

    return N
end

function derivative!(::Lagrange{1, Square, 1}, dN::Matrix, ξ::Vector)
    size(dN) == (1,2) || throw(ArgumentError("dN must have size (1,2)"))
    length(ξ) == 1 || throw(ArgumentError("ξ must have length 1"))

    @inbounds begin
        ξ_x = ξ[1]

        dN[1,1] = -0.5
        dN[1,2] =  0.5
    end

    return dN
end

#####################
# Lagrange 2 Square #
#####################

n_basefunctions(::Lagrange{2, Square, 1}) = 3

function value!(::Lagrange{2, Square, 1}, N::Vector, ξ::Vector)
    length(N) == 3 || throw(ArgumentError("N must have length 3"))
    length(ξ) == 1 || throw(ArgumentError("ξ must have length 1"))

    @inbounds begin
        ξ_x = ξ[1]

        N[1] = ξ_x * (ξ_x - 1) * 0.5
        N[2] = 1 - ξ_x^2
        N[3] = ξ_x * (ξ_x + 1) * 0.5
    end

    return N
end



function derivative!(::Lagrange{2, Square, 1}, dN::Matrix, ξ::Vector)
    size(dN) == (1,3) || throw(ArgumentError("dN must have size (1,3)"))
    length(ξ) == 1 || throw(ArgumentError("ξ must have length 1"))

    @inbounds begin
        ξ_x = ξ[1]

        dN[1,1] = ξ_x - 0.5
        dN[1,2] = -2 * ξ_x
        dN[1,3] = ξ_x + 0.5
    end

    return dN
end

###################
# Lagrange 1 Square
###################

n_basefunctions(::Lagrange{1, Square, 2}) = 4

function value!(::Lagrange{1, Square, 2}, N::Vector, ξ::Vector)
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

function derivative!(::Lagrange{1, Square, 2}, dN::Matrix, ξ::Vector)
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

n_basefunctions(::Lagrange{1, Triangle, 2}) = 3

function value!(::Lagrange{1, Triangle, 2}, N::Vector, ξ::Vector)
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

function derivative!(::Lagrange{1, Triangle, 2}, dN::Matrix, ξ::Vector)
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

#####################
# Lagrange 2 Triangle
#####################

n_basefunctions(::Lagrange{2, Triangle, 2}) = 6

function value!(::Lagrange{2, Triangle, 2}, N::Vector, ξ::Vector)
    length(N) == 6 || throw(ArgumentError("N must have length 6"))
    length(ξ) == 2 || throw(ArgumentError("ξ must have length 2"))

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

function derivative!(::Lagrange{2, Triangle, 2}, dN::Matrix, ξ::Vector)
    size(dN) == (2, 6) || throw(ArgumentError("dN must have size (2, 6)"))
    length(ξ) == 2 || throw(ArgumentError("ξ must have length 2"))

    @inbounds begin

        ξ_x = ξ[1]
        ξ_y = ξ[2]

        γ = 1 - ξ_x - ξ_y

        dN[1, 1] = 4ξ_x - 1
        dN[1, 2] = 0
        dN[1, 3] = -4γ + 1
        dN[1, 4] = 4ξ_y
        dN[1, 5] = -4ξ_y
        dN[1, 6] = 4(γ - ξ_x)

        dN[2, 1] = 0
        dN[2, 2] = 4ξ_y - 1
        dN[2, 3] = -4γ + 1
        dN[2, 4] = 4ξ_x
        dN[2, 5] = 4(γ - ξ_y)
        dN[2, 6] = -4ξ_x
    end

    return dN
end


#####################
# Lagrange 1 Square #
#####################

n_basefunctions(::Lagrange{1, Square, 3}) = 8

function value!(::Lagrange{1, Square, 3}, N::Vector, ξ::Vector)
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

function derivative!(fs::Lagrange{1, Square, 3}, dN::Matrix, ξ::Vector)

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

type Serendipity{order, shape, dim} <: FunctionSpace{order, shape, dim} end

n_basefunctions(::Serendipity{2, Square, 2}) = 8

function value!(::Serendipity{2, Square, 2}, N::Vector, ξ::Vector)
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

function derivative!(::Serendipity{2, Square, 2}, dN::Matrix, ξ::Vector)
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
