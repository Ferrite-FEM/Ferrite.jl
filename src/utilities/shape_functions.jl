# function format:
# x_y_z
# x: N = shape_function, dN = derivative shape function
# y: geometry, T = triangle, Q = quadratic
# z: polynomial order

"""
Computes the shape functions at a point for
a bilinear quadratic element
"""
N_Q_2(ξ::Vector) = N_Q_2!(zeros(4), ξ)

function N_Q_2!(N::Vector, ξ::Vector)
    length(N) == 4 || throw(ArgumentError("N must have length 4"))
    length(ξ) == 2 || throw(ArgumentError("ξ must have length 2"))

    ξ_x = ξ[1]
    ξ_y = ξ[2]

    N[1] = (1 + ξ_x) * (1 + ξ_y) * 0.25
    N[2] = (1 - ξ_x) * (1 + ξ_y) * 0.25
    N[3] = (1 - ξ_x) * (1 - ξ_y) * 0.25
    N[4] = (1 + ξ_x) * (1 - ξ_y) * 0.25

    return N
end

"""
Computes the derivatives of the shape functions at a point for
a bilinear quadratic element
"""
dN_Q_2(ξ::Vector) = dN_Q_2!(zeros(2, 4), ξ)

function dN_Q_2!(dN::Matrix, ξ::Vector)
    size(dN) == (2, 4) || throw(ArgumentError("dN must have size (2, 4)"))
    length(ξ) == 2 || throw(ArgumentError("ξ must have length 2"))

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

    return dN
end


"""
Computes the shape functions at a point for
a linear triangle element
"""
N_T_1(ξ::Vector) = N_Q_2!(zeros(3), ξ)

function N_T_1!(N::Vector, ξ::Vector)
    length(N) == 3 || throw(ArgumentError("N must have length 3"))
    length(ξ) == 2 || throw(ArgumentError("ξ must have length 2"))

    ξ_x = ξ[1]
    ξ_y = ξ[2]

    N[1] = ξ_x
    N[2] = ξ_y
    N[3] = 1.0 - ξ_x - ξ_y

    return N
end

"""
Computes the derivatives of the shape functions at a point for
a linear triangle element
"""
dN_T_1(ξ::Vector) = dN_T_1!(zeros(2, 3), ξ)

function dN_T_1!(dN::Matrix, ξ::Vector)
    size(dN) == (2, 3) || throw(ArgumentError("dN must have size (2, 3)"))
    length(ξ) == 2 || throw(ArgumentError("ξ must have length 2"))

    dN[1,1] =  1.0
    dN[2,1] =  0.0

    dN[1,2] = 0.0
    dN[2,2] = 1.0

    dN[1,3] = -1.0
    dN[2,3] = -1.0

    return dN
end