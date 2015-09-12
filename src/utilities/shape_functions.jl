# function format:
# x_y_z
# x: N = shape_function, dN = derivative shape function
# y: geometry, T = triangle, Q = quadratic
# z: polynomial order

evaluate_N(  ::Lagrange_S_1, ξ::Vector) = N_Q_1(ξ)
evaluate_dN( ::Lagrange_S_1, ξ::Vector) = dN_Q_1(ξ)
evaluate_N!( ::Lagrange_S_1, N::Vector, ξ::Vector) = N_Q_1!(N, ξ)
evaluate_dN!(::Lagrange_S_1, dN::Matrix, ξ::Vector) = dN_Q_1!(dN, ξ)

evaluate_N(  ::Lagrange_T_1, ξ::Vector) = N_T_1(ξ)
evaluate_dN( ::Lagrange_T_1, ξ::Vector) = dN_T_1(ξ)
evaluate_N!( ::Lagrange_T_1, N::Vector, ξ::Vector) = N_T_1!(N, ξ)
evaluate_dN!(::Lagrange_T_1, dN::Matrix, ξ::Vector) = dN_T_1!(dN, ξ)

evaluate_N(  ::Lagrange_S_2, ξ::Vector) = N_Q_2(ξ)
evaluate_dN( ::Lagrange_S_2, ξ::Vector) = dN_Q_2(ξ)
evaluate_N!( ::Lagrange_S_2, N::Vector, ξ::Vector) = N_Q_2!(N, ξ)
evaluate_dN!(::Lagrange_S_2, dN::Matrix, ξ::Vector) = dN_Q_2!(dN, ξ)

evaluate_N(  ::Lagrange_C_1, ξ::Vector) = N_C_1(ξ)
evaluate_dN( ::Lagrange_C_1, ξ::Vector) = dN_C_1(ξ)
evaluate_N!( ::Lagrange_C_1, N::Vector, ξ::Vector) = N_C_1!(N, ξ)
evaluate_dN!(::Lagrange_C_1, dN::Matrix, ξ::Vector) = dN_C_1!(dN, ξ)
"""
Computes the shape functions at a point for
a bilinear quadriterial element
"""
N_Q_1(ξ::Vector) = N_Q_1!(zeros(4), ξ)

function N_Q_1!(N::Vector, ξ::Vector)
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
a bilinear quadriterial element
"""
dN_Q_1(ξ::Vector) = dN_Q_1!(zeros(2, 4), ξ)

function dN_Q_1!(dN::Matrix, ξ::Vector)
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
N_T_1(ξ::Vector) = N_T_1!(zeros(3), ξ)

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

"""
Computes the shape functions at a point for
a quadratic quadrilateral element
"""
N_Q_2(ξ::Vector) = N_Q_2!(zeros(8), ξ)

function N_Q_2!(N::Vector, ξ::Vector)
    length(N) == 8 || throw(ArgumentError("N must have length 3"))
    length(ξ) == 2 || throw(ArgumentError("ξ must have length 2"))

    ξ_x = ξ[1]
    ξ_y = ξ[2]

    N[1] = (1 - ξ_x) * (1 - ξ_y) * 0.25(-ξ_x - ξ_y - 1)
    N[2] = (1 + ξ_x) * (1 - ξ_y) * 0.25( ξ_x - ξ_y - 1)
    N[3] = (1 + ξ_x) * (1 + ξ_y) * 0.25( ξ_x + ξ_y - 1)
    N[4] = (1 - ξ_x) * (1 + ξ_y) * 0.25(-ξ_x + ξ_y - 1)
    N[6] = 0.5(1 + ξ_x) * (1 - ξ_y * ξ_y)
    N[5] = 0.5(1 - ξ_x * ξ_x) * (1 - ξ_y)
    N[7] = 0.5(1 - ξ_x * ξ_x) * (1 + ξ_y)
    N[8] = 0.5(1 - ξ_x) * (1 - ξ_y * ξ_y)
    return N
end


"""
Computes the derivatives of the shape functions at a point for
a quadratic quadrilateral element
"""
dN_Q_2(ξ::Vector) = dN_Q_2!(zeros(2, 8), ξ)

function dN_Q_2!(dN::Matrix, ξ::Vector)
    size(dN) == (2, 8) || throw(ArgumentError("dN must have size (2, 8)"))
    length(ξ) == 2 || throw(ArgumentError("ξ must have length 2"))

    ξ_x = ξ[1]
    ξ_y = ξ[2]

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
    return dN
end

"""
Computes the shape functions at a point for
a linear cubic element
"""
N_C_1(ξ::Vector) = N_C_1!(zeros(8), ξ)

function N_C_1!(N::Vector, ξ::Vector)
    length(N) == 8 || throw(ArgumentError("N must have length 8"))
    length(ξ) == 3 || throw(ArgumentError("ξ must have length 3"))

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

    return N
end

"""
Computes the derivatives of the shape functions at a point for
a quadratic hexaedric element
"""
dN_C_1(ξ::Vector) = dN_C_1!(zeros(3, 8), ξ)

function dN_C_1!(dN::Matrix, ξ::Vector)

    size(dN) == (3, 8) || throw(ArgumentError("dN must have size (3, 8)"))
    length(ξ) == 3 || throw(ArgumentError("ξ must have length 3"))

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

    return dN
end
