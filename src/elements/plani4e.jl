# TODO, add plane stress support

"""
Computes the stiffness matrix for a four node isoparametric
quadraterial element
"""
function error_check_plan4(ex, ey, ptype, t, int_order, D)
    (ptype in (1,2)) || throw(ArgumentError("ptype must be 1, 2, 3, 4"))
    int_order <= 0 && throw(ArgumentError("integration order must be > 0"))
    t <= 0.0 && throw(ArgumentError("thickness must be > 0.0"))
end



function plani4e(ex::Vector, ey::Vector, ep::Vector, D::Matrix, eq::Vector=[0.0,0.0])
    ptype = convert(Int, ep[1])
    t = ep[2]
    int_order = convert(Int, ep[3])

    length(eq) == 2 || throw(ArgumentError("length of eq must be 2"))

    error_check_plan4(ex, ey, ptype, t, int_order, D)
    x = [ex ey]

    # Buffers
    Ke = zeros(8,8)
    fe = zeros(8)
    B = zeros(4, 8)
    dNdx = zeros(2, 4)
    J = zeros(2,2)

    qr = make_quadrule(int_order)
    for (ξ, w) in zip(qr.points, qr.weights)
        N = N_Q_2(ξ)
        dNdξ = dN_Q_2(ξ)
        @into! J = dNdξ * x
        Jinv = inv2x2(J)
        @into! dNdx = Jinv * dNdξ

        dV = det(J) * w * t

        for i in 1:4
            B[1, 2*i - 1] = B[4, 2*i] = dNdx[1, i]
            B[4, 2*i - 1] = B[2, 2*i] = dNdx[2, i]
        end

        Ke[:, :] += B' * D * B * dV
        fe[1:2:end] +=  N * eq[1] * dV
        fe[2:2:end] +=  N * eq[2] * dV
    end

    return Ke, fe
end

#=
"""
Integrates the stress and strain for a four node isoparametric
quadraterial element.
"""
function plani4s(ex::Vector, ey::Vector, ep::Vector, D::Matrix, ed::AbstractVecOrMat)
    ptype = convert(Int, ep[1])
    t = ep[2]
    int_order = convert(Int, ep[3])

    size(ed, 2) == 8 || throw(ArgumentError("ed must have 8 components"))
    error_check_plan4(ex, ey, ptype, t, int_order, D)

    x = [ex ey]

    # Buffers
    σs = zeros(4, size(ed, 1))
    εs = zeros(4, size(ed, 1))
    B = zeros(4, 8)

    qr = make_quadrule(int_order)
    for (ξ, w) in zip(qr.points, qr.weights)
        dNdξ = dN_Q_2(ξ)
        J = dNdξ * x
        Jinv = inv2x2(J)
        dNdx = Jinv * dNdξ
        detJ = det(J)
        dV = detJ * w * t

        for i in 1:4
            B[1, 2*i - 1] = B[4, 2*i] = dNdx[1, i]
            B[4, 2*i - 1] = B[2, 2*i] = dNdx[2, i]
        end

        println(B * ed' )
        println(D * εs)
        εs += B * ed'
        σs += D * εs
    end

    return σs', εs'
end
=#
