"""
Computes the stiffness matrix for a four node isoparametric
quadraterial element
"""
function plani4e{P, Q}(ex::Vector{P}, ey::Vector{Q},
                       ep, D::Matrix{Float64}, eq=[0.0,0.0])
    plani4e(convert(Vector{Float64}, ex),
            convert(Vector{Float64}, ey),
            ep, D, eq)
end

function plani4e(ex::Vector{Float64}, ey::Vector{Float64},
                 ep, D::Matrix{Float64}, eq=[0.0,0.0])
    ptype = convert(Int, ep[1])
    if !(ptype in (1,2,3))
        throw(ArgumentError("ptype must be 1, 2, 3"))
    end

    length(eq) == 2 || throw(ArgumentError("length of eq must be 2"))

    int_order = convert(Int, ep[3])
    int_order <= 0 && throw(ArgumentError("Integration order must be > 0"))

    t = ep[2]
    t <= 0.0 && throw(ArgumentError("Thickness must be > 0.0"))

    x = [ex ey]

    # Buffers
    Ke = zeros(8,8)
    fe = zeros(8)
    B = zeros(4, 8)

    qr = make_quadrule(int_order)
    for (ξ, w) in zip(qr.points, qr.weights)
        N = N_Q_2(ξ)
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

        Ke[:, :] += B' * D * B * dV
        # TODO: Rewrite this?
        fe[1:2:end] +=  N * eq[1] * dV
        fe[2:2:end] +=  N * eq[2] * dV
    end

    return Ke, fe
end
