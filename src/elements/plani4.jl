# TODO, add plane stress support

"""
Computes the stiffness matrix for a four node isoparametric
quadraterial element
"""
function plani4e{P, Q}(ex::AbstractVecOrMat{P}, ey::AbstractVecOrMat{Q},
                       ep, D::Matrix{Float64}, eq=[0.0,0.0])
    # Ugly but doing this now to deal with row/column ḿajor order
    # difference in Matlab and Julia // KC
    ex_mat = reshape(ex, (size(ex, 1), size(ex, 2)))
    ey_mat = reshape(ey, (size(ex, 1), size(ex, 2)))
    plani4e((@compat map(Float64, ex_mat)),
            (@compat map(Float64, ey_mat)), ep, D, eq)
end

function plani4e(ex::Matrix{Float64}, ey::Matrix{Float64},
                 ep::Array, D::Matrix{Float64}, eq::Vector=[0.0,0.0])
    ptype = convert(Int, ep[1])
    t = ep[2]
    int_order = convert(Int, ep[3])

    length(eq) == 2 || throw(ArgumentError("length of eq must be 2"))

    error_check_plan4(ex, ey, ptype, t, int_order, D)
    x = [ex ey]

    # Buffers
    Ke = zeros(8,8)
    fe = @static zeros(8)
    B = @static zeros(4, 8)
    dNdx = @static zeros(2, 4)
    dNdξ = @static zeros(2, 4)
    J = @static zeros(2,2)
    N = @static zeros(4)
    DB = @static zeros(4,8)
    BDB = @static zeros(8,8)
    N2 = @static zeros(8,2)
    Nb = @static zeros(8,1)

    compute_force = false
    if eq != zeros(2)
        compute_force = true
    end

    qr = get_quadrule(int_order)
    for (ξ, w) in zip(qr.points, qr.weights)

        dNdξ = dN_Q_2!(dNdξ, ξ)
        @into! J = dNdξ * x
        Jinv = inv2x2(J)
        @into! dNdx = Jinv * dNdξ

        dV = det2x2(J) * w * t

        for i in 1:4
            B[1, 2*i - 1] = B[4, 2*i] = dNdx[1, i]
            B[4, 2*i - 1] = B[2, 2*i] = dNdx[2, i]
        end

        @into! DB = D * B
        @into! BDB = B' * DB
        @devec Ke[:, :] += BDB .* dV

        if compute_force
            N = N_Q_2!(N, ξ)
            N2[1:2:end, 1] = N
            N2[2:2:end, 2] = N
            @into! Nb = N2 * eq
            Nbvec = vec(Nb)
            @devec fe += Nbvec .* dV
        end
    end

    return Ke, fe
end

function error_check_plan4(ex, ey, ptype, t, int_order, D)
    (ptype in (1,2,3)) || throw(ArgumentError("ptype must be 1, 2, 3"))
    int_order <= 0 && throw(ArgumentError("integration order must be > 0"))
    t <= 0.0 && throw(ArgumentError("thickness must be > 0.0"))
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
