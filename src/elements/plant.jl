# TODO, add plane stress support

"""
Computes the stiffness matrix for a three node isoparametric
triangular element
"""
function plante(ex::VecOrMat, ey::VecOrMat, ep, D::Matrix, eq::VecOrMat=[0.0,0.0])
    # Ugly but doing this now to deal with row/column ḿajor order
    # difference in Matlab and Julia // KC
    ex_mat = reshape(ex, (size(ex, 1), size(ex, 2)))
    ey_mat = reshape(ey, (size(ex, 1), size(ex, 2)))
    plante((@compat map(Float64, ex_mat)),
            (@compat map(Float64, ey_mat)), ep, D, eq)
end

function plante(ex::Matrix{Float64}, ey::Matrix{Float64},
                 ep::Array, D::Matrix{Float64}, eq::Vector=[0.0,0.0])
    ptype = convert(Int, ep[1])
    t = ep[2]
    int_order = convert(Int, ep[3])

    length(eq) == 2 || throw(ArgumentError("length of eq must be 2"))

    error_check_plant(ex, ey, ptype, t, int_order, D)
    x = [ex ey]

    # Buffers
    Ke = zeros(6,6)
    fe = @static zeros(6)
    B = @static zeros(4, 6)
    dNdx = @static zeros(2, 3)
    dNdξ = @static zeros(2, 3)
    J = @static zeros(2,2)
    N = @static zeros(3)
    DB = @static zeros(4,6)
    BDB = @static zeros(6,6)
    N2 = @static zeros(6,2)
    Nb = @static zeros(6,1)

    compute_force = false
    if eq != zeros(2)
        compute_force = true
    end

    qr = get_trirule(int_order)
    for (ξ, w) in zip(qr.points, qr.weights)

        dNdξ = dN_T_1!(dNdξ, ξ)
        @into! J = dNdξ * x
        Jinv = inv2x2(J)
        @into! dNdx = Jinv * dNdξ

        dV = det2x2(J) * w * t

        for i in 1:3
            B[1, 2*i - 1] = B[4, 2*i] = dNdx[1, i]
            B[4, 2*i - 1] = B[2, 2*i] = dNdx[2, i]
        end

        @into! DB = D * B
        @into! BDB = B' * DB
        @devec Ke[:, :] += BDB .* dV

        if compute_force
            N = N_T_1!(N, ξ)
            N2[1:2:end, 1] = N
            N2[2:2:end, 2] = N
            @into! Nb = N2 * eq
            Nbvec = vec(Nb)
            @devec fe += Nbvec .* dV
        end
    end

    return Ke, fe
end

function error_check_plant(ex, ey, ptype, t, int_order, D)
    (ptype in (1,2,3)) || throw(ArgumentError("ptype must be 1, 2, 3"))
    int_order <= 0 && throw(ArgumentError("integration order must be > 0"))
    t <= 0.0 && throw(ArgumentError("thickness must be > 0.0"))
end
