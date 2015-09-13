"""
    statcon(K, f, cd) -> K_cond, f_cond

Condenses out the dofs given in cd from K and f."""
function statcon(K::Matrix, f::VecOrMat, cd::Vector)
    n = chksquare(K)
    rd = setdiff(collect(1:length(f)), cd)
    K_AA = K[rd, rd]
    K_AB = K[rd, cd]
    K_BB = K[cd, cd]

    f_A = f[rd]
    f_B = f[cd]

    K_fact = factorize(K_BB)

    K_cd = K_AA - K_AB * (K_fact \ K_AB')
    f_cd = f_A - K_AB * (K_fact \ f_B)

    return K_cd, f_cd
end
