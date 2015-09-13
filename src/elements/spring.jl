"""
    spring1e(k) -> Ke

Computes the element stiffness matrix `Ke` for a
spring element with stiffness `k`.
"""
function spring1e(k)
    Ke = [k -k;
          -k k]
end

"""
    spring1s(k, u) -> fe

Computes the force `fe` for a spring element with stiffness
`k` and displacements `u`.
"""
function spring1s(k, u::VecOrMat)
    if length(u) != 2
        throw(ArgumentError("displacements for computing the spring force must" *
                            "be a vector of length 2"))
    end
    return k * (u[2] - u[1])
end
