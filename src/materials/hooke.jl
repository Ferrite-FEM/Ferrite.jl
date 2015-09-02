"""
Computes the material stiffness matrix for a
linear elastic and isotropic material.

**ptype**:

* 1: plane stress

* 2: plane strain

* 3: axisymmetry

* 4: 3D
"""
function hooke(ptype, E, v)
    G = E / (2(1 + v))
    f = E / ((1+v) * (1-2v))
    λ = f * v
    M = f * (1 - v)

    if ptype == 1
        D = [1    v   0        ;
             v    1   0        ;
             0    0   0.5*(1-v)]
        scale!(D, E / (1-v^2))
    elseif ptype == 2 || ptype == 3
        D = [M    λ     λ   0;
             λ    M     λ   0;
             λ    λ     M   0;
             0    0     0   G]
    else
        D = [M    λ    λ    0    0   0;
             λ    M    λ    0    0   0;
             λ    λ    M    0    0   0;
             0    0    0    G    0   0;
             0    0    0    0    G   0;
             0    0    0    0    0   G]
    end
    return D
end