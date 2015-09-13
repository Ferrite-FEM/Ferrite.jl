"""
    bar2e(ex, ey, elem_prop) -> Ke

Computes the element stiffness matrix `Ke` for a 2D bar element.
"""
function bar2e(ex::VecOrMat, ey::VecOrMat, elem_prop::VecOrMat)

    # Local element stiffness
    E = elem_prop[1];  A = elem_prop[2]

    dx = ex[2] - ex[1]
    dy = ey[2] - ey[1]
    L = sqrt(dx^2 + dy^2)

    k = E * A / L
    Kel = [ k -k
           -k  k]

    # Cosines
    c = dx/L; s = dy/L

    # Global element stiffness
    G = [c s 0 0
         0 0 c s]

    return G' * Kel * G

end

"""
    bar2s(ex, ey, elem_prop, el_disp) -> N

Computes the sectional force (normal force) `N` for a 2D bar element.
"""
function bar2s(ex::VecOrMat, ey::VecOrMat, elem_prop::VecOrMat, el_disp::VecOrMat)

    E = elem_prop[1];  A = elem_prop[2]

    dx = ex[2]-ex[1]
    dy = ey[2]-ey[1]
    L = sqrt(dx^2 + dy^2)

    k = E * A / L

    # Cosines
    c = dx/L; s = dy/L

    # Compute end displacements in local coordinate system
    G = [c s 0 0
         0 0 c s]
    u = G * vec(el_disp)

    return k * (u[2] - u[1])

end


const __bar2g_c1 = Float64[1  0 -1  0;
                           0  0  0  0;
                          -1  0  1  0;
                           0  0  0  0]

const __bar2g_c2 = Float64[0  0  0  0;
                           0  1  0 -1;
                           0  0  0  0;
                           0 -1  0  1];
"""
    bar2g(ex, ey, elem_prop, N) -> Ke

Computes the element stiffness matrix `Ke` for a
geometrically nonlinear 2D bar element.
"""
function bar2g(ex::VecOrMat, ey::VecOrMat, elem_prop::VecOrMat, N::Number)

    E = elem_prop[1];  A = elem_prop[2]

    dx = ex[2]-ex[1]
    dy = ey[2]-ey[1]
    L = sqrt(dx^2 + dy^2)

    # Cosines
    c = dx/L; s = dy/L

    G = [ c   s  0.0 0.0;
         -s   c  0.0 0.0;
         0.0 0.0  c   s ;
         0.0 0.0 -s   c]

    k = E * A / L

    return G' * (k * __bar2g_c1 + N/L * __bar2g_c2) * G

end

