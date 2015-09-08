"""
Computes the element stiffness matrix *Ke* for a 2D bar element.
"""
function bar2e(ex::Array, ey::Array, elem_prop::Array)

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
Computes the sectional force (normal force) *N* for a 2D bar element.
"""
function bar2s(ex::Array, ey::Array, elem_prop::Array, el_disp::Array)

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

