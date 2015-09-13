# Add docstrings

################
# Solid elements
################
"""
    plani4e(ex, ey, ep, D, [eq=[0.0,0.0]]) -> Ke, fe

Computes the stiffness matrix `Ke` and force vector `fe` for a
four node isoparametric quadraterial element.
"""
plani4e(ex, ey, ep, D, eq)

"""
    plani8e(ex, ey, ep, D, [eq=[0.0,0.0]]) -> Ke, fe

Computes the stiffness matrix `Ke` and force vector `fe`
for an eight node isoparametric quadraterial element.
"""
plani8e(ex, ey, ep, D, eq)

"""
    plante(ex, ey, ep, D, [eq=[0.0,0.0]]) -> Ke, fe

Computes the stiffness matrix `Ke` and force vector `fe` for a
three node isoparametric triangular element.
"""
plante(ex, ey, ep, D, eq)

"""
    soli8e(ex, ey, ez, ep, D, [eq=[0.0,0.0,0.0]]) -> Ke, fe

Computes the stiffness matrix `Ke` and force vector `fe` for a
eight node isoparametric hexahedron element.
"""
soli8e(ex, ey, ez, ep, D, eq)


################
# Heat elements
################
"""
    flw2i4e(ex, ey, ep, D, [eq=[0.0]]) -> Ke, fe

Computes the stiffness matrix `Ke` and force vector `fe` for a
four node isoparametric quadraterial heat transfer element.
"""
flw2i4e(ex, ey, ep, D, eq)

"""
    flw2i8e(ex, ey, ep, D, [eq=[0.0]]) -> Ke, fe

Computes the stiffness matrix `Ke` and force vector `fe` for an
eight node isoparametric quadraterial element.
"""
flw2i8e(ex, ey, ep, D, eq)

"""
    flw2te(ex, ey, ep, D, [eq=[0.0]]) -> Ke, fe

Computes the stiffness matrix and `Ke` and force vector `fe`
for a three node isoparametric triangular heat transfer element.
"""
flw2te(ex, ey, ep, D, eq)

"""
    flw3i8e(ex, ey, ez, ep, D, [eq=[0.0]]) -> Ke, fe

Computes the stiffness matrix `Ke` and force vector `fe` for a
eight node isoparametric hexahedron heat transfer  element.
"""
flw3i8e(ex, ey, ez, ep, D, eq=[0.0])


