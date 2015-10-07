# Add docstrings

################
# Solid elements
################
"""
    plani4e(ex, ey, ep, D, [eq = zeros(2)]) -> Ke, fe

Computes the stiffness matrix `Ke` and force vector `fe` for a
four node isoparametric quadraterial element with body load `eq`.
"""
plani4e(ex, ey, ep, D, eq)

"""
    plani4s(ex, ey, ep, D, ed) -> σs, εs, points

Computes the stresses and strains in the gauss points given by the
integration rule for a four node isoparametric quadraterial element.

Also returns the global coordinates `points` at the gauss points.
"""
plani4s(ex, ey, ep, D, ed)


"""
    plani8e(ex, ey, ep, D, [eq = zeros(2)]) -> Ke, fe

Computes the stiffness matrix `Ke` and force vector `fe`
for an eight node isoparametric quadraterial element with body load `eq`.
"""
plani8e(ex, ey, ep, D, eq)

"""
    plani8s(ex, ey, ep, D, ed) -> σs, εs, points

Computes the stresses and strains in the gauss points given by the
integration rule for an eight node isoparametric quadraterial element.

Also returns the global coordinates `points` at the gauss points.
"""
plani8s(ex, ey, ep, D, ed)

"""
    plante(ex, ey, ep, D, [eq = zeros(2)]) -> Ke, fe

Computes the stiffness matrix `Ke` and force vector `fe` for a
three node isoparametric triangular element with body load `eq`.
"""
plante(ex, ey, ep, D, eq)

"""
    plants(ex, ey, ep, D, ed) -> σs, εs, points

Computes the stresses and strains in the gauss points given by the
integration rule for a three node isoparametric triangular element.

Also returns the global coordinates `points` at the gauss points.
"""
plants(ex, ey, ep, D, ed)

"""
    soli8e(ex, ey, ez, ep, D, [eq = zeros(3)]) -> Ke, fe

Computes the stiffness matrix `Ke` and force vector `fe` for an
eight node isoparametric hexahedron element with body load `eq`.
"""
soli8e(ex, ey, ez, ep, D, eq)


"""
    soli8s(ex, ey, ez, ep, D, ed) -> σs, εs, points

Computes the stresses and strains in the gauss points given by the
integration rule for an eight node isoparametric hexahedron element.

Also returns the global coordinates `points` at the gauss points.
"""
soli8s(ex, ey, ep, D, ed)


################
# Heat elements
################
"""
    flw2i4e(ex, ey, ep, D, [eq=[0.0]]) -> Ke, fe

Computes the stiffness matrix `Ke` and force vector `fe` for a
four node isoparametric quadraterial heat transfer element with a heat source `eq`.
"""
flw2i4e(ex, ey, ep, D, eq)

"""
    flw2i4s(ex, ey, ep, D, ed) -> es, et, points

Computes the heat flows `es` and the gradients `et` in the gauss points given by the
integration rule for a four node isoparametric quadraterial heat transfer element.

Also returns the global coordinates `points` at the gauss points.
"""
flw2i4s(ex, ey, ep, D, eq)


"""
    flw2i8e(ex, ey, ep, D, [eq=[0.0]]) -> Ke, fe

Computes the stiffness matrix `Ke` and force vector `fe` for an
eight node isoparametric quadraterial heat transfer element with a heat source `eq`.
"""
flw2i8e(ex, ey, ep, D, eq)


"""
    flw2i8s(ex, ey, ep, D, ed) -> es, et, points

Computes the heat flows `es` and the gradients `et` in the gauss points given by the
integration rule for an eight node isoparametric quadraterial heat transfer element.

Also returns the global coordinates `points` at the gauss points.
"""
flw2i8s(ex, ey, ep, D, eq)

"""
    flw2te(ex, ey, ep, D, [eq=[0.0]]) -> Ke, fe

Computes the stiffness matrix and `Ke` and force vector `fe`
for a three node isoparametric triangular heat transfer element with a heat source `eq`.
"""
flw2te(ex, ey, ep, D, eq)

"""
    flw2ts(ex, ey, ep, D, ed) -> es, et, points

Computes the heat flows `es` and the gradients `et` in the gauss points given by the
integration rule for a three node isoparametric triangular heat transfer element.

Also returns the global coordinates `points` at the gauss points.
"""
flw2ts(ex, ey, ep, D, eq)

"""
    flw3i8e(ex, ey, ez, ep, D, [eq=[0.0]]) -> Ke, fe

Computes the stiffness matrix `Ke` and force vector `fe` for an
eight node isoparametric hexahedron heat transfer element with a heat source `eq`.
"""
flw3i8e(ex, ey, ez, ep, D, eq=[0.0])

"""
    flw3i8s(ex, ey, ez, ep, D, ed) -> es, et, points

Computes the heat flows `es` and the gradients `et` in the gauss points given by the
integration rule for an eight node isoparametric hexahedron heat transfer element

Also returns the global coordinates `points` at the gauss points.
"""
flw3i8s(ex, ey, ez, ep, D, eq)
