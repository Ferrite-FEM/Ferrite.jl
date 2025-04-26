"""
    Nedelec{refshape, order, vdim} <: VectorInterpolation

H(curl)-conforming Nedelec elements of the 1st kind.
The following interpolations are implemented:
 
* Nedelec{RefTriangle, 1}
* Nedelec{RefTriangle, 2}
* Nedelec{RefQuadrilateral, 1}
* Nedelec{RefTetrahedron, 1}
* Nedelec{RefHexahedron, 1}
"""
struct Nedelec{shape, order, vdim} <: VectorInterpolation{vdim, shape, order}
    function Nedelec{shape, order}() where {rdim, shape <: AbstractRefShape{rdim}, order}
        return new{shape, order, rdim}()
    end
end
mapping_type(::Nedelec) = CovariantPiolaMapping()
edgedof_indices(ip::Nedelec) = edgedof_interior_indices(ip)

# 2D refshape (rdim == vdim for Nedelec)
facedof_indices(ip::Nedelec{<:AbstractRefShape{2}}) = (ntuple(i -> i, getnbasefunctions(ip)),)

# RefTriangle, 1st order Lagrange
# https://defelement.org/elements/examples/triangle-nedelec1-lagrange-0.html
function reference_shape_value(ip::Nedelec{RefTriangle, 1}, ξ::Vec{2}, i::Int)
    x, y = ξ
    i == 1 && return Vec(- y, x)
    i == 2 && return Vec(- y, x - 1) # Changed sign, follow Ferrite's sign convention
    i == 3 && return Vec(1 - y, x)
    throw(ArgumentError("no shape function $i for interpolation $ip"))
end

getnbasefunctions(::Nedelec{RefTriangle, 1}) = 3
edgedof_interior_indices(::Nedelec{RefTriangle, 1}) = ((1,), (2,), (3,))
adjust_dofs_during_distribution(::Nedelec{RefTriangle, 1}) = false

function get_direction(::Nedelec{RefTriangle, 1}, shape_nr, cell)
    return get_edge_direction(cell, shape_nr) # shape_nr = edge_nr
end

# RefTriangle, 2nd order Lagrange
# https://defelement.org/elements/examples/triangle-nedelec1-lagrange-1.html
function reference_shape_value(ip::Nedelec{RefTriangle, 2}, ξ::Vec{2}, i::Int)
    x, y = ξ
    # Edge 1
    i == 1 && return Vec(2 * y * (1 - 4 * x), 4 * x * (2 * x - 1))
    i == 2 && return Vec(4 * y * (1 - 2 * y), 2 * x * (4 * y - 1))
    # Edge 2 (flip order and sign compared to defelement)
    i == 3 && return Vec(4 * y * (1 - 2 * y), 8 * x * y - 2 * x - 6 * y + 2)
    i == 4 && return Vec(2 * y * (4 * x + 4 * y - 3), -8 * x^2 - 8 * x * y + 12 * x + 6 * y - 4)
    # Edge 3
    i == 5 && return Vec(8 * x * y - 6 * x + 8 * y^2 - 12 * y + 4, 2 * x * (-4 * x - 4 * y + 3))
    i == 6 && return Vec(
        -8 * x * y + 6 * x + 2 * y - 2, 4 * x * (2 * x - 1)
    )
    # Face
    i == 7 && return Vec(8 * y * (-x - 2 * y + 2), 8 * x * (x + 2 * y - 1))
    i == 8 && return Vec(8 * y * (2 * x + y - 1), 8 * x * (-2 * x - y + 2))
    throw(ArgumentError("no shape function $i for interpolation $ip"))
end

getnbasefunctions(::Nedelec{RefTriangle, 2}) = 8
edgedof_interior_indices(::Nedelec{RefTriangle, 2}) = ((1, 2), (3, 4), (5, 6))
facedof_interior_indices(::Nedelec{RefTriangle, 2}) = ((7, 8),)
adjust_dofs_during_distribution(::Nedelec{RefTriangle, 2}) = true

function get_direction(::Nedelec{RefTriangle, 2}, shape_nr, cell)
    shape_nr > 6 && return 1
    edge_nr = (shape_nr + 1) ÷ 2
    return get_edge_direction(cell, edge_nr)
end

# RefQuadrilateral, 1st order Lagrange
# https://defelement.org/elements/examples/quadrilateral-nedelec1-lagrange-1.html
# Scaled by 1/2 as the reference edge length in Ferrite is length 2, but 1 in DefElement.
function reference_shape_value(ip::Nedelec{RefQuadrilateral, 1}, ξ::Vec{2, T}, i::Int) where {T}
    x, y = ξ
    nil = zero(T)

    i == 1 && return Vec((1 - y) / 4, nil)
    i == 2 && return Vec(nil, (1 + x) / 4)
    i == 3 && return Vec(-(1 + y) / 4, nil) # Changed sign, follow Ferrite's sign convention
    i == 4 && return Vec(nil, -(1 - x) / 4) # Changed sign, follow Ferrite's sign convention
    throw(ArgumentError("no shape function $i for interpolation $ip"))
end

getnbasefunctions(::Nedelec{RefQuadrilateral, 1}) = 4
edgedof_interior_indices(::Nedelec{RefQuadrilateral, 1}) = ((1,), (2,), (3,), (4,))
adjust_dofs_during_distribution(::Nedelec{RefQuadrilateral, 1}) = false

function get_direction(::Nedelec{RefQuadrilateral, 1}, shape_nr, cell)
    return get_edge_direction(cell, shape_nr) # shape_nr = edge_nr
end

# RefTetrahedron, 1st order Lagrange
# https://defelement.org/elements/examples/tetrahedron-nedelec1-lagrange-1.html
function reference_shape_value(ip::Nedelec{RefTetrahedron, 1}, ξ::Vec{3, T}, i::Int) where {T}
    x, y, z = ξ
    nil = zero(T)

    i == 1 && return Vec(1 - y - z, x, x)
    i == 2 && return Vec(-y, x, nil)
    i == 3 && return Vec(-y, x + z - 1, -y) # Changed sign, follow Ferrite's sign convention
    i == 4 && return Vec(z, z, 1 - x - y)
    i == 5 && return Vec(-z, nil, x)
    i == 6 && return Vec(nil, -z, y)
    throw(ArgumentError("no shape function $i for interpolation $ip"))
end

getnbasefunctions(::Nedelec{RefTetrahedron, 1}) = 6
edgedof_interior_indices(::Nedelec{RefTetrahedron, 1}) = ((1,), (2,), (3,), (4,), (5,), (6,))
facedof_indices(::Nedelec{RefTetrahedron, 1}) = ((1, 2, 3), (1, 4, 5), (2, 5, 6), (3, 4, 6))
adjust_dofs_during_distribution(::Nedelec{RefTetrahedron, 1}) = false

function get_direction(::Nedelec{RefTetrahedron, 1}, shape_nr, cell)
    return get_edge_direction(cell, shape_nr) # shape_nr = edge_nr
end

# RefHexahedron, 1st order Lagrange
# https://defelement.org/elements/examples/hexahedron-nedelec1-lagrange-1.html
function reference_shape_value(ip::Nedelec{RefHexahedron, 1}, ξ::Vec{3, T}, i::Int) where {T}
    x, y, z = ξ
    nil = zero(T)

    i == 1 && return Vec((1 - y) * (1 - z) / 8, nil, nil)
    i == 2 && return Vec(nil, (1 + x) * (1 - z) / 8, nil)
    i == 3 && return Vec(-(1 + y) * (1 - z) / 8, nil, nil) # Changed sign, follow Ferrite's sign convention
    i == 4 && return Vec(nil, -(1 - x) * (1 - z) / 8, nil)  # Changed sign, follow Ferrite's sign convention
    i == 5 && return Vec((1 - y) * (1 + z) / 8, nil, nil)
    i == 6 && return Vec(nil, (1 + x) * (1 + z) / 8, nil)
    i == 7 && return Vec(-(1 + y) * (1 + z) / 8, nil, nil)  # Changed sign, follow Ferrite's sign convention
    i == 8 && return Vec(nil, -(1 - x) * (1 + z) / 8, nil) # Changed sign, follow Ferrite's sign convention
    i == 9 && return Vec(nil, nil, (1 - x) * (1 - y) / 8)
    i == 10 && return Vec(nil, nil, (1 + x) * (1 - y) / 8)
    i == 11 && return Vec(nil, nil, (1 + x) * (1 + y) / 8)
    i == 12 && return Vec(nil, nil, (1 - x) * (1 + y) / 8)
    throw(ArgumentError("no shape function $i for interpolation $ip"))
end

getnbasefunctions(::Nedelec{RefHexahedron, 1}) = 12
edgedof_interior_indices(::Nedelec{RefHexahedron, 1}) = ((1,), (2,), (3,), (4,), (5,), (6,), (7,), (8,), (9,), (10,), (11,), (12,))
facedof_indices(::Nedelec{RefHexahedron, 1}) = ((1, 2, 3, 4), (1, 5, 9, 10), (2, 6, 10, 11), (3, 7, 11, 12), (4, 8, 9, 12), (5, 6, 7, 8))
adjust_dofs_during_distribution(::Nedelec{RefHexahedron, 1}) = false

function get_direction(::Nedelec{RefHexahedron, 1}, shape_nr, cell)
    return get_edge_direction(cell, shape_nr) # shape_nr = edge_nr
end
