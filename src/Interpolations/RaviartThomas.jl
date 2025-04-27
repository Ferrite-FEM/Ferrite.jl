"""
    RaviartThomas{refshape, order, vdim} <: VectorInterpolation

H(div)-conforming Raviart-Thomas elements.
The following interpolations are implemented:

* RaviartThomas{RefTriangle, 1}
* RaviartThomas{RefTriangle, 2}
* RaviartThomas{RefQuadrilateral, 1}
* RaviartThomas{RefTetrahedron, 1}
* RaviartThomas{RefHexahedron, 1}
"""
struct RaviartThomas{shape, order, vdim} <: VectorInterpolation{vdim, shape, order}
    function RaviartThomas{shape, order}() where {rdim, shape <: AbstractRefShape{rdim}, order}
        return new{shape, order, rdim}()
    end
end
mapping_type(::RaviartThomas) = ContravariantPiolaMapping()

# RefTriangle
edgedof_indices(ip::RaviartThomas{RefTriangle}) = edgedof_interior_indices(ip)
facedof_indices(ip::RaviartThomas{RefTriangle}) = (ntuple(i -> i, getnbasefunctions(ip)),)

# RefTriangle, 1st order Lagrange
# https://defelement.org/elements/examples/triangle-raviart-thomas-lagrange-0.html
function reference_shape_value(ip::RaviartThomas{RefTriangle, 1}, ξ::Vec{2}, i::Int)
    x, y = ξ
    i == 1 && return ξ                  # Flip sign
    i == 2 && return Vec(x - 1, y)      # Keep sign
    i == 3 && return Vec(x, y - 1)      # Flip sign
    throw(ArgumentError("no shape function $i for interpolation $ip"))
end

getnbasefunctions(::RaviartThomas{RefTriangle, 1}) = 3
edgedof_interior_indices(::RaviartThomas{RefTriangle, 1}) = ((1,), (2,), (3,))
facedof_interior_indices(::RaviartThomas{RefTriangle, 1}) = ((),)
adjust_dofs_during_distribution(::RaviartThomas) = false

function get_direction(::RaviartThomas{RefTriangle, 1}, shape_nr, cell)
    return get_edge_direction(cell, shape_nr) # shape_nr = edge_nr
end

# RefTriangle, 2st order Lagrange
# https://defelement.org/elements/examples/triangle-raviart-thomas-lagrange-1.html
function reference_shape_value(ip::RaviartThomas{RefTriangle, 2}, ξ::Vec{2}, i::Int)
    x, y = ξ
    # Face 1 (keep ordering, flip sign)
    i == 1 && return Vec(4x * (2x - 1), 2y * (4x - 1))
    i == 2 && return Vec(2x * (4y - 1), 4y * (2y - 1))
    # Face 2 (flip ordering, keep signs)
    i == 3 && return Vec(8x * y - 2x - 6y + 2, 4y * (2y - 1))
    i == 4 && return Vec(-8x^2 - 8x * y + 12x + 6y - 4, 2y * (-4x - 4y + 3))
    # Face 3 (keep ordering, flip sign)
    i == 5 && return Vec(2x * (3 - 4x - 4y), -8x * y + 6x - 8y^2 + 12y - 4)
    i == 6 && return Vec(4x * (2x - 1), 8x * y - 6x - 2y + 2)
    # Cell
    i == 7 && return Vec(8x * (-2x - y + 2), 8y * (-2x - y + 1))
    i == 8 && return Vec(8x * (-2y - x + 1), 8y * (-2y - x + 2))
    throw(ArgumentError("no shape function $i for interpolation $ip"))
end

getnbasefunctions(::RaviartThomas{RefTriangle, 2}) = 8
edgedof_interior_indices(::RaviartThomas{RefTriangle, 2}) = ((1, 2), (3, 4), (5, 6))
facedof_interior_indices(::RaviartThomas{RefTriangle, 2}) = ((7, 8),)
adjust_dofs_during_distribution(::RaviartThomas{RefTriangle, 2}) = true

function get_direction(::RaviartThomas{RefTriangle, 2}, shape_nr, cell)
    shape_nr > 6 && return 1
    edge_nr = (shape_nr + 1) ÷ 2
    return get_edge_direction(cell, edge_nr)
end

# RefQuadrilateral
edgedof_indices(ip::RaviartThomas{RefQuadrilateral}) = edgedof_interior_indices(ip)
facedof_indices(ip::RaviartThomas{RefQuadrilateral}) = (ntuple(i -> i, getnbasefunctions(ip)),)

# RefQuadrilateral, 1st order Lagrange
# https://defelement.org/elements/examples/quadrilateral-raviart-thomas-lagrange-1.html
function reference_shape_value(ip::RaviartThomas{RefQuadrilateral, 1}, ξ::Vec{2, T}, i::Int) where {T}
    x, y = ξ
    nil = zero(T)

    i == 1 && return Vec(nil, -(1 - y) / 4) # Changed sign, follow Ferrite's sign convention
    i == 2 && return Vec((1 + x) / 4, nil)  # Changed sign, follow Ferrite's sign convention
    i == 3 && return Vec(nil, (1 + y) / 4)
    i == 4 && return Vec((x - 1) / 4, nil)
    throw(ArgumentError("no shape function $i for interpolation $ip"))
end

getnbasefunctions(::RaviartThomas{RefQuadrilateral, 1}) = 4
edgedof_interior_indices(::RaviartThomas{RefQuadrilateral, 1}) = ((1,), (2,), (3,), (4,))
facedof_interior_indices(::RaviartThomas{RefQuadrilateral, 1}) = ((),)
adjust_dofs_during_distribution(::RaviartThomas{RefQuadrilateral, 1}) = false

function get_direction(::RaviartThomas{RefQuadrilateral, 1}, shape_nr, cell)
    return get_edge_direction(cell, shape_nr) # shape_nr = edge_nr
end

# RefTetrahedron, 1st order Lagrange
# https://defelement.com/elements/examples/tetrahedron-raviart-thomas-lagrange-1.html
function reference_shape_value(ip::RaviartThomas{RefTetrahedron, 1}, ξ::Vec{3}, i::Int)
    x, y, z = ξ
    i == 1 && return Vec(2 * x, 2 * y, 2 * (z - 1)) # Changed sign, follow Ferrite's sign convention
    i == 2 && return Vec(2 * x, 2 * (y - 1), 2 * z)
    i == 3 && return Vec(2 * x, 2 * y, 2 * z)
    i == 4 && return Vec(2 * (x - 1), 2 * y, 2 * z) # Changed sign, follow Ferrite's sign convention
    throw(ArgumentError("no shape function $i for interpolation $ip"))
end

getnbasefunctions(::RaviartThomas{RefTetrahedron, 1}) = 4
edgedof_interior_indices(::RaviartThomas{RefTetrahedron, 1}) = ntuple(_ -> (), 6)
edgedof_indices(ip::RaviartThomas{RefTetrahedron, 1}) = edgedof_interior_indices(ip)
facedof_interior_indices(::RaviartThomas{RefTetrahedron, 1}) = ((1,), (2,), (3,), (4,))
facedof_indices(ip::RaviartThomas{RefTetrahedron, 1}) = facedof_interior_indices(ip)
adjust_dofs_during_distribution(::RaviartThomas{RefTetrahedron, 1}) = false

function get_direction(::RaviartThomas{RefTetrahedron, 1}, shape_nr, cell)
    return get_face_direction(cell, shape_nr) # shape_nr = face_nr
end

# RefHexahedron, 1st order Lagrange
# https://defelement.com/elements/examples/hexahedron-raviart-thomas-lagrange-1.html
# Scale with factor 1/4 as Ferrite has 4 times defelement's face area.
function reference_shape_value(ip::RaviartThomas{RefHexahedron, 1}, ξ::Vec{3, T}, i::Int) where {T}
    x, y, z = ξ
    nil = zero(T)

    i == 1 && return Vec(nil, nil, (z - 1) / 8) # Changed sign, follow Ferrite's sign convention
    i == 2 && return Vec(nil, (y - 1) / 8, nil)
    i == 3 && return Vec((x + 1) / 8, nil, nil)
    i == 4 && return Vec(nil, (y + 1) / 8, nil) # Changed sign, follow Ferrite's sign convention
    i == 5 && return Vec((x - 1) / 8, nil, nil) # Changed sign, follow Ferrite's sign convention
    i == 6 && return Vec(nil, nil, (z + 1) / 8)
    throw(ArgumentError("no shape function $i for interpolation $ip"))
end

getnbasefunctions(::RaviartThomas{RefHexahedron, 1}) = 6
edgedof_interior_indices(::RaviartThomas{RefHexahedron, 1}) = ntuple(_ -> (), 12)
edgedof_indices(ip::RaviartThomas{RefHexahedron, 1}) = edgedof_interior_indices(ip)
facedof_interior_indices(::RaviartThomas{RefHexahedron, 1}) = ((1,), (2,), (3,), (4,), (5,), (6,))
facedof_indices(ip::RaviartThomas{RefHexahedron, 1}) = facedof_interior_indices(ip)
adjust_dofs_during_distribution(::RaviartThomas{RefHexahedron, 1}) = false

function get_direction(::RaviartThomas{RefHexahedron, 1}, shape_nr, cell)
    return get_face_direction(cell, shape_nr) # shape_nr = face_nr
end
