"""
    BrezziDouglasMarini{refshape, order, vdim} <: VectorInterpolation

H(div)-conforming Brezzi-Douglas-Marini elements.
The following interpolations are implemented:
 
* BrezziDouglasMarini{RefTriangle, 1}
"""
struct BrezziDouglasMarini{shape, order, vdim} <: VectorInterpolation{vdim, shape, order}
    function BrezziDouglasMarini{shape, order}() where {rdim, shape <: AbstractRefShape{rdim}, order}
        return new{shape, order, rdim}()
    end
end
mapping_type(::BrezziDouglasMarini) = ContravariantPiolaMapping()

# RefTriangle
edgedof_indices(ip::BrezziDouglasMarini{RefTriangle}) = edgedof_interior_indices(ip)
facedof_indices(ip::BrezziDouglasMarini{RefTriangle}) = (ntuple(i -> i, getnbasefunctions(ip)),)

# RefTriangle, 1st order Lagrange
# https://defelement.org/elements/examples/triangle-brezzi-douglas-marini-lagrange-1.html
function reference_shape_value(ip::BrezziDouglasMarini{RefTriangle, 1}, ξ::Vec{2}, i::Int)
    x, y = ξ
    # Edge 1: y=1-x, n = [1, 1]/√2 (Flip sign, pos. integration outwards)
    i == 1 && return Vec(4x, -2y) # N ⋅ n = (2√2 x - √2 (1-x)) = 3√2 x - √2
    i == 2 && return Vec(-2x, 4y) # N ⋅ n = (-√2x + 2√2 (1-x)) = 2√2 - 3√2x
    # Edge 2: x=0, n = [-1, 0] (reverse order to follow Ferrite convention)
    i == 3 && return Vec(-2x - 6y + 2, 4y) # N ⋅ n = (6y - 2)
    i == 4 && return Vec(4x + 6y - 4, -2y) # N ⋅ n = (4 - 6y)
    # Edge 3: y=0, n = [0, -1] (Flip sign, pos. integration outwards)
    i == 5 && return Vec(-2x, 6x + 4y - 4) # N ⋅ n = (4 - 6x)
    i == 6 && return Vec(4x, -6x - 2y + 2) # N ⋅ n = (6x - 2)
    throw(ArgumentError("no shape function $i for interpolation $ip"))
end

getnbasefunctions(::BrezziDouglasMarini{RefTriangle, 1}) = 6
edgedof_interior_indices(::BrezziDouglasMarini{RefTriangle, 1}) = ((1, 2), (3, 4), (5, 6))
adjust_dofs_during_distribution(::BrezziDouglasMarini{RefTriangle, 1}) = true

function get_direction(::BrezziDouglasMarini{RefTriangle, 1}, shape_nr, cell)
    edge_nr = (shape_nr + 1) ÷ 2
    return get_edge_direction(cell, edge_nr)
end
