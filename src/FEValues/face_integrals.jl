"""
    element_face_transformation(point::Vec{N, T}, cell_T::Type{<:AbstractRefShape}, face::Int) 

Transform point from face's reference (N-1)D coordinates to ND coordinates on the cell's face.
"""
element_face_transformation

"""
    weighted_normal(J::AbstractTensor, fv::FaceValues, face::Int)
    weighted_normal(J::AbstractTensor, cell_T::Type{<:AbstractRefShape}, face::Int)

Compute the vector normal to the face weighted by the area ration between the face and the reference face.
This is computed by taking cross product of the jacobian compenets that align to the face local axis.
"""
function weighted_normal(J::AbstractTensor, fv::FaceValues, face::Int)
    return weighted_normal(J, getrefshape(fv.func_interp), face)
end

"""
    create_face_quad_rule(cell_T::Type{RefShape}, w::Vector{T}, p::Vector{Vec{N, T}}
    create_face_quad_rule(cell_T::Type{RefShape}, quad_faces::Vector{Int}, w_quad::Vector{T}, p_quad::Vector{Vec{N, T}}, tri_faces::Vector{Int}, w_tri::Vector{T}, p_tri::Vector{Vec{N, T}}) 

Creates ["FaceQuadratureRule"](@ref) with the given cell type, weights and points. If the cell has faces of different shapes
(i.e. quadrilaterals and triangles) then each shape's faces indices, weights and points are passed separately.
"""
function create_face_quad_rule(cell_T::Type{RefShape}, w::Vector{T}, p::Vector{Vec{N, T}}) where {N, T, RefShape <: AbstractRefShape}
    n_points = length(w)
    face_quad_rule = QuadratureRule{RefShape, T, getdim(AbstractCell{cell_T})}[]
    for face in 1:nfaces(cell_T)
        new_points = [element_face_transformation(N != 0 ? p[i] : Vec(zero(T)), cell_T, face) for i in 1:n_points] # ξ = 1-t-s, η = s, ζ = 0
        push!(face_quad_rule, QuadratureRule{RefShape, T}(w, new_points))    
    end
    return FaceQuadratureRule(face_quad_rule)
end

# For cells with mixed faces
function create_face_quad_rule(cell_T::Type{RefShape}, quad_faces::Vector{Int}, w_quad::Vector{T}, p_quad::Vector{Vec{N, T}}, tri_faces::Vector{Int}, w_tri::Vector{T}, p_tri::Vector{Vec{N, T}}) where {N, T, RefShape <: Union{RefPrism, RefPyramid}}
    n_points_quad = length(w_quad)
    n_points_tri = length(w_tri)
    face_quad_rule = Array{QuadratureRule{RefShape, T, getdim(AbstractCell{cell_T})}}(undef, nfaces(cell_T))
    for face in quad_faces
        new_points = [element_face_transformation(N != 0 ? p_quad[i] : Vec(zero(T)), cell_T, face) for i in 1:n_points_quad]
        face_quad_rule[face] = QuadratureRule{RefShape, T}(w_quad, new_points)
    end
    for face in tri_faces
        new_points = [element_face_transformation(N != 0 ? p_tri[i] : T[], cell_T, face) for i in 1:n_points_tri]
        face_quad_rule[face] = QuadratureRule{RefShape, T}(w_tri, new_points)
    end
    return FaceQuadratureRule(face_quad_rule)
end

##################
# All 1D RefLine #
##################

# Mapping from to 0D node to 1D line vertex.
function element_face_transformation(::Vec{N, T}, cell_T::Type{RefLine}, face::Int) where {N, T}
    face == 1 && return Vec{1, T}(( -one(T),))
    face == 2 && return Vec{1, T}((  one(T),))
    throw(ArgumentError("unknown face number"))
end

function weighted_normal(::Tensor{2,1,T}, cell_T::Type{RefLine}, face::Int) where {T}
    face == 1 && return Vec{1,T}((-one(T),))
    face == 2 && return Vec{1,T}(( one(T),))
    throw(ArgumentError("unknown face number"))
end

###########################
# All 2D RefQuadrilateral #
###########################

# Mapping from 1D line to 2D face of a quadrilateral.
function element_face_transformation(point::Vec{1, T}, cell_T::Type{RefQuadrilateral}, face::Int) where T
    x = point[]
    face == 1 && return Vec{2, T}(( x,          -one(T)))
    face == 2 && return Vec{2, T}(( one(T),     x))
    face == 3 && return Vec{2, T}(( -x,         one(T)))
    face == 4 && return Vec{2, T}(( -one(T),    -x))
    throw(ArgumentError("unknown face number"))
end

function weighted_normal(J::Tensor{2,2}, cell_T::Type{RefQuadrilateral}, face::Int)
    @inbounds begin
        face == 1 && return Vec{2}(( J[2,1], -J[1,1]))
        face == 2 && return Vec{2}(( J[2,2], -J[1,2]))
        face == 3 && return Vec{2}((-J[2,1],  J[1,1]))
        face == 4 && return Vec{2}((-J[2,2],  J[1,2]))
    end
    throw(ArgumentError("unknown face number"))
end

######################
# All RefTriangle 2D #
######################

# Mapping from 1D line to 2D face of a triangle.
function element_face_transformation(point::Vec{1, T},  cell_T::Type{RefTriangle}, face::Int) where T
    x = (point[] + one(T)) / 2
    face == 1 && return Vec{2, T}(( one(T) - x,     x ))
    face == 2 && return Vec{2, T}(( zero(T),        one(T) -x))
    face == 3 && return Vec{2, T}(( x,              zero(T)))
    throw(ArgumentError("unknown face number"))
end

function weighted_normal(J::Tensor{2,2}, cell_T::Type{RefTriangle}, face::Int)
    @inbounds begin
        face == 1 && return Vec{2}((-(J[2,1] - J[2,2]), J[1,1] - J[1,2]))
        face == 2 && return Vec{2}((-J[2,2], J[1,2]))
        face == 3 && return Vec{2}((J[2,1], -J[1,1]))
    end
    throw(ArgumentError("unknown face number"))
end

########################
# All RefHexahedron 3D #
########################

# Mapping from 2D quadrilateral to 3D face of a hexahedron.
function element_face_transformation(point::Vec{2, T}, cell_T::Type{RefHexahedron}, face::Int) where T
    x,y = point
    face == 1 && return Vec{3, T}(( y,      x,          -one(T)))
    face == 2 && return Vec{3, T}(( x,      -one(T),    y))
    face == 3 && return Vec{3, T}(( one(T), x,          y))
    face == 4 && return Vec{3, T}(( -x,     one(T),     y))
    face == 5 && return Vec{3, T}((-one(T), y,          x))
    face == 6 && return Vec{3, T}(( x,      y,          one(T)))
    throw(ArgumentError("unknown face number"))
end

function weighted_normal(J::Tensor{2,3}, cell_T::Type{RefHexahedron}, face::Int)
    @inbounds begin
        face == 1 && return J[:,2] × J[:,1]
        face == 2 && return J[:,1] × J[:,3]
        face == 3 && return J[:,2] × J[:,3]
        face == 4 && return J[:,3] × J[:,1]
        face == 5 && return J[:,3] × J[:,2]
        face == 6 && return J[:,1] × J[:,2]
    end
    throw(ArgumentError("unknown face number"))
end

#########################
# All RefTetrahedron 3D #
#########################

# Mapping from 2D triangle to 3D face of a tetrahedon.
function element_face_transformation(point::Vec{2, T}, cell_T::Type{RefTetrahedron}, face::Int) where T
    x,y = point
    face == 1 && return Vec{3, T}( (one(T)-x-y,     y,              zero(T)))
    face == 2 && return Vec{3, T}( (y,              zero(T),        one(T)-x-y))
    face == 3 && return Vec{3, T}( (x,              y,              one(T)-x-y))
    face == 4 && return Vec{3, T}( (zero(T),        one(T)-x-y,     y))
    throw(ArgumentError("unknown face number"))
end

function weighted_normal(J::Tensor{2,3}, cell_T::Type{RefTetrahedron}, face::Int)
    @inbounds begin
        face == 1 && return J[:,2] × J[:,1]
        face == 2 && return J[:,1] × J[:,3]
        face == 3 && return (J[:,1]-J[:,3]) × (J[:,2]-J[:,3])
        face == 4 && return J[:,3] × J[:,2]
    end
    throw(ArgumentError("unknown face number"))
end

###################
# All RefPrism 3D #
###################

# Mapping from 2D quadrilateral/triangle to 3D face of a wedge.
function element_face_transformation(point::Vec{2, T}, cell_T::Type{RefPrism}, face::Int) where T
    # Note that for quadrilaterals the domain is [-1, 1]² but for triangles it is [0, 1]²
    x,y = point
    face == 1 && return Vec{3, T}(( one(T)-x-y,             y,                      zero(T)))
    face == 2 && return Vec{3, T}(( (one(T)+x)/2,           zero(T),                (one(T)+y)/2))
    face == 3 && return Vec{3, T}(( zero(T),                one(T)-(one(T)+x)/2,    (one(T)+y)/2))
    face == 4 && return Vec{3, T}(( one(T)-(one(T)+x)/2,   (one(T)+x)/2,            (one(T)+y)/2))
    face == 5 && return Vec{3, T}(( y,                      one(T)-x-y,             one(T)))
    throw(ArgumentError("unknown face number"))
end

function weighted_normal(J::Tensor{2,3}, cell_T::Type{RefPrism}, face::Int)
    @inbounds begin
        face == 1 && return J[:,2] × J[:,1]
        face == 2 && return J[:,1] × J[:,3]
        face == 3 && return J[:,3] × J[:,2]
        face == 4 && return (J[:,2]-J[:,1]) × J[:,3]
        face == 5 && return J[:,1] × J[:,2]
    end
    throw(ArgumentError("unknown face number"))
end

#####################
# All RefPyramid 3D #
#####################

# Mapping from 2D face to 3D face of a pyramid.
function element_face_transformation(point::Vec{2, T}, cell_T::Type{RefPyramid}, face::Int) where T
    x,y = point
    face == 1 && return Vec{3, T}(( (y+one(T))/2,   (x+one(T))/2,       zero(T)))
    face == 2 && return Vec{3, T}(( y,              zero(T),            one(T)-x-y))
    face == 3 && return Vec{3, T}(( zero(T),        one(T)-x-y,         y))
    face == 4 && return Vec{3, T}(( x+y,            y,                  one(T)-x-y))
    face == 5 && return Vec{3, T}(( one(T)-x-y,     one(T)-y,           y))
    throw(ArgumentError("unknown face number"))
end

function weighted_normal(J::Tensor{2,3}, cell_T::Type{RefPyramid}, face::Int)
    @inbounds begin
        face == 1 && return J[:,2] × J[:,1]
        face == 2 && return J[:,1] × J[:,3]
        face == 3 && return J[:,3] × J[:,2]
        face == 4 && return J[:,2] × (J[:,3]-J[:,1])
        face == 5 && return (J[:,3]-J[:,2]) × J[:,1]
    end
    throw(ArgumentError("unknown face number"))
end

