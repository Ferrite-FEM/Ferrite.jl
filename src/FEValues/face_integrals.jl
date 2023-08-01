"""
    weighted_normal(J::AbstractTensor, fv::FaceValues, face::Int)
    weighted_normal(J::AbstractTensor, ::Type{<:AbstractRefShape}, face::Int)

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
        new_points = [transfer_point_face_to_cell(N != 0 ? p[i] : Vec(zero(T)), cell_T, face) for i in 1:n_points] # ξ = 1-t-s, η = s, ζ = 0
        push!(face_quad_rule, QuadratureRule{RefShape, T}(w, new_points))    
    end
    return FaceQuadratureRule(face_quad_rule)
end

# For cells with mixed faces
function create_face_quad_rule(cell_T::Type{RefShape}, quad_faces::Vector{Int}, w_quad::Vector{T}, p_quad::Vector{Vec{N, T}}, tri_faces::Vector{Int}, w_tri::Vector{T}, p_tri::Vector{Vec{N, T}}) where {N, T, RefShape <: AbstractRefShape}
    n_points_quad = length(w_quad)
    n_points_tri = length(w_tri)
    face_quad_rule = Array{QuadratureRule{RefShape, T, getdim(AbstractCell{cell_T})}}(undef, nfaces(cell_T))
    for face in quad_faces
        new_points = [transfer_point_face_to_cell(N != 0 ? p_quad[i] : Vec(zero(T)), cell_T, face) for i in 1:n_points_quad]
        face_quad_rule[face] = QuadratureRule{RefShape, T}(w_quad, new_points)
    end
    for face in tri_faces
        new_points = [transfer_point_face_to_cell(N != 0 ? p_tri[i] : T[], cell_T, face) for i in 1:n_points_tri]
        face_quad_rule[face] = QuadratureRule{RefShape, T}(w_tri, new_points)
    end
    return FaceQuadratureRule(face_quad_rule)
end

##################
# All 1D RefLine #
##################

function weighted_normal(::Tensor{2,1,T}, ::Type{RefLine}, face::Int) where {T}
    face == 1 && return Vec{1,T}((-one(T),))
    face == 2 && return Vec{1,T}(( one(T),))
    throw(ArgumentError("unknown face number: $face"))
end

###########################
# All 2D RefQuadrilateral #
###########################

function weighted_normal(J::Tensor{2,2}, ::Type{RefQuadrilateral}, face::Int)
    @inbounds begin
        face == 1 && return Vec{2}(( J[2,1], -J[1,1]))
        face == 2 && return Vec{2}(( J[2,2], -J[1,2]))
        face == 3 && return Vec{2}((-J[2,1],  J[1,1]))
        face == 4 && return Vec{2}((-J[2,2],  J[1,2]))
    end
    throw(ArgumentError("unknown face number: $face"))
end

######################
# All RefTriangle 2D #
######################

function weighted_normal(J::Tensor{2,2}, ::Type{RefTriangle}, face::Int)
    @inbounds begin
        face == 1 && return Vec{2}((-(J[2,1] - J[2,2]), J[1,1] - J[1,2]))
        face == 2 && return Vec{2}((-J[2,2], J[1,2]))
        face == 3 && return Vec{2}((J[2,1], -J[1,1]))
    end
    throw(ArgumentError("unknown face number: $face"))
end

########################
# All RefHexahedron 3D #
########################

function weighted_normal(J::Tensor{2,3}, ::Type{RefHexahedron}, face::Int)
    @inbounds begin
        face == 1 && return J[:,2] × J[:,1]
        face == 2 && return J[:,1] × J[:,3]
        face == 3 && return J[:,2] × J[:,3]
        face == 4 && return J[:,3] × J[:,1]
        face == 5 && return J[:,3] × J[:,2]
        face == 6 && return J[:,1] × J[:,2]
    end
    throw(ArgumentError("unknown face number: $face"))
end

#########################
# All RefTetrahedron 3D #
#########################

function weighted_normal(J::Tensor{2,3}, ::Type{RefTetrahedron}, face::Int)
    @inbounds begin
        face == 1 && return J[:,2] × J[:,1]
        face == 2 && return J[:,1] × J[:,3]
        face == 3 && return (J[:,1]-J[:,3]) × (J[:,2]-J[:,3])
        face == 4 && return J[:,3] × J[:,2]
    end
    throw(ArgumentError("unknown face number: $face"))
end

###################
# All RefPrism 3D #
###################

function weighted_normal(J::Tensor{2,3}, ::Type{RefPrism}, face::Int)
    @inbounds begin
        face == 1 && return J[:,2] × J[:,1]
        face == 2 && return J[:,1] × J[:,3]
        face == 3 && return J[:,3] × J[:,2]
        face == 4 && return (J[:,2]-J[:,1]) × J[:,3]
        face == 5 && return J[:,1] × J[:,2]
    end
    throw(ArgumentError("unknown face number: $face"))
end

#####################
# All RefPyramid 3D #
#####################

function weighted_normal(J::Tensor{2,3}, ::Type{RefPyramid}, face::Int)
    @inbounds begin
        face == 1 && return J[:,2] × J[:,1]
        face == 2 && return J[:,1] × J[:,3]
        face == 3 && return J[:,3] × J[:,2]
        face == 4 && return J[:,2] × (J[:,3]-J[:,1])
        face == 5 && return (J[:,3]-J[:,2]) × J[:,1]
    end
    throw(ArgumentError("unknown face number: $face"))
end

