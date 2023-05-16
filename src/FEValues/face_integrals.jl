# TODO: Remove when Interpolations are ported to new system
_get_new_refshape(::Interpolation{1, RefCube}) = RefLine
_get_new_refshape(::Interpolation{2, RefCube}) = RefQuadrilateral
_get_new_refshape(::Interpolation{3, RefCube}) = RefHexahedron
_get_new_refshape(::Interpolation{2, RefTetrahedron}) = RefTriangle
_get_new_refshape(::Interpolation{3, RefTetrahedron}) = RefTetrahedron

function weighted_normal(J::AbstractTensor, fv::FaceValues, face::Int)
    getdim(::Interpolation{dim}) where {dim} = dim
    return weighted_normal(J, _get_new_refshape(fv.func_interp), face)
end

##################
# All 1D RefCube #
##################
function create_face_quad_rule(quad_rule::QuadratureRule{0,shape,T}, ::Interpolation{1,shape}) where {T,shape<:RefCube}
    w = getweights(quad_rule)
    face_quad_rule = QuadratureRule{1,shape,T}[]

    # Face 1
    new_points = [Vec{1,T}((-one(T),))] # ξ = -1
    push!(face_quad_rule, QuadratureRule{1,shape,T}(w, new_points))
    # Face 2
    new_points = [Vec{1,T}((one(T),))] # ξ = 1
    push!(face_quad_rule, QuadratureRule{1,shape,T}(w, new_points))

    return face_quad_rule
end

function weighted_normal(::Tensor{2,1,T}, ::Type{RefLine}, face::Int) where {T}
    face == 1 && return Vec{1,T}((-one(T),))
    face == 2 && return Vec{1,T}(( one(T),))
    throw(ArgumentError("unknown face number: $face"))
end

##################
# All 2D RefCube #
##################
function create_face_quad_rule(quad_rule::QuadratureRule{1,shape,T}, ::Interpolation{2,shape}) where {T,shape<:RefCube}
    w = getweights(quad_rule)
    p = getpoints(quad_rule)
    n_points = length(w)
    face_quad_rule = QuadratureRule{2,shape,T}[]

    # Face 1
    new_points = [Vec{2,T}((p[i][1], -one(T))) for i in 1:n_points] # ξ = t, η = -1
    push!(face_quad_rule, QuadratureRule{2,shape,T}(w, new_points))
    # Face 2
    new_points = [Vec{2,T}((one(T), p[i][1])) for i in 1:n_points] # ξ = 1, η = t
    push!(face_quad_rule, QuadratureRule{2,shape,T}(w, new_points))
    # Face 3
    new_points = [Vec{2,T}((p[i][1], one(T))) for i in 1:n_points] # ξ = t, η = 1
    push!(face_quad_rule, QuadratureRule{2,shape,T}(w, new_points))
    # Face 4
    new_points = [Vec{2,T}((-one(T), p[i][1])) for i in 1:n_points] # ξ = -1, η = t
    push!(face_quad_rule, QuadratureRule{2,shape,T}(w, new_points))

    return face_quad_rule
end

function weighted_normal(J::Tensor{2,2}, ::Type{RefQuadrilateral}, face::Int)
    @inbounds begin
        face == 1 && return Vec{2}(( J[2,1], -J[1,1]))
        face == 2 && return Vec{2}(( J[2,2], -J[1,2]))
        face == 3 && return Vec{2}((-J[2,1],  J[1,1]))
        face == 4 && return Vec{2}((-J[2,2],  J[1,2]))
    end
    throw(ArgumentError("unknown face number: $face"))
end

#########################
# All RefTetrahedron 2D #
#########################
function create_face_quad_rule(quad_rule::QuadratureRule{1,shape,T}, ::Interpolation{2,shape}) where {T,shape<:RefTetrahedron}
    w = getweights(quad_rule)
    p = getpoints(quad_rule)
    n_points = length(w)
    face_quad_rule = QuadratureRule{2,shape,T}[]

    # Face 1
    new_points = [Vec{2,T}((p[i][1], one(T)-p[i][1])) for i in 1:n_points] # ξ = t, η = 1-t
    push!(face_quad_rule, QuadratureRule{2,shape,T}(w, new_points))
    # Face 2
    new_points = [Vec{2,T}((zero(T), p[i][1])) for i in 1:n_points] # ξ = 0, η = t
    push!(face_quad_rule, QuadratureRule{2,shape,T}(w, new_points))
    # Face 3
    new_points = [Vec{2,T}((p[i][1], zero(T))) for i in 1:n_points] # ξ = t, η = 0
    push!(face_quad_rule, QuadratureRule{2,shape,T}(w, new_points))

    return face_quad_rule
end

function weighted_normal(J::Tensor{2,2}, ::Type{RefTriangle}, face::Int)
    @inbounds begin
        face == 1 && return Vec{2}((-(J[2,1] - J[2,2]), J[1,1] - J[1,2]))
        face == 2 && return Vec{2}((-J[2,2], J[1,2]))
        face == 3 && return Vec{2}((J[2,1], -J[1,1]))
    end
    throw(ArgumentError("unknown face number: $face"))
end

##################
# All RefCube 3D #
##################
function create_face_quad_rule(quad_rule::QuadratureRule{2,shape,T}, ::Interpolation{3,shape}) where {T,shape<:RefCube}
    w = getweights(quad_rule)
    p = getpoints(quad_rule)
    n_points = length(w)
    face_quad_rule = QuadratureRule{3,shape,T}[]

    # Face 1
    new_points = [Vec{3,T}((p[i][1], p[i][2], -one(T))) for i in 1:n_points] # ξ = t, η = s, ζ = -1
    push!(face_quad_rule, QuadratureRule{3,shape,T}(w, new_points))
    # Face 2
    new_points = [Vec{3,T}((p[i][1], -one(T), p[i][2])) for i in 1:n_points] # ξ = t, η = -1, ζ = s
    push!(face_quad_rule, QuadratureRule{3,shape,T}(w, new_points))
    # Face 3
    new_points = [Vec{3,T}((one(T), p[i][1], p[i][2])) for i in 1:n_points] # ξ = 1, η = t, ζ = s
    push!(face_quad_rule, QuadratureRule{3,shape,T}(w, new_points))
    # Face 4
    new_points = [Vec{3,T}((p[i][1], one(T), p[i][2])) for i in 1:n_points] # ξ = t, η = 1, ζ = s
    push!(face_quad_rule, QuadratureRule{3,shape,T}(w, new_points))
    # Face 5
    new_points = [Vec{3,T}((-one(T), p[i][1], p[i][2])) for i in 1:n_points] # ξ = -1, η = t, ζ = s
    push!(face_quad_rule, QuadratureRule{3,shape,T}(w, new_points))
    # Face 6
    new_points = [Vec{3,T}((p[i][1], p[i][2], one(T))) for i in 1:n_points] # ξ = t, η = s, ζ = 1
    push!(face_quad_rule, QuadratureRule{3,shape,T}(w, new_points))

    return face_quad_rule
end

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
function create_face_quad_rule(quad_rule::QuadratureRule{2,shape,T}, ::Interpolation{3,shape}) where {T,shape<:RefTetrahedron}
    w = getweights(quad_rule)
    p = getpoints(quad_rule)
    n_points = length(w)
    face_quad_rule = QuadratureRule{3,shape,T}[]

    # Face 1
    new_points = [Vec{3,T}((p[i][1], p[i][2], zero(T))) for i in 1:n_points] # ξ = t, η = s, ζ = 0
    push!(face_quad_rule, QuadratureRule{3,shape,T}(w, new_points))
    # Face 2
    new_points = [Vec{3,T}((p[i][1], zero(T), p[i][2])) for i in 1:n_points] # ξ = t, η = 0, ζ = s
    push!(face_quad_rule, QuadratureRule{3,shape,T}(w, new_points))
    # Face 3
    new_points = [Vec{3,T}((p[i][1], p[i][2], one(T)-p[i][1]-p[i][2])) for i in 1:n_points] # ξ = t, η = s, ζ = 1-t-s
    push!(face_quad_rule, QuadratureRule{3,shape,T}(w, new_points))
    # Face 4
    new_points = [Vec{3,T}((zero(T), p[i][1], p[i][2])) for i in 1:n_points] # ξ = 0, η = t, ζ = s
    push!(face_quad_rule, QuadratureRule{3,shape,T}(w, new_points))

    return face_quad_rule
end

function weighted_normal(J::Tensor{2,3}, ::Type{RefTetrahedron}, face::Int)
    @inbounds begin
        face == 1 && return J[:,2] × J[:,1]
        face == 2 && return J[:,1] × J[:,3]
        face == 3 && return (J[:,1]-J[:,3]) × (J[:,2]-J[:,3])
        face == 4 && return J[:,3] × J[:,2]
    end
    throw(ArgumentError("unknown face number: $face"))
end
