function weighted_normal(J::AbstractTensor, fv::FaceValues, face::Int)
    return weighted_normal(J, getrefshape(fv.func_interp), face)
end

##################
# All 1D RefLine #
##################
function create_face_quad_rule(quad_rule::QuadratureRule{0,RefLine,T}, ::Interpolation{1,RefLine}) where {T}
    w = getweights(quad_rule)
    face_quad_rule = QuadratureRule{1,RefLine,T}[]

    # Face 1
    new_points = [Vec{1,T}((-one(T),))] # ξ = -1
    push!(face_quad_rule, QuadratureRule{1,RefLine,T}(w, new_points))
    # Face 2
    new_points = [Vec{1,T}((one(T),))] # ξ = 1
    push!(face_quad_rule, QuadratureRule{1,RefLine,T}(w, new_points))

    return face_quad_rule
end

function weighted_normal(::Tensor{2,1,T}, ::Type{RefLine}, face::Int) where {T}
    face == 1 && return Vec{1,T}((-one(T),))
    face == 2 && return Vec{1,T}(( one(T),))
    throw(ArgumentError("unknown face number: $face"))
end

###########################
# All 2D RefQuadrilateral #
###########################
function create_face_quad_rule(quad_rule::QuadratureRule{1,RefQuadrilateral,T}, ::Interpolation{2,RefQuadrilateral}) where {T}
    w = getweights(quad_rule)
    p = getpoints(quad_rule)
    n_points = length(w)
    face_quad_rule = QuadratureRule{2,RefQuadrilateral,T}[]

    # Face 1
    new_points = [Vec{2,T}((p[i][1], -one(T))) for i in 1:n_points] # ξ = t, η = -1
    push!(face_quad_rule, QuadratureRule{2,RefQuadrilateral,T}(w, new_points))
    # Face 2
    new_points = [Vec{2,T}((one(T), p[i][1])) for i in 1:n_points] # ξ = 1, η = t
    push!(face_quad_rule, QuadratureRule{2,RefQuadrilateral,T}(w, new_points))
    # Face 3
    new_points = [Vec{2,T}((p[i][1], one(T))) for i in 1:n_points] # ξ = t, η = 1
    push!(face_quad_rule, QuadratureRule{2,RefQuadrilateral,T}(w, new_points))
    # Face 4
    new_points = [Vec{2,T}((-one(T), p[i][1])) for i in 1:n_points] # ξ = -1, η = t
    push!(face_quad_rule, QuadratureRule{2,RefQuadrilateral,T}(w, new_points))

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

######################
# All RefTriangle 2D #
######################
function create_face_quad_rule(quad_rule::QuadratureRule{1,RefTriangle,T}, ::Interpolation{2,RefTriangle}) where {T}
    w = getweights(quad_rule)
    p = getpoints(quad_rule)
    n_points = length(w)
    face_quad_rule = QuadratureRule{2,RefTriangle,T}[]

    # Face 1
    new_points = [Vec{2,T}((p[i][1], one(T)-p[i][1])) for i in 1:n_points] # ξ = t, η = 1-t
    push!(face_quad_rule, QuadratureRule{2,RefTriangle,T}(w, new_points))
    # Face 2
    new_points = [Vec{2,T}((zero(T), p[i][1])) for i in 1:n_points] # ξ = 0, η = t
    push!(face_quad_rule, QuadratureRule{2,RefTriangle,T}(w, new_points))
    # Face 3
    new_points = [Vec{2,T}((p[i][1], zero(T))) for i in 1:n_points] # ξ = t, η = 0
    push!(face_quad_rule, QuadratureRule{2,RefTriangle,T}(w, new_points))

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

########################
# All RefHexahedron 3D #
########################
function create_face_quad_rule(quad_rule::QuadratureRule{2,RefHexahedron,T}, ::Interpolation{3,RefHexahedron}) where {T}
    w = getweights(quad_rule)
    p = getpoints(quad_rule)
    n_points = length(w)
    face_quad_rule = QuadratureRule{3,RefHexahedron,T}[]

    # Face 1
    new_points = [Vec{3,T}((p[i][1], p[i][2], -one(T))) for i in 1:n_points] # ξ = t, η = s, ζ = -1
    push!(face_quad_rule, QuadratureRule{3,RefHexahedron,T}(w, new_points))
    # Face 2
    new_points = [Vec{3,T}((p[i][1], -one(T), p[i][2])) for i in 1:n_points] # ξ = t, η = -1, ζ = s
    push!(face_quad_rule, QuadratureRule{3,RefHexahedron,T}(w, new_points))
    # Face 3
    new_points = [Vec{3,T}((one(T), p[i][1], p[i][2])) for i in 1:n_points] # ξ = 1, η = t, ζ = s
    push!(face_quad_rule, QuadratureRule{3,RefHexahedron,T}(w, new_points))
    # Face 4
    new_points = [Vec{3,T}((p[i][1], one(T), p[i][2])) for i in 1:n_points] # ξ = t, η = 1, ζ = s
    push!(face_quad_rule, QuadratureRule{3,RefHexahedron,T}(w, new_points))
    # Face 5
    new_points = [Vec{3,T}((-one(T), p[i][1], p[i][2])) for i in 1:n_points] # ξ = -1, η = t, ζ = s
    push!(face_quad_rule, QuadratureRule{3,RefHexahedron,T}(w, new_points))
    # Face 6
    new_points = [Vec{3,T}((p[i][1], p[i][2], one(T))) for i in 1:n_points] # ξ = t, η = s, ζ = 1
    push!(face_quad_rule, QuadratureRule{3,RefHexahedron,T}(w, new_points))

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
function create_face_quad_rule(quad_rule::QuadratureRule{2,RefTetrahedron,T}, ::Interpolation{3,RefTetrahedron}) where {T}
    w = getweights(quad_rule)
    p = getpoints(quad_rule)
    n_points = length(w)
    face_quad_rule = QuadratureRule{3,RefTetrahedron,T}[]

    # Face 1
    new_points = [Vec{3,T}((p[i][1], p[i][2], zero(T))) for i in 1:n_points] # ξ = t, η = s, ζ = 0
    push!(face_quad_rule, QuadratureRule{3,RefTetrahedron,T}(w, new_points))
    # Face 2
    new_points = [Vec{3,T}((p[i][1], zero(T), p[i][2])) for i in 1:n_points] # ξ = t, η = 0, ζ = s
    push!(face_quad_rule, QuadratureRule{3,RefTetrahedron,T}(w, new_points))
    # Face 3
    new_points = [Vec{3,T}((p[i][1], p[i][2], one(T)-p[i][1]-p[i][2])) for i in 1:n_points] # ξ = t, η = s, ζ = 1-t-s
    push!(face_quad_rule, QuadratureRule{3,RefTetrahedron,T}(w, new_points))
    # Face 4
    new_points = [Vec{3,T}((zero(T), p[i][1], p[i][2])) for i in 1:n_points] # ξ = 0, η = t, ζ = s
    push!(face_quad_rule, QuadratureRule{3,RefTetrahedron,T}(w, new_points))

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
