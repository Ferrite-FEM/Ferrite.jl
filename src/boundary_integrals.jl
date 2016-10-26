##################
# All 1D RefCube #
##################
function create_boundary_quad_rule{T, shape <: RefCube}(quad_rule::QuadratureRule{0, shape, T}, ::FunctionSpace{1, shape})
    w = getweights(quad_rule)
    boundary_quad_rule = QuadratureRule{1, shape, T}[]

    # Boundary 1
    new_points = [Vec{1, T}((-one(T),))] # ξ = -1
    push!(boundary_quad_rule, QuadratureRule{1, shape, T}(w, new_points))
    # Boundary 2
    new_points = [Vec{1, T}((one(T),))] # ξ = 1
    push!(boundary_quad_rule, QuadratureRule{1, shape, T}(w, new_points))

    return boundary_quad_rule
end

detJ_boundary{T}(::Tensor{2, 1, T}, ::FunctionSpace{1, RefCube}, ::Int) = one(T)

##################
# All 2D RefCube #
##################
function create_boundary_quad_rule{T, shape <: RefCube}(quad_rule::QuadratureRule{1, shape, T}, ::FunctionSpace{2, shape})
    w = getweights(quad_rule)
    p = getpoints(quad_rule)
    n_points = length(w)
    boundary_quad_rule = QuadratureRule{2, shape, T}[]

    # Boundary 1
    new_points = [Vec{2, T}((p[i][1], -one(T))) for i in 1:n_points] # ξ = t, η = -1
    push!(boundary_quad_rule, QuadratureRule{2, shape, T}(w, new_points))
    # Boundary 2
    new_points = [Vec{2, T}((one(T), p[i][1])) for i in 1:n_points] # ξ = 1, η = t
    push!(boundary_quad_rule, QuadratureRule{2, shape, T}(w, new_points))
    # Boundary 3
    new_points = [Vec{2, T}((p[i][1], one(T))) for i in 1:n_points] # ξ = t, η = 1
    push!(boundary_quad_rule, QuadratureRule{2, shape, T}(w, new_points))
    # Boundary 4
    new_points = [Vec{2, T}((-one(T), p[i][1])) for i in 1:n_points] # ξ = -1, η = t
    push!(boundary_quad_rule, QuadratureRule{2, shape, T}(w, new_points))

    return boundary_quad_rule
end

function detJ_boundary(J::Tensor{2, 2}, ::FunctionSpace{2, RefCube}, boundary::Int)
    boundary == 1 && return sqrt(J[1,1]^2 + J[1,2]^2)
    boundary == 2 && return sqrt(J[2,1]^2 + J[2,2]^2)
    boundary == 3 && return sqrt(J[1,1]^2 + J[1,2]^2)
    boundary == 4 && return sqrt(J[2,1]^2 + J[2,2]^2)
end

#########################
# All RefTetrahedron 2D #
#########################
function create_boundary_quad_rule{T, shape <: RefTetrahedron}(quad_rule::QuadratureRule{1, shape, T}, ::FunctionSpace{2, shape})
    w = getweights(quad_rule)
    p = getpoints(quad_rule)
    n_points = length(w)
    boundary_quad_rule = QuadratureRule{2, shape, T}[]

    # Boundary 1
    new_points = [Vec{2, T}((p[i][1], one(T)-p[i][1])) for i in 1:n_points] # ξ = t, η = 1-t
    push!(boundary_quad_rule, QuadratureRule{2, shape, T}(w, new_points))
    # Boundary 2
    new_points = [Vec{2, T}((zero(T), p[i][1])) for i in 1:n_points] # ξ = 0, η = t
    push!(boundary_quad_rule, QuadratureRule{2, shape, T}(w, new_points))
    # Boundary 3
    new_points = [Vec{2, T}((p[i][1], zero(T))) for i in 1:n_points] # ξ = t, η = 0
    push!(boundary_quad_rule, QuadratureRule{2, shape, T}(w, new_points))

    return boundary_quad_rule
end

function detJ_boundary(J::Tensor{2, 2}, ::FunctionSpace{2, RefTetrahedron}, boundary::Int)
    boundary == 1 && return sqrt((J[1,1] - J[2,1])^2 + (J[1,2] - J[2,2])^2)
    boundary == 2 && return sqrt(J[2,1]^2 + J[2,2]^2)
    boundary == 3 && return sqrt(J[1,1]^2 + J[1,2]^2)
end

##################
# All RefCube 3D #
##################
function create_boundary_quad_rule{T, shape <: RefCube}(quad_rule::QuadratureRule{2, shape, T}, ::FunctionSpace{3, shape})
    w = getweights(quad_rule)
    p = getpoints(quad_rule)
    n_points = length(w)
    boundary_quad_rule = QuadratureRule{3, shape, T}[]

    # Boundary 1
    new_points = [Vec{3, T}((p[i][1], p[i][2], -one(T))) for i in 1:n_points] # ξ = t, η = s, ζ = -1
    push!(boundary_quad_rule, QuadratureRule{3, shape, T}(w, new_points))
    # Boundary 2
    new_points = [Vec{3, T}((p[i][1], -one(T), p[i][2])) for i in 1:n_points] # ξ = t, η = -1, ζ = s
    push!(boundary_quad_rule, QuadratureRule{3, shape, T}(w, new_points))
    # Boundary 3
    new_points = [Vec{3, T}((one(T), p[i][1], p[i][2])) for i in 1:n_points] # ξ = 1, η = t, ζ = s
    push!(boundary_quad_rule, QuadratureRule{3, shape, T}(w, new_points))
    # Boundary 4
    new_points = [Vec{3, T}((p[i][1], one(T), p[i][2])) for i in 1:n_points] # ξ = t, η = 1, ζ = s
    push!(boundary_quad_rule, QuadratureRule{3, shape, T}(w, new_points))
    # Boundary 5
    new_points = [Vec{3, T}((-one(T), p[i][1], p[i][2])) for i in 1:n_points] # ξ = -1, η = t, ζ = s
    push!(boundary_quad_rule, QuadratureRule{3, shape, T}(w, new_points))
    # Boundary 6
    new_points = [Vec{3, T}((p[i][1], p[i][2], one(T))) for i in 1:n_points] # ξ = t, η = s, ζ = 1
    push!(boundary_quad_rule, QuadratureRule{3, shape, T}(w, new_points))

    return boundary_quad_rule
end

function detJ_boundary(J::Tensor{2, 3}, ::FunctionSpace{3, RefCube}, boundary::Int)
    boundary == 1 && return norm(J[1,:] × J[2,:])
    boundary == 2 && return norm(J[1,:] × J[3,:])
    boundary == 3 && return norm(J[2,:] × J[3,:])
    boundary == 4 && return norm(J[1,:] × J[3,:])
    boundary == 5 && return norm(J[2,:] × J[3,:])
    boundary == 6 && return norm(J[1,:] × J[2,:])
end

#########################
# All RefTetrahedron 3D #
#########################
function create_boundary_quad_rule{T, shape <: RefTetrahedron}(quad_rule::QuadratureRule{2, shape, T}, ::FunctionSpace{3, shape})
    w = getweights(quad_rule)
    p = getpoints(quad_rule)
    n_points = length(w)
    boundary_quad_rule = QuadratureRule{3, shape, T}[]

    # Boundary 1
    new_points = [Vec{3, T}((p[i][1], p[i][2], zero(T))) for i in 1:n_points] # ξ = t, η = s, ζ = 0
    push!(boundary_quad_rule, QuadratureRule{3, shape, T}(w, new_points))
    # Boundary 2
    new_points = [Vec{3, T}((p[i][1], zero(T), p[i][2])) for i in 1:n_points] # ξ = t, η = 0, ζ = s
    push!(boundary_quad_rule, QuadratureRule{3, shape, T}(w, new_points))
    # Boundary 3
    new_points = [Vec{3, T}((p[i][1], p[i][2], one(T)-p[i][1]-p[i][2])) for i in 1:n_points] # ξ = t, η = s, ζ = 1-t-s
    push!(boundary_quad_rule, QuadratureRule{3, shape, T}(w, new_points))
    # Boundary 4
    new_points = [Vec{3, T}((zero(T), p[i][1], p[i][2])) for i in 1:n_points] # ξ = 0, η = t, ζ = s
    push!(boundary_quad_rule, QuadratureRule{3, shape, T}(w, new_points))

    return boundary_quad_rule
end

function detJ_boundary(J::Tensor{2, 3}, ::FunctionSpace{3, RefTetrahedron}, boundary::Int)
    boundary == 1 && return norm(J[1,:] × J[2,:])
    boundary == 2 && return norm(J[1,:] × J[3,:])
    boundary == 3 && return norm((J[1,:]-J[3,:]) × (J[2,:]-J[3,:]))
    boundary == 4 && return norm(J[2,:] × J[3,:])
end
