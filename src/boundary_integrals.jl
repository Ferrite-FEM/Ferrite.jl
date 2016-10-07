##################
# All 1D RefCube #
##################
function create_boundary_quad_rule{T}(::FunctionSpace{1, RefCube}, quad_rule::QuadratureRule{0, T})
    w = weights(quad_rule)
    boundary_quad_rule = QuadratureRule{1, T}[]

    # Boundary 1
    new_points = [-1.0]
    push!(boundary_quad_rule, QuadratureRule(w,reinterpret(Vec{1, T}, new_points, (length(w),))))
    # Boundary 2
    new_points = [1.0]
    push!(boundary_quad_rule, QuadratureRule(w,reinterpret(Vec{1, T}, new_points,(length(w),))))

    return boundary_quad_rule
end

detJ_boundary{T}(::FunctionSpace{1, RefCube}, ::Tensor{2, 1, T}, ::Int) = one(T)

##################
# All 2D RefCube #
##################
function create_boundary_quad_rule{T}(::FunctionSpace{2, RefCube}, quad_rule::QuadratureRule{1, T})
    w = weights(quad_rule)
    p = reinterpret(T,points(quad_rule),(1,length(w)))
    t = p[1,:]'
    boundary_quad_rule = QuadratureRule{2, T}[]

    # Boundary 1
    new_points = [t; -ones(t)] # ξ = t, η = -1
    push!(boundary_quad_rule, QuadratureRule(w,reinterpret(Vec{2, T},new_points,(length(w),))))
    # Boundary 2
    new_points = [ones(t); t] # ξ = 1, η = t
    push!(boundary_quad_rule, QuadratureRule(w,reinterpret(Vec{2, T},new_points,(length(w),))))
    # Boundary 3
    new_points = [t; ones(t)] # ξ = t, η = 1
    push!(boundary_quad_rule, QuadratureRule(w,reinterpret(Vec{2, T},new_points,(length(w),))))
    # Boundary 4
    new_points = [-ones(t); t] # ξ = -1, η = t
    push!(boundary_quad_rule, QuadratureRule(w,reinterpret(Vec{2, T},new_points,(length(w),))))

    return boundary_quad_rule
end

function detJ_boundary(::FunctionSpace{2, RefCube}, J::Tensor{2, 2}, boundary::Int)
    boundary == 1 && return sqrt(J[1,1]^2 + J[1,2]^2)
    boundary == 2 && return sqrt(J[2,1]^2 + J[2,2]^2)
    boundary == 3 && return sqrt(J[1,1]^2 + J[1,2]^2)
    boundary == 4 && return sqrt(J[2,1]^2 + J[2,2]^2)
end

#########################
# All RefTetrahedron 2D #
#########################
function create_boundary_quad_rule{T}(::FunctionSpace{2, RefTetrahedron}, quad_rule::QuadratureRule{1, T})
    w = weights(quad_rule)
    p = reinterpret(T,points(quad_rule),(1,length(w)))
    t = p[1,:]'
    boundary_quad_rule = QuadratureRule{2, T}[]

    # Boundary 1
    new_points = [t; one(T)-t] # ξ = t, η = 1-t
    push!(boundary_quad_rule, QuadratureRule(w,reinterpret(Vec{2, T},new_points,(length(w),))))
    # Boundary 2
    new_points = [zeros(t); t] # ξ = 0, η = t
    push!(boundary_quad_rule, QuadratureRule(w,reinterpret(Vec{2, T},new_points,(length(w),))))
    # Boundary 3
    new_points = [t; zeros(t)] # ξ = t, η = 0
    push!(boundary_quad_rule, QuadratureRule(w,reinterpret(Vec{2, T},new_points,(length(w),))))

    return boundary_quad_rule
end

function detJ_boundary(::FunctionSpace{2, RefTetrahedron}, J::Tensor{2, 2}, boundary::Int)
    boundary == 1 && return sqrt((J[1,1] - J[2,1])^2 + (J[1,2] - J[2,2])^2)
    boundary == 2 && return sqrt(J[2,1]^2 + J[2,2]^2)
    boundary == 3 && return sqrt(J[1,1]^2 + J[1,2]^2)
end

##################
# All RefCube 3D #
##################
function create_boundary_quad_rule{T}(::FunctionSpace{3, RefCube}, quad_rule::QuadratureRule{2, T})
    w = weights(quad_rule)
    p = reinterpret(T,points(quad_rule),(2,length(w)))
    t = p[1,:]'; s = p[2,:]'
    boundary_quad_rule = QuadratureRule{3, T}[]

    # Boundary 1
    new_points = [t; s; -ones(t)] # ξ = t, η = s, ζ = -1
    push!(boundary_quad_rule, QuadratureRule(w,reinterpret(Vec{3, T},new_points,(length(w),))))
    # Boundary 2
    new_points = [t; -ones(t); s] # ξ = t, η = -1, ζ = s
    push!(boundary_quad_rule, QuadratureRule(w,reinterpret(Vec{3, T},new_points,(length(w),))))
    # Boundary 3
    new_points = [ones(t); t; s] # ξ = 1, η = t, ζ = s
    push!(boundary_quad_rule, QuadratureRule(w,reinterpret(Vec{3, T},new_points,(length(w),))))
    # Boundary 4
    new_points = [t; ones(t); s] # ξ = t, η = 1, ζ = s
    push!(boundary_quad_rule, QuadratureRule(w,reinterpret(Vec{3, T},new_points,(length(w),))))
    # Boundary 5
    new_points = [-ones(t); t; s] # ξ = -1, η = t, ζ = s
    push!(boundary_quad_rule, QuadratureRule(w,reinterpret(Vec{3, T},new_points,(length(w),))))
    # Boundary 6
    new_points = [t; s; ones(t)] # ξ = t, η = s, ζ = 1
    push!(boundary_quad_rule, QuadratureRule(w,reinterpret(Vec{3, T},new_points,(length(w),))))

    return boundary_quad_rule
end

function detJ_boundary(::FunctionSpace{3, RefCube}, J::Tensor{2, 3}, boundary::Int)
    boundary == 1 && return norm(J[1,:] × J[2,:])
    boundary == 2 && return norm(J[1,:] × J[3,:])
    boundary == 3 && return norm(J[2,:] × J[3,:])
    boundary == 4 && return norm(J[1,:] × J[3,:])
    boundary == 5 && return norm(J[2,:] × J[2,:])
    boundary == 6 && return norm(J[1,:] × J[2,:])
end
