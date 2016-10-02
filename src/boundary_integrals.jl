#########################
# Lagrange{2,RefCube,1} #
#########################
function create_boundary_quad_rule{T}(::Lagrange{2, RefCube, 1}, quad_rule::QuadratureRule{1, T})
    w = weights(quad_rule)
    p = reinterpret(T,points(quad_rule),(1,length(w)))
    boundary_quad_rule = QuadratureRule{2, T}[]

    # Boundary 1
    new_points = [p; -ones(p)] # ξ = t, η = -1
    push!(boundary_quad_rule, QuadratureRule(w,reinterpret(Vec{2, T},new_points,(length(w),))))
    # Boundary 2
    new_points = [ones(p); p] # ξ = 1, η = t
    push!(boundary_quad_rule, QuadratureRule(w,reinterpret(Vec{2, T},new_points,(length(w),))))
    # Boundary 3
    new_points = [p; ones(p)] # ξ = t, η = 1
    push!(boundary_quad_rule, QuadratureRule(w,reinterpret(Vec{2, T},new_points,(length(w),))))
    # Boundary 4
    new_points = [-ones(p); p] # ξ = -1, η = t
    push!(boundary_quad_rule, QuadratureRule(w,reinterpret(Vec{2, T},new_points,(length(w),))))

    return boundary_quad_rule
end

function detJ_boundary(::Lagrange{2, RefCube, 1}, J::Tensor{2, 2}, boundary::Int)
    boundary == 1 && return sqrt(J[1,1]^2 + J[1,2]^2)
    boundary == 2 && return sqrt(J[2,1]^2 + J[2,2]^2)
    boundary == 3 && return sqrt(J[1,1]^2 + J[1,2]^2)
    boundary == 4 && return sqrt(J[2,1]^2 + J[2,2]^2)
end
