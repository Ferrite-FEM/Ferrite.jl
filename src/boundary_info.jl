function boundary_information{dim_qr,T}(fs::Lagrange{2,RefCube,1},quad_rule::QuadratureRule{dim_qr,T})
    dim = n_dim(fs)
    @assert dim == dim_qr + 1

    w = weights(quad_rule)
    p = reinterpret(T,points(quad_rule),(dim_qr,length(w)))

    boundary_quad_rule = QuadratureRule{dim,T}[]

    # Boundary 1
    new_points = [p; -ones(p)] # ξ = t, η = -1
    push!(boundary_quad_rule, QuadratureRule(w,reinterpret(Vec{dim,T},new_points,(length(w),))))

    # Boundary 2
    new_points = [ones(p); p] # ξ = 1, η = t
    push!(boundary_quad_rule, QuadratureRule(w,reinterpret(Vec{dim,T},new_points,(length(w),))))

    # Boundary 3
    new_points = [p; ones(p)] # ξ = t, η = 1
    push!(boundary_quad_rule, QuadratureRule(w,reinterpret(Vec{dim,T},new_points,(length(w),))))
    # Boundary 4
    new_points = [-ones(p); p] # ξ = -1, η = t
    push!(boundary_quad_rule, QuadratureRule(w,reinterpret(Vec{dim,T},new_points,(length(w),))))


    return boundary_quad_rule
end
