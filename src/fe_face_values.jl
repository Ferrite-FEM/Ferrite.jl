immutable FEFaceValues{dim, T <: Real, FS <: FunctionSpace, GS <: FunctionSpace}
    N::Vector{Vector{Vector{T}}}
    dNdx::Vector{Vector{Vector{Vec{dim, T}}}}
    dNdξ::Vector{Vector{Vector{Vec{dim, T}}}}
    detJdS::Vector{Vector{T}}
    quad_rule::Vector{QuadratureRule{dim, T}}
    function_space::FS
    dMdξ::Vector{Vector{Vector{Vec{dim, T}}}}
    geometric_space::GS
    current_boundary::Ref{Int}
end

FEFaceValues{dim_qr, FS <: FunctionSpace, GS <: FunctionSpace}(quad_rule::QuadratureRule{dim_qr}, func_space::FS, geom_space::GS=func_space) = FEFaceValues(Float64, quad_rule, func_space, geom_space)

function FEFaceValues{dim_qr, T, FS <: FunctionSpace, GS <: FunctionSpace}(::Type{T}, quad_rule::QuadratureRule{dim_qr}, func_space::FS, geom_space::GS=func_space)
    @assert n_dim(func_space) == n_dim(geom_space)
    @assert ref_shape(func_space) == ref_shape(geom_space)
    n_qpoints = length(points(quad_rule))
    dim = dim_qr + 1

    boundary_quad_rule = create_boundary_quad_rule(func_space,quad_rule)
    n_bounds = length(boundary_quad_rule)

    # Function interpolation
    n_func_basefuncs = n_basefunctions(func_space)
    N =    [[zeros(T, n_func_basefuncs) for i in 1:n_qpoints]                      for k in 1:n_bounds]
    dNdx = [[[zero(Vec{dim, T}) for i in 1:n_func_basefuncs] for j in 1:n_qpoints] for k in 1:n_bounds]
    dNdξ = [[[zero(Vec{dim, T}) for i in 1:n_func_basefuncs] for j in 1:n_qpoints] for k in 1:n_bounds]

    # Geometry interpolation
    n_geom_basefuncs = n_basefunctions(geom_space)
    dMdξ = [[[zero(Vec{dim, T}) for i in 1:n_geom_basefuncs] for j in 1:n_qpoints] for k in 1:n_bounds]
    for k in 1:n_bounds, (i, ξ) in enumerate(boundary_quad_rule[k].points)
        value!(func_space, N[k][i], ξ)
        derivative!(func_space, dNdξ[k][i], ξ)
        derivative!(geom_space, dMdξ[k][i], ξ)
    end

    detJdS = [zeros(T, n_qpoints) for i in 1:n_bounds]

    FEFaceValues(N, dNdx, dNdξ, detJdS, boundary_quad_rule, func_space, dMdξ, geom_space, Ref(0))
end

function reinit!{dim, T}(fe_fv::FEFaceValues{dim}, x::Vector{Vec{dim, T}}, boundary::Int)
    n_geom_basefuncs = n_basefunctions(get_geometricspace(fe_fv))
    n_func_basefuncs = n_basefunctions(get_functionspace(fe_fv))
    @assert length(x) == n_geom_basefuncs

    fe_fv.current_boundary[] = boundary
    cb = current_boundary(fe_fv)

    for i in 1:length(points(fe_fv.quad_rule[cb]))
        w = weights(fe_fv.quad_rule[cb])[i]
        fefv_J = zero(Tensor{2, dim})
        for j in 1:n_geom_basefuncs
            fefv_J += fe_fv.dMdξ[cb][i][j] ⊗ x[j]
        end
        Jinv = inv(fefv_J)
        for j in 1:n_func_basefuncs
            fe_fv.dNdx[cb][i][j] = Jinv ⋅ fe_fv.dNdξ[cb][i][j]
        end
        detJ = detJ_boundary(get_geometricspace(fe_fv),fefv_J,cb)
        detJ <= 0.0 && throw(ArgumentError("detJ is not positive: detJ = $(detJ)"))
        fe_fv.detJdS[cb][i] = detJ * w
    end
end

"""
The current active boundary of the `FEFaceValues` type.

    current_boundary(fe_fv::FEFaceValues)

** Arguments **

* `fe_face_values`: the `FEFaceValues` object

** Results **

* `::Int`: the current active boundary (from last `reinit!`).

"""
current_boundary(fe_fv::FEFaceValues) = fe_fv.current_boundary[]
