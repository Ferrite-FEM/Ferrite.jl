immutable FECellVectorValues{dim, T <: Real, FS <: FunctionSpace, GS <: FunctionSpace, shape <: AbstractRefShape, M} <: AbstractFECellScalarValues{dim, T, FS, GS}
    N::Matrix{Vec{dim, T}}
    dNdx::Matrix{Tensor{2, dim, T, M}}
    dNdξ::Matrix{Tensor{2, dim, T, M}}
    detJdV::Vector{T}
    quad_rule::QuadratureRule{dim, shape, T}
    function_space::FS
    M::Matrix{T}
    dMdξ::Matrix{Vec{dim, T}}
    geometric_space::GS
end

FECellVectorValues{dim, FS <: FunctionSpace, GS <: FunctionSpace}(quad_rule::QuadratureRule{dim}, func_space::FS, geom_space::GS=func_space) = FECellVectorValues(Float64, quad_rule, func_space, geom_space)
getnbasefunctions{dim}(fe_cvv::FECellVectorValues{dim}) = getnbasefunctions(fe_cvv.function_space) * dim

function FECellVectorValues{dim, T, FS <: FunctionSpace, GS <: FunctionSpace, shape <: AbstractRefShape}(::Type{T}, quad_rule::QuadratureRule{dim, shape}, func_space::FS, geom_space::GS=func_space)
    @assert getdim(func_space) == getdim(geom_space)
    @assert getrefshape(func_space) == getrefshape(geom_space) == shape
    n_qpoints = length(getweights(quad_rule))

    # Function interpolation
    n_func_basefuncs = getnbasefunctions(func_space) * dim
    N    = zeros(Vec{dim, T}, n_func_basefuncs, n_qpoints)
    dNdx = [zero(Tensor{2, dim, T}) for i in 1:n_func_basefuncs, j in 1:n_qpoints]
    dNdξ = [zero(Tensor{2, dim, T}) for i in 1:n_func_basefuncs, j in 1:n_qpoints]

    # Geometry interpolation
    n_geom_basefuncs = getnbasefunctions(geom_space)
    M = zeros(T, n_geom_basefuncs, n_qpoints)
    dMdξ = zeros(Vec{dim, T}, n_geom_basefuncs, n_qpoints)

    N_temp = zeros(getnbasefunctions(func_space))
    dNdξ_temp = zeros(Vec{dim, T}, getnbasefunctions(func_space))
    for (i, ξ) in enumerate(quad_rule.points)
        value!(func_space, N_temp, ξ)
        derivative!(func_space, dNdξ_temp, ξ)
        basefunc_count = 1
        for basefunc in 1:getnbasefunctions(func_space)
            for comp in 1:dim
                N_comp = zeros(T, dim)
                N_comp[comp] = N_temp[basefunc]
                N[basefunc_count, i] = Vec{dim, T}((N_comp...))

                dN_comp = zeros(T, dim, dim)
                dN_comp[comp, :] = dNdξ_temp[basefunc]
                dNdξ[basefunc_count, i] = Tensor{2, dim, T}((dN_comp...))
                basefunc_count += 1
            end
        end
        value!(geom_space, view(M, :, i), ξ)
        derivative!(geom_space, view(dMdξ, :, i), ξ)
    end
    detJdV = zeros(T, n_qpoints)
    FECellVectorValues(N, dNdx, dNdξ, detJdV, quad_rule, func_space, M, dMdξ, geom_space)
end

