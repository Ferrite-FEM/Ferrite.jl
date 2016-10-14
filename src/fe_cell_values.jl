"""
An `FECellValues` object facilitates the process of evaluating values shape functions, gradients of shape functions,
values of nodal functions, gradients and divergences of nodal functions etc. in the finite element cell

**Constructor**

    FECellValues([::Type{T}], quad_rule::QuadratureRule, function_space::FunctionSpace, [geometric_space::FunctionSpace])


**Arguments**

* `T` an optional argument to determine the type the internal data is stored as.
* `quad_rule` an instance of a [`QuadratureRule`](@ref)
* `function_space` an instance of a [`FunctionSpace`](@ref) used to interpolate the approximated function
* `geometric_space` an optional instance of a [`FunctionSpace`](@ref) which is used to interpolate the geometry 

** Common methods**

* [`get_quadrule`](@ref)
* [`get_functionspace`](@ref)
* [`get_geometricspace`](@ref)
* [`detJdV`](@ref)

* [`shape_value`](@ref)
* [`shape_gradient`](@ref)
* [`shape_divergence`](@ref)
* [`shape_derivative`](@ref)

* [`function_scalar_value`](@ref)
* [`function_vector_value`](@ref)
* [`function_scalar_gradient`](@ref)
* [`function_vector_divergence`](@ref)
* [`function_vector_gradient`](@ref)
* [`function_vector_symmetric_gradient`](@ref)
* [`spatial_coordinate`](@ref)
"""
immutable FECellValues{dim, T <: Real, FS <: FunctionSpace, GS <: FunctionSpace, shape <: AbstractRefShape} <: AbstractFEValues{dim, T, FS, GS}
    N::Vector{Vector{T}}
    dNdx::Vector{Vector{Vec{dim, T}}}
    dNdξ::Vector{Vector{Vec{dim, T}}}
    detJdV::Vector{T}
    quad_rule::QuadratureRule{dim, shape, T}
    function_space::FS
    M::Vector{Vector{T}}
    dMdξ::Vector{Vector{Vec{dim, T}}}
    geometric_space::GS
end

FECellValues{dim, FS <: FunctionSpace, GS <: FunctionSpace}(quad_rule::QuadratureRule{dim}, func_space::FS, geom_space::GS=func_space) = FECellValues(Float64, quad_rule, func_space, geom_space)

function FECellValues{dim, T, FS <: FunctionSpace, GS <: FunctionSpace, shape <: AbstractRefShape}(::Type{T}, quad_rule::QuadratureRule{dim, shape}, func_space::FS, geom_space::GS=func_space)
    @assert functionspace_n_dim(func_space) == functionspace_n_dim(geom_space)
    @assert functionspace_ref_shape(func_space) == functionspace_ref_shape(geom_space) == shape
    n_qpoints = length(weights(quad_rule))

    # Function interpolation
    n_func_basefuncs = n_basefunctions(func_space)
    N = [zeros(T, n_func_basefuncs) for i in 1:n_qpoints]
    dNdx = [[zero(Vec{dim, T}) for i in 1:n_func_basefuncs] for j in 1:n_qpoints]
    dNdξ = [[zero(Vec{dim, T}) for i in 1:n_func_basefuncs] for j in 1:n_qpoints]

    # Geometry interpolation
    n_geom_basefuncs = n_basefunctions(geom_space)
    M = [zeros(T, n_geom_basefuncs) for i in 1:n_qpoints]
    dMdξ = [[zero(Vec{dim, T}) for i in 1:n_geom_basefuncs] for j in 1:n_qpoints]

    for (i, ξ) in enumerate(quad_rule.points)
        value!(func_space, N[i], ξ)
        derivative!(func_space, dNdξ[i], ξ)
        value!(geom_space, M[i], ξ)
        derivative!(geom_space, dMdξ[i], ξ)
    end

    detJdV = zeros(T, n_qpoints)

    FECellValues(N, dNdx, dNdξ, detJdV, quad_rule, func_space, M, dMdξ, geom_space)
end


"""
Updates the `FECellValues` object for an element.

    reinit!{dim, T}(fe_cv::FECellValues{dim}, x::Vector{Vec{dim, T}})

** Arguments **

* `fe_cv`: the `FECellValues` object
* `x`: A `Vector` of `Vec`, one for each nodal position in the element.

** Result **

* nothing


**Details**


"""
function reinit!{dim, T}(fe_cv::FECellValues{dim}, x::Vector{Vec{dim, T}})
    n_geom_basefuncs = n_basefunctions(get_geometricspace(fe_cv))
    n_func_basefuncs = n_basefunctions(get_functionspace(fe_cv))
    @assert length(x) == n_geom_basefuncs

    for i in 1:length(points(fe_cv.quad_rule))
        w = weights(fe_cv.quad_rule)[i]
        fecv_J = zero(Tensor{2, dim})
        for j in 1:n_geom_basefuncs
            fecv_J += fe_cv.dMdξ[i][j] ⊗ x[j]
        end
        Jinv = inv(fecv_J)
        for j in 1:n_func_basefuncs
            fe_cv.dNdx[i][j] = Jinv ⋅ fe_cv.dNdξ[i][j]
        end
        detJ = det(fecv_J)
        detJ <= 0.0 && throw(ArgumentError("detJ is not positive: detJ = $(detJ)"))
        fe_cv.detJdV[i] = detJ * w
    end
end
