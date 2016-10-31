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

* [`getquadrule`](@ref)
* [`getfunctionspace`](@ref)
* [`getgeometricspace`](@ref)
* [`getdetJdV`](@ref)

* [`shape_value`](@ref)
* [`shape_gradient`](@ref)
* [`shape_divergence`](@ref)
* [`shape_derivative`](@ref)

* [`function_value`](@ref)
* [`function_gradient`](@ref)
* [`function_symmetric_gradient`](@ref)
* [`function_divergence`](@ref)
* [`spatial_coordinate`](@ref)
"""
immutable FECellValues{dim, T <: Real, FS <: FunctionSpace, GS <: FunctionSpace, shape <: AbstractRefShape} <: AbstractFECellValues{dim, T, FS, GS}
    N::Matrix{T}
    dNdx::Matrix{Vec{dim, T}}
    dNdξ::Matrix{Vec{dim, T}}
    detJdV::Vector{T}
    quad_rule::QuadratureRule{dim, shape, T}
    function_space::FS
    M::Matrix{T}
    dMdξ::Matrix{Vec{dim, T}}
    geometric_space::GS
end

FECellValues{dim, FS <: FunctionSpace, GS <: FunctionSpace}(quad_rule::QuadratureRule{dim}, func_space::FS, geom_space::GS=func_space) = FECellValues(Float64, quad_rule, func_space, geom_space)

function FECellValues{dim, T, FS <: FunctionSpace, GS <: FunctionSpace, shape <: AbstractRefShape}(::Type{T}, quad_rule::QuadratureRule{dim, shape}, func_space::FS, geom_space::GS=func_space)
    @assert getdim(func_space) == getdim(geom_space)
    @assert getrefshape(func_space) == getrefshape(geom_space) == shape
    n_qpoints = length(getweights(quad_rule))

    # Function interpolation
    n_func_basefuncs = getnbasefunctions(func_space)
    N = zeros(T, n_func_basefuncs, n_qpoints)
    dNdx = zeros(Vec{dim, T}, n_func_basefuncs, n_qpoints)
    dNdξ = zeros(Vec{dim, T}, n_func_basefuncs, n_qpoints)

    # Geometry interpolation
    n_geom_basefuncs = getnbasefunctions(geom_space)
    M = zeros(T, n_geom_basefuncs, n_qpoints)
    dMdξ = zeros(Vec{dim, T}, n_geom_basefuncs, n_qpoints)

    for (i, ξ) in enumerate(quad_rule.points)
        value!(func_space,  view(N, :, i), ξ)
        derivative!(func_space,  view(dNdξ, :, i), ξ)
        value!(geom_space,  view(M, :, i), ξ)
        derivative!(geom_space,  view(dMdξ, :, i), ξ)
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
function reinit!{dim, T}(fe_cv::AbstractFECellValues{dim}, x::Vector{Vec{dim, T}})
    n_geom_basefuncs = getnbasefunctions(getgeometricspace(fe_cv))
    n_func_basefuncs = getnbasefunctions(getfunctionspace(fe_cv))
    @assert length(x) == n_geom_basefuncs
    isa(fe_cv, FECellVectorValues) && (n_func_basefuncs *= dim)

    @inbounds for i in 1:length(getpoints(fe_cv.quad_rule))
        w = getweights(fe_cv.quad_rule)[i]
        fecv_J = zero(Tensor{2, dim})
        for j in 1:n_geom_basefuncs
            fecv_J += fe_cv.dMdξ[j, i] ⊗ x[j]
        end
        Jinv = inv(fecv_J)
        for j in 1:n_func_basefuncs
            fe_cv.dNdx[j,i] = Jinv ⋅ fe_cv.dNdξ[j, i]
        end
        detJ = det(fecv_J)
        detJ <= 0.0 && throw(ArgumentError("detJ is not positive: detJ = $(detJ)"))
        fe_cv.detJdV[i] = detJ * w
    end
end
