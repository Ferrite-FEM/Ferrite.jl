"""
An `CellScalarValues` object facilitates the process of evaluating values shape functions, gradients of shape functions,
values of nodal functions, gradients and divergences of nodal functions etc. in the finite element cell

**Constructor**

    CellScalarValues([::Type{T}], quad_rule::QuadratureRule, function_space::FunctionSpace, [geometric_space::FunctionSpace])


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
immutable CellScalarValues{dim, T <: Real, FS <: FunctionSpace, GS <: FunctionSpace, shape <: AbstractRefShape} <: CellValues{dim, T, FS, GS}
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

CellScalarValues{dim, FS <: FunctionSpace, GS <: FunctionSpace}(quad_rule::QuadratureRule{dim}, func_space::FS, geom_space::GS=func_space) = CellScalarValues(Float64, quad_rule, func_space, geom_space)
getnbasefunctions{dim}(cv::CellScalarValues{dim}) = getnbasefunctions(cv.function_space)

function CellScalarValues{dim, T, FS <: FunctionSpace, GS <: FunctionSpace, shape <: AbstractRefShape}(::Type{T}, quad_rule::QuadratureRule{dim, shape}, func_space::FS, geom_space::GS=func_space)
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

    CellScalarValues(N, dNdx, dNdξ, detJdV, quad_rule, func_space, M, dMdξ, geom_space)
end



