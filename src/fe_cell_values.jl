"""
An `FEValues` object facilitates the process of evaluating values shape functions, gradients of shape functions,
values of nodal functions, gradients and divergences of nodal functions etc.

**Constructor**

    FEValues([::Type{T}], quad_rule::QuadratureRule, function_space::FunctionSpace, [geometric_space::FunctionSpace])


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
"""
immutable FEValues{dim, T <: Real, FS <: FunctionSpace, GS <: FunctionSpace}
    N::Vector{Vector{T}}
    dNdx::Vector{Vector{Vec{dim, T}}}
    dNdξ::Vector{Vector{Vec{dim, T}}}
    detJdV::Vector{T}
    quad_rule::QuadratureRule{dim, T}
    function_space::FS
    dMdξ::Vector{Vector{Vec{dim, T}}}
    geometric_space::GS
end

FEValues{dim, FS <: FunctionSpace, GS <: FunctionSpace}(quad_rule::QuadratureRule{dim}, func_space::FS, geom_space::GS=func_space) = FEValues(Float64, quad_rule, func_space, geom_space)

function FEValues{dim, T, FS <: FunctionSpace, GS <: FunctionSpace}(::Type{T}, quad_rule::QuadratureRule{dim}, func_space::FS, geom_space::GS=func_space)
    @assert n_dim(func_space) == n_dim(geom_space)
    @assert ref_shape(func_space) == ref_shape(geom_space)
    n_qpoints = length(weights(quad_rule))

    # Function interpolation
    n_func_basefuncs = n_basefunctions(func_space)
    N = [zeros(T, n_func_basefuncs) for i in 1:n_qpoints]
    dNdx = [[zero(Vec{dim, T}) for i in 1:n_func_basefuncs] for j in 1:n_qpoints]
    dNdξ = [[zero(Vec{dim, T}) for i in 1:n_func_basefuncs] for j in 1:n_qpoints]

    # Geometry interpolation
    n_geom_basefuncs = n_basefunctions(geom_space)
    dMdξ = [[zero(Vec{dim, T}) for i in 1:n_geom_basefuncs] for j in 1:n_qpoints]

    for (i, ξ) in enumerate(quad_rule.points)
        value!(func_space, N[i], ξ)
        derivative!(func_space, dNdξ[i], ξ)
        derivative!(geom_space, dMdξ[i], ξ)
    end

    detJdV = zeros(T, n_qpoints)

    FEValues(N, dNdx, dNdξ, detJdV, quad_rule, func_space, dMdξ, geom_space)
end


"""
Updates the `FEValues` object for an element.

    reinit!{dim, T}(fe_v::FEValues{dim}, x::Vector{Vec{dim, T}})

** Arguments **

* `fe_values`: the `FEValues` object
* `x`: A `Vector` of `Vec`, one for each nodal position in the element.

** Result **

* nothing


**Details**


"""
function reinit!{dim, T}(fe_v::FEValues{dim}, x::Vector{Vec{dim, T}})
    n_geom_basefuncs = n_basefunctions(get_geometricspace(fe_v))
    n_func_basefuncs = n_basefunctions(get_functionspace(fe_v))
    @assert length(x) == n_geom_basefuncs

    for i in 1:length(points(fe_v.quad_rule))
        w = weights(fe_v.quad_rule)[i]
        fev_J = zero(Tensor{2, dim})
        for j in 1:n_geom_basefuncs
            fev_J += fe_v.dMdξ[i][j] ⊗ x[j]
        end
        Jinv = inv(fev_J)
        for j in 1:n_func_basefuncs
            fe_v.dNdx[i][j] = Jinv ⋅ fe_v.dNdξ[i][j]
        end
        detJ = det(fev_J)
        detJ <= 0.0 && throw(ArgumentError("detJ is not positive: detJ = $(detJ)"))
        fe_v.detJdV[i] = detJ * w
    end
end
