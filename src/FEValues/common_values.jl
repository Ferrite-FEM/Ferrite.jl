# Common methods for all `Values` objects

using Base: @propagate_inbounds

abstract type FieldTrait end
struct VectorValued <: FieldTrait end
struct ScalarValued <: FieldTrait end

FieldTrait(::Type{<:CellScalarValues}) = ScalarValued()
FieldTrait(::Type{<:FaceScalarValues}) = ScalarValued()
FieldTrait(::Type{<:PointScalarValuesInternal}) = ScalarValued()
FieldTrait(::Type{<:PointScalarValues}) = ScalarValued()
FieldTrait(::Type{<:PointVectorValues}) = VectorValued()
FieldTrait(::Type{<:CellVectorValues}) = VectorValued()
FieldTrait(::Type{<:FaceVectorValues}) = VectorValued()

@noinline throw_detJ_not_pos(detJ) = throw(ArgumentError("det(J) is not positive: det(J) = $(detJ)"))

getnbasefunctions(cv::Values) = size(cv.N, 1)
getngeobasefunctions(cv::Values) = size(cv.M, 1)

getn_scalarbasefunctions(cv::T) where T = getn_scalarbasefunctions(FieldTrait(T), cv)
getn_scalarbasefunctions(::ScalarValued, cv::Values) = size(cv.N, 1)
getn_scalarbasefunctions(::VectorValued, cv::Values{dim}) where {dim} = size(cv.N, 1) ÷ dim

function checkquadpoint(cv::Union{CellScalarValues,FaceScalarValues,CellVectorValues,FaceVectorValues,PointScalarValues}, qp::Int)
    0 < qp <= getnquadpoints(cv) || error("quadrature point out of range")
    return nothing
end
checkquadpoint(_, _::Int) = nothing

@noinline function throw_incompatible_dof_length(length_ue, n_base_funcs)
    throw(ArgumentError(
        "the number of base functions ($(n_base_funcs)) does not match the length " *
        "of the vector ($(length_ue)). Perhaps you passed the global vector, " *
        "or forgot to pass a dof_range?"
    ))
end
@noinline function throw_incompatible_coord_length(length_x, n_base_funcs)
    throw(ArgumentError(
        "the number of (geometric) base functions ($(n_base_funcs)) does not match " *
        "the number of coordinates in the vector ($(length_x)). Perhaps you forgot to " *
        "use an appropriate geometric interpolation when creating FE values? See " *
        "https://github.com/Ferrite-FEM/Ferrite.jl/issues/265 for more details."
    ))
end

"""
    reinit!(cv::CellValues, x::Vector)
    reinit!(bv::FaceValues, x::Vector, face::Int)

Update the `CellValues`/`FaceValues` object for a cell or face with coordinates `x`.
The derivatives of the shape functions, and the new integration weights are computed.
"""
reinit!

"""
    getnquadpoints(fe_v::Values)

Return the number of quadrature points for the `Values` object.
"""
getnquadpoints(fe::Values) = length(fe.qr.weights)

"""
    getdetJdV(fe_v::Values, q_point::Int)

Return the product between the determinant of the Jacobian and the quadrature
point weight for the given quadrature point: ``\\det(J(\\mathbf{x})) w_q``

This value is typically used when one integrates a function on a
finite element cell or face as

``\\int\\limits_\\Omega f(\\mathbf{x}) d \\Omega \\approx \\sum\\limits_{q = 1}^{n_q} f(\\mathbf{x}_q) \\det(J(\\mathbf{x})) w_q``
``\\int\\limits_\\Gamma f(\\mathbf{x}) d \\Gamma \\approx \\sum\\limits_{q = 1}^{n_q} f(\\mathbf{x}_q) \\det(J(\\mathbf{x})) w_q``

"""
@propagate_inbounds getdetJdV(cv::CellValues, q_point::Int) = cv.detJdV[q_point]
@propagate_inbounds getdetJdV(bv::FaceValues, q_point::Int) = bv.detJdV[q_point, bv.current_face[]]

"""
    shape_value(fe_v::Values, q_point::Int, base_function::Int)

Return the value of shape function `base_function` evaluated in
quadrature point `q_point`.
"""
@propagate_inbounds shape_value(cv::CellValues, q_point::Int, base_func::Int) = cv.N[base_func, q_point]
@propagate_inbounds shape_value(bv::FaceValues, q_point::Int, base_func::Int) = bv.N[base_func, q_point, bv.current_face[]]

@propagate_inbounds geometric_value(cv::CellValues, q_point::Int, base_func::Int) = cv.M[base_func, q_point]
@propagate_inbounds geometric_value(bv::FaceValues, q_point::Int, base_func::Int) = bv.M[base_func, q_point, bv.current_face[]]

"""
    shape_gradient(fe_v::Values, q_point::Int, base_function::Int)

Return the gradient of shape function `base_function` evaluated in
quadrature point `q_point`.
"""
@propagate_inbounds shape_gradient(cv::CellValues, q_point::Int, base_func::Int) = cv.dNdx[base_func, q_point]
@propagate_inbounds shape_gradient(bv::FaceValues, q_point::Int, base_func::Int) = bv.dNdx[base_func, q_point, bv.current_face[]]

"""
    shape_symmetric_gradient(fe_v::Values, q_point::Int, base_function::Int)

Return the symmetric gradient of shape function `base_function` evaluated in
quadrature point `q_point`.
"""
@propagate_inbounds shape_symmetric_gradient(cv::CellVectorValues, q_point::Int, base_func::Int) = symmetric(shape_gradient(cv, q_point, base_func))
const shape_derivative = shape_gradient

"""
    shape_divergence(fe_v::Values, q_point::Int, base_function::Int)

Return the divergence of shape function `base_function` evaluated in
quadrature point `q_point`.
"""
@propagate_inbounds shape_divergence(cv::CellScalarValues, q_point::Int, base_func::Int) = sum(cv.dNdx[base_func, q_point])
@propagate_inbounds shape_divergence(bv::FaceScalarValues, q_point::Int, base_func::Int) = sum(bv.dNdx[base_func, q_point, bv.current_face[]])
@propagate_inbounds shape_divergence(cv::CellVectorValues, q_point::Int, base_func::Int) = tr(cv.dNdx[base_func, q_point])
@propagate_inbounds shape_divergence(bv::FaceVectorValues, q_point::Int, base_func::Int) = tr(bv.dNdx[base_func, q_point, bv.current_face[]])

shape_curl(cv::T, q_point, base_func) where T = shape_curl(FieldTrait(T), cv, q_point, base_func)

function shape_curl(::VectorValued, cv::Values, q_point::Int, base_func::Int)
    return curl_from_gradient(shape_gradient(cv, q_point, base_func))
end
curl_from_gradient(∇v::SecondOrderTensor{3}) = Vec{3}((∇v[3,2] - ∇v[2,3], ∇v[1,3] - ∇v[3,1], ∇v[2,1] - ∇v[1,2]))

"""
    function_value(fe_v::Values, q_point::Int, u::AbstractVector)

Compute the value of the function in a quadrature point. `u` is a vector with values
for the degrees of freedom. For a scalar valued function, `u` contains scalars.
For a vector valued function, `u` can be a vector of scalars (for use of `VectorValues`)
or `u` can be a vector of `Vec`s (for use with ScalarValues).

The value of a scalar valued function is computed as ``u(\\mathbf{x}) = \\sum\\limits_{i = 1}^n N_i (\\mathbf{x}) u_i``
where ``u_i`` are the value of ``u`` in the nodes. For a vector valued function the value is calculated as
``\\mathbf{u}(\\mathbf{x}) = \\sum\\limits_{i = 1}^n N_i (\\mathbf{x}) \\mathbf{u}_i`` where ``\\mathbf{u}_i`` are the
nodal values of ``\\mathbf{u}``.
"""
function_value(fe_v::T, args...) where T <: Values = function_value(FieldTrait(T), fe_v, args...)

function function_value(::FieldTrait, fe_v::Values{dim}, q_point::Int, u::AbstractVector{T}, dof_range = eachindex(u)) where {dim,T}
    n_base_funcs = getnbasefunctions(fe_v)
    length(dof_range) == n_base_funcs || throw_incompatible_dof_length(length(dof_range), n_base_funcs)
    @boundscheck checkbounds(u, dof_range)
    @boundscheck checkquadpoint(fe_v, q_point)
    val = zero(_valuetype(fe_v, u))
    @inbounds for (i, j) in enumerate(dof_range)
        val += shape_value(fe_v, q_point, i) * u[j]
    end
    return val
end

function function_value(::VectorValued, fe_v::Values{dim}, q_point::Int, u::AbstractVector{Vec{dim,T}}) where {dim,T}
    n_base_funcs = getn_scalarbasefunctions(fe_v)
    length(u) == n_base_funcs || throw_incompatible_dof_length(length(u), n_base_funcs)
    @boundscheck checkquadpoint(fe_v, q_point)
    val = zero(Vec{dim, T})
    basefunc = 1
    @inbounds for i in 1:n_base_funcs
        for j in 1:dim
            val += shape_value(fe_v, q_point, basefunc) * u[i][j]
            basefunc += 1
        end
    end
    return val
end

_valuetype(t::T, v) where T = _valuetype(FieldTrait(T), t, v)
_valuetype(::ScalarValued, ::Values{dim}, ::AbstractVector{T}) where {dim,T} = T
_valuetype(::ScalarValued, ::Values{dim}, ::AbstractVector{Vec{dim,T}}) where {dim,T} = Vec{dim,T}
_valuetype(::VectorValued, ::Values{dim}, ::AbstractVector{T}) where {dim,T} = Vec{dim,T}
# _valuetype(::VectorValues{dim}, ::AbstractVector{Vec{dim,T}}) where {dim,T} = Vec{dim,T}

"""
    function_gradient(fe_v::Values{dim}, q_point::Int, u::AbstractVector)

Compute the gradient of the function in a quadrature point. `u` is a vector with values
for the degrees of freedom. For a scalar valued function, `u` contains scalars.
For a vector valued function, `u` can be a vector of scalars (for use of `VectorValues`)
or `u` can be a vector of `Vec`s (for use with ScalarValues).

The gradient of a scalar function or a vector valued function with use of `VectorValues` is computed as
``\\mathbf{\\nabla} u(\\mathbf{x}) = \\sum\\limits_{i = 1}^n \\mathbf{\\nabla} N_i (\\mathbf{x}) u_i`` or
``\\mathbf{\\nabla} u(\\mathbf{x}) = \\sum\\limits_{i = 1}^n \\mathbf{\\nabla} \\mathbf{N}_i (\\mathbf{x}) u_i`` respectively,
where ``u_i`` are the nodal values of the function.
For a vector valued function with use of `ScalarValues` the gradient is computed as
``\\mathbf{\\nabla} \\mathbf{u}(\\mathbf{x}) = \\sum\\limits_{i = 1}^n \\mathbf{u}_i \\otimes \\mathbf{\\nabla} N_i (\\mathbf{x})``
where ``\\mathbf{u}_i`` are the nodal values of ``\\mathbf{u}``.
"""
function_gradient(fe_v::T, args...) where T <: Values = function_gradient(FieldTrait(T), fe_v, args...)

function function_gradient(::FieldTrait, fe_v::Values{dim}, q_point::Int, u::AbstractVector{T}, dof_range = eachindex(u)) where {dim,T}
    n_base_funcs = getnbasefunctions(fe_v)
    length(dof_range) == n_base_funcs || throw_incompatible_dof_length(length(dof_range), n_base_funcs)
    @boundscheck checkbounds(u, dof_range)
    @boundscheck checkquadpoint(fe_v, q_point)
    grad = zero(_gradienttype(fe_v, u))
    @inbounds for (i, j) in enumerate(dof_range)
        grad += shape_gradient(fe_v, q_point, i) * u[j]
    end
    return grad
end

_gradienttype(values::T, v) where T = _gradienttype(FieldTrait(T), values, v)
 _gradienttype(::ScalarValued, ::Values{dim}, ::AbstractVector{T}) where {dim,T} = Vec{dim,T}
_gradienttype(::VectorValued, ::Values{dim}, ::AbstractVector{T}) where {dim,T} = Tensor{2,dim,T}

function function_gradient(::ScalarValued, fe_v::Values{dim}, q_point::Int, u::AbstractVector{Vec{dim,T}}) where {dim,T}
    n_base_funcs = getn_scalarbasefunctions(fe_v)
    length(u) == n_base_funcs || throw_incompatible_dof_length(length(u), n_base_funcs)
    @boundscheck checkquadpoint(fe_v, q_point)
    grad = zero(Tensor{2,dim,T})
    @inbounds for i in 1:n_base_funcs
        grad += u[i] ⊗ shape_gradient(fe_v, q_point, i)
    end
    return grad
end

function function_gradient(::VectorValued, fe_v::Values{dim}, q_point::Int, u::AbstractVector{Vec{dim,T}}) where {dim,T}
    n_base_funcs = getn_scalarbasefunctions(fe_v)
    length(u) == n_base_funcs || throw_incompatible_dof_length(length(u), n_base_funcs)
    @boundscheck checkquadpoint(fe_v, q_point)
    grad = zero(Tensor{2,dim,T})
    basefunc_count = 1
    @inbounds for i in 1:n_base_funcs
        for j in 1:dim
            grad += u[i][j] * shape_gradient(fe_v, q_point, basefunc_count)
            basefunc_count += 1
        end
    end
    return grad
end


const function_derivative = function_gradient

"""
    function_symmetric_gradient(fe_v::Values, q_point::Int, u::AbstractVector)

Compute the symmetric gradient of the function, see [`function_gradient`](@ref).
Return a `SymmetricTensor`.

The symmetric gradient of a scalar function is computed as
``\\left[ \\mathbf{\\nabla}  \\mathbf{u}(\\mathbf{x_q}) \\right]^\\text{sym} =  \\sum\\limits_{i = 1}^n  \\frac{1}{2} \\left[ \\mathbf{\\nabla} N_i (\\mathbf{x}_q) \\otimes \\mathbf{u}_i + \\mathbf{u}_i  \\otimes  \\mathbf{\\nabla} N_i (\\mathbf{x}_q) \\right]``
where ``\\mathbf{u}_i`` are the nodal values of the function.
"""
function function_symmetric_gradient(fe_v::Values, q_point::Int, u::AbstractVector, dof_range::UnitRange = 1:length(u))
    grad = function_gradient(fe_v, q_point, u, dof_range)
    return symmetric(grad)
end

function function_symmetric_gradient(fe_v::Values, q_point::Int, u::AbstractVector{Vec{dim, T}}) where {dim, T}
    grad = function_gradient(fe_v, q_point, u)
    return symmetric(grad)
end

"""
    function_divergence(fe_v::Values, q_point::Int, u::AbstractVector)

Compute the divergence of the vector valued function in a quadrature point.

The divergence of a vector valued functions in the quadrature point ``\\mathbf{x}_q)`` is computed as
``\\mathbf{\\nabla} \\cdot \\mathbf{u}(\\mathbf{x_q}) = \\sum\\limits_{i = 1}^n \\mathbf{\\nabla} N_i (\\mathbf{x_q}) \\cdot \\mathbf{u}_i``
where ``\\mathbf{u}_i`` are the nodal values of the function.
"""
function_divergence(fe_v::T, args...) where T <: Values = function_divergence(FieldTrait(T), fe_v, args...)

function function_divergence(::ScalarValued, fe_v::Values{dim}, q_point::Int, u::AbstractVector{Vec{dim,T}}) where {dim,T}
    n_base_funcs = getn_scalarbasefunctions(fe_v)
    length(u) == n_base_funcs || throw_incompatible_dof_length(length(u), n_base_funcs)
    @boundscheck checkquadpoint(fe_v, q_point)
    diverg = zero(T)
    @inbounds for i in 1:n_base_funcs
        diverg += shape_gradient(fe_v, q_point, i) ⋅ u[i]
    end
    return diverg
end

function_divergence(::VectorValued, fe_v::Values{dim}, q_point::Int, u::AbstractVector{T}, dof_range = eachindex(u)) where {dim,T} =
    tr(function_gradient(fe_v, q_point, u, dof_range))

function_divergence(::VectorValued, fe_v::Values{dim}, q_point::Int, u::AbstractVector{Vec{dim,T}}) where {dim,T} =
    tr(function_gradient(fe_v, q_point, u))

function_curl(fe_v::Values, q_point::Int, u::AbstractVector, dof_range = eachindex(u)) =
    curl_from_gradient(function_gradient(fe_v, q_point, u, dof_range))

function_curl(fe_v::Values, q_point::Int, u::AbstractVector{Vec{3, T}}) where T =
    curl_from_gradient(function_gradient(fe_v, q_point, u))

"""
    spatial_coordinate(fe_v::Values{dim}, q_point::Int, x::AbstractVector)

Compute the spatial coordinate in a quadrature point. `x` contains the nodal
coordinates of the cell.

The coordinate is computed, using the geometric interpolation, as
``\\mathbf{x} = \\sum\\limits_{i = 1}^n M_i (\\mathbf{x}) \\mathbf{\\hat{x}}_i``
"""
function spatial_coordinate(fe_v::Values{dim}, q_point::Int, x::AbstractVector{Vec{dim,T}}) where {dim,T}
    n_base_funcs = getngeobasefunctions(fe_v)
    length(x) == n_base_funcs || throw_incompatible_coord_length(length(x), n_base_funcs)
    @boundscheck checkquadpoint(fe_v, q_point)
    vec = zero(Vec{dim,T})
    @inbounds for i in 1:n_base_funcs
        vec += geometric_value(fe_v, q_point, i) * x[i]
    end
    return vec
end

function Base.show(io::IO, ::MIME"text/plain", fe_v::Values)
    print(io, "$(typeof(fe_v)) with $(getnbasefunctions(fe_v)) shape functions and $(getnquadpoints(fe_v)) quadrature points")
end

# copy
for ValueType in (CellScalarValues, CellVectorValues, FaceScalarValues, FaceVectorValues)
    args = [:(copy(cv.$fname)) for fname in fieldnames(ValueType)]
    @eval begin
        function Base.copy(cv::$ValueType)
            return typeof(cv)($(args...))
        end
    end
end
