"""
A `BoundaryScalarValues` object facilitates the process of evaluating values shape functions, gradients of shape functions,
values of nodal functions, gradients and divergences of nodal functions etc. on the finite element boundary

**Constructor**

    BoundaryScalarValues([::Type{T}], quad_rule::QuadratureRule, function_space::FunctionSpace, [geometric_space::FunctionSpace])


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
immutable BoundaryScalarValues{dim, T <: Real, FS <: FunctionSpace, GS <: FunctionSpace, shape <: AbstractRefShape} <: BoundaryValues{dim, T, FS, GS}
    N::Array{T, 3}
    dNdx::Array{Vec{dim, T}, 3}
    dNdξ::Array{Vec{dim, T}, 3}
    detJdV::Matrix{T}
    quad_rule::Vector{QuadratureRule{dim, shape, T}}
    function_space::FS
    M::Array{T, 3}
    dMdξ::Array{Vec{dim, T}, 3}
    geometric_space::GS
    current_boundary::Ref{Int}
end

BoundaryScalarValues{dim_qr, FS <: FunctionSpace, GS <: FunctionSpace}(quad_rule::QuadratureRule{dim_qr}, func_space::FS, geom_space::GS=func_space) =
    BoundaryScalarValues(Float64, quad_rule, func_space, geom_space)

function BoundaryScalarValues{dim_qr, T, FS <: FunctionSpace, GS <: FunctionSpace, shape <: AbstractRefShape}(
                        ::Type{T}, quad_rule::QuadratureRule{dim_qr, shape}, func_space::FS, geom_space::GS=func_space)
    @assert getdim(func_space) == getdim(geom_space)
    @assert getrefshape(func_space) == getrefshape(geom_space) == shape
    n_qpoints = length(getweights(quad_rule))
    dim = dim_qr + 1

    boundary_quad_rule = create_boundary_quad_rule(quad_rule, func_space)
    n_bounds = length(boundary_quad_rule)

    # Function interpolation
    n_func_basefuncs = getnbasefunctions(func_space)
    N =    zeros(T, n_func_basefuncs, n_qpoints, n_bounds)
    dNdx = zeros(Vec{dim, T}, n_func_basefuncs, n_qpoints, n_bounds)
    dNdξ = zeros(Vec{dim, T}, n_func_basefuncs, n_qpoints, n_bounds)

    # Geometry interpolation
    n_geom_basefuncs = getnbasefunctions(geom_space)
    M =    zeros(T, n_geom_basefuncs, n_qpoints, n_bounds)
    dMdξ = zeros(Vec{dim, T}, n_geom_basefuncs, n_qpoints, n_bounds)

    for k in 1:n_bounds, (i, ξ) in enumerate(boundary_quad_rule[k].points)
        value!(func_space, view(N, :, i, k), ξ)
        derivative!(func_space, view(dNdξ, :, i, k), ξ)
        value!(geom_space, view(M, :, i, k), ξ)
        derivative!(geom_space, view(dMdξ, :, i, k), ξ)
    end

    detJdV = zeros(T, n_qpoints, n_bounds)

    BoundaryScalarValues(N, dNdx, dNdξ, detJdV, boundary_quad_rule, func_space, M, dMdξ, geom_space, Ref(0))
end

"""
Updates the `BoundaryScalarValues` object for a given boundary

    reinit!{dim, T}(bv::BoundaryScalarValues{dim}, x::Vector{Vec{dim, T}}, boundary::Int)

** Arguments **

* `bv`: the `BoundaryScalarValues` object
* `x`: A `Vector` of `Vec`, one for each nodal position in the element.
* `boundary`: The boundary number for the element

** Result **

* nothing


**Details**


"""
function reinit!{dim, T}(bv::BoundaryScalarValues{dim}, x::Vector{Vec{dim, T}}, boundary::Int)
    n_geom_basefuncs = getnbasefunctions(getgeometricspace(bv))
    n_func_basefuncs = getnbasefunctions(getfunctionspace(bv))
    @assert length(x) == n_geom_basefuncs

    bv.current_boundary[] = boundary
    cb = getcurrentboundary(bv)

    for i in 1:length(getpoints(bv.quad_rule[cb]))
        w = getweights(bv.quad_rule[cb])[i]
        febv_J = zero(Tensor{2, dim})
        for j in 1:n_geom_basefuncs
            febv_J += bv.dMdξ[j, i, cb] ⊗ x[j]
        end
        Jinv = inv(febv_J)
        for j in 1:n_func_basefuncs
            bv.dNdx[j, i, cb] = Jinv ⋅ bv.dNdξ[j, i, cb]
        end
        detJ = detJ_boundary(febv_J, getgeometricspace(bv), cb)
        detJ <= 0.0 && throw(ArgumentError("detJ is not positive: detJ = $(detJ)"))
        bv.detJdV[i, cb] = detJ * w
    end
end

"""
The current active boundary of the `BoundaryScalarValues` type.

    getcurrentboundary(bv::BoundaryScalarValues)

** Arguments **

* `fe_boundary_values`: the `BoundaryScalarValues` object

** Results **

* `::Int`: the current active boundary (from last `reinit!`).

"""
getcurrentboundary(bv::BoundaryScalarValues) = bv.current_boundary[]
"""
The boundary number for a cell, typically used to get the boundary number which is needed
to `reinit!` a `BoundaryScalarValues` object for  boundary integration

    getboundarynumber(boundary_nodes, cell_nodes, fs::FunctionSpace)

** Arguments **

* `boundary_nodes`: the node numbers of the nodes on the boundary of the cell
* `cell_nodes`: the node numbers of the cell
* `fs`: the `FunctionSpace` for the cell

** Results **

* `::Int`: the corresponding boundary
"""
function getboundarynumber(boundary_nodes::Vector{Int}, cell_nodes::Vector{Int}, fs::FunctionSpace)
    @assert length(boundary_nodes) == getnboundarynodes(fs)
    @assert length(cell_nodes) == getnbasefunctions(fs)

    tmp = zeros(boundary_nodes)
    for i in 1:length(boundary_nodes)
        tmp[i] = findfirst(j -> j == boundary_nodes[i], cell_nodes)
    end

    if 0 in tmp
        throw(ArgumentError("at least one boundary node: $boundary_nodes not in cell nodes: $cell_nodes"))
    end
    sort!(tmp)
    boundary_nodes_sorted = ntuple(i -> tmp[i], Val{getnboundarynodes(fs)})
    for (i, boundary) in enumerate(getboundarylist(fs))
        boundary_nodes_sorted == boundary && return i
    end

    throw(ArgumentError("invalid node numbers for boundary"))
end
