"""
An `FEBoundaryValues` object facilitates the process of evaluating values shape functions, gradients of shape functions,
values of nodal functions, gradients and divergences of nodal functions etc. on the finite element boundary

**Constructor**

    FEBoundaryValues([::Type{T}], quad_rule::QuadratureRule, function_space::FunctionSpace, [geometric_space::FunctionSpace])


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
immutable FEBoundaryValues{dim, T <: Real, FS <: FunctionSpace, GS <: FunctionSpace, shape <: AbstractRefShape} <: AbstractFEValues{dim, T, FS, GS}
    N::Vector{Vector{Vector{T}}}
    dNdx::Vector{Vector{Vector{Vec{dim, T}}}}
    dNdξ::Vector{Vector{Vector{Vec{dim, T}}}}
    detJdV::Vector{Vector{T}}
    quad_rule::Vector{QuadratureRule{dim, shape, T}}
    function_space::FS
    M::Vector{Vector{Vector{T}}}
    dMdξ::Vector{Vector{Vector{Vec{dim, T}}}}
    geometric_space::GS
    current_boundary::Ref{Int}
end

FEBoundaryValues{dim_qr, FS <: FunctionSpace, GS <: FunctionSpace}(quad_rule::QuadratureRule{dim_qr}, func_space::FS, geom_space::GS=func_space) = FEBoundaryValues(Float64, quad_rule, func_space, geom_space)

function FEBoundaryValues{dim_qr, T, FS <: FunctionSpace, GS <: FunctionSpace, shape <: AbstractRefShape}(::Type{T}, quad_rule::QuadratureRule{dim_qr, shape}, func_space::FS, geom_space::GS=func_space)
    @assert functionspace_n_dim(func_space) == functionspace_n_dim(geom_space)
    @assert functionspace_ref_shape(func_space) == functionspace_ref_shape(geom_space) == shape
    n_qpoints = length(weights(quad_rule))
    dim = dim_qr + 1

    boundary_quad_rule = create_boundary_quad_rule(quad_rule, func_space)
    n_bounds = length(boundary_quad_rule)

    # Function interpolation
    n_func_basefuncs = n_basefunctions(func_space)
    N =    [[zeros(T, n_func_basefuncs) for i in 1:n_qpoints]                      for k in 1:n_bounds]
    dNdx = [[[zero(Vec{dim, T}) for i in 1:n_func_basefuncs] for j in 1:n_qpoints] for k in 1:n_bounds]
    dNdξ = [[[zero(Vec{dim, T}) for i in 1:n_func_basefuncs] for j in 1:n_qpoints] for k in 1:n_bounds]

    # Geometry interpolation
    n_geom_basefuncs = n_basefunctions(geom_space)
    M =    [[zeros(T, n_geom_basefuncs) for i in 1:n_qpoints]                      for k in 1:n_bounds]
    dMdξ = [[[zero(Vec{dim, T}) for i in 1:n_geom_basefuncs] for j in 1:n_qpoints] for k in 1:n_bounds]

    for k in 1:n_bounds, (i, ξ) in enumerate(boundary_quad_rule[k].points)
        value!(func_space, N[k][i], ξ)
        derivative!(func_space, dNdξ[k][i], ξ)
        value!(geom_space, M[k][i], ξ)
        derivative!(geom_space, dMdξ[k][i], ξ)
    end

    detJdV = [zeros(T, n_qpoints) for i in 1:n_bounds]

    FEBoundaryValues(N, dNdx, dNdξ, detJdV, boundary_quad_rule, func_space, M, dMdξ, geom_space, Ref(0))
end

"""
Updates the `FEBoundaryValues` object for a given boundary

    reinit!{dim, T}(fe_bv::FEBoundaryValues{dim}, x::Vector{Vec{dim, T}}, boundary::Int)

** Arguments **

* `fe_bv`: the `FEBoundaryValues` object
* `x`: A `Vector` of `Vec`, one for each nodal position in the element.
* `boundary`: The boundary number for the element

** Result **

* nothing


**Details**


"""
function reinit!{dim, T}(fe_bv::FEBoundaryValues{dim}, x::Vector{Vec{dim, T}}, boundary::Int)
    n_geom_basefuncs = n_basefunctions(get_geometricspace(fe_bv))
    n_func_basefuncs = n_basefunctions(get_functionspace(fe_bv))
    @assert length(x) == n_geom_basefuncs

    fe_bv.current_boundary[] = boundary
    cb = current_boundary(fe_bv)

    for i in 1:length(points(fe_bv.quad_rule[cb]))
        w = weights(fe_bv.quad_rule[cb])[i]
        febv_J = zero(Tensor{2, dim})
        for j in 1:n_geom_basefuncs
            febv_J += fe_bv.dMdξ[cb][i][j] ⊗ x[j]
        end
        Jinv = inv(febv_J)
        for j in 1:n_func_basefuncs
            fe_bv.dNdx[cb][i][j] = Jinv ⋅ fe_bv.dNdξ[cb][i][j]
        end
        detJ = detJ_boundary(febv_J, get_geometricspace(fe_bv), cb)
        detJ <= 0.0 && throw(ArgumentError("detJ is not positive: detJ = $(detJ)"))
        fe_bv.detJdV[cb][i] = detJ * w
    end
end

"""
The current active boundary of the `FEBoundaryValues` type.

    current_boundary(fe_bv::FEBoundaryValues)

** Arguments **

* `fe_boundary_values`: the `FEBoundaryValues` object

** Results **

* `::Int`: the current active boundary (from last `reinit!`).

"""
current_boundary(fe_bv::FEBoundaryValues) = fe_bv.current_boundary[]
"""
The boundary number for a cell, typically used to get the boundary number which is needed
to `reinit!` a `FEBoundaryValues` object for  boundary integration

    get_boundarynumber(boundary_nodes, cell_nodes, fs::FunctionSpace)

** Arguments **

* `boundary_nodes`: the node numbers of the nodes on the boundary of the cell
* `cell_nodes`: the node numbers of the cell
* `fs`: the `FunctionSpace` for the cell

** Results **

* `::Int`: the corresponding boundary
"""
function get_boundarynumber(boundary_nodes::Vector{Int}, cell_nodes::Vector{Int}, fs::FunctionSpace)
    @assert length(boundary_nodes) == n_boundarynodes(fs)
    @assert length(cell_nodes) == n_basefunctions(fs)

    tmp = zeros(boundary_nodes)
    for i in 1:length(boundary_nodes)
        tmp[i] = findfirst(j -> j == boundary_nodes[i], cell_nodes)
    end

    if 0 in tmp
        throw(ArgumentError("at least one boundary node: $boundary_nodes not in cell nodes: $cell_nodes"))
    end
    sort!(tmp)
    boundary_nodes_sorted = ntuple(i -> tmp[i], Val{n_boundarynodes(fs)})
    for (i, boundary) in enumerate(boundarylist(fs))
        boundary_nodes_sorted == boundary && return i
    end

    throw(ArgumentError("invalid node numbers for boundary"))
end
