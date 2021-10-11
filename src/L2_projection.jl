
abstract type AbstractProjector end

struct L2Projector <: AbstractProjector
    func_ip::Interpolation
    geom_ip::Interpolation
    M_cholesky #::SuiteSparse.CHOLMOD.Factor{Float64}
    dh::MixedDofHandler
    set::Vector{Int}
    node2dof_map::Dict{Int64, Array{Int64,N} where N}
    fe_values::Union{CellValues,Nothing} # only used for deprecated constructor
    qr_rhs::Union{QuadratureRule,Nothing}    # only used for deprecated constructor
end

function L2Projector(fe_values::Ferrite.Values, interp::Interpolation,
    grid::Ferrite.AbstractGrid, set=1:getncells(grid), fe_values_mass::Ferrite.Values=fe_values)

    Base.depwarn("L2Projector(fe_values, interp, grid) is deprecated, " *
                 "use L2Projector(qr, interp, grid) instead.", :L2Projector)

    dim, T, shape = typeof(fe_values).parameters

    # Create an internal scalar valued field. This is enough since the projection is done on a component basis, hence a scalar field.
    dh = MixedDofHandler(grid)
    field = Field(:_, interp, 1)
    fh = FieldHandler([field], Set(set))
    push!(dh, fh)
    _, vertex_dict, _, _ = __close!(dh)

    M = _assemble_L2_matrix(fe_values_mass, set, dh)  # the "mass" matrix
    M_cholesky = cholesky(M)  # TODO maybe have a lazy eval instead of precomputing? / JB
    dummy = Lagrange{1,RefCube,1}()
    return L2Projector(dummy, dummy, M_cholesky, dh, collect(set), vertex_dict[1], fe_values, nothing)
end

function L2Projector(qr::QuadratureRule, func_ip::Interpolation,
    grid::Ferrite.AbstractGrid, set=1:getncells(grid), qr_mass::QuadratureRule=_mass_qr(func_ip),
    geom_ip::Interpolation = default_interpolation(typeof(grid.cells[first(set)])))
    Base.depwarn("L2Projector(qr, func_ip, grid) is deprecated, " *
                 "use L2Projector(func_ip, grid) instead.", :L2Projector)
    return L2Projector(func_ip, grid; qr_lhs=qr_mass, set=set, geom_ip=geom_ip, qr_rhs=qr)
end

"""
    L2Projector(func_ip::Interpolation, grid::AbstractGrid; kwargs...)

Create an `L2Projector` used for projecting quadrature data. `func_ip`
is the function interpolation used for the projection and `grid` the grid
over which the projection is applied.

Keyword arguments:
 - `qr_lhs`: quadrature for the left hand side. Defaults to a quadrature which exactly
   integrates a mass matrix with `func_ip` as the interpolation.
 - `set`: element set over which the projection applies. Defaults to all elements in the grid.
 - `geom_ip`: geometric interpolation. Defaults to the default interpolation for the grid.


The `L2Projector` acts as the integrated left hand side of the projection equation:
Find projection ``u \\in L_2(\\Omega)`` such that
```math
\\int v u \\ \\mathrm{d}\\Omega = \\int v f \\ \\mathrm{d}\\Omega \\quad \\forall v \\in L_2(\\Omega),
```
where ``f`` is the data to project.

Use [`project`](@ref) to integrate the right hand side and solve for the system.
"""
function L2Projector(
        func_ip::Interpolation,
        grid::AbstractGrid;
        qr_lhs::QuadratureRule = _mass_qr(func_ip),
        set = 1:getncells(grid),
        geom_ip::Interpolation = default_interpolation(typeof(grid.cells[first(set)])),
        qr_rhs::Union{QuadratureRule,Nothing}=nothing, # deprecated
    )

    _check_same_celltype(grid, collect(set)) # TODO this does the right thing, but gives the wrong error message if it fails

    fe_values_mass = CellScalarValues(qr_lhs, func_ip, geom_ip)

    # Create an internal scalar valued field. This is enough since the projection is done on a component basis, hence a scalar field.
    dh = MixedDofHandler(grid)
    field = Field(:_, func_ip, 1) # we need to create the field, but the interpolation is not used here
    fh = FieldHandler([field], Set(set))
    push!(dh, fh)
    _, vertex_dict, _, _ = __close!(dh)

    M = _assemble_L2_matrix(fe_values_mass, set, dh)  # the "mass" matrix
    M_cholesky = cholesky(M)

    # For deprecated API
    fe_values = qr_rhs === nothing ? nothing :
                CellScalarValues(qr_rhs, func_ip, geom_ip)

    return L2Projector(func_ip, geom_ip, M_cholesky, dh, collect(set), vertex_dict[1], fe_values, qr_rhs)
end

# Quadrature sufficient for integrating a mass matrix
function _mass_qr(::Lagrange{dim, shape, order}) where {dim, shape, order}
    return QuadratureRule{dim,shape}(order + 1)
end
function _mass_qr(::Lagrange{dim, RefTetrahedron, 2}) where {dim}
    return QuadratureRule{dim,RefTetrahedron}(4)
end

function _assemble_L2_matrix(fe_values, set, dh)

    n = Ferrite.getn_scalarbasefunctions(fe_values)
    M = create_symmetric_sparsity_pattern(dh)
    assembler = start_assemble(M)

    Me = zeros(n, n)
    cell_dofs = zeros(Int, n)

    function symmetrize_to_lower!(K)
       for i in 1:size(K, 1)
           for j in i+1:size(K, 1)
               K[j, i] = K[i, j]
           end
       end
    end

    ## Assemble contributions from each cell
    for cellnum in set
        celldofs!(cell_dofs, dh, cellnum)

        fill!(Me, 0)
        Xe = getcoordinates(dh.grid, cellnum)
        reinit!(fe_values, Xe)

        ## ∭( v ⋅ u )dΩ
        for q_point = 1:getnquadpoints(fe_values)
            dΩ = getdetJdV(fe_values, q_point)
            for j = 1:n
                v = shape_value(fe_values, q_point, j)
                for i = 1:j
                    u = shape_value(fe_values, q_point, i)
                    Me[i, j] += v ⋅ u * dΩ
                end
            end
        end
        symmetrize_to_lower!(Me)
        assemble!(assembler, cell_dofs, Me)
    end
    return M
end

function project(vars::Vector{Vector{T}}, proj::L2Projector;
                 project_to_nodes=true) where T <: Union{Number, AbstractTensor}
    Base.depwarn("project(vars, proj::L2Projector) is deprecated, " *
                 "use project(proj, vars, qr) instead.", :project)
    return project(proj, vars; project_to_nodes=project_to_nodes)
end


"""
    project(proj::L2Projector, vals, qr_rhs::QuadratureRule; project_to_nodes=true)

Makes a L2 projection of data `vals` to the nodes of the grid using the projector `proj`
(see [`L2Projector`](@ref)).

`project` integrates the right hand side, and solves the projection ``u`` from the following projection equation:
Find projection ``u \\in L_2(\\Omega)`` such that
```math
\\int v u \\ \\mathrm{d}\\Omega = \\int v f \\ \\mathrm{d}\\Omega \\quad \\forall v \\in L_2(\\Omega),
```
where ``f`` is the data to project, i.e. `vals`.

The data `vals` should be a vector, with length corresponding to number of elements, of vectors,
with length corresponding to number of quadrature points per element, matching the number of points in `qr_rhs`.
Alternatively, `vals` can be a matrix, with number of columns corresponding to number of elements,
and number of rows corresponding to number of points in `qr_rhs`.
Example (scalar) input data:
```julia
vals = [
    [0.44, 0.98, 0.32], # data for quadrature point 1, 2, 3 of element 1
    [0.29, 0.48, 0.55], # data for quadrature point 1, 2, 3 of element 2
    # ...
]
```
or equivalent in matrix form:
```julia
vals = [
    0.44 0.29 # ...
    0.98 0.48 # ...
    0.32 0.55 # ...
]
```
Supported data types to project are `Number`s and `AbstractTensor`s.

If the parameter `project_to_nodes` is `true`, then the projection returns the values in the order of the mesh nodes
(suitable format for exporting). If `false`, it returns the values corresponding to the degrees of freedom for a scalar
field over the domain, which is useful if one wants to interpolate the projected values.
"""
function project(proj::L2Projector,
                 vars::AbstractVector{<:AbstractVector{T}},
                 qr_rhs::Union{QuadratureRule,Nothing}=nothing;
                 project_to_nodes::Bool=true) where T <: Union{Number, AbstractTensor}

    # For using the deprecated API
    fe_values = qr_rhs === nothing ?
        proj.fe_values :
        CellScalarValues(qr_rhs, proj.func_ip, proj.geom_ip)

    M = T <: AbstractTensor ? length(vars[1][1].data) : 1

    projected_vals = _project(vars, proj, fe_values, M, T)::Vector{T}
    if project_to_nodes
        # NOTE we may have more projected values than verticies in the mesh => not all values are returned
        nnodes = getnnodes(proj.dh.grid)
        reordered_vals = fill(convert(T, NaN * zero(T)), nnodes)
        for node = 1:nnodes
            if (k = get(proj.node2dof_map, node, nothing); k !== nothing)
                @assert length(k) == 1
                reordered_vals[node] = projected_vals[k[1]]
            end
        end
        return reordered_vals
    else
        return projected_vals
    end
end
function project(p::L2Projector, vars::AbstractMatrix, qr_rhs::QuadratureRule; project_to_nodes=true)
    # TODO: Random access into vars is required for now, hence the collect
    return project(p, collect(eachcol(vars)), qr_rhs; project_to_nodes=project_to_nodes)
end

function _project(vars, proj::L2Projector, fe_values::Values, M::Integer, ::Type{T}) where {T}
    # Assemble the multi-column rhs, f = ∭( v ⋅ x̂ )dΩ
    # The number of columns corresponds to the length of the data-tuple in the tensor x̂.

    f = zeros(ndofs(proj.dh), M)
    n = getnbasefunctions(fe_values)
    fe = zeros(n, M)

    cell_dofs = zeros(Int, n)
    nqp = getnquadpoints(fe_values)

    get_data(x::AbstractTensor, i) = x.data[i]
    get_data(x::Number, i) = x

    ## Assemble contributions from each cell
    for cellnum in proj.set
        celldofs!(cell_dofs, proj.dh, cellnum)
        fill!(fe, 0)
        Xe = getcoordinates(proj.dh.grid, cellnum)
        cell_vars = vars[cellnum]
        reinit!(fe_values, Xe)

        for q_point = 1:nqp
            dΩ = getdetJdV(fe_values, q_point)
            qp_vars = cell_vars[q_point]
            for i = 1:n
                v = shape_value(fe_values, q_point, i)
                for j in 1:M
                    fe[i, j] += v * get_data(qp_vars, j) * dΩ
                end
            end
        end

        # Assemble cell contribution
        for (num, dof) in enumerate(cell_dofs)
            f[dof, :] += fe[num, :]
        end
    end

    # solve for the projected nodal values
    projected_vals = proj.M_cholesky \ f

    # Recast to original input type
    make_T(vals) = T <: AbstractTensor ? T(Tuple(vals)) : vals[1]
    return T[make_T(x) for x in eachrow(projected_vals)]
end
