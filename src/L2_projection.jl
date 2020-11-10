
abstract type AbstractProjector end

struct L2Projector{CV<:CellValues} <: AbstractProjector
    fe_values::CV
    M_cholesky #::SuiteSparse.CHOLMOD.Factor{Float64}
    dh::MixedDofHandler
    set::Vector{Int}
    node2dof_map::Dict{Int64, Array{Int64,N} where N}
end

function L2Projector(fe_values::JuAFEM.Values, interp::Interpolation,
    grid::JuAFEM.AbstractGrid, set=1:getncells(grid), fe_values_mass::JuAFEM.Values=fe_values)

    dim, T, shape = typeof(fe_values).parameters

    # Create an internal scalar valued field. This is enough since the projection is done on a component basis, hence a scalar field.
    dh = MixedDofHandler(grid)
    field = Field(:_, interp, 1)
    fh = FieldHandler([field], Set(set))
    push!(dh, fh)
    _, vertex_dict, _, _ = __close!(dh)

    M = _assemble_L2_matrix(fe_values_mass, set, dh)  # the "mass" matrix
    M_cholesky = cholesky(M)  # TODO maybe have a lazy eval instead of precomputing? / JB
    return L2Projector(fe_values, M_cholesky, dh, collect(set), vertex_dict[1])
end

function L2Projector(qr::QuadratureRule, func_ip::Interpolation,
    grid::JuAFEM.AbstractGrid, set=1:getncells(grid), qr_mass::QuadratureRule=qr,
    geom_ip::Interpolation = default_interpolation(typeof(grid.cells[first(set)])))

    _check_same_celltype(grid, collect(set)) # TODO this does the right thing, but gives the wrong error message if it fails

    fe_values = CellScalarValues(qr, func_ip, geom_ip)
    fe_values_mass = CellScalarValues(qr_mass, func_ip, geom_ip)

    # Create an internal scalar valued field. This is enough since the projection is done on a component basis, hence a scalar field.
    dh = MixedDofHandler(grid)
    field = Field(:_, func_ip, 1) # we need to create the field, but the interpolation is not used here
    fh = FieldHandler([field], Set(set))
    push!(dh, fh)
    _, vertex_dict, _, _ = __close!(dh)

    M = _assemble_L2_matrix(fe_values_mass, set, dh)  # the "mass" matrix
    M_cholesky = cholesky(M)  # TODO maybe have a lazy eval instead of precomputing? / JB
    return L2Projector(fe_values, M_cholesky, dh, collect(set), vertex_dict[1])
end


function _assemble_L2_matrix(fe_values, set, dh)

    n = JuAFEM.getn_scalarbasefunctions(fe_values)
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


"""
    project(Vector{Vector{<:Tensor}}}, L2Projector, project_to_nodes=true)

Makes a L2 projection of tensor values to the nodes of the grid. This is commonly used for turning values, computed at integration points, to nodal values, so they can be visualized easier. It is also useful for error estimation and recovering more accurate estimations of secondary unknowns.

If the parameter `project_to_nodes` is true, then the projection returns the values in the order of the mesh nodes. If false, it returns the values corresponding to the degrees of freedom for a scalar field over the domain, which is useful if one wants to interpolate the projected values.
"""
function project(
    vars::Array{Array{T,1},1},
    proj::L2Projector;
    project_to_nodes=true) where {T<:AbstractTensor}

    M = length(vars[1][1].data)
    projected_vals = _project(vars, proj, M)
    if project_to_nodes
        # NOTE we may have more projected values than verticies in the mesh => not all values are returned
        nnodes = getnnodes(proj.dh.grid)
        reordered_vals = fill(NaN, nnodes, size(projected_vals, 2))
        for node = 1:nnodes
            if haskey(proj.node2dof_map, node)
                reordered_vals[node, :] = projected_vals[proj.node2dof_map[node], :]
            end
        end
        return [T(Tuple(reordered_vals[i,:])) for i=1:size(reordered_vals, 1)]
    else
        # convert back to the original tensor type
        return [T(Tuple(reordered_vals[i,:])) for i=1:size(reordered_vals, 1)]
    end
end

function _project(vars, proj::L2Projector, M::Integer)
    # Assemble the multi-column rhs, f = ∭( v ⋅ x̂ )dΩ
    # The number of columns corresponds to the length of the data-tuple in the tensor x̂.

    f = zeros(ndofs(proj.dh), M)
    fe_values = proj.fe_values
    n = getnbasefunctions(proj.fe_values)
    fe = zeros(n, M)

    cell_dofs = zeros(Int, n)
    nqp = getnquadpoints(fe_values)

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
                fe[i, :] += v * [qp_vars.data[i] for i=1:M] * dΩ
            end
        end

        # Assemble cell contribution
        for (num, dof) in enumerate(cell_dofs)
            f[dof, :] += fe[num, :]
        end
    end

    # solve for the projected nodal values
    return proj.M_cholesky\f

end
