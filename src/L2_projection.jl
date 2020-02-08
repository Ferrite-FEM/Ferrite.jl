
abstract type AbstractProjector end

struct L2Projector <: AbstractProjector
    fe_values::CellValues
    M_cholesky # ::SuiteSparse.CHOLMOD.Factor{Float64}
    dh::DofHandler
    set::Vector{Integer}
end

function L2Projector(fe_values::JuAFEM.Values, interp::Interpolation, grid::JuAFEM.AbstractGrid, set=1:getncells(grid))

    dim, T, shape = typeof(fe_values).parameters

    # Create an internal scalar valued field. This is enough since the projection is done on a component basis, hence a scalar field.
    dh = DofHandler(grid)
    push!(dh, :_, 1, interp)
    close!(dh)

    M = _assemble_L2_matrix(fe_values, set, dh)  # the "mass" matrix
    M_cholesky = cholesky(M)  # TODO maybe have a lazy eval instead of precomputing? / JB
    return L2Projector(fe_values, M_cholesky, dh, set)
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
    project(Vector{Vector{<:Tensor}}}, L2Projector)

Makes a L2 projection of tensor values to the nodes of the grid. This is commonly used for turning values, computed at integration points, to nodal values, so they can be visualized easier. It is also useful for error estimation and recovering more accurate estimations of secondary unknowns.
"""
function project(
    vars::Array{Array{Tensor{order,dim,T,M},1},1},
    proj::L2Projector
    ) where {order,dim,T,M}

    projected_vals = _project(vars, proj, M)
    # convert back to the original tensor type
    return Tensor{order,dim,T}.(eachrow(projected_vals))
end

function project(
    vars::Array{Array{SymmetricTensor{order,dim,T,M},1},1},
    proj::L2Projector
    ) where {order,dim,T,M}

    projected_vals = _project(vars, proj, M)
    # convert back to the original tensor type
    return SymmetricTensor{order,dim,T}.(eachrow(projected_vals))
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
                fe[i, :] += v * qp_vars[1:M] * dΩ
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
