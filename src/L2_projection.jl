
abstract type AbstractProjector end

struct L2Projector <: AbstractProjector
    func_ip::Interpolation
    geom_ip::Interpolation
    M_cholesky #::SuiteSparse.CHOLMOD.Factor{Float64}
    dh::DofHandler
    set::Vector{Int}
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
        geom_ip::Interpolation = default_interpolation(getcelltype(grid, first(set))),
    )

    # TODO: Maybe this should not be allowed? We always assume to project scalar entries.
    if func_ip isa VectorizedInterpolation
        func_ip = func_ip.ip
    end

    _check_same_celltype(grid, set)

    fe_values_mass = CellValues(qr_lhs, func_ip, geom_ip)

    # Create an internal scalar valued field. This is enough since the projection is done on a component basis, hence a scalar field.
    dh = DofHandler(grid)
    sdh = SubDofHandler(dh, Set(set))
    add!(sdh, :_, func_ip) # we need to create the field, but the interpolation is not used here
    close!(dh)

    M = _assemble_L2_matrix(fe_values_mass, set, dh)  # the "mass" matrix
    M_cholesky = cholesky(Symmetric(M))

    return L2Projector(func_ip, geom_ip, M_cholesky, dh, collect(set))
end

# Quadrature sufficient for integrating a mass matrix
function _mass_qr(::Lagrange{shape, order}) where {shape <: AbstractRefShape, order}
    return QuadratureRule{shape}(order + 1)
end
function _mass_qr(::Lagrange{shape, 2}) where {shape <: RefSimplex}
    return QuadratureRule{shape}(4)
end
_mass_qr(ip::VectorizedInterpolation) = _mass_qr(ip.ip)

function Base.show(io::IO, ::MIME"text/plain", proj::L2Projector)
    println(io, typeof(proj))
    println(io, "  projection on:           ", length(proj.set), "/", getncells(get_grid(proj.dh)), " cells in grid")
    println(io, "  function interpolation:  ", proj.func_ip)
    println(io, "  geometric interpolation: ", proj.geom_ip)
end

function _assemble_L2_matrix(fe_values, set, dh)

    n = Ferrite.getnbasefunctions(fe_values)
    M = Symmetric(create_matrix(create_sparsity_pattern(dh; nnz_per_col = 2 * n)))
    assembler = start_assemble(M)

    Me = zeros(n, n)
    cell_dofs = zeros(Int, n)

    function symmetrize_to_lower!(K::Matrix)
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
        Xe = getcoordinates(get_grid(dh), cellnum)
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
    project(proj::L2Projector, vals, qr_rhs::QuadratureRule)

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

The order of the returned data correspond to the order of the `L2Projector`s internal
`DofHandler`. To export the result, use `vtk_point_data(vtk, proj, projected_data)`.
"""
function project(proj::L2Projector,
                 vars::AbstractVector{<:AbstractVector{T}},
                 qr_rhs::QuadratureRule) where T <: Union{Number, AbstractTensor}

    # For using the deprecated API
    fe_values = CellValues(qr_rhs, proj.func_ip, proj.geom_ip)

    M = T <: AbstractTensor ? length(vars[1][1].data) : 1

    projected_vals = _project(vars, proj, fe_values, M, T)::Vector{T}

    return projected_vals
end
function project(p::L2Projector, vars::AbstractMatrix, qr_rhs::QuadratureRule)
    # TODO: Random access into vars is required for now, hence the collect
    return project(p, collect(eachcol(vars)), qr_rhs)
end

function _project(vars, proj::L2Projector, fe_values::AbstractValues, M::Integer, ::Type{T}) where {T}
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
    for (ic,cellnum) in enumerate(proj.set)
        celldofs!(cell_dofs, proj.dh, cellnum)
        fill!(fe, 0)
        Xe = getcoordinates(get_grid(proj.dh), cellnum)
        cell_vars = vars[ic]
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

function WriteVTK.vtk_point_data(vtk::WriteVTK.DatasetFile, proj::L2Projector, vals::Vector{T}, name::AbstractString) where T
    data = _evaluate_at_grid_nodes(proj, vals, #=vtk=# Val(true))::Matrix
    @assert size(data, 2) == getnnodes(get_grid(proj.dh))
    vtk_point_data(vtk, data, name; component_names=component_names(T))
    return vtk
end

evaluate_at_grid_nodes(proj::L2Projector, vals::AbstractVector) =
    _evaluate_at_grid_nodes(proj, vals, Val(false))

# Numbers can be handled by the method for DofHandler
_evaluate_at_grid_nodes(proj::L2Projector, vals::AbstractVector{<:Number}, vtk) =
    _evaluate_at_grid_nodes(proj.dh, vals, only(getfieldnames(proj.dh)), vtk)

# Deal with projected tensors
function _evaluate_at_grid_nodes(
    proj::L2Projector, vals::AbstractVector{S}, ::Val{vtk}
) where {order, dim, T, M, S <: Union{Tensor{order,dim,T,M}, SymmetricTensor{order,dim,T,M}}, vtk}
    dh = proj.dh
    # The internal dofhandler in the projector is a scalar field, but the values in vals
    # can be any tensor field, however, the number of dofs should always match the length of vals
    @assert ndofs(dh) == length(vals)
    if vtk
        nout = S <: Vec{2} ? 3 : M # Pad 2D Vec to 3D
        data = fill(T(NaN), nout, getnnodes(get_grid(dh)))
    else
        data = fill(NaN * zero(S), getnnodes(get_grid(dh)))
    end
    ip, gip = proj.func_ip, proj.geom_ip
    refdim, refshape = getdim(ip), getrefshape(ip)
    local_node_coords = reference_coordinates(gip)
    qr = QuadratureRule{refshape}(zeros(length(local_node_coords)), local_node_coords)
    cv = CellValues(qr, ip)
    # Function barrier
    return _evaluate_at_grid_nodes!(data, cv, dh, proj.set, vals)
end
function _evaluate_at_grid_nodes!(data, cv, dh, set, u::AbstractVector{S}) where S
    ue = zeros(S, getnbasefunctions(cv))
    for cell in CellIterator(dh, set)
        @assert getnquadpoints(cv) == length(cell.nodes)
        for (i, I) in pairs(cell.dofs)
            ue[i] = u[I]
        end
        for (qp, nodeid) in pairs(cell.nodes)
            # Loop manually over the shape functions since function_value
            # doesn't like scalar base functions with tensor dofs
            val = zero(S)
            for i in 1:getnbasefunctions(cv)
                val += shape_value(cv, qp, i) * ue[i]
            end
            if data isa Matrix # VTK
                dataview = @view data[:, nodeid]
                fill!(dataview, 0) # purge the NaN
                toparaview!(dataview, val)
            else
                data[nodeid] = val
            end
        end
    end
    return data
end
