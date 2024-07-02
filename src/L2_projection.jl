abstract type AbstractProjector end

mutable struct L2Projector <: AbstractProjector
    M_cholesky #::SuiteSparse.CHOLMOD.Factor{Float64}
    dh::DofHandler
    qrs_lhs::Vector{<:QuadratureRule}
    qrs_rhs::Vector{<:QuadratureRule}
end
isclosed(proj::L2Projector) = isclosed(proj.dh)

function Base.show(io::IO, ::MIME"text/plain", proj::L2Projector)
    dh = proj.dh
    print(io, typeof(proj))
    isclosed(proj) || (print(io, " (not closed)"); return nothing)
    println(io)
    ncells = sum(length(sdh.cellset) for sdh in dh.subdofhandlers)
    println(io, "  projection on:           ", ncells, "/", getncells(get_grid(dh)), " cells in grid")
    if length(dh.subdofhandlers) == 1 # Same as before
        sdh = dh.subdofhandlers[1]
        println(io, "  function interpolation:  ", only(sdh.field_interpolations))
        println(io, "  geometric interpolation: ", geometric_interpolation(getcelltype(sdh)))
    else
        println(io, "  Split into ", length(dh.subdofhandlers), " sets")
    end
    return nothing
end

"""
    L2Projector(grid::AbstractGrid)

Initiate an `L2Projector` for projecting quadrature data onto
a function space. To define the function space, add interpolations for
differents cell sets with `add!` before `close!`ing the projector,
see the example below.

The `L2Projector` acts as the integrated left hand side of the projection equation:
Find projection ``u \\in U_h(\\Omega) \\subset L_2(\\Omega)`` such that
```math
\\int v u \\ \\mathrm{d}\\Omega = \\int v f \\ \\mathrm{d}\\Omega \\quad \\forall v \\in U_h(\\Omega),
```
where ``f \\in L_2(\\Omega)`` is the data to project. The function space ``U_h(\\Omega)``
is the finite element approximation given by the interpolations `add!`ed to the `L2Projector`.

### Example
```julia
proj = L2Projector(grid)
qr_quad = QuadratureRule{RefQuadrilateral}(2)
add!(proj, quad_set, Lagrange{RefQuadrilateral, 1}(); qr_rhs = qr_quad)
qr_tria = QuadratureRule{RefTriangle}(1)
add!(proj, tria_set, Lagrange{RefTriangle, 1}(); qr_rhs = qr_tria)
close!(proj)

vals = Dict{Int, Vector{Float64}}() # Can also be Vector{Vector},
                                    # indexed with cellnr
for (set, qr) in ((quad_set, qr_quad), (tria_set, qr_tria))
    nqp = getnquadpoints(qr)
    for cellnr in set
        vals[cellnr] = rand(nqp)
    end
end

projected = project(proj, vals)
```
where `projected` can be used in e.g. `evaluate_at_points` with the [`PointEvalHandler`](@ref),
or with [`evaluate_at_grid_nodes`](@ref).
"""
function L2Projector(grid::AbstractGrid)
    dh = DofHandler(grid)
    return L2Projector(nothing, dh, QuadratureRule[], QuadratureRule[])
end

"""
    L2Projector(ip::Interpolation, grid::AbstractGrid; [qr_lhs], [set])

A quick way to initiate an `L2Projector`, add an interpolation `ip` on the `set` to
it, and then `close!` it so that it can be used to `project`. The optional
keyword argument `set` defaults to all cells in the `grid`, while `qr_lhs` defaults
to a quadrature rule that integrates the mass matrix exactly for the interpolation `ip`.
"""
function L2Projector(
        ip::Interpolation,
        grid::AbstractGrid;
        qr_lhs::QuadratureRule = _mass_qr(ip),
        set = OrderedSet(1:getncells(grid)),
        geom_ip = nothing,
    )
    geom_ip === nothing || @warn("Providing geom_ip is deprecated, the geometric interpolation of the cells with always be used")
    proj = L2Projector(grid)
    add!(proj, set, ip; qr_lhs, qr_rhs = nothing)
    close!(proj)
    return proj
end

"""
    add!(proj::L2Projector, set::AbstractVecOrSet{Int}, ip::Interpolation;
        qr_rhs, [qr_lhs])

Add an interpolation `ip` on the cells in `set` to the `L2Projector` `proj`.

* `qr_rhs` sets the quadrature rule used to later integrate the right-hand-side of the
  projection equation, when calling [`project`](@ref). It should match the quadrature points used when
  creating the quadrature-point variables to project.
* The *optional* `qr_lhs` sets the quadrature rule used to integrate the left-hand-side of the projection equation,
  and defaults to a quadrature rule that integrates the mass-matrix exactly for the given interpolation `ip`.

"""
function add!(proj::L2Projector, set::AbstractVecOrSet{Int}, ip::Interpolation;
        qr_rhs::Union{QuadratureRule, Nothing}, qr_lhs::QuadratureRule = _mass_qr(ip)
        )
    # Validate user input
    if qr_rhs !== nothing
        getrefshape(ip) == getrefshape(qr_rhs) || error("The reference shape of the interpolation and the qr_rhs must be the same")
        push!(proj.qrs_rhs, qr_rhs)
    end
    getrefshape(ip) == getrefshape(qr_lhs) || error("The reference shape of the interpolation and the qr_lhs must be the same")

    sdh = SubDofHandler(proj.dh, set)
    add!(sdh, :_, ip isa VectorizedInterpolation ? ip.ip : ip)
    push!(proj.qrs_lhs, qr_lhs)

    return proj
end

"""
    close!(proj::L2Projector)

Close `proj` which assembles and calculates the left-hand-side of the projection equation, before doing a Cholesky factorization
of the mass-matrix.
"""
function close!(proj::L2Projector)
    close!(proj.dh)
    M = _assemble_L2_matrix(proj.dh, proj.qrs_lhs)
    proj.M_cholesky = cholesky(Symmetric(M))
    return proj
end

# Quadrature sufficient for integrating a mass matrix
function _mass_qr(::Lagrange{shape, order}) where {shape <: AbstractRefShape, order}
    return QuadratureRule{shape}(order + 1)
end
function _mass_qr(::Lagrange{shape, 2}) where {shape <: RefSimplex}
    return QuadratureRule{shape}(4)
end
_mass_qr(ip::VectorizedInterpolation) = _mass_qr(ip.ip)

function _assemble_L2_matrix(dh::DofHandler, qrs_lhs::Vector{<:QuadratureRule})
    M = Symmetric(allocate_matrix(dh))
    assembler = start_assemble(M)
    for (sdh, qr_lhs) in zip(dh.subdofhandlers, qrs_lhs)
        ip_fun = only(sdh.field_interpolations)
        ip_geo = geometric_interpolation(getcelltype(sdh))
        cv = CellValues(qr_lhs, ip_fun, ip_geo; update_gradients = false)
        _assemble_L2_matrix!(assembler, cv, sdh)
    end
    return M
end

function _assemble_L2_matrix!(assembler, cellvalues::CellValues, sdh::SubDofHandler)

    n = getnbasefunctions(cellvalues)
    Me = zeros(n, n)

    function symmetrize_to_lower!(K::Matrix)
       for i in 1:size(K, 1)
           for j in i+1:size(K, 1)
               K[j, i] = K[i, j]
           end
       end
    end

    ## Assemble contributions from each cell
    for cell in CellIterator(sdh)
        fill!(Me, 0)
        reinit!(cellvalues, cell)

        ## ∭( v ⋅ u )dΩ
        for q_point = 1:getnquadpoints(cellvalues)
            dΩ = getdetJdV(cellvalues, q_point)
            for j = 1:n
                v = shape_value(cellvalues, q_point, j)
                for i = 1:j
                    u = shape_value(cellvalues, q_point, i)
                    Me[i, j] += v ⋅ u * dΩ
                end
            end
        end
        symmetrize_to_lower!(Me)
        assemble!(assembler, celldofs(cell), Me)
    end
    return assembler
end

"""
    project(proj::L2Projector, vals, [qr_rhs::QuadratureRule])

Makes a L2 projection of data `vals` to the nodes of the grid using the projector `proj`
(see [`L2Projector`](@ref)).

`project` integrates the right hand side, and solves the projection ``u`` from the following projection equation:
Find projection ``u \\in U_h(\\Omega) \\subset L_2(\\Omega)`` such that
```math
\\int v u \\ \\mathrm{d}\\Omega = \\int v f \\ \\mathrm{d}\\Omega \\quad \\forall v \\in U_h(\\Omega),
```
where ``f \\in L_2(\\Omega)`` is the data to project. The function space ``U_h(\\Omega)``
is the finite element approximation given by the interpolations in `proj`.

The data `vals` should be an `AbstractVector` or `AbstractDict` that is indexed by the cell number.
Each index in `vals` should give an `AbstractVector` with one element for each cell quadrature point.

If `proj` was created by calling `L2Projector(ip, grid, set)`, `qr_rhs` must be given. Otherwise, this
is added for each domain when calling `add!(proj, args...)`.

Alternatively, `vals` can be a matrix, with the column index referring the cell number,
and the row index corresponding to quadrature point number.
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

!!! note
    The order of the returned data correspond to the order of the `L2Projector`'s internal
    `DofHandler`. The data can be further analyzed with [`evaluate_at_points`](@ref) and
    [`evaluate_at_grid_nodes`](@ref). Use [`write_projected`](@ref) to export the result.

"""
function project(proj::L2Projector, vars::Union{AbstractVector, AbstractDict})
    return _project(proj, vars, proj.qrs_rhs)
end
# Old-style providing quadrature rule to project
function project(p::L2Projector, vars::Union{AbstractVector, AbstractDict}, qr_rhs::QuadratureRule)
    length(p.dh.subdofhandlers) == 1 || error("For multiple domains, provide the right-hand-side quadrature rule to the L2Projector")
    return _project(p, vars, [qr_rhs])
end
# Providing matrix data instead of Vector / Dict
function project(p::L2Projector, vars::AbstractMatrix, args...)
    # TODO: Random access into vars is required for now, hence the collect
    return project(p, collect(eachcol(vars)), args...)
end

function _project(proj::L2Projector, vars::Union{AbstractVector{TC}, AbstractDict{Int, TC}}, qrs_rhs::Vector{<:QuadratureRule}) where
        {TC <: AbstractVector{T}} where T <: Union{Number, AbstractTensor}

    # Sanity checks for user input
    isclosed(proj) || error("The L2Projector is not closed")
    length(qrs_rhs) == 0 && error("The right-hand-side quadrature rule must be provided, unless already given to the L2Projector")
    length(qrs_rhs) == length(proj.dh.subdofhandlers) || error("Number of qrs_rhs must match the number of `add!`ed sets")
    for (qr_rhs, sdh) in zip(qrs_rhs, proj.dh.subdofhandlers)
        if getrefshape(qr_rhs) !== getrefshape(getcelltype(sdh))
            error("Reference shape of quadrature rule and cells doesn't match. Please ensure that `qrs_rhs` has the same order as sets are added to the L2Projector")
        end
    end
    # Catch if old input-style giving vars indexed by the set index, instead of the cell id
    if isa(vars, AbstractVector) && length(vars) != getncells(get_grid(proj.dh))
        error("vars is indexed by the cellid, not the index in the set: length(vars) != number of cells")
    end

    M = T <: AbstractTensor ? Tensors.n_components(Tensors.get_base(T)) : 1

    return _project(proj, qrs_rhs, vars, M, T)::Vector{T}
end

function _project(proj::L2Projector, qrs_rhs::Vector{<:QuadratureRule}, vars::Union{AbstractVector, AbstractDict}, M::Integer, ::Type{T}) where T
    f = zeros(ndofs(proj.dh), M)
    for (sdh, qr_rhs) in zip(proj.dh.subdofhandlers, qrs_rhs)
        ip_fun = only(sdh.field_interpolations)
        ip_geo = geometric_interpolation(getcelltype(sdh))
        cv = CellValues(qr_rhs, ip_fun, ip_geo; update_gradients = false)
        assemble_proj_rhs!(f, cv, sdh, vars)
    end

    # solve for the projected nodal values
    projected_vals = proj.M_cholesky \ f

    # Recast to original input type
    make_T(vals) = T <: AbstractTensor ? T(Tuple(vals)) : vals[1]
    return T[make_T(x) for x in Base.eachrow(projected_vals)]
end

function assemble_proj_rhs!(f::Matrix, cellvalues::CellValues, sdh::SubDofHandler, vars::Union{AbstractVector, AbstractDict})
    # Assemble the multi-column rhs, f = ∭( v ⋅ x̂ )dΩ
    # The number of columns corresponds to the length of the data-tuple in the tensor x̂.
    M = size(f, 2)
    n = getnbasefunctions(cellvalues)
    fe = zeros(n, M)
    nqp = getnquadpoints(cellvalues)

    get_data(x::AbstractTensor, i) = x.data[i]
    get_data(x::Number, _) = x

    ## Assemble contributions from each cell
    for cell in CellIterator(sdh)
        fill!(fe, 0)
        cell_vars = vars[cellid(cell)]
        length(cell_vars) == nqp || error("The number of variables per cell doesn't match the number of quadrature points")
        reinit!(cellvalues, cell)

        for q_point = 1:nqp
            dΩ = getdetJdV(cellvalues, q_point)
            qp_vars = cell_vars[q_point]
            for i = 1:n
                v = shape_value(cellvalues, q_point, i)
                for j in 1:M
                    fe[i, j] += v * get_data(qp_vars, j) * dΩ
                end
            end
        end

        # Assemble cell contribution
        for (num, dof) in enumerate(celldofs(cell))
            f[dof, :] += fe[num, :]
        end
    end
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
        data = fill(T(NaN) * zero(S), getnnodes(get_grid(dh)))
    end
    for sdh in dh.subdofhandlers
        ip = only(sdh.field_interpolations)
        gip = geometric_interpolation(getcelltype(sdh))
        RefShape = getrefshape(ip)
        local_node_coords = reference_coordinates(gip)
        qr = QuadratureRule{RefShape}(zeros(length(local_node_coords)), local_node_coords)
        cv = CellValues(qr, ip, gip; update_detJdV = false, update_gradients = false)
        _evaluate_at_grid_nodes!(data, cv, sdh, vals)
    end
    return data
end

function _evaluate_at_grid_nodes!(data, cv, sdh, u::AbstractVector{S}) where S
    ue = zeros(S, getnbasefunctions(cv))
    for cell in CellIterator(sdh)
        reinit!(cv, cell)
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
