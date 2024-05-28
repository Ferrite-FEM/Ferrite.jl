using Ferrite, FerriteGmsh, SparseArrays
grid = generate_grid(Quadrilateral, (8,8));
function random_deformation_field(x)
    if any(x .≈ -1.0) || any(x .≈ 1.0)
        return x
    else
        Vec{2}(x .+ (rand(2).-0.5)*0.15)
    end
end
transform_coordinates!(grid, random_deformation_field)
grid  = ForestBWG(grid,10)

analytical_solution(x) = atan(2*(norm(x)-0.5)/0.02)
analytical_rhs(x) = -laplace(analytical_solution,x)

function assemble_cell!(ke, fe, cellvalues, ue, coords)
    fill!(ke, 0.0)
    fill!(fe, 0.0)

    n_basefuncs = getnbasefunctions(cellvalues)
    for q_point in 1:getnquadpoints(cellvalues)
        x = spatial_coordinate(cellvalues, q_point, coords)
        dΩ = getdetJdV(cellvalues, q_point)
        for i in 1:n_basefuncs
            Nᵢ = shape_value(cellvalues, q_point, i)
            ∇Nᵢ = shape_gradient(cellvalues, q_point, i)
            fe[i] += analytical_rhs(x) * Nᵢ * dΩ
            for j in 1:n_basefuncs
                ∇Nⱼ = shape_gradient(cellvalues, q_point, j)
                ke[i, j] += ∇Nⱼ ⋅ ∇Nᵢ * dΩ
            end
        end
    end
end

function assemble_global!(K, f, a, dh, cellvalues)
    ## Allocate the element stiffness matrix and element force vector
    n_basefuncs = getnbasefunctions(cellvalues)
    ke = zeros(n_basefuncs, n_basefuncs)
    fe = zeros(n_basefuncs)
    ## Create an assembler
    assembler = start_assemble(K, f)
    ## Loop over all cells
    for cell in CellIterator(dh)
        reinit!(cellvalues, cell)
        @views ue = a[celldofs(cell)]
        ## Compute element contribution
        coords = getcoordinates(cell)
        assemble_cell!(ke, fe, cellvalues, ue, coords)
        ## Assemble ke and fe into K and f
        assemble!(assembler, celldofs(cell), ke, fe)
    end
    return K, f
end

function solve(grid, field_name)
    dim = 2
    order = 1
    ip = Lagrange{RefQuadrilateral, order}()
    qr = QuadratureRule{RefQuadrilateral}(2)
    cellvalues = CellValues(qr, ip);

    dh = DofHandler(grid)
    add!(dh, field_name, ip)
    close!(dh);

    ch = ConstraintHandler(dh)
    add!(ch, ConformityConstraint(field_name))
    add!(ch, Dirichlet(field_name, getfacetset(grid, "top"), (x, t) -> 0.0))
    add!(ch, Dirichlet(field_name, getfacetset(grid, "right"), (x, t) -> 0.0))
    add!(ch, Dirichlet(field_name, getfacetset(grid, "left"), (x, t) -> 0.0))
    add!(ch, Dirichlet(field_name, getfacetset(grid, "bottom"), (x, t) -> 0.0))
    close!(ch);

    K = create_sparsity_pattern(dh,ch)
    f = zeros(ndofs(dh))
    a = zeros(ndofs(dh))
    assemble_global!(K, f, a, dh, cellvalues);
    apply!(K, f, ch)
    u = K \ f;
    apply!(u,ch)
    return u,dh,ch,cellvalues
end

function _internal_cv_field(dh::AbstractDofHandler, field_name::Symbol)
    @assert length(dh.field_names) == 1 "Multiple fields not supported yet for L2ZZErrorEstimator."
    @assert length(dh.subdofhandlers) == 1 "Multiple subdofhandlers not supported yet for L2ZZErrorEstimator."
    ip = dh.subdofhandlers[1].field_interpolations[1]
    order = getorder(ip)
    qr = QuadratureRule{getrefshape(ip)}(max(2order-1,2))
    return CellValues(qr, ip)
end

function _internal_cv_flux(dh::AbstractDofHandler, field_name::Symbol)
    @assert length(dh.field_names) == 1 "Multiple fields not supported yet for L2ZZErrorEstimator."
    @assert length(dh.subdofhandlers) == 1 "Multiple subdofhandlers not supported yet for L2ZZErrorEstimator."
    sdim = getdim(dh.grid)
    ip = getlowerorder(dh.subdofhandlers[1].field_interpolations[1])^sdim
    order = getorder(ip)
    qr = QuadratureRule{getrefshape(ip)}(max(2order-1,2))
    return CellValues(qr, ip)
end

function _zz_compute_fluxes(dh::AbstractDofHandler, u::AbstractVector, field_name::Symbol)
    cellvalues = _internal_cv_flux(dh, field_name)
    ## Superconvergent point
    qr_sc = QuadratureRule{RefQuadrilateral}(1)
    cellvalues_sc = CellValues(qr_sc, cellvalues);
    ## Buffers
    σ_gp = Vector{Vector{Vec{2,Float64}}}()
    σ_gp_loc = Vector{Vec{2,Float64}}()
    σ_gp_sc = Vector{Vector{Vec{2,Float64}}}()
    σ_gp_sc_loc = Vector{Vec{2,Float64}}()
    for (cellid,cell) in enumerate(CellIterator(dh))
        @views ue = u[celldofs(cell)]

        reinit!(cellvalues, cell)
        for q_point in 1:getnquadpoints(cellvalues)
            gradu = function_gradient(cellvalues, q_point, ue)
            push!(σ_gp_loc, gradu)
        end
        push!(σ_gp,copy(σ_gp_loc))
        empty!(σ_gp_loc)

        reinit!(cellvalues_sc, cell)
        for q_point in 1:getnquadpoints(cellvalues_sc)
            gradu = function_gradient(cellvalues_sc, q_point, ue)
            push!(σ_gp_sc_loc, gradu)
        end
        push!(σ_gp_sc,copy(σ_gp_sc_loc))
        empty!(σ_gp_sc_loc)
    end
    return σ_gp, σ_gp_sc
end

"""
    L2ZZErrorEstimator(dh::AbstractGrid)

This strategy computes the error via nodal L2 projection of the fluxes.
"""
mutable struct L2ZZErrorEstimator{DHType <: Ferrite.AbstractDofHandler}
    dh::DHType
    errors::Vector{Float64}
end

function L2ZZErrorEstimator(dh::AbstractGrid)
    L2ZZErrorEstimator(dh, Float64[])
end

function zz_flux_differencing!(errors, dh, field_name, σ1, σ2)
    @assert length(dh.field_names) == 1 "Multiple fields not supported yet for L2ZZErrorEstimator."
    @assert length(dh.subdofhandlers) == 1 "Multiple subdofhandlers not supported yet for L2ZZErrorEstimator."
    ip = dh.subdofhandlers[1].field_interpolations[1]

    sdim = getdim(dh.grid)
    ip = Lagrange{RefQuadrilateral, 1}()^2
    qr_sc = QuadratureRule{RefQuadrilateral}(1)
    cellvalues_flux = _internal_cv_flux()
    dh, field_name,
    _zz_flux_differencing!
end

# Function barrier
function _zz_flux_differencing!(errors, dh, field_name, cellvalues_flux, σ_projected, σ_superconvergent)
    for (cellid,cell) in enumerate(CellIterator(dh))
        reinit!(cv, cell)
        @views σe = σ_projected[celldofs(cell)]
        errors[cellid] = 0.0
        for q_point in 1:getnquadpoints(cellvalues_flux)
            σ_projected_at_sc = function_value(cellvalues_flux, q_point, σe)
            errors[cellid] += norm((σ_superconvergent[cellid][q_point] - σ_projected_at_sc ))
            errors[cellid] *= getdetJdV(cellvalues_flux,q_point)
        end
    end
end

function update_error_estimate!(estimator::L2ZZErrorEstimator, dh::AbstractDofHandler, u::AbstractVector, field_name::Symbol)
    @assert length(dh.field_names) == 1 "Multiple fields not supported yet for L2ZZErrorEstimator."
    @assert length(dh.subdofhandlers) == 1 "Multiple subdofhandlers not supported yet for L2ZZErrorEstimator."
    @assert field_name ∈ dh.field_names "Field $field_name not found in dof handler with following fields: $(dh.field_names)."

    errors = estimator.errors
    grid = get_grid(dh)
    resize!(errors, getncells(grid))
    #Compute fluxes
    σ_qp, σ_gp_sc = _zz_compute_fluxes(dh, u, field_name)

    ip = dh.subdofhandlers[1].field_interpolations[1]
    # Should be exact for all cases. We can query this via dispatch.
    qr = QuadratureRule{getrefshape(ip)}(max(2getorder(ip)-1,2))

    # Project fluxes to nodes
    projector = L2Projector(ip, transfered_grid)
    σ_dof = project(projector, σ_qp, qr)

    zz_flux_differencing!(errors, dh, field_name, σ_gp_sc, σ_dof)

    return nothing
end

get_errors(estimator::L2ZZErrorEstimator) = estimator.errors

abstract type AbstractMarkingStrategy end
struct ThresholdMarking <: AbstractMarkingStrategy
    refinement_threshold::Float64
    markers::Vector{Bool}
end

function solve_adaptive(initial_grid)
    refinement_threshold = 0.001

    ip = Lagrange{RefQuadrilateral, 1}()^2
    qr_sc = QuadratureRule{RefQuadrilateral}(1)
    cellvalues_flux = CellValues(qr_sc, ip);
    finished = false
    i = 1
    grid = deepcopy(initial_grid)
    pvd = VTKFileCollection("heat_amr.pvd",grid);
    while !finished && i<=10
        @show i
        # Solve the problem
        transfered_grid = Ferrite.creategrid(grid)
        u,dh,ch,cv = solve(transfered_grid, :u)

        # Estimate the errors
        estimator = L2ZZErrorEstimator(dh, u, :u)
        errors = get_errors(estimator)
        for (cellid,local_error) in 
            cells_to_refine = Int[]
            if errors > refinement_threshold
                push!(cells_to_refine,cellid)
            end
        end

        addstep!(pvd, i, dh) do vtk
            write_solution(vtk, dh, u)
            # write_projection(vtk, projector, σ_dof, "flux")
            # write_cell_data(vtk, getindex.(collect(Iterators.flatten(σ_gp_sc)),1), "flux sc x")
            # write_cell_data(vtk, getindex.(collect(Iterators.flatten(σ_gp_sc)),2), "flux sc y")
            write_cell_data(vtk, errors, "error")
        end

        Ferrite.refine!(grid, cells_to_refine)
        Ferrite.balanceforest!(grid)

        i += 1
        if isempty(cells_to_refine)
            finished = true
        end
    end
    close(pvd);
end

solve_adaptive(grid)
