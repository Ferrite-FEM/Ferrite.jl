using Ferrite, Test

module ConvergenceTestHelper

using Ferrite, SparseArrays, ForwardDiff, Test
import LinearAlgebra: diag

get_geometry(::Ferrite.Interpolation{RefLine}) = Line
get_geometry(::Ferrite.Interpolation{RefQuadrilateral}) = Quadrilateral
get_geometry(::Ferrite.Interpolation{RefTriangle}) = Triangle
get_geometry(::Ferrite.Interpolation{RefPrism}) = Wedge
get_geometry(::Ferrite.Interpolation{RefHexahedron}) = Hexahedron
get_geometry(::Ferrite.Interpolation{RefTetrahedron}) = Tetrahedron

get_quadrature_order(::Lagrange{shape, order}) where {shape, order} = 2*order
get_quadrature_order(::Serendipity{shape, order}) where {shape, order} = 2*order
get_quadrature_order(::CrouzeixRaviart{shape, order}) where {shape, order} = 2*order+1

get_N(::Ferrite.Interpolation{shape, 1}) where {shape} = 19
get_N(::Ferrite.Interpolation{shape, 2}) where {shape} = 12
get_N(::Ferrite.Interpolation{shape, 3}) where {shape} = 8
get_N(::Ferrite.Interpolation{shape, 4}) where {shape} = 5
get_N(::Ferrite.Interpolation{shape, 5}) where {shape} = 3

analytical_solution(x) = cot(norm(x))
analytical_rhs(x) = -sum(diag(ForwardDiff.hessian(analytical_solution,x)))

# Standard assembly copy pasta for Poisson problem
function assemble_element!(Ke::Matrix, fe::Vector, cellvalues::CellValues, coords)
    n_basefuncs = getnbasefunctions(cellvalues)
    ## Reset to 0
    fill!(Ke, 0)
    fill!(fe, 0)
    ## Loop over quadrature points
    for q_point in 1:getnquadpoints(cellvalues)
        ## Get the quadrature weight
        dΩ = getdetJdV(cellvalues, q_point)
        x = spatial_coordinate(cellvalues, q_point, coords)
        ## Loop over test shape functions
        for i in 1:n_basefuncs
            δu  = shape_value(cellvalues, q_point, i)
            ∇δu = shape_gradient(cellvalues, q_point, i)
            ## Add contribution to fe
            fe[i] += analytical_rhs(x) * δu * dΩ
            ## Loop over trial shape functions
            for j in 1:n_basefuncs
                ∇u = shape_gradient(cellvalues, q_point, j)
                ## Add contribution to Ke
                Ke[i, j] += (∇δu ⋅ ∇u) * dΩ
            end
        end
    end
    #@show norm(Ke), norm(fe)
    return Ke, fe
end

# Standard assembly copy pasta for Poisson problem
function assemble_global(cellvalues::CellValues, K::SparseMatrixCSC, dh::DofHandler)
    ## Allocate the element stiffness matrix and element force vector
    n_basefuncs = getnbasefunctions(cellvalues)
    Ke = zeros(n_basefuncs, n_basefuncs)
    fe = zeros(n_basefuncs)
    ## Allocate global force vector f
    f = zeros(ndofs(dh))
    ## Create an assembler
    assembler = start_assemble(K, f)
    ## Loop over all cels
    for cell in CellIterator(dh)
        ## Reinitialize cellvalues for this cell
        reinit!(cellvalues, cell)
        coords = getcoordinates(cell)
        ## Compute element contribution
        assemble_element!(Ke, fe, cellvalues, coords)
        ## Assemble Ke and fe into K and f
        assemble!(assembler, celldofs(cell), Ke, fe)
    end
    return K, f
end

# Check L2 convergence
function check_and_compute_convergence(dh, u, cellvalues, testatol, forest)
    L2norm = 0.0
    L∞norm = 0.0
    marked_cells = Int[]
    for (cellid,cell) in enumerate(CellIterator(dh))
        L2loc = 0.0
        L∞loc = 0.0
        reinit!(cellvalues, cell)
        n_basefuncs = getnbasefunctions(cellvalues)
        coords = getcoordinates(cell)
        uₑ = u[celldofs(cell)]
        for q_point in 1:getnquadpoints(cellvalues)
            dΩ = getdetJdV(cellvalues, q_point)
            x = spatial_coordinate(cellvalues, q_point, coords)
            uₐₙₐ    = prod(cos, x*π/2)
            uₐₚₚᵣₒₓ = function_value(cellvalues, q_point, uₑ)
            L2norm += norm(uₐₙₐ-uₐₚₚᵣₒₓ)*dΩ
            L∞norm = max(L∞norm, norm(uₐₙₐ-uₐₚₚᵣₒₓ))
            L2loc += norm(uₐₙₐ-uₐₚₚᵣₒₓ)*dΩ
            L∞loc = max(L∞norm, norm(uₐₙₐ-uₐₚₚᵣₒₓ))
            #@test isapprox(uₐₙₐ, uₐₚₚᵣₒₓ; atol=testatol)
        end
        if L2loc > 2e-1
            push!(marked_cells,cellid)
        end
    end
    Ferrite.refine!(forest,marked_cells)
    Ferrite.balanceforest!(forest)
    L2norm, L∞norm
end

# Assemble and solve
function solve(dh, ch, cellvalues)
    K, f = assemble_global(cellvalues, create_sparsity_pattern(dh,ch), dh);
    apply!(K, f, ch)
    u = K \ f;
    apply!(u,ch)
end

function setup_poisson_problem(grid, interpolation, interpolation_geo, qr, N)
    # Construct Ferrite stuff
    dh = DofHandler(grid)
    add!(dh, :u, interpolation)
    close!(dh);

    ch = ConstraintHandler(dh);
    ∂Ω = union(
        values(Ferrite.getfacetsets(grid))...
    );
    dbc = Dirichlet(:u, ∂Ω, (x, t) -> analytical_solution(x))
    add!(ch, dbc);
    add!(ch, ConformityConstraint(:u))
    close!(ch);

    cellvalues = CellValues(qr, interpolation, interpolation_geo);

    return dh, ch, cellvalues
end

end # module ConvergenceTestHelper

function compute_fluxes(cellvalues::CellValues{<:ScalarInterpolation}, dh::DofHandler, a::AbstractVector{T}) where T

    n = getnbasefunctions(cellvalues)
    cell_dofs = zeros(Int, n)
    nqp = getnquadpoints(cellvalues)

    # Allocate storage for the fluxes to store
    q = [Vec{2,T}[] for _ in 1:getncells(dh.grid)]

    for (cell_num, cell) in enumerate(CellIterator(dh))
        q_cell = q[cell_num]
        celldofs!(cell_dofs, dh, cell_num)
        aᵉ = a[cell_dofs]
        reinit!(cellvalues, cell)

        for q_point in 1:nqp
            q_qp = - function_gradient(cellvalues, q_point, aᵉ)
            push!(q_cell, q_qp)
        end
    end
    return q
end

@testset "convergence analysis" begin
    @testset "$interpolation" for interpolation in (
        Lagrange{RefQuadrilateral, 1}(),
        #Lagrange{RefHexahedron, 1}(),
    )
        L2norm = Inf
        # Generate a grid ...
        geometry = ConvergenceTestHelper.get_geometry(interpolation)
        interpolation_geo = interpolation
        N = ConvergenceTestHelper.get_N(interpolation)
        grid = generate_grid(geometry, ntuple(x->10, Ferrite.getrefdim(geometry)));
        adaptive_grid = ForestBWG(grid,7)
        # ... a suitable quadrature rule ...
        qr_order = ConvergenceTestHelper.get_quadrature_order(interpolation)
        qr = QuadratureRule{Ferrite.getrefshape(interpolation)}(qr_order)
        # ... and then pray to the gods of convergence.
        i = 0
        while L2norm > 1e-3 && i < 8
            grid_transfered = Ferrite.creategrid(adaptive_grid)
            dh, ch, cellvalues = ConvergenceTestHelper.setup_poisson_problem(grid_transfered, interpolation, interpolation_geo, qr, N)
            u = ConvergenceTestHelper.solve(dh, ch, cellvalues)
            # q_gp = compute_fluxes(cellvalues, dh, u);
            # projector = L2Projector(interpolation, grid_transfered);
            # q_projected = project(projector, q_gp, qr);
            L2norm, _ = ConvergenceTestHelper.check_and_compute_convergence(dh, u, cellvalues, 1e-2, adaptive_grid)
            # vtk_grid("p4est_test$(i).vtu",dh) do vtk
            #     vtk_point_data(vtk,dh,u)
            #     vtk_point_data(vtk, projector, q_projected, "q")
            # end
            i += 1
        end
    end
end
