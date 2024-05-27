using Ferrite, Test
import Ferrite: getrefdim, default_interpolation

module ConvergenceTestHelper

using Ferrite, SparseArrays, ForwardDiff, Test
import LinearAlgebra: diag

get_geometry(::Ferrite.Interpolation{RefLine}) = Line
get_geometry(::Ferrite.Interpolation{RefQuadrilateral}) = Quadrilateral
get_geometry(::Ferrite.Interpolation{RefTriangle}) = Triangle
get_geometry(::Ferrite.Interpolation{RefPrism}) = Wedge
get_geometry(::Ferrite.Interpolation{RefHexahedron}) = Hexahedron
get_geometry(::Ferrite.Interpolation{RefTetrahedron}) = Tetrahedron
get_geometry(::Ferrite.Interpolation{RefPyramid}) = Pyramid

get_quadrature_order(::Lagrange{shape, order}) where {shape, order} = 2*order
get_quadrature_order(::Serendipity{shape, order}) where {shape, order} = 2*order
get_quadrature_order(::CrouzeixRaviart{shape, order}) where {shape, order} = 2*order+1
get_quadrature_order(::BubbleEnrichedLagrange{shape, order}) where {shape, order} = 2*order

get_num_elements(::Ferrite.Interpolation{shape, 1}) where {shape} = 21
get_num_elements(::Ferrite.Interpolation{shape, 2}) where {shape} = 7
get_num_elements(::Ferrite.Interpolation{RefHexahedron, 1}) = 11
get_num_elements(::Ferrite.Interpolation{RefHexahedron, 2}) = 4
get_num_elements(::Ferrite.Interpolation{shape, 3}) where {shape} = 8
get_num_elements(::Ferrite.Interpolation{shape, 4}) where {shape} = 5
get_num_elements(::Ferrite.Interpolation{shape, 5}) where {shape} = 3

analytical_solution(x) = prod(cos, x*π/2)
analytical_rhs(x) = -Tensors.laplace(analytical_solution,x)

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

# Compute norms
function check_and_compute_convergence_norms(dh, u, cellvalues, testatol)
    L2norm = 0.0
    ∇L2norm = 0.0
    L∞norm = 0.0
    for cell in CellIterator(dh)
        reinit!(cellvalues, cell)
        n_basefuncs = getnbasefunctions(cellvalues)
        coords = getcoordinates(cell)
        uₑ = u[celldofs(cell)]
        for q_point in 1:getnquadpoints(cellvalues)
            dΩ = getdetJdV(cellvalues, q_point)
            x = spatial_coordinate(cellvalues, q_point, coords)
            uₐₙₐ    = prod(cos, x*π/2)
            uₐₚₚᵣₒₓ = function_value(cellvalues, q_point, uₑ)
            L∞norm = max(L∞norm, norm(uₐₙₐ-uₐₚₚᵣₒₓ))
            L2norm  += norm(uₐₙₐ-uₐₚₚᵣₒₓ)^2*dΩ

            ∇uₐₙₐ    = gradient(x-> prod(cos, x*π/2), x)
            ∇uₐₚₚᵣₒₓ = function_gradient(cellvalues, q_point, uₑ)
            ∇L2norm += norm(∇uₐₙₐ-∇uₐₚₚᵣₒₓ)^2*dΩ

            # Pointwise convergence
            @test uₐₙₐ ≈ uₐₚₚᵣₒₓ atol=testatol
        end
    end
    √(L2norm), √(∇L2norm), L∞norm
end

# Assemble and solve
function solve(dh, ch, cellvalues)
    K, f = assemble_global(cellvalues, create_sparsity_pattern(dh), dh);
    apply!(K, f, ch)
    u = K \ f;
end

function setup_poisson_problem(grid, interpolation, interpolation_geo, qr)
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
    close!(ch);

    cellvalues = CellValues(qr, interpolation, interpolation_geo);

    return dh, ch, cellvalues
end

end # module ConvergenceTestHelper

# These test only for convergence within margins
@testset "convergence analysis" begin
    @testset "$interpolation" for interpolation in (
        Lagrange{RefTriangle, 3}(),
        Lagrange{RefTriangle, 4}(),
        Lagrange{RefTriangle, 5}(),
        Lagrange{RefHexahedron, 1}(),
        Lagrange{RefTetrahedron, 1}(),
        Lagrange{RefPrism, 1}(),
        Lagrange{RefPyramid, 1}(),
        #
        Serendipity{RefQuadrilateral, 2}(),
        Serendipity{RefHexahedron, 2}(),
        #
        BubbleEnrichedLagrange{RefTriangle, 1}(),
        #
        CrouzeixRaviart{RefTriangle, 1}(),
    )
        # Generate a grid ...
        geometry = ConvergenceTestHelper.get_geometry(interpolation)
        interpolation_geo = default_interpolation(geometry)
        N = ConvergenceTestHelper.get_num_elements(interpolation)
        grid = generate_grid(geometry, ntuple(x->N, getrefdim(geometry)));
        # ... a suitable quadrature rule ...
        qr_order = ConvergenceTestHelper.get_quadrature_order(interpolation)
        qr = QuadratureRule{getrefshape(interpolation)}(qr_order)
        # ... and then pray to the gods of convergence.
        dh, ch, cellvalues = ConvergenceTestHelper.setup_poisson_problem(grid, interpolation, interpolation_geo, qr)
        u = ConvergenceTestHelper.solve(dh, ch, cellvalues)
        ConvergenceTestHelper.check_and_compute_convergence_norms(dh, u, cellvalues, 1e-2)
    end
end

# These test also for correct convergence rates
@testset "convergence rate" begin
    @testset "$interpolation" for interpolation in (
        Lagrange{RefLine, 1}(),
        Lagrange{RefLine, 2}(),
        Lagrange{RefQuadrilateral, 1}(),
        Lagrange{RefQuadrilateral, 2}(),
        Lagrange{RefQuadrilateral, 3}(),
        Lagrange{RefTriangle, 1}(),
        Lagrange{RefTriangle, 2}(),
        Lagrange{RefHexahedron, 2}(),
        Lagrange{RefTetrahedron, 2}(),
        Lagrange{RefPrism, 2}(),
    )
        # Generate a grid ...
        geometry = ConvergenceTestHelper.get_geometry(interpolation)
        interpolation_geo = default_interpolation(geometry)
        # "Coarse case"
        N₁ = ConvergenceTestHelper.get_num_elements(interpolation)
        grid = generate_grid(geometry, ntuple(x->N₁, getrefdim(geometry)));
        # ... a suitable quadrature rule ...
        qr_order = ConvergenceTestHelper.get_quadrature_order(interpolation)
        qr = QuadratureRule{getrefshape(interpolation)}(qr_order)
        # ... and then pray to the gods of convergence.
        dh, ch, cellvalues = ConvergenceTestHelper.setup_poisson_problem(grid, interpolation, interpolation_geo, qr)
        u = ConvergenceTestHelper.solve(dh, ch, cellvalues)
        L2₁, H1₁, _ = ConvergenceTestHelper.check_and_compute_convergence_norms(dh, u, cellvalues, 1e-2)

        # "Fine case"
        N₂ = 2*N₁
        grid = generate_grid(geometry, ntuple(x->N₂, getrefdim(geometry)));
        # ... a suitable quadrature rule ...
        qr_order = ConvergenceTestHelper.get_quadrature_order(interpolation)
        qr = QuadratureRule{getrefshape(interpolation)}(qr_order)
        # ... and then pray to the gods of convergence.
        dh, ch, cellvalues = ConvergenceTestHelper.setup_poisson_problem(grid, interpolation, interpolation_geo, qr)
        u = ConvergenceTestHelper.solve(dh, ch, cellvalues)
        L2₂, H1₂, _ = ConvergenceTestHelper.check_and_compute_convergence_norms(dh, u, cellvalues, 5e-3)

        @test -(log(L2₂)-log(L2₁))/(log(N₂)-log(N₁)) ≈ Ferrite.getorder(interpolation)+1 atol=0.1
        @test -(log(H1₂)-log(H1₁))/(log(N₂)-log(N₁)) ≈ Ferrite.getorder(interpolation) atol=0.1
    end
end
