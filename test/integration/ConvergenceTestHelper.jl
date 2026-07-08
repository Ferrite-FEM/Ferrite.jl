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

get_quadrature_order(::Lagrange{shape, order}) where {shape, order} = max(2 * order - 1, 2)
get_quadrature_order(::Lagrange{RefTriangle, 5}) = 8
get_quadrature_order(::Lagrange{RefPrism, order}) where {order} = 2 * order # Don't know why
get_quadrature_order(::Serendipity{shape, order}) where {shape, order} = max(2 * order - 1, 2)
get_quadrature_order(::CrouzeixRaviart{shape, order}) where {shape, order} = max(2 * order - 1, 2)
get_quadrature_order(::RannacherTurek{shape, order}) where {shape, order} = max(2 * order - 1, 2)
get_quadrature_order(::BubbleEnrichedLagrange{shape, order}) where {shape, order} = max(2 * order - 1, 2)

get_num_elements(::Ferrite.Interpolation{shape, 1}) where {shape} = 21
get_num_elements(::Ferrite.Interpolation{shape, 2}) where {shape} = 7
get_num_elements(::Ferrite.Interpolation{RefHexahedron, 1}) = 11
get_num_elements(::Ferrite.RannacherTurek{RefQuadrilateral, 1}) = 15
get_num_elements(::Ferrite.RannacherTurek{RefHexahedron, 1}) = 13
get_num_elements(::Ferrite.Interpolation{RefHexahedron, 2}) = 4
get_num_elements(::Ferrite.Interpolation{shape, 3}) where {shape} = 8
get_num_elements(::Ferrite.Interpolation{shape, 4}) where {shape} = 5
get_num_elements(::Ferrite.Interpolation{shape, 5}) where {shape} = 3

get_test_tolerance(ip) = 1.0e-2
get_test_tolerance(ip::RannacherTurek) = 4.0e-2
get_test_tolerance(ip::CrouzeixRaviart) = 4.0e-2

analytical_solution(x) = prod(cos, x * π / 2)
analytical_rhs(x) = -Tensors.laplace(analytical_solution, x)

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
            δu = shape_value(cellvalues, q_point, i)
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
            uₐₙₐ = prod(cos, x * π / 2)
            uₐₚₚᵣₒₓ = function_value(cellvalues, q_point, uₑ)
            L∞norm = max(L∞norm, norm(uₐₙₐ - uₐₚₚᵣₒₓ))
            L2norm += norm(uₐₙₐ - uₐₚₚᵣₒₓ)^2 * dΩ

            ∇uₐₙₐ = gradient(x -> prod(cos, x * π / 2), x)
            ∇uₐₚₚᵣₒₓ = function_gradient(cellvalues, q_point, uₑ)
            ∇L2norm += norm(∇uₐₙₐ - ∇uₐₚₚᵣₒₓ)^2 * dΩ

            # Pointwise convergence
            @test uₐₙₐ ≈ uₐₚₚᵣₒₓ atol = testatol
        end
    end
    return √(L2norm), √(∇L2norm), L∞norm
end

# Assemble and solve
function solve(dh, ch, cellvalues)
    K, f = assemble_global(cellvalues, allocate_matrix(dh, ch), dh)
    apply!(K, f, ch)
    u = K \ f
    apply!(u, ch)
    return u
end

function setup_poisson_problem(grid, interpolation, interpolation_geo, qr)
    # Construct Ferrite stuff
    dh = DofHandler(grid)
    add!(dh, :u, interpolation)
    close!(dh)

    ch = ConstraintHandler(dh)
    ∂Ω = union(
        values(Ferrite.getfacetsets(grid))...
    )
    dbc = Dirichlet(:u, ∂Ω, (x, t) -> analytical_solution(x))
    add!(ch, dbc)
    # Hanging nodes on adaptively refined grids must be constrained to
    # their masters for the approximation to stay H¹-conforming.
    grid isa Ferrite.NonConformingGrid && add!(ch, ConformityConstraint(:u))
    close!(ch)

    cellvalues = CellValues(qr, interpolation, interpolation_geo)

    return dh, ch, cellvalues
end

end # module ConvergenceTestHelper
