using Ferrite, SparseArrays, ForwardDiff, Test
import LinearAlgebra: diag

dotest = false # test pointwise convergenve
testatol = 1e-2 # absolute tolerance for 

# Solution parameters
analytical_solution(x) = prod(cos, x*π/2)

# Finite element approximation parameters
element = Wedge
N = 21
ip_order = 1
qe_order = 3

# Get the RHS analytically
analytical_rhs(x) = -sum(diag(ForwardDiff.hessian(analytical_solution,x)))

# Construct Ferrite stuff
ip_geo = Ferrite.default_interpolation(element)
dim = Ferrite.getdim(ip_geo)
grid = generate_grid(element, ntuple(x->N, dim));
refshape = Ferrite.getrefshape(ip_geo)
ip = Lagrange{dim, refshape, ip_order}()
qr = QuadratureRule{dim, refshape}(qe_order)
cellvalues = CellScalarValues(qr, ip, ip_geo);
dh = DofHandler(grid)
add!(dh, :u, 1, ip)
close!(dh);
ch = ConstraintHandler(dh);
∂Ω = union(
    values(getfacesets(grid))...
);
dbc = Dirichlet(:u, ∂Ω, (x, t) -> analytical_solution(x))
add!(ch, dbc);
close!(ch)

# Standard assembly copy pasta for Poisson problem
function assemble_element!(Ke::Matrix, fe::Vector, cellvalues::CellScalarValues, coords)
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
function assemble_global(cellvalues::CellScalarValues, K::SparseMatrixCSC, dh::DofHandler)
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

# Assemble and solve
function solve(dh, ch, cellvalues)
    K, f = assemble_global(cellvalues, create_sparsity_pattern(dh), dh);
    apply!(K, f, ch)
    u = K \ f;
end

# Check L2 convergence
function check_and_compute_convergence(dh, u, cellvalues)
    L2norm = 0.0
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
            L2norm += norm(uₐₙₐ-uₐₚₚᵣₒₓ)*dΩ
            L∞norm = max(L∞norm, norm(uₐₙₐ-uₐₚₚᵣₒₓ))
            dotest && @test isapprox(uₐₙₐ, uₐₚₚᵣₒₓ; atol=testatol)
        end
    end
    L2norm, L∞norm
end

check_and_compute_convergence(dh, solve(dh, ch, cellvalues), cellvalues)
