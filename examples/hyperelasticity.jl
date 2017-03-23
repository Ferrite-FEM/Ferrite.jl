module HyperElasticity

using JuAFEM
using Tensors
using KrylovMethods
using TimerOutputs
import ProgressMeter

const ∇ = Tensors.gradient

# NeoHook
immutable NeoHook{T}
    μ::T
    λ::T
end

function compute_2nd_PK(mp::NeoHook, E)
    I = one(E)
    C = 2E + one(E)
    invC = inv(C)
    J = sqrt(det(C))
    return mp.μ *(I - invC) + mp.λ * log(J) * invC
end

function constitutive_driver(mp::NeoHook, E)
    ∂S∂E, SPK = ∇(E -> compute_2nd_PK(mp, E), E, :all)
    return SPK, ∂S∂E
end

############
# Assembly #
############

# calculate global residual g and stiffness k
function assemble{dim}(grid::Grid{dim}, dh::DofHandler, K, cv, fv, mp, u)
    # cache some stuff
    n = ndofs_per_cell(dh)
    Ke = zeros(n, n)
    fe = zeros(n)

    f = zeros(ndofs(dh))
    assembler = start_assemble(K, f)

    global_dofs = zeros(Int, n)

    # loop over all cells in the grid
    @timeit "assemble" for cell in CellIterator(dh)
        # reset
        fill!(Ke, 0)
        fill!(fe, 0)

        celldofs!(global_dofs, cell)
        ue = u[global_dofs] # element dofs
        @timeit "inner assemble" assemble_element!(Ke, fe, cell, cv, fv, mp, ue)

        assemble!(assembler, fe, Ke, global_dofs)
    end

    return f, K
end

function assemble_element!(ke, fe, cell, cv, fv, mp, ue, assemble_tangent = true)
    ndofs = getnbasefunctions(cv)
    reinit!(cv, cell)
    fill!(ke, 0.0)
    fill!(fe, 0.0)
    δE = Vector{SymmetricTensor{2, 3, eltype(ue), 6}}(ndofs)

    for qp in 1:getnquadpoints(cv)
        ∇u = function_gradient(cv, qp, ue)
        dΩ = getdetJdV(cv, qp)

        # strain and stress + tangent
        F = one(∇u) + ∇u
        E = symmetric(1/2 * (F' ⋅ F - one(F)))

        S, ∂S∂E = constitutive_driver(mp, E)

        # Hoist computations of δE
        for i in 1:ndofs
            δFi = shape_gradient(cv, qp, i)
            δE[i] = symmetric(1/2*(δFi'⋅F + F'⋅δFi))
        end

        for i in 1:ndofs
            δFi = shape_gradient(cv, qp, i)
            fe[i] += (δE[i] ⊡ S) * dΩ
            δE∂S∂E = δE[i] ⊡ ∂S∂E
            S∇δu = S ⋅ δFi'
            for j in 1:ndofs
                δ∇uj = shape_gradient(cv, qp, j)
                ke[i, j] += (δE∂S∂E ⊡ δE[j] + S∇δu ⊡ δ∇uj' ) * dΩ
            end
        end
    end

end

# solve the problem
function solve()
    reset_timer!()

    const dim = 3

    # Generate a grid
    N = 20
    L = 1.0
    left = zero(Vec{dim})
    right = L * ones(Vec{dim})
    grid = generate_grid(Tetrahedron, ntuple(x->N, dim), left, right)

    # Node sets
    addnodeset!(grid, "clamped", x -> norm(x[1]) ≈ 1)
    addnodeset!(grid, "rotation", x -> norm(x[1]) ≈ 0)

    # Material parameters
    E = 10.0
    ν = 0.3
    μ = E / (2(1 + ν))
    λ = (E * ν) / ((1 + ν) * (1 - 2ν))
    mp = NeoHook(μ, λ)

    # finite element base
    ip = Lagrange{dim, RefTetrahedron, 1}()
    qr = QuadratureRule{dim, RefTetrahedron}(1)
    qr_face = QuadratureRule{dim-1, RefTetrahedron}(1)
    cv = CellVectorValues(qr, ip)
    fv = FaceVectorValues(qr_face, ip)

    # DofHandler
    dh = DofHandler(grid)
    push!(dh, :u, dim) # Add a displacement field
    close!(dh)

    function rotation(X, t, θ = deg2rad(60.0))
        x, y, z = X
        return t * Vec{dim}(
            (0.0,
            L/2 - y + (y-L/2)*cos(θ) - (z-L/2)*sin(θ),
            L/2 - z + (y-L/2)*sin(θ) + (z-L/2)*cos(θ)
            ))
    end

    dbc = DirichletBoundaryConditions(dh)
    # Add a homogenoush boundary condition on the "clamped" edge
    add!(dbc, :u, getnodeset(grid, "clamped"), (x,t) -> [0.0, 0.0, 0.0], collect(1:dim))
    add!(dbc, :u, getnodeset(grid, "rotation"), (x,t) -> rotation(x, t), collect(1:dim))
    close!(dbc)
    t = 0.5
    JuAFEM.update!(dbc, t)

    println("Analysis with ", length(grid.cells), " elements")

    # pre-allocate
    _ndofs = ndofs(dh)
    un = zeros(_ndofs) # previous solution vector
    u  = zeros(_ndofs)
    Δu = zeros(_ndofs)

    apply!(un, dbc)

    K = create_sparsity_pattern(dh)

    newton_itr = -1
    NEWTON_TOL = 1e-8
    prog = ProgressMeter.ProgressThresh(NEWTON_TOL, "Solving:")
   
    while true; newton_itr += 1
        u = un + Δu
        f, K = assemble(grid, dh, K, cv, fv, mp, u)
        normg = norm(f[JuAFEM.free_dofs(dbc)])
        apply_zero!(K, f, dbc)
        ProgressMeter.update!(prog, normg; showvalues = [(:iter, newton_itr)])
     
        if normg < NEWTON_TOL
            break
        end

        if newton_itr > 30
            error("Reached maximum Newton iterations, aborting")
            break
        end

        @timeit "linear solve" ΔΔu, flag, relres, iter, resvec = cg(K, f; maxIter = 1000, tol = min(1e-3, normg))
        @assert flag == 0
        
        apply_zero!(ΔΔu, dbc)
        Δu -= ΔΔu
    end

    # save the solution
    @timeit "export" begin
        vtk = vtk_grid("hyperelasticity", dh, u)
        vtk_save(vtk)
    end

    print_timer()
end

end # module