# # [Incompressible elasticity](@id tutorial-incompressible-elasticity)
#
#-
#md # !!! tip
#md #     This example is also available as a Jupyter notebook:
#md #     [`incompressible_elasticity.ipynb`](@__NBVIEWER_ROOT_URL__/examples/incompressible_elasticity.ipynb).
#-
#
# ## Introduction
#
# Mixed elements can be used to overcome locking when the material becomes
# incompressible. However, for an element to be stable, it needs to fulfill
# the LBB condition.
# In this example we will consider two different element formulations
# - linear displacement with linear pressure approximation (does *not* fulfill LBB)
# - quadratic displacement with linear pressure approximation (does fulfill LBB)
# The quadratic/linear element is also known as the Taylor-Hood element.
# We will consider Cook's Membrane with an applied traction on the right hand side.
#-
# ## Commented program
#
# What follows is a program spliced with comments.
#md # The full program, without comments, can be found in the next
#md # [section](@ref incompressible_elasticity-plain-program).
using Ferrite
using BlockArrays, SparseArrays, LinearAlgebra

# First we generate a simple grid, specifying the 4 corners of Cooks membrane.
function create_cook_grid(nx, ny)
    corners = [Vec{2}((0.0,   0.0)),
               Vec{2}((48.0, 44.0)),
               Vec{2}((48.0, 60.0)),
               Vec{2}((0.0,  44.0))]
    grid = generate_grid(Triangle, (nx, ny), corners);
    ## facesets for boundary conditions
    addfaceset!(grid, "clamped", x -> norm(x[1]) ≈ 0.0);
    addfaceset!(grid, "traction", x -> norm(x[1]) ≈ 48.0);
    return grid
end;


# We create a DofHandler, with two fields, `:u` and `:p`,
# with possibly different interpolations
function create_dofhandler(grid, ipu, ipp)
    dh = DofHandler(grid)
    add!(dh, :u, ipu) # displacement
    add!(dh, :p, ipp) # pressure
    close!(dh)
    return dh
end;

# We also need to add Dirichlet boundary conditions on the `"clamped"` faceset.
# We specify a homogeneous Dirichlet bc on the displacement field, `:u`.
function create_bc(dh)
    dbc = ConstraintHandler(dh)
    add!(dbc, Dirichlet(:u, getfaceset(dh.grid, "clamped"), (x,t) -> zero(Vec{2}), [1,2]))
    close!(dbc)
    t = 0.0
    update!(dbc, t)
    return dbc
end;

# The material is linear elastic, which is here specified by the shear and bulk moduli
struct LinearElasticity{T}
    G::T
    K::T
end

# Now to the assembling of the stiffness matrix. This mixed formulation leads to a blocked
# element matrix. Since Ferrite does not force us to use any particular matrix type we will
# use a `PseudoBlockArray` from `BlockArrays.jl`.
function doassemble(
    cellvalues::MultiCellValues,
    facevalues_u::FaceValues{<:VectorInterpolation},
    K::SparseMatrixCSC, grid::Grid, dh::DofHandler, mp::LinearElasticity
)
    f = zeros(ndofs(dh))
    assembler = start_assemble(K, f)
    nu = getnbasefunctions(cellvalues[:u])
    np = getnbasefunctions(cellvalues[:p])

    fe = PseudoBlockArray(zeros(nu + np), [nu, np]) # local force vector
    ke = PseudoBlockArray(zeros(nu + np, nu + np), [nu, np], [nu, np]) # local stiffness matrix

    ## traction vector
    t = Vec{2}((0.0, 1/16))
    ## cache ɛdev outside the element routine to avoid some unnecessary allocations
    ɛdev = [zero(SymmetricTensor{2, 2}) for i in 1:getnbasefunctions(cellvalues[:u])]

    for cell in CellIterator(dh)
        fill!(ke, 0)
        fill!(fe, 0)
        assemble_up!(ke, fe, cell, cellvalues, facevalues_u, grid, mp, ɛdev, t)
        assemble!(assembler, celldofs(cell), fe, ke)
    end

    return K, f
end;

# The element routine integrates the local stiffness and force vector for all elements.
# Since the problem results in a symmetric matrix we choose to only assemble the lower part,
# and then symmetrize it after the loop over the quadrature points.
function assemble_up!(Ke, fe, cell, cellvalues, facevalues_u, grid, mp, ɛdev, t)

    n_basefuncs_u = getnbasefunctions(cellvalues[:u])
    n_basefuncs_p = getnbasefunctions(cellvalues[:p])
    u▄, p▄ = 1, 2
    reinit!(cellvalues, cell)

    ## We only assemble lower half triangle of the stiffness matrix and then symmetrize it.
    for q_point in 1:getnquadpoints(cellvalues)
        for i in 1:n_basefuncs_u
            ɛdev[i] = dev(symmetric(shape_gradient(cellvalues[:u], q_point, i)))
        end
        dΩ = getdetJdV(cellvalues, q_point)
        for i in 1:n_basefuncs_u
            divδu = shape_divergence(cellvalues[:u], q_point, i)
            δu = shape_value(cellvalues[:u], q_point, i)
            for j in 1:i
                Ke[BlockIndex((u▄, u▄), (i, j))] += 2 * mp.G * ɛdev[i] ⊡ ɛdev[j] * dΩ
            end
        end

        for i in 1:n_basefuncs_p
            δp = shape_value(cellvalues[:p], q_point, i)
            for j in 1:n_basefuncs_u
                divδu = shape_divergence(cellvalues[:u], q_point, j)
                Ke[BlockIndex((p▄, u▄), (i, j))] += -δp * divδu * dΩ
            end
            for j in 1:i
                p = shape_value(cellvalues[:p], q_point, j)
                Ke[BlockIndex((p▄, p▄), (i, j))] += - 1/mp.K * δp * p * dΩ
            end

        end
    end

    symmetrize_lower!(Ke)

    ## We integrate the Neumann boundary using the facevalues.
    ## We loop over all the faces in the cell, then check if the face
    ## is in our `"traction"` faceset.
    for face in 1:nfaces(cell)
        if onboundary(cell, face) && (cellid(cell), face) ∈ getfaceset(grid, "traction")
            reinit!(facevalues_u, cell, face)
            for q_point in 1:getnquadpoints(facevalues_u)
                dΓ = getdetJdV(facevalues_u, q_point)
                for i in 1:n_basefuncs_u
                    δu = shape_value(facevalues_u, q_point, i)
                    fe[i] += (δu ⋅ t) * dΓ
                end
            end
        end
    end
end

function symmetrize_lower!(K)
    for i in 1:size(K,1)
        for j in i+1:size(K,1)
            K[i,j] = K[j,i]
        end
    end
end;

# Now we have constructed all the necessary components, we just need a function
# to put it all together.

function solve(ν, interpolation_u, interpolation_p)
    ## material
    Emod = 1.
    Gmod = Emod / 2(1 + ν)
    Kmod = Emod * ν / ((1+ν) * (1-2ν))
    mp = LinearElasticity(Gmod, Kmod)

    ## grid, dofhandler, boundary condition
    n = 50
    grid = create_cook_grid(n, n)
    dh = create_dofhandler(grid, interpolation_u, interpolation_p)
    dbc = create_bc(dh)

    ## facevalues (for Neumann boundary conditions)
    interpolation_geom = Lagrange{RefTriangle,1}()^2
    facevalues_u = FaceValues(FaceQuadratureRule{RefTriangle}(3), interpolation_u, interpolation_geom)
    
    ## cellvalues
    cellvalues = MultiCellValues(dh; qr=3)

    ## assembly and solve
    K = create_sparsity_pattern(dh);
    K, f = doassemble(cellvalues, facevalues_u, K, grid, dh, mp);
    apply!(K, f, dbc)
    u = Symmetric(K) \ f;

    ## export
    filename = "cook_" * (isa(interpolation_u, Lagrange{RefTriangle,1}) ? "linear" : "quadratic") *
                         "_linear"
    vtk_grid(filename, dh) do vtkfile
        vtk_point_data(vtkfile, dh, u)
    end
    return u
end

# We now define the interpolation for displacement and pressure. We use (scalar) Lagrange
# interpolation as a basis for both, and for the displacement, which is a vector, we
# vectorize it to 2 dimensions such that we obtain vector shape functions (and 2nd order
# tensors for the gradients).

linear_p    = Lagrange{RefTriangle,1}()
linear_u    = Lagrange{RefTriangle,1}()^2
quadratic_u = Lagrange{RefTriangle,2}()^2

# All that is left is to solve the problem. We choose a value of Poissons
# ratio that is near incompressibility -- $ν = 0.5$ -- and thus expect the
# linear/linear approximation to return garbage, and the quadratic/linear
# approximation to be stable.

u1 = solve(0.4999999, linear_u,    linear_p)
u2 = solve(0.4999999, quadratic_u, linear_p);

## test the result                 #src
using Test                         #src
@test norm(u2) ≈ 919.2122668839389 #src

#md # ## [Plain program](@id incompressible_elasticity-plain-program)
#md #
#md # Here follows a version of the program without any comments.
#md # The file is also available here:
#md # [`incompressible_elasticity.jl`](incompressible_elasticity.jl).
#md #
#md # ```julia
#md # @__CODE__
#md # ```
