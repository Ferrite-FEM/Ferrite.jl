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
using Ferrite, Tensors
using SparseArrays, LinearAlgebra

# First we generate a simple grid, specifying the 4 corners of Cooks membrane.
function create_cook_grid(nx, ny)
    corners = [Vec{2}(( 0.0,  0.0)),
               Vec{2}((48.0, 44.0)),
               Vec{2}((48.0, 60.0)),
               Vec{2}(( 0.0, 44.0))]
    grid = generate_grid(Triangle, (nx, ny), corners)
    ## facesets for boundary conditions
    addfaceset!(grid, "clamped", x -> norm(x[1]) ≈ 0.0)
    addfaceset!(grid, "traction", x -> norm(x[1]) ≈ 48.0)
    return grid
end;

# Next we define a function to set up our cell- and facevalues.
function create_values(interpolation_u, interpolation_p)
    ## Quadrature rules
    qr      = QuadratureRule{RefTriangle}(3)
    face_qr = FaceQuadratureRule{RefTriangle}(3)

    ## Cell values, for both fields
    cellvalues = CellMultiValues(qr, (u=interpolation_u, p=interpolation_p))    
    
    ## Face values (only for the displacement, u)
    facevalues_u = FaceValues(face_qr, interpolation_u)

    return cellvalues, facevalues_u
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
    add!(dbc, Dirichlet(:u, getfaceset(dh.grid, "clamped"), x -> zero(x), [1, 2]))
    close!(dbc)
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
        cellvalues::CellMultiValues, facevalues_u::FaceValues,
        K::SparseMatrixCSC, grid::Grid, dh::DofHandler, mp::LinearElasticity
        )

    f = zeros(ndofs(dh))
    assembler = start_assemble(K, f)

    n = ndofs_per_cell(dh)
    fe = zeros(n)    # local force vector
    ke = zeros(n, n) # local stiffness matrix

    ## traction vector
    t = Vec{2}((0.0, 1 / 16))
    ## cache ɛdev outside the element routine to avoid some unnecessary allocations
    ɛdev = [zero(SymmetricTensor{2, 2}) for i in 1:getnbasefunctions(cellvalues[:u])]

    ## local dof ranges for each field
    dofrange_u = dof_range(dh, :u)
    dofrange_p = dof_range(dh, :p)

    for cell in CellIterator(dh)
        fill!(ke, 0)
        fill!(fe, 0)
        assemble_up!(ke, fe, cell, cellvalues, facevalues_u, grid, mp, ɛdev, t, dofrange_u, dofrange_p)
        assemble!(assembler, celldofs(cell), fe, ke)
    end

    return K, f
end;

# The element routine integrates the local stiffness and force vector for all elements.
# Since the problem results in a symmetric matrix we choose to only assemble the lower part,
# and then symmetrize it after the loop over the quadrature points.
function assemble_up!(Ke, fe, cell, cellvalues, facevalues_u, grid, mp, ɛdev, t, dofrange_u, dofrange_p)
    reinit!(cellvalues, cell)
    ## We only assemble lower half triangle of the stiffness matrix and then symmetrize it.
    for q_point in 1:getnquadpoints(cellvalues)
        for i in keys(dofrange_u)
            ɛdev[i] = dev(symmetric(shape_gradient(cellvalues[:u], q_point, i)))
        end
        dΩ = getdetJdV(cellvalues, q_point)
        for (iᵤ, Iᵤ) in pairs(dofrange_u)
            for (jᵤ, Jᵤ) in pairs(dofrange_u[1:iᵤ])
                Ke[Iᵤ, Jᵤ] += 2 * mp.G * ɛdev[iᵤ] ⊡ ɛdev[jᵤ] * dΩ
            end
        end

        for (iₚ, Iₚ) in pairs(dofrange_p)
            δp = shape_value(cellvalues[:p], q_point, iₚ)
            for (jᵤ, Jᵤ) in pairs(dofrange_u)
                divδu = shape_divergence(cellvalues[:u], q_point, jᵤ)
                Ke[Iₚ, Jᵤ] += -δp * divδu * dΩ
            end
            for (jₚ, Jₚ) in pairs(dofrange_p[1:iₚ])
                p = shape_value(cellvalues[:p], q_point, jₚ)
                Ke[Iₚ, Jₚ] += - 1 / mp.K * δp * p * dΩ
            end

        end
    end

    symmetrize_lower!(Ke)

    ## We integrate the Neumann boundary using the facevalues.
    ## We loop over all the faces in the cell, then check if the face
    ## is in our `"traction"` faceset.
    for face in 1:nfaces(cell)
        if (cellid(cell), face) ∈ getfaceset(grid, "traction")
            reinit!(facevalues_u, cell, face)
            for q_point in 1:getnquadpoints(facevalues_u)
                dΓ = getdetJdV(facevalues_u, q_point)
                for (iᵤ, Iᵤ) in pairs(dofrange_u)
                    δu = shape_value(facevalues_u, q_point, iᵤ)
                    fe[Iᵤ] += (δu ⋅ t) * dΓ
                end
            end
        end
    end
end

function symmetrize_lower!(Ke)
    for i in 1:size(Ke, 1)
        for j in i+1:size(Ke, 1)
            Ke[i, j] = Ke[j, i]
        end
    end
end;

# To evaluate the stresses after solving the problem we once again loop over the cells in
# the grid. Stresses are evaluated in the quadrature points, however, for
# export/visualization you typically want values in the nodes of the mesh, or as single data
# points per cell. For the former you can project the quadrature point data to a finite
# element space (see the example with the `L2Projector` in [Post processing and
# visualization](@ref howto-postprocessing)). In this example we choose to compute the mean
# value of the stress within each cell, and thus end up with one data point per cell. The
# mean value is computed as
# ```math
# \bar{\boldsymbol{\sigma}}_i = \frac{1}{ |\Omega_i|}
# \int_{\Omega_i} \boldsymbol{\sigma}\, \mathrm{d}\Omega, \quad
# |\Omega_i| = \int_{\Omega_i} 1\, \mathrm{d}\Omega
# ```
# where $\Omega_i$ is the domain occupied by cell number $i$, and $|\Omega_i|$ the volume
# (area) of the cell. The integrals are evaluated using numerical quadrature with the help
# of cellvalues for u and p, just like in the assembly procedure.
#
# Note that even though all strain components in the out-of-plane direction are zero (plane
# strain) the stress components are not. Specifically, $\sigma_{33}$ will be non-zero in
# this formulation. Therefore we expand the strain to a 3D tensor, and then compute the (3D)
# stress tensor.

function compute_stresses(cellvalues::CellMultiValues, dh::DofHandler, mp::LinearElasticity, a::Vector)
    ae = zeros(ndofs_per_cell(dh)) # local solution vector
    u_range = dof_range(dh, :u)    # local range of dofs corresponding to u
    p_range = dof_range(dh, :p)    # local range of dofs corresponding to p
    ## Allocate storage for the stresses
    σ = zeros(SymmetricTensor{2, 3}, getncells(dh.grid))
    ## Loop over the cells and compute the cell-average stress
    for cc in CellIterator(dh)
        ## Update cellvalues
        reinit!(cellvalues, cc)
        ## Extract the cell local part of the solution
        for (i, I) in pairs(cc.dofs)
            ae[i] = a[I]
        end
        ## Loop over the quadrature points
        σΩi = zero(SymmetricTensor{2, 3}) # stress integrated over the cell
        Ωi = 0.0                          # cell volume (area)
        for qp in 1:getnquadpoints(cellvalues)
            dΩ = getdetJdV(cellvalues, qp)
            ## Evaluate the strain and the pressure
            ε = function_symmetric_gradient(cellvalues[:u], qp, ae, u_range)
            p = function_value(cellvalues[:p], qp, ae, p_range)
            ## Expand strain to 3D
            ε3D = SymmetricTensor{2, 3}((i, j) -> i < 3 && j < 3 ? ε[i, j] : 0.0)
            ## Compute the stress in this quadrature point
            σqp  = 2 * mp.G * dev(ε3D) - one(ε3D) * p
            σΩi += σqp * dΩ
            Ωi  += dΩ
        end
        ## Store the value
        σ[cellid(cc)] = σΩi / Ωi
    end
    return σ
end;

# Now we have constructed all the necessary components, we just need a function
# to put it all together.

function solve(ν, interpolation_u, interpolation_p)
    ## material
    Emod = 1.0
    Gmod = Emod / 2(1 + ν)
    Kmod = Emod * ν / ((1 + ν) * (1 - 2ν))
    mp = LinearElasticity(Gmod, Kmod)

    ## Grid, dofhandler, boundary condition
    n = 50
    grid = create_cook_grid(n, n)
    dh = create_dofhandler(grid, interpolation_u, interpolation_p)
    dbc = create_bc(dh)

    ## CellValues
    cellvalues, facevalues_u = create_values(interpolation_u, interpolation_p)

    ## Assembly and solve
    K = create_sparsity_pattern(dh)
    K, f = doassemble(cellvalues, facevalues_u, K, grid, dh, mp)
    apply!(K, f, dbc)
    u = K \ f

    ## Compute the stress
    σ = compute_stresses(cellvalues, dh, mp, u)
    σvM = map(x -> √(3/2 * dev(x) ⊡ dev(x)), σ) # von Mise effective stress

    ## Export the solution and the stress
    filename = "cook_" * (interpolation_u == Lagrange{RefTriangle, 1}()^2 ? "linear" : "quadratic") *
                         "_linear"
    vtk_grid(filename, dh) do vtkfile
        vtk_point_data(vtkfile, dh, u)
        for i in 1:3, j in 1:3
            σij = [x[i, j] for x in σ]
            vtk_cell_data(vtkfile, σij, "sigma_$(i)$(j)")
        end
        vtk_cell_data(vtkfile, σvM, "sigma von Mises")
    end
    return u
end
#md nothing # hide

# We now define the interpolation for displacement and pressure. We use (scalar) Lagrange
# interpolation as a basis for both, and for the displacement, which is a vector, we
# vectorize it to 2 dimensions such that we obtain vector shape functions (and 2nd order
# tensors for the gradients).

linear_p    = Lagrange{RefTriangle,1}()
linear_u    = Lagrange{RefTriangle,1}()^2
quadratic_u = Lagrange{RefTriangle,2}()^2
#md nothing # hide

# All that is left is to solve the problem. We choose a value of Poissons
# ratio that results in incompressibility ($ν = 0.5$) and thus expect the
# linear/linear approximation to return garbage, and the quadratic/linear
# approximation to be stable.

u1 = solve(0.5, linear_u,    linear_p);
u2 = solve(0.5, quadratic_u, linear_p);

## test the result                 #src
using Test                         #src
@test norm(u2) ≈ 919.2121476140703 #src

#md # ## [Plain program](@id incompressible_elasticity-plain-program)
#md #
#md # Here follows a version of the program without any comments.
#md # The file is also available here:
#md # [`incompressible_elasticity.jl`](incompressible_elasticity.jl).
#md #
#md # ```julia
#md # @__CODE__
#md # ```
