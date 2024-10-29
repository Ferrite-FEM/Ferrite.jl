if isdefined(Main, :is_ci) #hide
    IS_CI = Main.is_ci     #hide
else                       #hide
    IS_CI = false          #hide
end                        #hide
nothing                    #hide

using Ferrite, FerriteGmsh
using BlockArrays, SparseArrays, LinearAlgebra, WriteVTK

struct GrayScottMaterial{T}
    D₁::T
    D₂::T
    F::T
    k::T
end;

function assemble_element_mass!(Me::Matrix, cellvalues::CellValues)
    n_basefuncs = getnbasefunctions(cellvalues)
    # The mass matrices between the reactions are not coupled, so we get a blocked-strided matrix.
    num_reactants = 2
    r₁range = 1:num_reactants:(num_reactants * n_basefuncs)
    r₂range = 2:num_reactants:(num_reactants * n_basefuncs)
    Me₁ = @view Me[r₁range, r₁range]
    Me₂ = @view Me[r₂range, r₂range]
    # Reset to 0
    fill!(Me, 0)
    # Loop over quadrature points
    for q_point in 1:getnquadpoints(cellvalues)
        # Get the quadrature weight
        dΩ = getdetJdV(cellvalues, q_point)
        # Loop over test shape functions
        for i in 1:n_basefuncs
            δuᵢ = shape_value(cellvalues, q_point, i)
            # Loop over trial shape functions
            for j in 1:n_basefuncs
                δuⱼ = shape_value(cellvalues, q_point, j)
                # Add contribution to Ke
                Me₁[i, j] += (δuᵢ * δuⱼ) * dΩ
                Me₂[i, j] += (δuᵢ * δuⱼ) * dΩ
            end
        end
    end
    return nothing
end

function assemble_element_diffusion!(De::Matrix, cellvalues::CellValues, material::GrayScottMaterial)
    n_basefuncs = getnbasefunctions(cellvalues)
    D₁ = material.D₁
    D₂ = material.D₂
    # The diffusion between the reactions is not coupled, so we get a blocked-strided matrix.
    num_reactants = 2
    r₁range = 1:num_reactants:(num_reactants * n_basefuncs)
    r₂range = 2:num_reactants:(num_reactants * n_basefuncs)
    De₁ = @view De[r₁range, r₁range]
    De₂ = @view De[r₂range, r₂range]
    # Reset to 0
    fill!(De, 0)
    # Loop over quadrature points
    for q_point in 1:getnquadpoints(cellvalues)
        # Get the quadrature weight
        dΩ = getdetJdV(cellvalues, q_point)
        # Loop over test shape functions
        for i in 1:n_basefuncs
            ∇δuᵢ = shape_gradient(cellvalues, q_point, i)
            # Loop over trial shape functions
            for j in 1:n_basefuncs
                ∇δuⱼ = shape_gradient(cellvalues, q_point, j)
                # Add contribution to Ke
                De₁[i, j] += D₁ * (∇δuᵢ ⋅ ∇δuⱼ) * dΩ
                De₂[i, j] += D₂ * (∇δuᵢ ⋅ ∇δuⱼ) * dΩ
            end
        end
    end
    return nothing
end

function assemble_matrices!(M::SparseMatrixCSC, D::SparseMatrixCSC, cellvalues::CellValues, dh::DofHandler, material::GrayScottMaterial)
    n_basefuncs = getnbasefunctions(cellvalues)

    # Allocate the element stiffness matrix and element force vector
    Me = zeros(2 * n_basefuncs, 2 * n_basefuncs)
    De = zeros(2 * n_basefuncs, 2 * n_basefuncs)

    # Create an assembler
    M_assembler = start_assemble(M)
    D_assembler = start_assemble(D)
    # Loop over all cels
    for cell in CellIterator(dh)
        # Reinitialize cellvalues for this cell
        reinit!(cellvalues, cell)
        # Compute element contribution
        assemble_element_mass!(Me, cellvalues)
        assemble!(M_assembler, celldofs(cell), Me)

        assemble_element_diffusion!(De, cellvalues, material)
        assemble!(D_assembler, celldofs(cell), De)
    end
    return nothing
end;

function setup_initial_conditions!(u₀::Vector, cellvalues::CellValues, dh::DofHandler)
    u₀ .= ones(ndofs(dh))
    u₀[2:2:end] .= 0.0

    n_basefuncs = getnbasefunctions(cellvalues)

    for cell in CellIterator(dh)
        reinit!(cellvalues, cell)

        coords = getcoordinates(cell)
        dofs = celldofs(cell)
        uₑ = @view u₀[dofs]
        rv₀ₑ = reshape(uₑ, (2, n_basefuncs))

        for i in 1:n_basefuncs
            if coords[i][3] > 0.9
                rv₀ₑ[1, i] = 0.5
                rv₀ₑ[2, i] = 0.25
            end
        end
    end

    return u₀ .+= 0.01 * rand(ndofs(dh))
end;

function create_embedded_sphere(refinements)
    gmsh.initialize()

    # Add a unit sphere in 3D space
    gmsh.model.occ.addSphere(0.0, 0.0, 0.0, 1.0)
    gmsh.model.occ.synchronize()

    # Generate nodes and surface elements only, hence we need to pass 2 into generate
    gmsh.model.mesh.generate(2)

    # To get good solution quality refine the elements several times
    for _ in 1:refinements
        gmsh.model.mesh.refine()
    end

    # Now we create a Ferrite grid out of it. Note that we also call toelements
    # with our surface element dimension to obtain these.
    nodes = tonodes()
    elements, _ = toelements(2)
    gmsh.finalize()
    return grid = Grid(elements, nodes)
end

function gray_scott_on_sphere(material::GrayScottMaterial, Δt::Real, T::Real, refinements::Integer)
    # We start by setting up grid, dof handler and the matrices for the heat problem.
    grid = create_embedded_sphere(refinements)

    # Next we are creating our element assembly helper for surface elements.
    # The only change which we need to introduce here is to pass in a geometrical
    # interpolation with the same dimension as the physical space into which our
    # elements are embedded into, which is in this example 3.
    ip = Lagrange{RefTriangle, 1}()
    qr = QuadratureRule{RefTriangle}(2)
    cellvalues = CellValues(qr, ip, ip^3)

    # We have two options to add the reactants to the dof handler, which will give us slightly
    # different resulting dof distributions:
    # A) We can add a scalar-valued interpolation for each reactant.
    # B) We can add one vectorized interpolation whose dimension is the number of reactants
    #    number of reactants.
    # In this tutorial we opt for B, because the dofs are distributed per cell entity -- or
    # to be specific for this tutorial, we use an isoparametric concept such that the nodes
    # of our grid and the nodes of our solution approximation coincide. This way a reaction
    # we can create simply reshape the solution vector u to a matrix where the inner index
    # corresponds to the index of the reactant. Note that we will still use the scalar
    # interpolation for the assembly procedure.
    dh = DofHandler(grid)
    add!(dh, :reactants, ip^2)
    close!(dh)

    # We can save some memory by telling the sparsity pattern that the matrices are not coupled.
    M = allocate_matrix(dh; coupling = [true false; false true])
    D = allocate_matrix(dh; coupling = [true false; false true])

    # Since the heat problem is linear and has no time dependent parameters, we precompute the
    # decomposition of the system matrix to speed up the linear system solver.
    assemble_matrices!(M, D, cellvalues, dh, material)
    A = M + Δt .* D
    cholA = cholesky(A)

    # Now we setup buffers for the time dependent solution and fill the initial condition.
    uₜ = zeros(ndofs(dh))
    uₜ₋₁ = ones(ndofs(dh))
    setup_initial_conditions!(uₜ₋₁, cellvalues, dh)

    # And prepare output for visualization.
    pvd = paraview_collection("reactive-surface")
    VTKGridFile("reactive-surface-0", dh) do vtk
        write_solution(vtk, dh, uₜ₋₁)
        pvd[0.0] = vtk
    end

    # This is now the main solve loop.
    F = material.F
    k = material.k
    for (iₜ, t) in enumerate(Δt:Δt:T)
        # First we solve the heat problem
        uₜ .= cholA \ (M * uₜ₋₁)

        # Then we solve the point-wise reaction problem with the solution of
        # the heat problem as initial guess. 2 is the number of reactants.
        num_individual_reaction_dofs = ndofs(dh) ÷ 2
        rvₜ = reshape(uₜ, (2, num_individual_reaction_dofs))
        for i in 1:num_individual_reaction_dofs
            r₁ = rvₜ[1, i]
            r₂ = rvₜ[2, i]
            rvₜ[1, i] += Δt * (-r₁ * r₂^2 + F * (1 - r₁))
            rvₜ[2, i] += Δt * (r₁ * r₂^2 - r₂ * (F + k))
        end

        # The solution is then stored every 10th step to vtk files for
        # later visualization purposes.
        if (iₜ % 10) == 0
            VTKGridFile("reactive-surface-$(iₜ)", dh) do vtk
                write_solution(vtk, dh, uₜ₋₁)
                pvd[t] = vtk
            end
        end

        # Finally we totate the solution to initialize the next timestep.
        uₜ₋₁ .= uₜ
    end

    return vtk_save(pvd)
end

# This parametrization gives the spot pattern shown in the gif above.
material = GrayScottMaterial(0.00016, 0.00008, 0.06, 0.062)
    gray_scott_on_sphere(material, 10.0, 32000.0, 3)

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl
