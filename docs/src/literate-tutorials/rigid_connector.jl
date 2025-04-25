using Ferrite
using FerriteGmsh


grid = togrid("docs/src/literate-tutorials/plate_hole.geo")

grid = Grid(
    Ferrite.AbstractCell[grid.cells...],
    grid.nodes,
    facetsets = grid.facetsets,
    cellsets = grid.cellsets
)
rigidbody_node = Node(Vec((5.0, 5.0)))
rigidbody_cellid = getncells(grid) + 1
push!(grid.nodes, rigidbody_node)
push!(grid.cells, Ferrite.Point((getnnodes(grid),)))

addcellset!(grid, "rigidbody", [getncells(grid)])
addvertexset!(grid, "rigidvertex", x -> x ≈ rigidbody_node.x)

ip_u = Lagrange{RefTriangle, 1}()^2
ip_rb_u = Lagrange{Ferrite.RefPoint, 1}()^2
ip_rb_θ = Lagrange{Ferrite.RefPoint, 1}()

qr = QuadratureRule{RefTriangle}(2)
cellvalues = CellValues(qr, ip_u)

dh = DofHandler(grid)
sdh = SubDofHandler(dh, getcellset(grid, "PlateWithHole"))
add!(sdh, :u, ip_u)

sdh = SubDofHandler(dh, getcellset(grid, "rigidbody"))
add!(sdh, :u, ip_rb_u)
add!(sdh, :θ, ip_rb_θ)
close!(dh)

ch = ConstraintHandler(dh)
rb = Ferrite.RigidConnector(;
    rigidbody_cellid = rigidbody_cellid,
    facets = getfacetset(grid, "HoleInterior"),
)
add!(ch, rb)
add!(ch, Dirichlet(:u, getfacetset(grid, "PlateRightLeft"), x -> (0.0, 0.0)))
add!(ch, Dirichlet(:u, getvertexset(grid, "rigidvertex"), x -> (0.0, 0.0)))
add!(ch, Dirichlet(:θ, getvertexset(grid, "rigidvertex"), x -> (0.1)))
close!(ch)

Emod = 200.0e3 # Young's modulus [MPa]
ν = 0.3        # Poisson's ratio [-]
Gmod = Emod / (2(1 + ν))  # Shear modulus
Kmod = Emod / (3(1 - 2ν)) # Bulk modulus
C = gradient(ϵ -> 2 * Gmod * dev(ϵ) + 3 * Kmod * vol(ϵ), zero(SymmetricTensor{2, 2}));

function assemble_cell!(ke, cellvalues, C)
    for q_point in 1:getnquadpoints(cellvalues)
        ## Get the integration weight for the quadrature point
        dΩ = getdetJdV(cellvalues, q_point)
        for i in 1:getnbasefunctions(cellvalues)
            ## Gradient of the test function
            ∇Nᵢ = shape_gradient(cellvalues, q_point, i)
            for j in 1:getnbasefunctions(cellvalues)
                ## Symmetric gradient of the trial function
                ∇ˢʸᵐNⱼ = shape_symmetric_gradient(cellvalues, q_point, j)
                ke[i, j] += (∇Nᵢ ⊡ C ⊡ ∇ˢʸᵐNⱼ) * dΩ
            end
        end
    end
    return ke
end

function assemble_global!(K, dh, cellvalues, C, cellset)
    ## Allocate the element stiffness matrix
    n_basefuncs = getnbasefunctions(cellvalues)
    ke = zeros(n_basefuncs, n_basefuncs)
    ## Create an assembler
    assembler = start_assemble(K)
    ## Loop over all cells
    for cell in CellIterator(dh, cellset)
        ## Update the shape function gradients based on the cell coordinates
        reinit!(cellvalues, cell)
        ## Reset the element stiffness matrix
        fill!(ke, 0.0)
        ## Compute element contribution
        assemble_cell!(ke, cellvalues, C)
        ## Assemble ke into K
        assemble!(assembler, celldofs(cell), ke)
    end
    return K
end

K = allocate_matrix(dh, ch)
f_ext = zeros(Float64, ndofs(dh))

assemble_global!(K, dh, cellvalues, C, getcellset(grid, "PlateWithHole"));

apply!(K, f_ext, ch)
a = K \ f_ext;

apply!(a, ch)


function calculate_stresses(grid, dh, cv, u, C, cellset)
    qp_stresses = [
        [zero(SymmetricTensor{2, 2}) for _ in 1:getnquadpoints(cv)]
            for _ in 1:getncells(grid)
    ]
    avg_cell_stresses = tuple((zeros(length(cellset)) for _ in 1:3)...)
    for cell in CellIterator(dh, cellset)
        reinit!(cv, cell)
        cell_stresses = qp_stresses[cellid(cell)]
        for q_point in 1:getnquadpoints(cv)
            ε = function_symmetric_gradient(cv, q_point, u, celldofs(cell))
            cell_stresses[q_point] = C ⊡ ε
        end
        σ_avg = sum(cell_stresses) / getnquadpoints(cv)
        avg_cell_stresses[1][cellid(cell)] = σ_avg[1, 1]
        avg_cell_stresses[2][cellid(cell)] = σ_avg[2, 2]
        avg_cell_stresses[3][cellid(cell)] = σ_avg[1, 2]
    end
    return qp_stresses, avg_cell_stresses
end

qp_stresses, avg_cell_stresses = calculate_stresses(grid, dh, cellvalues, a, C, getcellset(grid, "PlateWithHole"));

# We now use the the L2Projector to project the stress-field onto the piecewise linear
# finite element space that we used to solve the problem.
proj = L2Projector(grid)
add!(proj, getcellset(grid, "PlateWithHole"), ip_u; qr_rhs = qr)
close!(proj)

projected = project(proj, qp_stresses)

VTKGridFile("rigid_con", grid) do vtk
    write_solution(vtk, dh, a)
    write_projection(vtk, proj, projected, "stress")
end
