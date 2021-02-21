# # Linear shell
#
# ## Introduction
# 
# In this example we show how shell elements can be implemented in JuAFEM. 
# The first part of the tutorial explains how to distribute dofs (displacement and rotational)
# and how boundary condition can be applied to edges and vertices.
# In the second part, we give an brief description on the shell element used for this tutorial.

# ## Set up of the problem

using JuAFEM
using ForwardDiff

# Define a main function which we later call to solve the problem

function main()

    # Next we generate a mesh. There is currently no builet-in function for generating 
    # a shell mesh, so we create one our. See `generate_shell_grid` further down in this file.
    nels = (10,10)
    size = (10.0, 5.0)
    grid = generate_shell_grid(nels, size)

    # Here we define the interpolation and precompute the shape values/derivatives using `CellScalarValues`. 
    # Bi-Linear shape functions inplane is used (note how 2d-interpolation and quadrature rule is used).
    # A quadrature rule for the out-of-plane direction is also defined here. 
    ip = Lagrange{2,RefCube,1}()
    qr_inplane = QuadratureRule{2,RefCube}(2)
    cv = CellScalarValues(qr_inplane, ip)

    qr_ooplane = QuadratureRule{1,RefCube}(2)

    # Distribute displacement dofs, `u = (x,y,z)` and rotation-dofs, `:θ = (θ₁,  θ₂)`.
    dh = DofHandler(grid)
    push!(dh, :u, 3, ip)
    push!(dh, :θ, 2, ip)
    close!(dh)

    # Next we define the boundary conditions. 
    #
    # On the left edge, we lock the displacements in the x- and z- directions, and the rotations.
    ch = ConstraintHandler(dh)
    dbc = Dirichlet(:u, getedgeset(grid, "left"), (x, t) -> (0.0,0.0), [1,3])
    add!(ch, dbc)
    dbc = Dirichlet(:θ, getedgeset(grid, "left"), (x, t) -> (0.0,0.0), [1,2])
    add!(ch, dbc)

    # On the right edge, we also lock the displacements in the x- and z- directions, but apply a 
    # precribed roation.
    dbc = Dirichlet(:u, getedgeset(grid, "right"), (x, t) -> (0.0,0.0), [1,3])
    add!(ch, dbc)
    dbc = Dirichlet(:θ, getedgeset(grid, "right"), (x, t) -> (pi/10,0.0), [1,2])
    add!(ch, dbc)

    # In order to not get rigid body motion, we lock the y-displacement in one fo the corners (`getvertexset).
    dbc = Dirichlet(:θ, getvertexset(grid, "corner"), (x, t) -> (0.0), [2])
    add!(ch, dbc)

    close!(ch)
    update!(ch, 0.0)


    # Now we define the sitffness matrix for the material. In this linear shell, plane stress
    # is assumed, ie $\\sigma_{zz} = 0 $. Thererfor, the stiffness matrix is 5x5 (opposed to the normal 6x6.)
    
    E = 210.0
    ν = 0.3
    a = (1-ν)/2
    material = E/(1-ν^2) * [1 ν 0 0 0;
                            ν 1 0 0 0;
                            0 0 a 0 0;
                            0 0 0 a 0;
                            0 0 0 0 a]
    
    # Define the a named tuple with some data related to the shell: thickness, shear_factor, and material.
    data = (thickness = 1.0, C = material, shear_factor = 5/6)
    
    # Main assembly loop
    nnodes = getnbasefunctions(ip)
    ndofs_shell = ndofs_per_cell(dh)

    K = create_sparsity_pattern(dh)
    f = zeros(Float64, ndofs(dh))
   
    ke = zeros(ndofs_shell, ndofs_shell)
    fe = zeros(ndofs_shell) 

    celldofs = zeros(Int, ndofs_shell)
    cellcoords = zeros(Vec{3,Float64}, nnodes)

    assembler = start_assemble(K, f)
    for cellid in 1:getncells(grid)
        fill!(ke, 0.0)

        celldofs!(celldofs, dh, cellid)
        getcoordinates!(cellcoords, grid, cellid)

        integrate_shell!(ke, cv, qr_ooplane, cellcoords, data)

        assemble!(assembler, celldofs, fe, ke)
    end

    # Apply BC and solve.
    apply!(K, f, ch)
    a = K\f

    # Output results.
    vtk_grid("linear_shell", dh) do vtk
        vtk_point_data(vtk, dh, a)
    end
end

# Since there is no builed-in function for generating a shell mesh, we have to 
# create a function here. It simply generates a 2d-quadrature mesh, and appends
# a third coordinate (z-direction) to the node-positions. It also adds 
# edge- and vertex-sets.
function generate_shell_grid(nels, size)

    _grid = generate_grid(Quadrilateral, nels, Vec((0.0,0.0)), Vec(size))
    nodes = [(n.x[1], n.x[2], 0.0) |> Vec{3} |> Node  for n in _grid.nodes]
    cells = [Quadrilateral3D(cell.nodes) for cell in _grid.cells]

    grid = Grid(cells, nodes)

    addedgeset!(grid, "left", (x) -> x[1] ≈ 0.0)
    addedgeset!(grid, "right", (x) -> x[1] ≈ size[1])
    addvertexset!(grid, "corner", (x) -> x[1] ≈ 0.0 && x[2] ≈ 0.0 && x[3] ≈ 0.0)
    return grid
end

JuAFEM.cell_to_vtkcell(::Type{Quadrilateral3D}) = VTKCellTypes.VTK_QUAD

# ## The shell element. 

function integrate_shell!(ke, cv, qr_ooplane, X, data)
    
    nnodes = getnbasefunctions(cv)
    ndofs = nnodes*5
    h = data.thickness

    p = zeros(Vec{3}, nnodes)
    for i in 1:nnodes
        a = Vec{3}((0.0, 0.0, 1.0)) 
        p[i] = a/norm(a)
    end

    fibercoord = fiber_coordsys.(p)
    ef1 = getindex.(fibercoord, 1)
    ef2 = getindex.(fibercoord, 2)

    for iqp in 1:getnquadpoints(cv)

        dNdξ = [Vec{3}((cv.dNdξ[i,iqp]..., 0.0)) for i in 1:nnodes]
        N = cv.N[:,iqp]

        for oqp in 1:length(qr_ooplane.weights)

            ζ = qr_ooplane.points[oqp][1]
            
            q = lamina_coordsys(dNdξ, ζ, X, p, h)
            J = getjacobian(q, N, dNdξ, ζ, X, p, h)
            Jinv = inv(J)   
            
            dζdx = Vec{3}((0.0, 0.0, 1)) ⋅ Jinv
            dNdx = [dNdξ[i] ⋅ Jinv for i in 1:nnodes]

            B = ForwardDiff.jacobian( (a) -> strain(a, N, dNdx, ζ, dζdx, q, ef1, ef2, h), zeros(Float64, ndofs) )

            dV = det(J) * cv.qr_weights[iqp] * qr_ooplane.weights[oqp]
            ke .+= B'*data.C*B * dV
        end
    end
end

function strain(dofvec::Vector{T}, N, dNdx, ζ, dζdx, q, ef1, ef2, h) where T

    u = reinterpret(Vec{3,T}, dofvec[1:12])
    θ = reinterpret(Vec{2,T}, dofvec[13:20])

    dudx = zeros(T, 3, 3)
    for a in 1:length(N)
        for i in 1:3, j in 1:3
            dudx[i,j] += dNdx[a][j] * u[a][i] + h/2 * (dNdx[a][j]*ζ + N[a]*dζdx[j]) * (θ[a][1]*ef1[a][i] - θ[a][2]*ef2[a][i])
        end
    end

    dudx = q'*dudx
    ε = [dudx[1,1], dudx[2,2], dudx[1,2]+dudx[2,1], dudx[2,3]+dudx[3,2], dudx[1,3]+dudx[3,1]]
    return ε

end

function getjacobian(q, N, dNdξ, ζ, X, p, h)

    J = zeros(3,3)
    for a in 1:length(N)
        for i in 1:3, j in 1:3
            _dNdξ = (j==3) ? 0.0 : dNdξ[a][j]
            _dζdξ = (j==3) ? 1.0 : 0.0
            _N = N[a]

            J[i,j] += _dNdξ * X[a][i]  +  (_dNdξ*ζ + _N*_dζdξ) * h/2 * p[a][i]
        end
    end

    return (q' * J) |> Tensor{2,3,Float64}
end

function lamina_coordsys(dNdξ, ζ, x, p, h)

    e1 = zero(Vec{3})
    e2 = zero(Vec{3})

    for i in 1:length(dNdξ)
        e1 += dNdξ[i][1] * x[i] + 0.5*h*ζ * dNdξ[i][1] * p[i]
        e2 += dNdξ[i][2] * x[i] + 0.5*h*ζ * dNdξ[i][1] * p[i]
    end

    ez = Tensors.cross(e1,e2)
    ez /= norm(ez)

    a = e1/norm(e1) + e2/norm(e2)
    b = Tensors.cross(ez,a)

    ex = (a-b)/norm(a-b)
    ey = (a+b)/norm(a+b)

    return Tensor{2,3}(hcat(ex,ey,ez))
end

function fiber_coordsys(P)

    a = collect(P)
    j = 1
    if a[1] > a[3]; a[3] = a[1]; j = 2; end
    if a[2] > a[3]; j = 3; end
    
    e3 = P
    e2 = Tensors.cross(P, basevec(Vec{3}, j))
    e2 /= norm(e2)
    e1 = Tensors.cross(P, e2)

    return e1, e2

end
