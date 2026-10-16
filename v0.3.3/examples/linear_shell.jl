using Ferrite
using ForwardDiff

function main() #wrap everything in a function...

nels = (10,10)
size = (10.0, 10.0)
grid = generate_shell_grid(nels, size)

ip = Lagrange{2,RefCube,1}()
qr_inplane = QuadratureRule{2,RefCube}(1)
qr_ooplane = QuadratureRule{1,RefCube}(2)
cv = CellScalarValues(qr_inplane, ip)

dh = DofHandler(grid)
push!(dh, :u, 3, ip)
push!(dh, :θ, 2, ip)
close!(dh)

addedgeset!(grid, "left",  (x) -> x[1] ≈ 0.0)
addedgeset!(grid, "right", (x) -> x[1] ≈ size[1])
addvertexset!(grid, "corner", (x) -> x[1] ≈ 0.0 && x[2] ≈ 0.0 && x[3] ≈ 0.0)

ch = ConstraintHandler(dh)
add!(ch,  Dirichlet(:u, getedgeset(grid, "left"), (x, t) -> (0.0, 0.0), [1,3])  )
add!(ch,  Dirichlet(:θ, getedgeset(grid, "left"), (x, t) -> (0.0, 0.0), [1,2])  )

add!(ch,  Dirichlet(:u, getedgeset(grid, "right"), (x, t) -> (0.0, 0.0), [1,3])  )
add!(ch,  Dirichlet(:θ, getedgeset(grid, "right"), (x, t) -> (0.0, pi/10), [1,2])  )

add!(ch,  Dirichlet(:θ, getvertexset(grid, "corner"), (x, t) -> (0.0), [2])  )

close!(ch)
update!(ch, 0.0)

κ = 5/6 # Shear correction factor
E = 210.0
ν = 0.3
a = (1-ν)/2
C = E/(1-ν^2) * [1 ν 0   0   0;
                ν 1 0   0   0;
                0 0 a*κ 0   0;
                0 0 0   a*κ 0;
                0 0 0   0   a*κ]


data = (thickness = 1.0, C = C); #Named tuple

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

    #Call the element routine
    integrate_shell!(ke, cv, qr_ooplane, cellcoords, data)

    assemble!(assembler, celldofs, fe, ke)
end

apply!(K, f, ch)
a = K\f

vtk_grid("linear_shell", dh) do vtk
    vtk_point_data(vtk, dh, a)
end

end; #end main functions

function generate_shell_grid(nels, size)
    _grid = generate_grid(Quadrilateral, nels, Vec((0.0,0.0)), Vec(size))
    nodes = [(n.x[1], n.x[2], 0.0) |> Vec{3} |> Node  for n in _grid.nodes]
    cells = [Quadrilateral3D(cell.nodes) for cell in _grid.cells]

    grid = Grid(cells, nodes)

    return grid
end;

function fiber_coordsys(Ps::Vector{Vec{3,Float64}})

    ef1 = Vec{3,Float64}[]
    ef2 = Vec{3,Float64}[]
    ef3 = Vec{3,Float64}[]
    for P in Ps
        a = abs.(P)
        j = 1
        if a[1] > a[3]; a[3] = a[1]; j = 2; end
        if a[2] > a[3]; j = 3; end

        e3 = P
        e2 = Tensors.cross(P, basevec(Vec{3}, j))
        e2 /= norm(e2)
        e1 = Tensors.cross(e2, P)

        push!(ef1, e1)
        push!(ef2, e2)
        push!(ef3, e3)
    end
    return ef1, ef2, ef3

end;

function lamina_coordsys(dNdξ, ζ, x, p, h)

    e1 = zero(Vec{3})
    e2 = zero(Vec{3})

    for i in 1:length(dNdξ)
        e1 += dNdξ[i][1] * x[i] + 0.5*h*ζ * dNdξ[i][1] * p[i]
        e2 += dNdξ[i][2] * x[i] + 0.5*h*ζ * dNdξ[i][1] * p[i]
    end

    e1 /= norm(e1)
    e2 /= norm(e2)

    ez = Tensors.cross(e1,e2)
    ez /= norm(ez)

    a = 0.5*(e1 + e2)
    a /= norm(a)

    b = Tensors.cross(ez,a)
    b /= norm(b)

    ex = sqrt(2)/2 * (a - b)
    ey = sqrt(2)/2 * (a + b)

    return Tensor{2,3}(hcat(ex,ey,ez))
end;

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
end;

function strain(dofvec::Vector{T}, N, dNdx, ζ, dζdx, q, ef1, ef2, h) where T

    u = reinterpret(Vec{3,T}, dofvec[1:12])
    θ = reinterpret(Vec{2,T}, dofvec[13:20])

    dudx = zeros(T, 3, 3)
    for m in 1:3, j in 1:3
        for a in 1:length(N)
            dudx[m,j] += dNdx[a][j] * u[a][m] + h/2 * (dNdx[a][j]*ζ + N[a]*dζdx[j]) * (θ[a][2]*ef1[a][m] - θ[a][1]*ef2[a][m])
        end
    end

    dudx = q*dudx
    ε = [dudx[1,1], dudx[2,2], dudx[1,2]+dudx[2,1], dudx[2,3]+dudx[3,2], dudx[1,3]+dudx[3,1]]
    return ε

end;

function integrate_shell!(ke, cv, qr_ooplane, X, data)
    nnodes = getnbasefunctions(cv)
    ndofs = nnodes*5
    h = data.thickness

    #Create the directors in each node.
    #Note: For a more general case, the directors should
    #be input parameters for the element routine.
    p = zeros(Vec{3}, nnodes)
    for i in 1:nnodes
        a = Vec{3}((0.0, 0.0, 1.0))
        p[i] = a/norm(a)
    end

    ef1, ef2, ef3 = fiber_coordsys(p)

    for iqp in 1:getnquadpoints(cv)

        dNdξ = cv.dNdξ[:,iqp]
        N = cv.N[:,iqp]

        for oqp in 1:length(qr_ooplane.weights)

            ζ = qr_ooplane.points[oqp][1]

            q = lamina_coordsys(dNdξ, ζ, X, p, h)
            J = getjacobian(q, N, dNdξ, ζ, X, p, h)
            Jinv = inv(J)

            dζdx = Vec{3}((0.0, 0.0, 1.0)) ⋅ Jinv
            dNdx = [Vec{3}((dNdξ[i][1], dNdξ[i][2], 0.0)) ⋅ Jinv for i in 1:nnodes]


            #For simplicity, use automatic differentiation to construct the B-matrix from the strain.
            B = ForwardDiff.jacobian(
                (a) -> strain(a, N, dNdx, ζ, dζdx, q, ef1, ef2, h), zeros(Float64, ndofs) )

            dV = det(J) * cv.qr_weights[iqp] * qr_ooplane.weights[oqp]
            ke .+= B'*data.C*B * dV
        end
    end
end;

main()

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl

