# isoparametric approximation
mesh = generate_grid(QuadraticQuadrilateral, (20, 20))
f(x) = x[1]^2
nodal_vals = [f(p.x) for p in mesh.nodes]
ip_f = Lagrange{2,RefCube,2}()
ip_g = Lagrange{2,RefCube,2}()

dh = MixedDofHandler(mesh)
field = Field(:u, ip, 2)
fh = FieldHandler([field], Set{Int}(1:getncells(mesh)))
push!(dh, fh)
close!(dh)

points = [Vec((x, 0.52)) for x in range(0.0, 1.0, length=100)]

ph = Ferrite.PointEvalHandler(dh, [ip_f], points, [ip_g])

vals = Ferrite.get_point_values(ph, nodal_vals)

# can recover a quadratic field by a quadratic approximation
f.(points) ≈ vals

#############################################################
# superparametric approximation
mesh = generate_grid(QuadraticQuadrilateral, (20, 20))
f(x) = x[1]^2
nodal_vals = [f(p.x) for p in mesh.nodes]

ip_f = Lagrange{2,RefCube,2}()
ip_g = Lagrange{2,RefCube,2}()

# do a L2Projection for getting values in dofs
projector = L2Projector(ip_f, mesh)

qr = QuadratureRule{2, RefCube}(2) # exactly approximate quadratic field
cv = CellScalarValues(qr, ip_f, ip_g)
qp_vals = [Vector{Float64}(undef, getnquadpoints(cv)) for i=1:getncells(mesh)]
for cellid in eachindex(mesh.cells)
    xe = getcoordinates(mesh, cellid)
    reinit!(cv, xe)
    for qp in 1:getnquadpoints(cv)
        qp_vals[cellid][qp] = f(spatial_coordinate(cv, qp, xe))
    end
end
dof_vals = project(projector, qp_vals, qr; project_to_nodes=false)

ph = Ferrite.PointEvalHandler(projector.dh, [ip_f], points, [ip_g])
vals = Ferrite.get_point_values(ph, dof_vals, projector)

f.(points) ≈ vals

nodal_vals = project(projector, qp_vals, qr; project_to_nodes=true)

## MWE for L2Projection bug
mesh = generate_grid(Quadrilateral, (2, 2))
f(x) = x[1]
ip_f = Lagrange{2,RefCube,1}()
ip_g = Lagrange{2,RefCube,1}()

projector = L2Projector(ip_f, mesh)

qr = QuadratureRule{2, RefCube}(2) # exactly approximate quadratic field
cv = CellScalarValues(qr, ip_f, ip_g)
qp_vals = [Vector{Float64}(undef, getnquadpoints(cv)) for i=1:getncells(mesh)]
for cellid in eachindex(mesh.cells)
    xe = getcoordinates(mesh, cellid)
    reinit!(cv, xe)
    for qp in 1:getnquadpoints(cv)
        qp_vals[cellid][qp] = f(spatial_coordinate(cv, qp, xe))
    end
end

projected_vals = project(projector, qp_vals, qr; project_to_nodes=true)