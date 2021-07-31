
# isoparametric approximation
mesh = generate_grid(QuadraticQuadrilateral, (20, 20))
f(x) = x[1]^2
nodal_vals = [f(p.x) for p in mesh.nodes]

ip_f = Lagrange{2,RefCube,2}() # function interpolation
ip_g = Lagrange{2,RefCube,2}() # geometry interpolation

# compute values in quadrature points
qr = QuadratureRule{2, RefCube}(3) # exactly approximate quadratic field
cv = CellScalarValues(qr, ip_f, ip_g)
qp_vals = [Vector{Float64}(undef, getnquadpoints(cv)) for i=1:getncells(mesh)]
for cellid in eachindex(mesh.cells)
    xe = getcoordinates(mesh, cellid)
    reinit!(cv, xe)
    for qp in 1:getnquadpoints(cv)
        qp_vals[cellid][qp] = f(spatial_coordinate(cv, qp, xe))
    end
end

# do a L2Projection for getting values in dofs
projector = L2Projector(ip_f, mesh)
dof_vals = project(projector, qp_vals, qr; project_to_nodes=false)

# points where we want to retrieve field values
points = [Vec((x, 0.52)) for x in range(0.0, 1.0, length=100)]

# set up PointEvalHandler and retrieve values
ph = Ferrite.PointEvalHandler(projector.dh, [ip_f], points, [ip_g])
vals = Ferrite.get_point_values(ph, dof_vals, projector)

@test f.(points) ≈ vals




# superparametric approximation
mesh = generate_grid(QuadraticQuadrilateral, (20, 20))
f(x) = x[1]^2
nodal_vals = [f(p.x) for p in mesh.nodes]
ip_f = Lagrange{2,RefCube,2}() # function interpolation
ip_g = Lagrange{2,RefCube,1}() # geometry interpolation

# construct MixedDofHandler
dh = MixedDofHandler(mesh)
field = Field(:u, ip_f, 2) # the interpolation here is unused
fh = FieldHandler([field], Set{Int}(1:getncells(mesh)))
push!(dh, fh)
close!(dh)

# points where we want to retrieve field values
points = [Vec((x, 0.52)) for x in range(0.0, 1.0, length=100)]

# set up PointEvalHandler and retrieve values
ph = Ferrite.PointEvalHandler(dh, [ip_f], points, [ip_g])
vals = Ferrite.get_point_values(ph, nodal_vals)

# can recover a quadratic field by a quadratic approximation
@test f.(points) ≈ vals




## Mixed grid where not all cells have the same fields 

# 5_______6
# |\      | 
# |   \   |
# 3______\4
# |       |
# |       |
# 1_______2 

nodes = [Node((0.0, 0.0)),
        Node((1.0, 0.0)),
        Node((0.0, 1.0)),
        Node((1.0, 1.0)),
        Node((0.0, 2.0)),
        Node((1.0, 2.0))]

cells = Ferrite.AbstractCell[Quadrilateral((1,2,4,3)),
        Triangle((3,4,6)),
        Triangle((3,6,5))]

mesh = Grid(cells, nodes)
addcellset!(mesh, "quads", Set{Int}((1,)))
addcellset!(mesh, "tris", Set{Int}((2, 3)))

ip_quad = Lagrange{2,RefCube,1}()
ip_tri = Lagrange{2,RefTetrahedron,1}()

f(x) = x[1]

# compute values in quadrature points for quad
qr = QuadratureRule{2, RefCube}(2)
cv = CellScalarValues(qr, ip_quad)
qp_vals_quads = [Vector{Float64}(undef, getnquadpoints(cv)) for cell in getcellset(mesh, "quads")]
for (local_cellid, global_cellid) in enumerate(getcellset(mesh, "quads"))
    xe = getcoordinates(mesh, global_cellid)
    reinit!(cv, xe)
    for qp in 1:getnquadpoints(cv)
        qp_vals_quads[local_cellid][qp] = f(spatial_coordinate(cv, qp, xe))
    end
end

# construct projector 
projector = L2Projector(ip_quad, mesh; set=getcellset(mesh, "quads"))

points = [Vec((x, 2x)) for x in range(0.0, 1.0, length=10)]

# first alternative: project to nodes and interpolate from nodal values
dof_vals = project(projector, qp_vals_quads, qr; project_to_nodes = false)
peh = PointEvalHandler(projector.dh, [ip_quad], points)
vals = Ferrite.get_point_values(peh, dof_vals, projector)

# second alternative: project to nodes, but obtain dof-order
# this is for the case of getting values from the result of a simulation
# construct DofHandler
dh = MixedDofHandler(mesh)
field = Field(:q, ip_quad, 2) # the interpolation here is unused
fh_quad = FieldHandler([field], getcellset(mesh, "quads"))
push!(dh, fh_quad)
field = Field(:t, ip_tri, 2) # the interpolation here is unused
fh_tri = FieldHandler([field], getcellset(mesh, "tris"))
push!(dh, fh_tri)
close!(dh)

nodal_vals = project(projector, qp_vals_quads, qr)
peh = PointEvalHandler(dh, [ip_quad, ip_tri], points)
vals = Ferrite.get_point_values(peh, nodal_vals)

mesh = generate_grid(QuadraticQuadrilateral, (20, 20))
dh = DofHandler(mesh)
push!(dh,:u,1)
close!(dh)
points = [Vec((x, 0.52)) for x in range(0.0, 1.0, length=100)]
peh = Ferrite.PointEvalHandler(dh,[ip_f],points)
f(x) = x[1]^2
nodal_vals = [f(p.x) for p in mesh.nodes]
vals = Ferrite.get_point_values(peh, nodal_vals)
