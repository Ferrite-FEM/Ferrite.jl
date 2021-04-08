mesh = generate_grid(Quadrilateral, (20, 20))
f(x) = x[1]^2
nodal_vals = [f(p.x) for p in mesh.nodes]
ip = Lagrange{2,RefCube,1}()

dh = MixedDofHandler(mesh)
field = Field(:u, ip, 2)
fh = FieldHandler([field], Set{Int}(1:getncells(mesh)))
push!(dh, fh)
close!(dh)

points = [Vec((x, 0.52)) for x in range(0.0, 1.0, length=100)]

ph = Ferrite.PointEvalHandler(dh, [ip], points)

Ferrite.get_point_values(ph, nodal_vals)
