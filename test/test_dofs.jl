# misc dofhandler unit tests
@testset "dofs" begin

# set up a test DofHandler
grid = generate_grid(Triangle, (10, 10))
dh = DofHandler(grid)
push!(dh, :u, 2, Lagrange{2,RefTetrahedron,2}())
push!(dh, :p, 1, Lagrange{2,RefTetrahedron,1}())
close!(dh)

# renumber!
perm = randperm(ndofs(dh))
iperm = invperm(perm)
dofs = copy(dh.cell_dofs)
JuAFEM.renumber!(JuAFEM.renumber!(dh, perm), iperm)
@test dofs == dh.cell_dofs


# dof_range
@test (@inferred dof_range(dh, :u)) == 1:12
@test (@inferred dof_range(dh, :p)) == 13:15
# dof_range for FieldHandler (use with MixedDofHandler)
ip = Lagrange{2, RefTetrahedron, 1}()
field_u = Field(:u, ip, 2)
field_c = Field(:c, ip, 1)
fh = FieldHandler([field_u, field_c], Set(1:getncells(grid)))
@test dof_range(fh, :u) == 1:6
@test dof_range(fh, :c) == 7:9

end # testset
