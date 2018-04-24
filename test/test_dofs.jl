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

end # testset
