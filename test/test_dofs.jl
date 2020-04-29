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

end # testset

@testset "dofs Line2" begin

nodes = [Node{2,Float64}(Vec(0.0,0.0)), Node{2,Float64}(Vec(1.0,1.0)), Node{2,Float64}(Vec(2.0,0.0))]
cells = [Line2((1,2)), Line2((2,3))]
grid = Grid(cells,nodes)

dh = DofHandler(grid)
push!(dh, :x, 2) #2d line with 1d interpolation
close!(dh)

@test celldofs(dh,1) == [1,2,3,4]
@test celldofs(dh,2) == [3,4,5,6]

end