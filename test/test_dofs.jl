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

@testset "Dofs for Line2" begin

nodes = [Node{2,Float64}(Vec(0.0,0.0)), Node{2,Float64}(Vec(1.0,1.0)), Node{2,Float64}(Vec(2.0,0.0))]
cells = [Line2d((1,2)), Line2d((2,3))]
grid = Grid(cells,nodes)

#2d line with 1st order 1d interpolation
dh = DofHandler(grid)
push!(dh, :x, 2)
close!(dh)

@test celldofs(dh,1) == [1,2,3,4]
@test celldofs(dh,2) == [3,4,5,6]

#2d line with 2nd order 1d interpolation
dh = DofHandler(grid)
push!(dh, :x, 2, Lagrange{1,RefCube,2}()) 
close!(dh)

@test celldofs(dh,1) == [1,2,3,4,5,6]
@test celldofs(dh,2) == [3,4,7,8,9,10]

#3d line with 2nd order 1d interpolation
dh = DofHandler(grid)
push!(dh, :u, 3, Lagrange{1,RefCube,2}()) 
push!(dh, :θ, 3, Lagrange{1,RefCube,2}()) 
close!(dh)

@test celldofs(dh,1) == collect(1:18)
@test celldofs(dh,2) == [4,5,6,19,20,21,22,23,24,    # u
                         13,14,15,25,26,27,28,29,30] # θ
end

@testset "Dofs for quad in 3d (shell)" begin

nodes = [Node{3,Float64}(Vec(0.0,0.0,0.0)), Node{3,Float64}(Vec(1.0,0.0,0.0)), 
            Node{3,Float64}(Vec(1.0,1.0,0.0)), Node{3,Float64}(Vec(1.0,0.0,0.0)),
            Node{3,Float64}(Vec(2.0,0.0,0.0)), Node{3,Float64}(Vec(2.0,2.0,0.0))]

cells = [Quadrilateral3d((1,2,3,4)), Quadrilateral3d((2,5,6,3))]
grid = Grid(cells,nodes)

#3d quad with 1st order 2d interpolation
dh = DofHandler(grid)
push!(dh, :u, 3, Lagrange{2,RefCube,1}())
push!(dh, :θ, 3, Lagrange{2,RefCube,1}())
close!(dh)

@test celldofs(dh,1) == collect(1:24)
@test celldofs(dh,2) == [4,5,6,25,26,27,28,29,30,7,8,9, # u
                         16,17,18,31,32,33,34,35,36,19,20,21]# θ

#3d quad with 2nd order 1d interpolation
dh = DofHandler(grid)
push!(dh, :x, 3, Lagrange{2,RefCube,2}()) 
close!(dh)

@test celldofs(dh,1) == [1,2,3,4,5,6,7,8,9,10,11,12, 13,14,15,16,17,18,19,20,21,22,23,24, 25,26,27]
@test celldofs(dh,2) == [4,5,6,28,29,30,31,32,33,7,8,9, 34,35,36,37,38,39,40,41,42,16,17,18, 43,44,45]

end