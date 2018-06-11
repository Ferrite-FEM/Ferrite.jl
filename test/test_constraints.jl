# misc constraint tests
@testset "node bc" begin
    grid = generate_grid(Triangle, (1, 1))
    addnodeset!(grid, "nodeset", x-> x[2] == -1 || x[1] == -1)
    dh = DofHandler(grid)
    push!(dh, :u, 2)
    push!(dh, :p, 1)
    close!(dh)
    ch = ConstraintHandler(dh)
    dbc1 = Dirichlet(:u, getnodeset(grid, "nodeset"), (x,t) -> x, [1, 2])
    dbc2 = Dirichlet(:p, getnodeset(grid, "nodeset"), (x,t) -> 0, 1)
    add!(ch, dbc1)
    add!(ch, dbc2)
    close!(ch)
    update!(ch)

    @test ch.prescribed_dofs == collect(1:9)
    @test ch.values == [-1, -1, 1, -1, -1, 1, 0, 0, 0]
end

using Revise, JuAFEM
grid = generate_grid(Triangle, (2, 2))
dh = DofHandler(grid)
push!(dh, :u, 2)
close!(dh)


ch = ConstraintHandler(dh)
dbc = Dirichlet(:u, getfaceset(grid, "left"), (x,t) -> 0, [1, 2])
add!(ch, dbc)
close!(ch)
update!(ch)
