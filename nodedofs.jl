using Revise, JuAFEM
grid = generate_grid(Triangle, (1, 1))
addnodeset!(grid, "nodeset", x-> x[2] == -1 || x[1] == -1)

dh = DofHandler(grid)
push!(dh, :u, 2)
push!(dh, :p, 1)
close!(dh)


let ch = ConstraintHandler(dh)
    dbc1 = Dirichlet(:u, getnodeset(grid, "nodeset"), (x,t) -> x, [1, 2])
    dbc2 = Dirichlet(:p, getnodeset(grid, "nodeset"), (x,t) -> 0, 1)
    add!(ch, dbc1)
    add!(ch, dbc2)
    close!(ch)
    update!(ch)
end

ch = ConstraintHandler(dh)
dbc1 = Dirichlet(:u, getnodeset(grid, "nodeset"), (x,t) -> x, [1, 2])
dbc2 = Dirichlet(:p, getnodeset(grid, "nodeset"), (x,t) -> 0, 1)
@code_warntype add!(ch, dbc1)
add!(ch, dbc1)
@code_warntype add!(ch, dbc2)
add!(ch, dbc2)

@code_warntype close!(ch)
close!(ch)
