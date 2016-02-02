facts("Constraints") do

context("blah") do

    c = Constraints()

    add_constraint!(c, 1, 2.0)

    @fact is_constrained(c, 1) --> true
    @fact get_value(c, 1) --> 2.0

    indices = [3,5,7]
    values = [2.0, 5.0, 8.0]

    add_constraint!(c, indices, values)

    @fact is_constrained(c, 5) --> true
    @fact get_value(c, 7) --> 8.0

    f = zeros(5)

    edof = [2, 1, 3, 4, 5]
    fe = [1.0, 2.0, 3.0, 4.0]

    assemble(edof, f, fe, c)

    @fact f[1] --> get_value(c, 1)
    @fact f[3] --> get_value(c, 3)
    @fact f[4] --> 3.0
    @fact f[5] --> get_value(c, 5)


    c = Constraints()

    add_constraint!(c, 1, 2.0)
    add_constraint!(c, 3, 4.0)

    a = start_assemble()

    Ke = rand(4,4)
    edof2 = [2, 2, 3, 4, 5]

    f = zeros(5)

    edof = [2, 1, 3, 4, 5]
    fe = [1.0, 2.0, 3.0, 4.0]
    fe2 = [2.0, 3.0, 4.0, 5.0]

    assemble(edof, f, fe, c)
    assemble(edof2, f, fe2, c)

    assemble(edof, a, Ke, c)
    assemble(edof2, a, Ke, c)

    K = end_assemble(a)

    println(a.constrained_dofs_added)

    u = K \ f

    println(f)

    @fact u[1] --> get_value(c, 1)
    @fact u[3] --> get_value(c, 3)

    println(full(K))


end


end
