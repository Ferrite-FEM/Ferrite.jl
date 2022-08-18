@testset "assemble" begin
    dofs = [1, 3, 5, 7]

    # residual
    ge = rand(4)
    g = zeros(8)
    assemble!(g, dofs, ge)
    @test g[1] == ge[1]
    @test g[3] == ge[2]
    @test g[5] == ge[3]
    @test g[7] == ge[4]

    # stiffness
    a = start_assemble()
    Ke = rand(4, 4)
    assemble!(a, dofs, Ke)
    K = end_assemble(a)
    @test K[1,1] == Ke[1,1]
    @test K[1,5] == Ke[1,3]
    @test K[5,1] == Ke[3,1]

    # assemble with different row and col dofs
    rdofs = [1,4,6]
    cdofs = [1,7]
    a = start_assemble()
    Ke = rand(length(rdofs), length(cdofs))
    assemble!(a, rdofs, cdofs, Ke)
    K = end_assemble(a)
    @test (K[rdofs,cdofs] .== Ke) |> all

end
