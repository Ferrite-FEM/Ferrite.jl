@testset "fevalues" begin
    a = start_assemble()
    dofs = [1, 3, 5, 7]
    Ke = rand(4, 4)
    assemble(dofs, a, Ke)
    K = end_assemble(a)
    @test K[1,1] == Ke[1,1]
    @test K[1,5] == Ke[1,3]
    @test K[5,1] == Ke[3,1]
end
