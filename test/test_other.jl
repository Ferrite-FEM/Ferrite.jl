
@testset "utils/other" begin
    K = sparse([1,2], [1,2], [1.0, 2.0], 2, 2)
    
    Ferrite._addindex_sparsematrix!(K, 2.0, 1, 1)
    @test K[1,1] = 3.0
    #@test_throws Ferrite._addindex_sparsematrix!(K, 2.0, 2, 1)
end