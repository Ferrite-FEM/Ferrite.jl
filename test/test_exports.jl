@testset "Exports" begin
    # Test that all exported symbols are defined
    @test all(x -> isdefined(Ferrite, x), names(Ferrite))
    # # See which is not defined if fails
    # for name in names(Ferrite)
    #     isdefined(Ferrite, name) || @warn "Ferrite.$name is not defined but $name is exported"
    # end
end
